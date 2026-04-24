#!/usr/bin/env python3
"""
test_filter_moving.py — 动态点云过滤测试
=========================================
原理：
    对比相邻两帧的「世界坐标点云」，利用局部深度差异找到动态区域。
    由于相机静止，静态像素的背投影世界坐标不会变化；
    动态区域（如正在运动的门）对应像素深度会改变，差异超过阈值即视为动态。

算法流程：
    1. 将每帧 Depth Map 反投影为世界坐标点云。
    2. 对 (prev_depth, curr_depth) 做差，取绝对值。
    3. 差异 > DEPTH_DIFF_THRESH 的像素视为动态区域。
    4. 从当前帧点云中提取动态区域的 3D 点。
    5. 使用 OpenCV 实时展示动态 mask，并用 rerun 展示 3D 动态点云。

使用方法：
    python test_filter_moving.py
    python test_filter_moving.py --data_dir ../record_pull_arc_arm_invisible
    python test_filter_moving.py --diff_thresh 0.01 --min_area 500
"""

import sys
import argparse
import math
import time
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# ── 可选：Rerun 3D 可视化 ──────────────────────────────────────────────────────
try:
    import rerun as rr
    HAVE_RERUN = True
except ImportError:
    HAVE_RERUN = False
    print("[WARN] rerun-sdk 未安装，跳过 3D 可视化。")

# =============================================================================
# 数据加载
# =============================================================================

def load_camera_info(data_dir: Path):
    info = {}
    with open(data_dir / "camera_info.txt") as f:
        for line in f:
            key, _, val = line.strip().partition(":")
            info[key.strip()] = float(val.strip())
    W = int(info["width"]); H = int(info["height"])
    fx, fy = info["fx"], info["fy"]
    cx, cy = info["cx"], info["cy"]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return W, H, K


def load_camera_poses(data_dir: Path):
    """返回 {frame_id: c2w_4x4} 字典，c2w = camera-to-world。"""
    poses = {}
    with open(data_dir / "camera_pose.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            fid = int(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = rot
            c2w[:3, 3] = [tx, ty, tz]
            poses[fid] = c2w
    return poses


def depth_to_world_points(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray):
    """
    将深度图反投影为世界坐标点云。
    返回 (H, W, 3) float32，invalid 区域为 [0,0,0]。
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    valid = depth > 0

    # 相机坐标 (OpenCV 约定：z 向前)
    x_cam = np.where(valid, (uu - cx) / fx * depth, 0.0)
    y_cam = np.where(valid, (vv - cy) / fy * depth, 0.0)
    z_cam = np.where(valid, depth, 0.0)

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
    pts_flat = pts_cam.reshape(-1, 3)

    # 转到世界坐标
    ones = np.ones((pts_flat.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts_flat, ones])  # (N, 4)
    pts_world = (c2w @ pts_h.T).T[:, :3].reshape(H, W, 3).astype(np.float32)

    pts_world[~valid] = 0.0
    return pts_world


# =============================================================================
# 动态过滤核心
# =============================================================================

def compute_dynamic_mask(depth_prev: np.ndarray,
                          depth_curr: np.ndarray,
                          diff_thresh: float = 0.015,
                          min_area: int = 200) -> np.ndarray:
    """
    返回 bool mask (H, W)，True 表示动态像素。
    算法：|depth_curr - depth_prev| > diff_thresh，且两帧均有效。
    """
    valid = (depth_prev > 0) & (depth_curr > 0)
    diff  = np.abs(depth_curr.astype(np.float32) - depth_prev.astype(np.float32))
    raw_mask = valid & (diff > diff_thresh)

    # 形态学去噪：先膨胀填孔洞，再腐蚀去噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # 过滤面积太小的连通域（噪声）
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    result = np.zeros_like(cleaned)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            result[labels == lbl] = 1

    return result.astype(bool)


def extract_dynamic_points(world_pts: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
    """从 (H,W,3) 点云中提取 mask 对应的点，返回 (N,3)。"""
    pts = world_pts[mask]
    # 过滤无效点
    valid = np.linalg.norm(pts, axis=1) > 0
    return pts[valid]


# =============================================================================
# 可视化工具
# =============================================================================

def make_vis_frame(rgb: np.ndarray,
                   mask: np.ndarray,
                   diff_img: np.ndarray,
                   frame_idx: int,
                   n_moving_pts: int,
                   fps: float) -> np.ndarray:
    """合成一张可视化图：原图 | Diff 热力图 | 动态区域叠加。"""
    H, W = rgb.shape[:2]

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Diff 热力图
    diff_norm = np.clip(diff_img * 50, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # 动态 mask 叠加
    overlay = bgr.copy()
    overlay[mask] = [0, 0, 255]  # 动态区域红色
    blended = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)

    # 拼接三列
    vis = np.hstack([bgr, diff_color, blended])

    # 文字信息
    info = (f"Frame {frame_idx:04d}  |  Moving pts: {n_moving_pts:5d}  |  {fps:.1f} FPS")
    cv2.putText(vis, info, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 1, cv2.LINE_AA)

    # 列标签
    for i, label in enumerate(["RGB", "Depth Diff", "Dynamic Mask"]):
        cv2.putText(vis, label, (i * W + 6, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    return vis


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="对比相邻两帧的世界点云，提取动态区域",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",    default=str(Path(__file__).parent / "record_pull_arc_with_joint_0"))
    parser.add_argument("--diff_thresh", type=float, default=0.001,
                        help="深度差异阈值（米），超过此值视为动态")
    parser.add_argument("--min_area",    type=int,   default=300,
                        help="动态区域最小面积（像素数），过滤噪点")
    parser.add_argument("--start_frame", type=int,   default=0)
    parser.add_argument("--max_frames",  type=int,   default=0,   help="0=全部")
    parser.add_argument("--save_video",  action="store_true")
    parser.add_argument("--no_display",  action="store_true")
    parser.add_argument("--no_rerun",    action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    W, H, K  = load_camera_info(data_dir)
    poses    = load_camera_poses(data_dir)

    depth_files = sorted((data_dir / "depth").glob("depth_*.npy"))
    rgb_files   = sorted((data_dir / "rgb").glob("rgb_*.png"))
    total = len(depth_files)
    if args.max_frames > 0:
        total = min(total, args.start_frame + args.max_frames)

    print(f"[Init] 数据集: {data_dir.name}  共 {len(depth_files)} 帧")
    print(f"  diff_thresh={args.diff_thresh}m  min_area={args.min_area}px")
    print(f"  处理帧: {args.start_frame} → {total - 1}")

    # ── Rerun ─────────────────────────────────────────────────────────────────
    if HAVE_RERUN and not args.no_rerun:
        rr.init("DynamicFilter", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # ── 视频输出 ───────────────────────────────────────────────────────────────
    writer = None
    if args.save_video:
        out_path = data_dir.parent / f"dynamic_filter_{data_dir.name}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 10, (W * 3, H))
        print(f"[Init] 视频输出: {out_path}")

    # ── 窗口 ──────────────────────────────────────────────────────────────────
    WIN = "Dynamic Filter  [ RGB | Depth Diff | Mask ]"
    if not args.no_display:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, W * 3, H)

    # ── 主循环 ────────────────────────────────────────────────────────────────
    prev_depth = None
    fps_t = time.perf_counter()
    fps = 0.0

    for frame_idx in range(args.start_frame, total):
        curr_depth = np.load(str(depth_files[frame_idx])).astype(np.float32)
        curr_rgb   = cv2.cvtColor(cv2.imread(str(rgb_files[frame_idx])), cv2.COLOR_BGR2RGB)

        fid  = int(depth_files[frame_idx].stem.split("_")[-1])
        c2w  = poses.get(fid, poses[min(poses.keys())])  # fallback

        # 第一帧只缓存，没有 prev 可对比
        if prev_depth is None:
            prev_depth = curr_depth
            continue

        # 1. 计算深度差异
        diff = np.abs(curr_depth - prev_depth)

        # 2. 动态 mask
        dyn_mask = compute_dynamic_mask(prev_depth, curr_depth,
                                         diff_thresh=args.diff_thresh,
                                         min_area=args.min_area)

        # 3. 反投影当前帧为世界点云，提取动态点
        world_pts  = depth_to_world_points(curr_depth, K, c2w)
        dyn_pts    = extract_dynamic_points(world_pts, dyn_mask)
        static_pts = world_pts[(~dyn_mask) & (curr_depth > 0)].reshape(-1, 3)

        # 随机采样静态点以减少 Rerun 数据量
        if len(static_pts) > 5000:
            idx = np.random.choice(len(static_pts), 5000, replace=False)
            static_pts = static_pts[idx]

        # FPS
        fps_t_now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(fps_t_now - fps_t, 1e-6))
        fps_t = fps_t_now

        # 4. 可视化
        vis = make_vis_frame(curr_rgb, dyn_mask, diff, frame_idx,
                             len(dyn_pts), fps)

        if not args.no_display:
            cv2.imshow(WIN, vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户按 Q 退出。")
                break

        if writer is not None:
            writer.write(vis)

        # 5. Rerun 3D
        if HAVE_RERUN and not args.no_rerun:
            rr.set_time_sequence("frame", frame_idx)
            rr.log("world/camera/rgb", rr.Image(curr_rgb))

            if len(dyn_pts) > 0:
                # 动态点：亮红色
                dyn_colors = np.tile([255, 60, 60], (len(dyn_pts), 1)).astype(np.uint8)
                rr.log("world/dynamic_pts",
                       rr.Points3D(dyn_pts, colors=dyn_colors, radii=0.008))

            if len(static_pts) > 0:
                # 静态点：灰色半透明
                sta_colors = np.tile([120, 120, 120], (len(static_pts), 1)).astype(np.uint8)
                rr.log("world/static_pts",
                       rr.Points3D(static_pts, colors=sta_colors, radii=0.003))

        prev_depth = curr_depth

        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx:4d}/{total}  "
                  f"dyn_pts={len(dyn_pts):5d}  "
                  f"mask_ratio={dyn_mask.mean()*100:.1f}%  "
                  f"FPS={fps:.1f}")

    # ── 清理 ──────────────────────────────────────────────────────────────────
    if not args.no_display:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"视频已保存。")
    print("✅ 完成。")


if __name__ == "__main__":
    main()
