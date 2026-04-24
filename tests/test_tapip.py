#!/usr/bin/env python3
"""
test_tapip.py  —  TAPIP3D 在线逐帧追踪 + 实时可视化
=====================================================
模拟实时系统：每来一帧就跑一次推理，实时显示追踪结果。

工作原理
--------
1. 第一帧：在一个 init_frames 大小的初始窗口上建立查询点。
2. 之后每帧：维护一个 context_frames 大小的滑动缓冲区，
   对缓冲区跑 TAPIP3D 推理，取最后一帧的追踪坐标实时显示。
3. OpenCV 窗口实时展示当前帧 RGB + 彩色追踪点。
4. 同时把追踪结果写入 MP4 视频。

运行方法（tapip3d conda 环境）：
    cd /home/wby/active_vision/vlm_based/TAPIP3D
    python ../test_tapip.py

参数：
    --data_dir          数据集目录（含 rgb/ depth/ camera_*.txt）
    --context_frames    每次推理使用的历史帧数（建议 8-24，越小越快）
    --init_frames       初始化查询点所用帧数（建议 ≥ context_frames）
    --resolution_factor 推理分辨率缩放（1 = 最省显存，2 = 更精细）
    --query_grid        查询点网格大小（边长，总点数=grid²）
    --max_tracks        可视化时最多显示的追踪点数
    --fps_limit         可视化帧率上限（0 = 不限）
    --save_video        是否将可视化结果写成 MP4
    --no_display        不弹出 OpenCV 窗口（适合无显示器环境）
"""

import sys
import argparse
import math
import time
import collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import cv2
import torch
import rerun as rr
from scipy.spatial.transform import Rotation as R

# ── TAPIP3D 模块路径 ──────────────────────────────────────────────────────────
TAPIP3D_DIR = Path(__file__).parent / "TAPIP3D"
if str(TAPIP3D_DIR) not in sys.path:
    sys.path.insert(0, str(TAPIP3D_DIR))

from utils.inference_utils import load_model, inference, get_grid_queries, resize_depth_bilinear
from datasets.data_ops import _filter_one_depth


# =============================================================================
# 数据加载
# =============================================================================

def load_camera_info(data_dir: Path):
    info = {}
    with open(data_dir / "camera_info.txt") as f:
        for line in f:
            key, _, val = line.strip().partition(":")
            info[key.strip()] = float(val.strip())
    return (int(info["width"]), int(info["height"]),
            info["fx"], info["fy"], info["cx"], info["cy"])


def load_camera_poses(data_dir: Path):
    """返回 (T, 4, 4) world_to_cam extrinsics，按帧号排序。"""
    rows = []
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
            rows.append((fid, np.linalg.inv(c2w).astype(np.float32)))  # world_to_cam
    rows.sort(key=lambda x: x[0])
    return np.stack([r[1] for r in rows])


def iter_frames(data_dir: Path, max_frames: int = 0):
    """
    逐帧迭代器，每次 yield (rgb, depth, extrinsics_row)。
    rgb: (H,W,3) uint8, depth: (H,W) float32, extrinsics_row: (4,4) float32
    """
    data_dir = Path(data_dir)
    extrinsics_all = load_camera_poses(data_dir)

    rgb_files   = sorted((data_dir / "rgb").glob("rgb_*.png"))
    depth_files = sorted((data_dir / "depth").glob("depth_*.npy"))
    assert len(rgb_files) == len(depth_files), \
        f"RGB/Depth 帧数不一致: {len(rgb_files)} vs {len(depth_files)}"
def preprocess_one_frame(rgb_np, depth_np, intr_np, extr_np, inference_res, device="cuda"):
    """单帧图像处理：Resize + Filter + Tensor Conversion"""
    orig_H, orig_W = depth_np.shape
    H_inf, W_inf   = inference_res

    # Resize
    rgb_r   = cv2.resize(rgb_np, (W_inf, H_inf), interpolation=cv2.INTER_LINEAR)
    depth_r = resize_depth_bilinear(depth_np, (W_inf, H_inf))

    # Scale intrinsics
    intr = intr_np.copy()
    intr[0, :] *= (W_inf - 1) / (orig_W - 1)
    intr[1, :] *= (H_inf - 1) / (orig_H - 1)

    # Filter
    depth_r = _filter_one_depth(depth_r, 0.08, 15, intr)

    # Convert to Tensor
    rgb_t   = (torch.from_numpy(rgb_r).permute(2, 0, 1).float() / 255.0).to(device)
    depth_t = torch.from_numpy(depth_r).float().to(device)
    intr_t  = torch.from_numpy(intr).float().to(device)
    extr_t  = torch.from_numpy(extr_np.astype(np.float32)).to(device)

    return rgb_t, depth_t, intr_t, extr_t


# =============================================================================
# 推理工具
# =============================================================================

def run_inference_on_buffer(model, video_t, depths_t, intr_t, extr_t,
                             query_point, num_iters, support_grid_size,
                             vis_threshold, depth_roi=None):
    """
    在缓冲区上运行 TAPIP3D，返回全部过程轨迹。
    返回 coords (T, N, 3), visibs (T, N) —— CPU numpy。
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        coords, visibs = inference(
            model=model,
            video=video_t,
            depths=depths_t,
            intrinsics=intr_t,
            extrinsics=extr_t,
            query_point=query_point,
            num_iters=num_iters,
            grid_size=support_grid_size,
            vis_threshold=vis_threshold,
            bidrectional=False,
            depth_roi=depth_roi,
        )
    return coords.cpu().numpy(), visibs.cpu().numpy()


# =============================================================================
# 可视化工具
# =============================================================================

# 固定颜色表，每个追踪点颜色固定（不随帧变化）
_PALETTE = None

def get_colors(N):
    global _PALETTE
    if _PALETTE is None or len(_PALETTE) < N:
        np.random.seed(42)
        _PALETTE = [tuple(int(c) for c in col)
                    for col in np.random.randint(30, 230, (max(N, 1024), 3))]
    return _PALETTE[:N]


def project_world_to_pixel(coords_world, extrinsics, intrinsics):
    """
    将世界坐标 (N,3) 投影到当前帧像素坐标 (N,2)。
    extrinsics: (4,4) world_to_cam
    intrinsics: (3,3) 推理分辨率下的内参
    忽略在相机后方的点（返回 NaN）。
    """
    N = len(coords_world)
    ones = np.ones((N, 1), dtype=np.float32)
    pts_h = np.hstack([coords_world.astype(np.float32), ones])  # (N,4)
    pts_cam = (extrinsics @ pts_h.T).T[:, :3]                   # (N,3)

    valid = pts_cam[:, 2] > 0
    px = np.full(N, np.nan, dtype=np.float32)
    py = np.full(N, np.nan, dtype=np.float32)

    z = pts_cam[valid, 2]
    px[valid] = pts_cam[valid, 0] / z * intrinsics[0, 0] + intrinsics[0, 2]
    py[valid] = pts_cam[valid, 1] / z * intrinsics[1, 1] + intrinsics[1, 2]
    return px, py


def draw_tracks(frame_bgr, px, py, visibs, colors,
                sel_idx, radius=5, trail_buf=None, trail_len=0):
    """
    在 frame_bgr 上绘制当前帧的追踪点（及可选尾迹）。
    px, py  : (N,) 像素坐标（可含 nan）
    visibs  : (N,) bool
    sel_idx : 要画的点的下标列表
    trail_buf: deque of (px, py, visibs) —— 过去帧，用于画尾迹
    """
    H, W = frame_bgr.shape[:2]

    # 尾迹（越老越透明）
    if trail_buf and trail_len > 0:
        for age, (tpx, tpy, tv) in enumerate(reversed(list(trail_buf)[-trail_len:])):
            alpha = (age + 1) / (trail_len + 1)
            r_t = max(1, int(radius * alpha * 0.6))
            for i, si in enumerate(sel_idx):
                if not tv[si] or np.isnan(tpx[si]) or np.isnan(tpy[si]):
                    continue
                cx, cy = int(round(tpx[si])), int(round(tpy[si]))
                if 0 <= cx < W and 0 <= cy < H:
                    c = colors[i]
                    overlay = frame_bgr.copy()
                    cv2.circle(overlay, (cx, cy), r_t, c, -1)
                    cv2.addWeighted(overlay, alpha * 0.5, frame_bgr, 1 - alpha * 0.5, 0, frame_bgr)

    # 当前帧的点
    for i, si in enumerate(sel_idx):
        if not visibs[si] or np.isnan(px[si]) or np.isnan(py[si]):
            continue
        cx, cy = int(round(px[si])), int(round(py[si]))
        if 0 <= cx < W and 0 <= cy < H:
            col = colors[i]
            cv2.circle(frame_bgr, (cx, cy), radius, col, -1)
            cv2.circle(frame_bgr, (cx, cy), radius + 1, (255, 255, 255), 1)

    return frame_bgr


# =============================================================================
# 主逻辑
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TAPIP3D 在线逐帧追踪 + 实时可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",    default=str(Path(__file__).parent / "record_pull_arc_0"))
    parser.add_argument("--checkpoint",  default=str(TAPIP3D_DIR / "tapip3d_final.pth"))
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--context_frames", type=int, default=72,
                        help="每次推理使用的历史帧数（显存允许范围内越大越精准）")
    parser.add_argument("--init_frames",    type=int, default=72,
                        help="初始化查询点的预热帧数")
    parser.add_argument("--resolution_factor", type=int, default=1,
                        help="推理分辨率缩放（1=省显存，2=更精细）")
    parser.add_argument("--support_grid_size", type=int, default=4)
    parser.add_argument("--num_iters",   type=int, default=6)
    parser.add_argument("--query_grid",  type=int, default=20,
                        help="查询点网格边长（总点数 = grid²）")
    parser.add_argument("--max_tracks",  type=int, default=300,
                        help="可视化显示的最大追踪点数")
    parser.add_argument("--trail_len",   type=int, default=8,
                        help="尾迹显示帧数（0 = 不显示尾迹）")
    parser.add_argument("--fps_limit",   type=float, default=0,
                        help="可视化帧率上限（0 = 不限速）")
    parser.add_argument("--max_frames",  type=int, default=0,
                        help="处理最多多少帧（0=全部）")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--vis_threshold", type=float, default=0.5)
    parser.add_argument("--save_video", action="store_true",
                        help="将可视化结果保存为 MP4")
    parser.add_argument("--no_display", action="store_true",
                        help="不弹出 OpenCV 窗口")
    parser.add_argument("--no_rerun", action="store_true",
                        help="禁用 Rerun 3D 可视化")
    parser.add_argument("--rerun_spawn", action="store_true", default=True,
                        help="自动启动 Rerun 视图")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ── 读取相机参数 ─────────────────────────────────────────────────────────
    width, height, fx, fy, cx, cy = load_camera_info(data_dir)
    K_orig = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]], dtype=np.float32)

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    print("[Init] 加载 TAPIP3D 模型...")
    model = load_model(args.checkpoint)
    model.to(args.device)
    model.eval()

    inference_res = (
        int(model.image_size[0] * math.sqrt(args.resolution_factor)),
        int(model.image_size[1] * math.sqrt(args.resolution_factor)),
    )
    H_inf, W_inf = inference_res
    print(f"  推理分辨率: {W_inf}×{H_inf}")

    # ── 初始化 Rerun ────────────────────────────────────────────────────────
    if not args.no_rerun:
        rr.init("TAPIP3D_Online", spawn=args.rerun_spawn)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True) # OpenCV 约定
        # 记录静态内参（用于 3D 重建）
        rr.log("world/camera", rr.Pinhole(
            resolution=[width, height],
            focal_length=[fx, fy],
            principal_point=[cx, cy]
        ), static=True)

    # ── 计算推理分辨率下的内参 ────────────────────────────────────────────────
    K_inf = K_orig.copy()
    K_inf[0, :] *= (W_inf - 1) / (width  - 1)
    K_inf[1, :] *= (H_inf - 1) / (height - 1)

    # ── 准备帧迭代器 ─────────────────────────────────────────────────────────
    rgb_files   = sorted((data_dir / "rgb").glob("rgb_*.png"))
    depth_files = sorted((data_dir / "depth").glob("depth_*.npy"))
    extrinsics_all = load_camera_poses(data_dir)
    T_total = len(rgb_files)
    if args.max_frames > 0:
        T_total = min(T_total, args.max_frames)
    print(f"[Init] 数据集: {T_total} 帧")

    # ── 滑动缓冲区 ── (存 Tensor 以提速) ──────────────────────────────────────────
    ctx = args.context_frames
    rgb_t_buf   = collections.deque(maxlen=ctx)
    depth_t_buf = collections.deque(maxlen=ctx)
    intr_t_buf  = collections.deque(maxlen=ctx)
    extr_t_buf  = collections.deque(maxlen=ctx)

    # 为了可视化，保留最近一帧的原始图像
    last_rgb_np = None

    # 尾迹缓冲区
    trail_buf = collections.deque(maxlen=args.trail_len + 1)

    # ── 输出视频 ─────────────────────────────────────────────────────────────
    video_writer = None
    if args.save_video:
        out_path = data_dir.parent / "tapip_results" / \
                   f"online_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_path), fourcc, 10, (W_inf, H_inf))
        print(f"[Init] 视频输出: {out_path}")

    # ── OpenCV 窗口 ──────────────────────────────────────────────────────────
    WIN = "TAPIP3D Online Tracking"
    if not args.no_display:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, W_inf * 2, H_inf * 2)

    query_point = None   # (N, 4) tensor，在 init_frames 后初始化
    depth_roi   = None   # 预计算的深度范围
    sel_idx     = None   # 可视化选择的点下标
    colors      = None
    fps_counter = {"t": time.perf_counter(), "count": 0, "fps": 0.0}

    print("\n[Track] 开始逐帧追踪... (按 'q' 退出)")
    print(f"  context_frames={ctx}, init_frames={args.init_frames}, "
          f"resolution_factor={args.resolution_factor}")

    for frame_idx in range(T_total):
        t_frame_start = time.perf_counter()

        # ── 1. 读取并前处理新帧 ──
        rgb   = cv2.cvtColor(cv2.imread(str(rgb_files[frame_idx])), cv2.COLOR_BGR2RGB)
        depth = np.load(str(depth_files[frame_idx])).astype(np.float32)
        extr  = extrinsics_all[frame_idx]  # world_to_cam (4,4)
        last_rgb_np = rgb

        rgb_t, depth_t, intr_t, extr_t = preprocess_one_frame(
            rgb, depth, K_orig, extr, inference_res, args.device
        )
        
        rgb_t_buf.append(rgb_t)
        depth_t_buf.append(depth_t)
        intr_t_buf.append(intr_t)
        extr_t_buf.append(extr_t)

        # ── 2. 初始化查询点 ──
        if frame_idx + 1 < args.init_frames:
            # 等待积攒足够的帧
            if not args.no_display:
                disp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                disp = cv2.resize(disp, (W_inf, H_inf))
                cv2.putText(disp, f"Buffered: {frame_idx+1}/{args.init_frames}",
                            (10, H_inf - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.imshow(WIN, disp); cv2.waitKey(1)
            continue

        if query_point is None:
            print(f"  [Frame {frame_idx}] 确认查询点...")
            # 取当前积攒的所有帧构建第一个窗口
            d_init = torch.stack(list(depth_t_buf), dim=0)  # (T, H, W)
            k_init = torch.stack(list(intr_t_buf), dim=0)   # (T, 3, 3)
            e_init = torch.stack(list(extr_t_buf), dim=0)   # (T, 4, 4)
            
            query_point = get_grid_queries(
                grid_size=args.query_grid,
                depths=d_init, intrinsics=k_init, extrinsics=e_init
            )
            
            # 手动计算一次 depth_roi，之后复用
            _d = d_init[d_init > 0].reshape(-1)
            if len(_d) > 0:
                q25 = torch.kthvalue(_d, int(0.25 * len(_d))).values
                q75 = torch.kthvalue(_d, int(0.75 * len(_d))).values
                depth_roi = torch.tensor([1e-7, (q75 + 1.5*(q75-q25)).item()], device=args.device)
            else:
                depth_roi = torch.tensor([1e-7, 10.0], device=args.device)

            N = query_point.shape[0]; sel = np.arange(N)
            if N > args.max_tracks: sel = np.random.choice(N, args.max_tracks, replace=False)
            sel_idx = sel.tolist(); colors = get_colors(len(sel_idx))
            print(f"  查询点数: {N} (可视化 {len(sel_idx)})")

        # ── 3. 在线推理 (使用缓存的 TensorStack) ──
        video_win = torch.stack(list(rgb_t_buf), dim=0)
        depth_win = torch.stack(list(depth_t_buf), dim=0)
        intr_win  = torch.stack(list(intr_t_buf), dim=0)
        extr_win  = torch.stack(list(extr_t_buf), dim=0)

        coords_win, visibs_win = run_inference_on_buffer(
            model, video_win, depth_win, intr_win, extr_win,
            query_point,
            num_iters=args.num_iters,
            support_grid_size=args.support_grid_size,
            vis_threshold=args.vis_threshold,
            depth_roi=depth_roi,
        )
        torch.cuda.empty_cache()

        # ── 3.5 链式更新 (平滑模式) ──
        # 当前窗口是 [t-15, t]。我们要为下一帧 [t-14, t+1] 准备查询点。
        # 下一帧的 index 0 对应当前窗口的 index 1。
        # 使用当前对 index 1 帧的预测作为下一轮的起点。
        if coords_win.shape[0] > 1:
            query_point[:, 1:4] = torch.from_numpy(coords_win[1]).to(args.device)
        else:
            query_point[:, 1:4] = torch.from_numpy(coords_win[0]).to(args.device)

        coords_last = coords_win[-1]
        visibs_last = visibs_win[-1]

        # ── 4. 投影与可视化 ──
        px, py = project_world_to_pixel(coords_last, extr, K_inf)
        trail_buf.append((px.copy(), py.copy(), visibs_last.copy()))

        disp = cv2.cvtColor(last_rgb_np, cv2.COLOR_RGB2BGR)
        disp = cv2.resize(disp, (W_inf, H_inf))

        draw_tracks(disp, px, py, visibs_last, colors, sel_idx,
                    radius=5, trail_buf=trail_buf, trail_len=args.trail_len)

        # FPS 统计
        fps_counter["count"] += 1
        elapsed = time.perf_counter() - fps_counter["t"]
        if elapsed >= 1.0:
            fps_counter["fps"]   = fps_counter["count"] / elapsed
            fps_counter["count"] = 0
            fps_counter["t"]     = time.perf_counter()

        n_vis = int(np.sum(visibs_last[sel_idx]))
        info  = (f"Frame {frame_idx+1}/{T_total}"
                 f"  Buf:{len(rgb_t_buf):2d}"
                 f"  Vis:{n_vis}/{len(sel_idx)}"
                 f"  {fps_counter['fps']:5.1f} FPS")
        cv2.rectangle(disp, (0, 0), (W_inf, 28), (0, 0, 0), -1)
        cv2.putText(disp, info, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)

        # ── 输出 ──
        if not args.no_display:
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户按 Q 退出。")
                break

        if video_writer is not None:
            video_writer.write(disp)

        # ── Rerun 日志 ───────────────────────────────────────────────────────
        if not args.no_rerun:
            # 修正 Rerun 时间 API: 使用推荐的 set_time 格式
            rr.set_time_sequence("frame", frame_idx)
            
            # 相机位姿 (extr 是 world_to_cam)
            c2w = np.linalg.inv(extr)
            # 将 Transform 和 Pinhole 分离到推荐层级
            rr.log("world/camera", rr.Transform3D(
                translation=c2w[:3, 3],
                mat3x3=c2w[:3, :3]
            ))
            rr.log("world/camera", rr.Pinhole(
                resolution=[width, height],
                focal_length=[fx, fy],
                principal_point=[cx, cy]
            ))
            rr.log("world/camera/rgb", rr.Image(rgb))
            rr.log("world/camera/depth", rr.DepthImage(depth, meter=1.0))
            
            if visibs_last is not None:
                vis_mask = visibs_last[sel_idx]
                tracked_pts = coords_last[sel_idx][vis_mask]
                tracked_cols = np.array(colors)[vis_mask]
                rr.log("world/tracks", rr.Points3D(tracked_pts, colors=tracked_cols, radii=0.015))

        # 帧率限制
        if args.fps_limit > 0:
            target = 1.0 / args.fps_limit
            elapsed_frame = time.perf_counter() - t_frame_start
            if elapsed_frame < target:
                time.sleep(target - elapsed_frame)

        if (frame_idx + 1) % 10 == 0:
            print(f"  Frame {frame_idx+1}/{T_total}"
                  f"  FPS={fps_counter['fps']:.1f}"
                  f"  Visible={n_vis}/{len(sel_idx)}")

    # ── 清理 ─────────────────────────────────────────────────────────────────
    if not args.no_display:
        cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()
        print(f"\n视频已保存: {out_path}")
    print("\n✅ 追踪完成。")


if __name__ == "__main__":
    main()
