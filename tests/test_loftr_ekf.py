#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import kornia as KN
from kornia.feature import LoFTR
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =============================================================================
# Config
# =============================================================================
DATA_ROOT = Path("record_pull_arc_with_joint_0")

DIFF_THRESH       = 0.002
MIN_AREA          = 3000
MIN_DYN_MATCHES   = 15
LOFTR_CONF_THRESH = 0.5
LOFTR_SIZE        = (640, 480)   # (W, H) LoFTR 输入分辨率
MIN_INIT_DISP_M   = 0.08         # 初始化所需最小累积平均 3D 位移 (m)

R_ARM, R_FOREARM, R_HAND, R_FINGER = 0.15, 0.13, 0.07, 0.03
CONNECTIONS = [
    ('panda_link0', 'panda_link1', R_ARM),
    ('panda_link1', 'panda_link2', R_ARM),
    ('panda_link2', 'panda_link3', R_ARM),
    ('panda_link3', 'panda_link4', R_ARM),
    ('panda_link4', 'panda_link5', R_FOREARM),
    ('panda_link5', 'panda_link6', R_FOREARM),
    ('panda_link6', 'panda_link7', R_FOREARM),
    ('panda_link7', 'panda_link8', R_FOREARM),
    ('panda_hand',  'panda_leftfinger',  R_HAND),
    ('panda_hand',  'panda_rightfinger', R_HAND),
    ('panda_leftfinger',  'panda_leftfinger',  R_FINGER),
    ('panda_rightfinger', 'panda_rightfinger', R_FINGER),
]

Q_DIAG = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])
R_DIAG = np.array([2e-2, 2e-2, 2e-2, 5e-2, 5e-2, 1e-2])

# =============================================================================
# 数据加载
# =============================================================================

def load_camera_info(root: Path):
    info = {}
    with open(root / "camera_info.txt") as f:
        for line in f:
            key, _, val = line.strip().partition(":")
            info[key.strip()] = float(val.strip())
    K = np.array([[info['fx'],       0, info['cx']],
                  [      0, info['fy'], info['cy']],
                  [      0,       0,        1     ]], dtype=np.float32)
    return K, int(info['width']), int(info['height'])


def load_camera_poses(root: Path):
    poses = {}
    with open(root / "camera_pose.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            fid = int(parts[0])
            t = np.array([float(x) for x in parts[1:4]])
            q = np.array([float(x) for x in parts[4:8]])
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R_scipy.from_quat(q).as_matrix()
            c2w[:3, 3] = t
            poses[fid] = c2w
    return poses


def load_link_poses(root: Path):
    all_lp = {}
    with open(root / "link_poses.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            fid  = int(parts[0])
            name = parts[1]
            pos  = np.array([float(x) for x in parts[2:5]])
            if fid not in all_lp:
                all_lp[fid] = {}
            all_lp[fid][name] = pos
    return all_lp


def load_ee_poses(root: Path):
    poses = {}
    with open(root / "ee_pose.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            fid = int(parts[0])
            poses[fid] = np.array([float(x) for x in parts[1:4]])
    return poses

# =============================================================================
# 点云工具
# =============================================================================

def dist_to_segments(pts: np.ndarray, segments) -> np.ndarray:
    inside = np.zeros(len(pts), dtype=bool)
    for A, B, radius in segments:
        AB = B - A
        mag_sq = float(np.dot(AB, AB))
        if mag_sq < 1e-6:
            dist = np.linalg.norm(pts - A, axis=1)
        else:
            AP = pts - A
            t  = np.clip(np.sum(AP * AB, axis=1) / mag_sq, 0., 1.)
            closest = A + t[:, None] * AB
            dist = np.linalg.norm(pts - closest, axis=1)
        inside |= (dist < radius)
    return inside


def get_robot_segments(lp_dict: dict):
    segs = []
    for s, e, r in CONNECTIONS:
        if s in lp_dict and e in lp_dict:
            segs.append((lp_dict[s], lp_dict[e], r))
    return segs

# =============================================================================
# 动态掩码
# =============================================================================

def get_dynamic_mask(prev_depth: np.ndarray, curr_depth: np.ndarray,
                     diff_thresh: float = DIFF_THRESH,
                     min_area: int = MIN_AREA) -> np.ndarray:
    """返回 2D bool 掩码，True = 深度变化显著的动态像素。"""
    valid    = (prev_depth > 0) & (curr_depth > 0)
    diff     = np.abs(curr_depth.astype(np.float32) - prev_depth.astype(np.float32))
    raw_mask = valid & (diff > diff_thresh)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    result = np.zeros_like(cleaned)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            result[labels == lbl] = 1
    return result.astype(bool)

# =============================================================================
# LoFTR 特征匹配
# =============================================================================

def img_to_loftr_tensor(bgr_img: np.ndarray, device: torch.device,
                        size: tuple = LOFTR_SIZE) -> torch.Tensor:
    """BGR 图像 → LoFTR 灰度 tensor [1, H, W]."""
    resized = cv2.resize(bgr_img, size)
    t = KN.image_to_tensor(resized, keepdim=False).float() / 255.
    return KN.color.rgb_to_grayscale(t).to(device)


def match_loftr(t0: torch.Tensor, t1: torch.Tensor, matcher,
                conf_thresh: float = LOFTR_CONF_THRESH):
    """返回 (mkpts0, mkpts1) in LOFTR_SIZE 像素坐标，已按置信度过滤。"""
    with torch.no_grad():
        corr = matcher({"image0": t0, "image1": t1})
    mkpts0 = corr['keypoints0'].cpu().numpy()
    mkpts1 = corr['keypoints1'].cpu().numpy()
    conf   = corr['confidence'].cpu().numpy()
    mask   = conf > conf_thresh
    return mkpts0[mask], mkpts1[mask]

# =============================================================================
# 3D 对应点提取（LoFTR 匹配 → 反投影 → 过滤）
# =============================================================================

def get_3d_correspondences(mkpts0: np.ndarray, mkpts1: np.ndarray,
                            depth0: np.ndarray, depth1: np.ndarray,
                            K: np.ndarray,
                            c2w0: np.ndarray, c2w1: np.ndarray,
                            dyn_mask_curr: np.ndarray,
                            robot_segs):
    """
    将 LoFTR 2D 匹配反投影到世界坐标系 3D 对应点对。
    过滤: 深度有效 + mkpts1 在动态掩码内 + 不在机械臂上。
    返回 (src_world, tgt_world) float32，或 (None, None)。
    """
    H, W  = depth0.shape
    lw, lh = LOFTR_SIZE
    sx = W / lw
    sy = H / lh

    # LoFTR 坐标 → 原图坐标
    px0 = mkpts0[:, 0] * sx;  py0 = mkpts0[:, 1] * sy
    px1 = mkpts1[:, 0] * sx;  py1 = mkpts1[:, 1] * sy

    ix0 = np.clip(np.round(px0).astype(int), 0, W - 1)
    iy0 = np.clip(np.round(py0).astype(int), 0, H - 1)
    ix1 = np.clip(np.round(px1).astype(int), 0, W - 1)
    iy1 = np.clip(np.round(py1).astype(int), 0, H - 1)

    z0 = depth0[iy0, ix0].astype(np.float64)
    z1 = depth1[iy1, ix1].astype(np.float64)

    # 有效深度 + 当前帧匹配点在动态区域
    valid = (z0 > 0) & (z1 > 0)
    if dyn_mask_curr is not None:
        valid &= dyn_mask_curr[iy1, ix1]

    if not np.any(valid):
        return None, None

    def unproj_world(px, py, z, c2w):
        xc = (px - K[0, 2]) / K[0, 0] * z
        yc = (py - K[1, 2]) / K[1, 1] * z
        pts_c = np.stack([xc, yc, z], axis=1)
        ones  = np.ones((len(pts_c), 1))
        return (c2w.astype(np.float64) @ np.hstack([pts_c, ones]).T).T[:, :3]

    src_w = unproj_world(px0[valid], py0[valid], z0[valid], c2w0)
    tgt_w = unproj_world(px1[valid], py1[valid], z1[valid], c2w1)

    # 去除机械臂点
    if robot_segs:
        bad = dist_to_segments(src_w, robot_segs) | dist_to_segments(tgt_w, robot_segs)
        src_w = src_w[~bad]
        tgt_w = tgt_w[~bad]

    if len(src_w) < MIN_DYN_MATCHES:
        return None, None

    return src_w.astype(np.float32), tgt_w.astype(np.float32)

# =============================================================================
# Kabsch SVD 刚体变换（替代 ICP）
# =============================================================================

def kabsch_transform(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """
    src, tgt: (N,3) 对应 3D 点对。
    返回 T_4x4 float32，将 src 映射到 tgt。
    """
    src = src.astype(np.float64);  tgt = tgt.astype(np.float64)
    c_s = src.mean(0);             c_t = tgt.mean(0)
    H   = (src - c_s).T @ (tgt - c_t)
    U, S, Vt = np.linalg.svd(H)
    R_est = Vt.T @ U.T
    if np.linalg.det(R_est) < 0:
        Vt[-1] *= -1
        R_est = Vt.T @ U.T
    t_est = c_t - R_est @ c_s
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_est.astype(np.float32)
    T[:3,  3] = t_est.astype(np.float32)
    return T

# =============================================================================
# Screw Theory
# =============================================================================

def rot_matrix_np(n: np.ndarray, theta: float) -> np.ndarray:
    n = n / (np.linalg.norm(n) + 1e-9)
    K_mat = np.array([[    0, -n[2],  n[1]],
                      [ n[2],     0, -n[0]],
                      [-n[1],  n[0],     0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K_mat + (1 - np.cos(theta)) * (K_mat @ K_mat)


def screw_transform_np(p: np.ndarray, n: np.ndarray, theta: float) -> np.ndarray:
    R_mat = rot_matrix_np(n, theta)
    t_vec = (np.eye(3) - R_mat) @ p
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3,  3] = t_vec
    return T.astype(np.float32)


def screw_from_transform(T: np.ndarray, p_prior: np.ndarray = None):
    R_mat = T[:3, :3].astype(np.float64)
    t_vec = T[:3,  3].astype(np.float64)

    trace = np.clip((np.trace(R_mat) - 1) / 2, -1., 1.)
    theta = float(np.arccos(trace))

    if abs(theta) < 1e-4:
        n = t_vec / (np.linalg.norm(t_vec) + 1e-9)
        perp = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
        perp -= np.dot(perp, n) * n
        perp /= np.linalg.norm(perp)
        return perp * 5.0, n, theta

    n = np.array([R_mat[2,1] - R_mat[1,2],
                  R_mat[0,2] - R_mat[2,0],
                  R_mat[1,0] - R_mat[0,1]]) / (2 * np.sin(theta))
    n /= (np.linalg.norm(n) + 1e-9)

    t_perp = t_vec - np.dot(t_vec, n) * n
    A = np.eye(3) - R_mat

    if p_prior is None:
        p, _, _, _ = np.linalg.lstsq(A, t_perp, rcond=None)
    else:
        lam = max(1e-3, min(1.0, (0.05 / (abs(theta) + 1e-6))**2))
        sqL = np.sqrt(lam)
        A_reg = np.vstack([A, sqL * np.eye(3)])
        b_reg = np.hstack([t_perp, sqL * p_prior])
        p, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)

    return p, n, float(theta)

# =============================================================================
# EKF
# =============================================================================

def az_el_to_n(az: float, el: float) -> np.ndarray:
    return np.array([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)], dtype=np.float64)


class AxisEKF:
    """状态: x = [px, py, pz, az, el, theta]"""

    def __init__(self, p0: np.ndarray, n0: np.ndarray, theta0: float):
        n0  = n0 / (np.linalg.norm(n0) + 1e-9)
        az0 = float(np.arctan2(n0[1], n0[0]))
        el0 = float(np.arcsin(np.clip(n0[2], -1., 1.)))
        self.x = np.array([p0[0], p0[1], p0[2], az0, el0, theta0], dtype=np.float64)
        self.P = np.diag([0.05, 0.05, 0.05, 0.2, 0.2, 0.05])
        self.Q = np.diag(Q_DIAG)
        self.R = np.diag(R_DIAG)

    def predict(self, delta_theta: float):
        self.x[5] += delta_theta
        self.P = self.P + self.Q

    def update(self, T_accumulated: np.ndarray):
        p_obs, n_obs, theta_obs = screw_from_transform(T_accumulated)

        n_obs  = n_obs / (np.linalg.norm(n_obs) + 1e-9)
        az_obs = float(np.arctan2(n_obs[1], n_obs[0]))
        el_obs = float(np.arcsin(np.clip(n_obs[2], -1., 1.)))

        # 轴方向符号歧义处理
        n_cur = az_el_to_n(self.x[3], self.x[4])
        if np.dot(n_obs, n_cur) < 0:
            n_obs  = -n_obs
            az_obs = float(np.arctan2(n_obs[1], n_obs[0]))
            el_obs = float(np.arcsin(np.clip(n_obs[2], -1., 1.)))
            theta_obs = -theta_obs

        z = np.array([p_obs[0], p_obs[1], p_obs[2], az_obs, el_obs, theta_obs])
        H = np.eye(6)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        inn = z - self.x
        for idx in [3, 4, 5]:
            while inn[idx] >  np.pi: inn[idx] -= 2 * np.pi
            while inn[idx] < -np.pi: inn[idx] += 2 * np.pi

        self.x = self.x + K @ inn
        self.P = (np.eye(6) - K @ H) @ self.P

    def get_axis(self):
        px, py, pz, az, el, theta = self.x
        return np.array([px, py, pz]), az_el_to_n(az, el), float(theta)

# =============================================================================
# Matplotlib Visualization for Initialization (Replacement for Open3D)
# =============================================================================

def visualize_init_matplotlib(src_pts, tgt_pts, pivot, n_dir):
    """
    使用 Matplotlib 可视化 LoFTR 匹配的点对、估计的转轴及世界坐标系。
    """
    print(f"\n[Visualizer] 正在准备可视化: src={len(src_pts)} pts, tgt={len(tgt_pts)} pts")
    print("[Visualizer] 正在显示初始化可视化... 请查看弹窗窗口。关闭窗口以继续运行 EKF。")
    
    # 关闭交互模式以实现阻塞显示
    was_interactive = plt.isinteractive()
    plt.ioff()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 绘制点云 (采样以提高绘图速度)
    skip = max(1, len(src_pts) // 500)
    ax.scatter(src_pts[::skip, 0], src_pts[::skip, 1], src_pts[::skip, 2], 
               c='red', s=2, alpha=0.5, label='Start (Ref)')
    ax.scatter(tgt_pts[::skip, 0], tgt_pts[::skip, 1], tgt_pts[::skip, 2], 
               c='green', s=2, alpha=0.5, label='Current (Init)')
    
    # 2. 绘制转轴
    ext = 1.0
    p0 = pivot - n_dir * ext
    p1 = pivot + n_dir * ext
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 
            'b-', linewidth=3, label='Est. Axis')
    ax.scatter(pivot[0], pivot[1], pivot[2], c='blue', s=100, marker='X')
    
    # 3. 绘制坐标轴原点
    ax.quiver(0,0,0, 0.2,0,0, color='r')
    ax.quiver(0,0,0, 0,0.2,0, color='g')
    ax.quiver(0,0,0, 0,0,0.2, color='b')
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("EKF Initialization: Start(Red) vs End(Green)")
    ax.legend()
    
    # 设置合理的视角范围
    all_pts = np.vstack([src_pts, tgt_pts])
    center = all_pts.mean(0)
    max_range = np.ptp(all_pts, axis=0).max() / 2.0
    ax.set_xlim(center[0]-max_range, center[0]+max_range)
    ax.set_ylim(center[1]-max_range, center[1]+max_range)
    ax.set_zlim(center[2]-max_range, center[2]+max_range)
    
    plt.show() # 阻塞调用
    
    if was_interactive:
        plt.ion()

# =============================================================================
# Visualization
# =============================================================================

class Visualizer:
    def __init__(self):
        plt.ion()
        self.fig   = plt.figure(figsize=(14, 6))
        self.ax3d  = self.fig.add_subplot(131, projection='3d')
        self.ax2d  = self.fig.add_subplot(132)
        self.ax_pc = self.fig.add_subplot(133)
        self.val_hist   = []
        self.frame_hist = []

    def update(self, frame_id: int, dyn_pts, pivot, n_dir, theta: float,
               rgb_img=None, n_matches: int = 0, init_disp_cm: float = None):
        self.ax3d.cla()
        ax = self.ax3d
        if dyn_pts is not None and len(dyn_pts) > 0:
            ax.scatter(dyn_pts[:,0], dyn_pts[:,1], dyn_pts[:,2],
                       s=3, c='orange', alpha=0.7, label=f'Matched ({len(dyn_pts)})')
        if pivot is not None:
            ext = 0.5
            p0  = pivot - n_dir * ext
            p1  = pivot + n_dir * ext
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    'c-', linewidth=3, label='Est. Axis')
            ax.scatter(*pivot, s=120, c='red', zorder=6, label='Pivot')
        ax.quiver(0,0,0, 0.3,0,0, color='r', arrow_length_ratio=0.3)
        ax.quiver(0,0,0, 0,0.3,0, color='g', arrow_length_ratio=0.3)
        ax.quiver(0,0,0, 0,0,0.3, color='b', arrow_length_ratio=0.3)
        ax.set_xlim(0.2, 1.6); ax.set_ylim(0.0, 1.5); ax.set_zlim(0.2, 1.4)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        if init_disp_cm is not None:
            title = f'Frame {frame_id}\nInit avg_disp={init_disp_cm:.1f}cm  matches={n_matches}'
        else:
            title = f'Frame {frame_id}\nθ={np.rad2deg(theta):.1f}°  matches={n_matches}'
        ax.set_title(title)
        if pivot is not None:
            ax.legend(fontsize=7, loc='upper left')

        # 折线图：EKF 阶段显示 θ，初始化阶段显示 avg_disp
        self.val_hist.append(init_disp_cm if init_disp_cm is not None else np.rad2deg(theta))
        self.frame_hist.append(frame_id)
        self.ax2d.cla()
        self.ax2d.plot(self.frame_hist, self.val_hist, 'b-o', markersize=3)
        self.ax2d.set_xlabel('Frame')
        self.ax2d.set_ylabel('avg_disp (cm) / θ (°)')
        self.ax2d.set_title('Init disp / EKF θ')
        self.ax2d.grid(True)

        self.ax_pc.cla()
        if rgb_img is not None:
            self.ax_pc.imshow(rgb_img)
            self.ax_pc.set_title(f'RGB Frame {frame_id}')
            self.ax_pc.axis('off')

        plt.tight_layout()
        try:
            plt.pause(0.001)
        except Exception:
            pass

# =============================================================================
# Main
# =============================================================================

def main():
    print(f"[Main] 加载数据集: {DATA_ROOT}")
    K, img_W, img_H = load_camera_info(DATA_ROOT)
    cam_poses        = load_camera_poses(DATA_ROOT)
    link_poses_all   = load_link_poses(DATA_ROOT)
    ee_poses         = load_ee_poses(DATA_ROOT)

    depth_files = sorted(DATA_ROOT.glob("depth/depth_*.npy"))
    rgb_dir     = DATA_ROOT / "rgb"
    print(f"[Main] 共 {len(depth_files)} 帧")

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[LoFTR] 使用设备: {device}")
    matcher = LoFTR(pretrained='outdoor').to(device).eval()

    vis = Visualizer()

    prev_depth    = None
    prev_rgb_bgr  = None
    prev_frame_id = None

    ref_depth     = None   # 参考帧（动态点云第一帧）
    ref_rgb_bgr   = None
    ref_c2w       = None
    ref_frame_id  = None

    ekf           = None
    initialized   = False
    T_accumulated = np.eye(4, dtype=np.float32)

    for depth_file in depth_files:
        frame_id = int(depth_file.stem.split("_")[-1])
        if frame_id not in cam_poses or frame_id not in link_poses_all:
            continue

        c2w        = cam_poses[frame_id]
        lp_dict    = link_poses_all[frame_id]
        robot_segs = get_robot_segments(lp_dict)

        curr_depth = np.load(str(depth_file)).astype(np.float32)
        rgb_path   = rgb_dir / f"rgb_{frame_id:06d}.png"
        curr_rgb_bgr = cv2.imread(str(rgb_path)) if rgb_path.exists() else None
        rgb_img    = cv2.cvtColor(curr_rgb_bgr, cv2.COLOR_BGR2RGB) \
                     if curr_rgb_bgr is not None else None

        # 第一帧：仅存储状态
        if prev_depth is None:
            prev_depth    = curr_depth
            prev_rgb_bgr  = curr_rgb_bgr
            prev_c2w      = c2w
            prev_frame_id = frame_id
            continue

        # 计算动态掩码
        dyn_mask = get_dynamic_mask(prev_depth, curr_depth)

        if np.sum(dyn_mask) < MIN_AREA:
            print(f"[Frame {frame_id}] 动态区域不足 ({np.sum(dyn_mask)} px), 等待...")
            if initialized:
                pivot, n_dir, theta = ekf.get_axis()
                vis.update(frame_id, None, pivot, n_dir, theta, rgb_img)
            else:
                vis.update(frame_id, None, None, None, 0., rgb_img)
            prev_depth    = curr_depth
            prev_rgb_bgr  = curr_rgb_bgr
            prev_c2w      = c2w
            prev_frame_id = frame_id
            continue

        if curr_rgb_bgr is None or prev_rgb_bgr is None:
            prev_depth    = curr_depth
            prev_rgb_bgr  = curr_rgb_bgr
            prev_c2w      = c2w
            prev_frame_id = frame_id
            continue

        # LoFTR 匹配 (帧间匹配用于后续 EKF 更新)
        t0 = img_to_loftr_tensor(prev_rgb_bgr, device)
        t1 = img_to_loftr_tensor(curr_rgb_bgr, device)
        mkpts0, mkpts1 = match_loftr(t0, t1, matcher)

        # 反投影到 3D 对应点
        src_w, tgt_w = get_3d_correspondences(
            mkpts0, mkpts1,
            prev_depth, curr_depth,
            K, prev_c2w, c2w,
            dyn_mask, robot_segs
        )

        if src_w is None:
            print(f"[Frame {frame_id}] LoFTR 动态匹配点不足, 跳过")
            prev_depth    = curr_depth
            prev_rgb_bgr  = curr_rgb_bgr
            prev_frame_id = frame_id
            prev_c2w      = c2w.copy()
            continue

        # Kabsch 刚体变换
        T_step = kabsch_transform(src_w, tgt_w)

        # =====================================================================
        # 阶段一: 几何初始化（直接匹配：当前帧 vs 参考帧）
        # =====================================================================
        if not initialized:
            if ref_depth is None:
                # 第一次检测到有效动态：记录为参考帧
                ref_depth    = curr_depth.copy()
                ref_rgb_bgr  = curr_rgb_bgr.copy()
                ref_c2w      = c2w.copy()
                ref_frame_id = frame_id
                print(f"[Init] 记录第 {frame_id} 帧为参考帧 (Start)")
                
                prev_depth    = curr_depth
                prev_rgb_bgr  = curr_rgb_bgr
                prev_frame_id = frame_id
                vis.update(frame_id, None, None, None, 0., rgb_img)
                continue

            # 直接匹配：当前帧 vs 参考帧
            t_ref = img_to_loftr_tensor(ref_rgb_bgr, device)
            t_cur = img_to_loftr_tensor(curr_rgb_bgr, device)
            mkpts_ref, mkpts_cur = match_loftr(t_ref, t_cur, matcher)

            # 获得 3D 对应点对 (参考帧 -> 当前帧)
            src_w_init, tgt_w_init = get_3d_correspondences(
                mkpts_ref, mkpts_cur,
                ref_depth, curr_depth,
                K, ref_c2w, c2w,
                dyn_mask, robot_segs
            )

            if src_w_init is None:
                print(f"[Init] Frame {frame_id}: 直接匹配点不足, 跳过")
                prev_depth    = curr_depth
                prev_rgb_bgr  = curr_rgb_bgr
                prev_frame_id = frame_id
                vis.update(frame_id, None, None, None, 0., rgb_img)
                continue

            # 计算这组点对的平均位移
            avg_disp = float(np.mean(np.linalg.norm(tgt_w_init - src_w_init, axis=1)))
            print(f"[Init] Frame {frame_id}: direct_matches={len(src_w_init)}, "
                  f"avg_disp={avg_disp*100:.2f}cm (目标={MIN_INIT_DISP_M*100:.0f}cm)")

            if avg_disp >= MIN_INIT_DISP_M:
                # 使用直接匹配的点对进行 Kabsch 和 Screw 分解
                T_direct = kabsch_transform(src_w_init, tgt_w_init)
                p0, n0, theta0 = screw_from_transform(T_direct, src_w_init.mean(0))
                
                print(f"[Init] 满足位移阈值！p0={np.round(p0,3)}  n0={np.round(n0,3)}  θ0={np.rad2deg(theta0):.2f}°")
                
                ekf           = AxisEKF(p0, n0, float(theta0))
                initialized   = True
                T_accumulated = T_direct # 此时的累积变换即为 direct 变换
                print("[Init] EKF 已初始化!")

                # Matplotlib 3D 可视化：只显示这组 init 点对
                visualize_init_matplotlib(src_w_init, tgt_w_init, p0, n0)

            prev_depth    = curr_depth
            prev_rgb_bgr  = curr_rgb_bgr
            prev_frame_id = frame_id
            vis.update(frame_id, tgt_w_init, None, None, 0., rgb_img,
                       n_matches=len(src_w_init), init_disp_cm=avg_disp * 100)
            continue

        # =====================================================================
        # 阶段二: EKF 闭环
        # =====================================================================

        # 2a. 用夹爪位移估计 delta_theta（预测步）
        delta_theta_pred = 0.0
        if prev_frame_id in ee_poses and frame_id in ee_poses:
            ee_delta        = ee_poses[frame_id] - ee_poses[prev_frame_id]
            pivot_cur, n_cur, _ = ekf.get_axis()
            perp   = ee_delta - np.dot(ee_delta, n_cur) * n_cur
            r_vec  = ee_poses[frame_id] - pivot_cur
            r_vec -= np.dot(r_vec, n_cur) * n_cur
            radius = np.linalg.norm(r_vec)
            if radius > 0.02:
                delta_theta_pred = np.linalg.norm(perp) / radius
                cross = np.cross(r_vec / (radius + 1e-9),
                                 perp  / (np.linalg.norm(perp) + 1e-9))
                if np.dot(cross, n_cur) < 0:
                    delta_theta_pred = -delta_theta_pred

        # 2b. EKF Predict
        ekf.predict(delta_theta_pred)

        # 2c. 累积变换 & EKF Update
        T_accumulated = T_step @ T_accumulated
        ekf.update(T_accumulated)

        pivot, n_dir, theta = ekf.get_axis()
        print(f"[Frame {frame_id}] p={np.round(pivot,3)}  n={np.round(n_dir,3)}  "
              f"θ={np.rad2deg(theta):.2f}°  matches={len(src_w)}")

        prev_depth    = curr_depth
        prev_rgb_bgr  = curr_rgb_bgr
        prev_c2w      = c2w.copy()
        prev_frame_id = frame_id

        vis.update(frame_id, tgt_w, pivot, n_dir, theta, rgb_img, n_matches=len(src_w))

    print("\n全部帧处理完毕。关闭图窗退出。")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
