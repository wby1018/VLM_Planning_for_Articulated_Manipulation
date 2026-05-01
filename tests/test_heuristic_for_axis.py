#!/usr/bin/env python3
"""
test_heuristic_for_axis.py — Heuristic-based Rotation/Translation Axis Estimation
================================================================================
原理：
  1. 记录第一帧出现的合法动态点云作为「参考帧点云」(Reference PC)。
  2. 对于后续每一帧，将当前帧动态点云通过 ICP 配准到参考帧点云，得到累计变换 T_ref_to_curr。
  3. 对 T_ref_to_curr 进行 Screw Theory 分解，直接提取转轴参数 p, n 和 θ。
  4. 不使用 EKF，纯粹基于当前帧与初始帧的相对位姿进行启发式估计。

使用方法:
  /home/wby/active_vision/openpi/.venv/bin/python vlm_based/test_heuristic_for_axis.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R_scipy

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Config
# =============================================================================
DATA_ROOT = Path("record_pull_arc_with_joint_0")

DIFF_THRESH    = 0.002    # 深度差异阈值 (m)
MIN_AREA       = 3000     # 动态区域最小面积 (px)
MIN_DYN_PTS    = 3000     # 合法动态点数量阈值

ICP_MAX_ITER   = 50       # ICP 最大迭代次数
ICP_MAX_DIST   = 0.1      # ICP 最大匹配距离 (m)

# 机械臂圆柱半径
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

# =============================================================================
# 数据加载 (复用自 test_ekf_for_axis.py)
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
            if len(parts) < 8: continue
            fid = int(parts[0])
            t = np.array([float(x) for x in parts[1:4]])
            q = np.array([float(x) for x in parts[4:8]])  # xyzw
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
            if len(parts) < 9: continue
            fid  = int(parts[0])
            name = parts[1]
            pos  = np.array([float(x) for x in parts[2:5]])
            if fid not in all_lp: all_lp[fid] = {}
            all_lp[fid][name] = pos
    return all_lp

# =============================================================================
# 点云工具
# =============================================================================

def depth_to_world(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray):
    H, W   = depth.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    valid  = depth > 0
    x_cam  = np.where(valid, (uu - K[0,2]) / K[0,0] * depth, 0.)
    y_cam  = np.where(valid, (vv - K[1,2]) / K[1,1] * depth, 0.)
    z_cam  = np.where(valid, depth, 0.)
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)
    ones    = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
    pts_w   = (c2w @ np.hstack([pts_cam, ones]).T).T[:, :3]
    pts_w   = pts_w.reshape(H, W, 3).astype(np.float32)
    pts_w[~valid] = 0.
    return pts_w

def dist_to_segments(pts: np.ndarray, segments) -> np.ndarray:
    inside = np.zeros(len(pts), dtype=bool)
    for A, B, radius in segments:
        AB    = B - A
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

def extract_dynamic_pc(prev_depth: np.ndarray, curr_depth: np.ndarray, K: np.ndarray, c2w: np.ndarray, robot_segs):
    valid    = (prev_depth > 0) & (curr_depth > 0)
    diff     = np.abs(curr_depth.astype(np.float32) - prev_depth.astype(np.float32))
    raw_mask = valid & (diff > DIFF_THRESH)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    result = np.zeros_like(cleaned)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= MIN_AREA:
            result[labels == lbl] = 1
    dyn_mask = result.astype(bool)
    if not np.any(dyn_mask): return None, None
    world_pts_map = depth_to_world(curr_depth, K, c2w)
    dyn_pts       = world_pts_map[dyn_mask]
    valid_mask    = np.linalg.norm(dyn_pts, axis=1) > 0
    dyn_pts       = dyn_pts[valid_mask]
    if len(dyn_pts) == 0: return None, dyn_mask
    if robot_segs:
        robot_mask = dist_to_segments(dyn_pts, robot_segs)
        dyn_pts    = dyn_pts[~robot_mask]
    if len(dyn_pts) < 5: return None, dyn_mask
    return dyn_pts, dyn_mask

# =============================================================================
# ICP (Point-to-Point, SVD)
# =============================================================================

def icp_with_centroid_init(src: np.ndarray, tgt: np.ndarray, max_iter: int = ICP_MAX_ITER, max_dist: float = ICP_MAX_DIST):
    """
    src: 当前帧点云
    tgt: 参考帧点云 (Reference)
    将 src 配准到 tgt，返回 T_src_to_tgt。
    """
    src_d = src.astype(np.float64)
    tgt_d = tgt.astype(np.float64)

    # 1. 质心对齐初始化
    c_src = src_d.mean(0)
    c_tgt = tgt_d.mean(0)
    t_init = c_tgt - c_src
    src_cur = src_d + t_init

    T_total = np.eye(4, dtype=np.float64)
    T_total[:3, 3] = t_init

    prev_rmse = 1e9
    for _ in range(max_iter):
        tree = KDTree(tgt_d)
        dists, idx = tree.query(src_cur, k=1, workers=-1)
        inliers = dists < max_dist
        if np.sum(inliers) < 6: break
        
        s_in = src_cur[inliers]
        t_in = tgt_d[idx[inliers]]
        c_s = s_in.mean(0); c_t = t_in.mean(0)
        H = (s_in - c_s).T @ (t_in - c_t)
        U, S, Vt = np.linalg.svd(H)
        R_est = Vt.T @ U.T
        if np.linalg.det(R_est) < 0:
            Vt[-1] *= -1
            R_est = Vt.T @ U.T
        t_est = c_t - R_est @ c_s

        T_step = np.eye(4)
        T_step[:3, :3] = R_est
        T_step[:3, 3]  = t_est
        T_total = T_step @ T_total
        src_cur = (R_est @ src_cur.T).T + t_est

        rmse = float(np.sqrt(np.mean(dists[inliers]**2)))
        if abs(prev_rmse - rmse) < 1e-6: break
        prev_rmse = rmse

    return T_total.astype(np.float32), prev_rmse

# =============================================================================
# Screw Theory
# =============================================================================

def screw_from_transform(T: np.ndarray, p_prior: np.ndarray = None):
    R_mat = T[:3, :3].astype(np.float64)
    t_vec = T[:3, 3].astype(np.float64)
    trace = np.clip((np.trace(R_mat) - 1) / 2, -1., 1.)
    theta = float(np.arccos(trace))

    if abs(theta) < 1e-4:
        n = t_vec / (np.linalg.norm(t_vec) + 1e-9)
        perp = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
        perp -= np.dot(perp, n) * n
        perp /= (np.linalg.norm(perp) + 1e-9)
        p = perp * 5.0
        return p, n, theta

    n = np.array([R_mat[2,1] - R_mat[1,2],
                  R_mat[0,2] - R_mat[2,0],
                  R_mat[1,0] - R_mat[0,1]]) / (2 * np.sin(theta))
    n /= (np.linalg.norm(n) + 1e-9)

    t_perp = t_vec - np.dot(t_vec, n) * n
    A = np.eye(3) - R_mat

    if p_prior is not None:
        # 正则化：当 theta 小时防止 p 飞走
        lam = max(1e-3, min(1.0, (0.05 / (abs(theta) + 1e-6))**2))
        sqL = np.sqrt(lam)
        A_reg = np.vstack([A, sqL * np.eye(3)])
        b_reg = np.hstack([t_perp, sqL * p_prior])
        p, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
    else:
        p, _, _, _ = np.linalg.lstsq(A, t_perp, rcond=None)
    
    return p, n, theta

# =============================================================================
# Visualization
# =============================================================================

class Visualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 6))
        self.ax3d  = self.fig.add_subplot(131, projection='3d')
        self.ax2d  = self.fig.add_subplot(132)
        self.ax_pc = self.fig.add_subplot(133)
        self.theta_hist = []
        self.frame_hist = []

    def update(self, frame_id: int, dyn_pts, pivot, n_dir, theta: float, rgb_img=None, rmse: float = 0.):
        self.ax3d.cla()
        ax = self.ax3d
        if dyn_pts is not None:
            ax.scatter(dyn_pts[:,0], dyn_pts[:,1], dyn_pts[:,2], s=2, c='orange', alpha=0.5)
        if pivot is not None:
            ext = 0.5
            p0 = pivot - n_dir * ext
            p1 = pivot + n_dir * ext
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'c-', linewidth=3)
            ax.scatter(*pivot, s=100, c='red')
        
        ax.quiver(0,0,0, 0.2,0,0, color='r')
        ax.quiver(0,0,0, 0,0.2,0, color='g')
        ax.quiver(0,0,0, 0,0,0.2, color='b')
        ax.set_xlim(0.2, 1.4); ax.set_ylim(0.0, 1.4); ax.set_zlim(0.2, 1.4)
        ax.set_title(f"Heuristic Axis Estimation - Frame {frame_id}\nTheta: {np.degrees(theta):.2f} deg")

        self.theta_hist.append(np.degrees(theta))
        self.frame_hist.append(frame_id)
        self.ax2d.cla()
        self.ax2d.plot(self.frame_hist, self.theta_hist, 'r-')
        self.ax2d.set_title("Theta over time")
        self.ax2d.grid(True)

        if rgb_img is not None:
            self.ax_pc.cla()
            self.ax_pc.imshow(rgb_img)
            self.ax_pc.axis('off')

        plt.pause(0.01)

# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Loading dataset: {DATA_ROOT}")
    K, _, _       = load_camera_info(DATA_ROOT)
    cam_poses     = load_camera_poses(DATA_ROOT)
    link_poses_all = load_link_poses(DATA_ROOT)

    depth_files = sorted(DATA_ROOT.glob("depth/depth_*.npy"))
    rgb_dir     = DATA_ROOT / "rgb"

    vis = Visualizer()

    ref_dyn_pts = None
    ref_frame_id = None
    prev_depth = None

    for depth_file in depth_files:
        frame_id = int(depth_file.stem.split("_")[-1])
        if frame_id not in cam_poses or frame_id not in link_poses_all: continue

        curr_depth = np.load(str(depth_file)).astype(np.float32)
        c2w = cam_poses[frame_id]
        robot_segs = get_robot_segments(link_poses_all[frame_id])

        rgb_path = rgb_dir / f"rgb_{frame_id:06d}.png"
        rgb_img = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB) if rgb_path.exists() else None

        if prev_depth is None:
            prev_depth = curr_depth
            continue

        # 提取动态点云
        dyn_pts, _ = extract_dynamic_pc(prev_depth, curr_depth, K, c2w, robot_segs)
        prev_depth = curr_depth

        if dyn_pts is None or len(dyn_pts) < MIN_DYN_PTS:
            vis.update(frame_id, dyn_pts, None, None, 0.0, rgb_img)
            continue

        # 记录第一帧合法的动态点云作为参考
        if ref_dyn_pts is None:
            ref_dyn_pts = dyn_pts
            ref_frame_id = frame_id
            print(f"Set Reference Frame: {ref_frame_id} (pts: {len(dyn_pts)})")
            vis.update(frame_id, dyn_pts, None, None, 0.0, rgb_img)
            continue

        # 启发式估计：当前帧 vs 参考帧
        # 将当前帧配准到参考帧，得到 T_curr_to_ref
        T_curr_to_ref, rmse = icp_with_centroid_init(dyn_pts, ref_dyn_pts)
        
        # 将其逆转为物体的运动 T_ref_to_curr
        T_ref_to_curr = np.linalg.inv(T_curr_to_ref)

        # Screw theory 提取转轴
        # 使用参考帧点云的质心作为正则化项，防止小位移时 Pivot 乱跑
        p_prior = ref_dyn_pts.mean(0)
        pivot, n_dir, theta = screw_from_transform(T_ref_to_curr, p_prior=p_prior)

        print(f"[Frame {frame_id}] Theta: {np.degrees(theta):.2f} deg, RMSE: {rmse:.4f}")
        vis.update(frame_id, dyn_pts, pivot, n_dir, theta, rgb_img, rmse)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
