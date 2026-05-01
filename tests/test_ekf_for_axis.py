#!/usr/bin/env python3
"""
test_ekf_for_axis.py — EKF-based Rotation/Translation Axis Estimation
=======================================================================
两阶段方法：
  阶段一: 累积有效动态点云帧，用 ICP + Screw Theory 初始化转轴参数
  阶段二: EKF 闭环，每帧用预测性 ICP 观测修正转轴

EKF 状态: x = [px, py, pz, az, el, theta]  (6-dim)
  (px, py, pz): 转轴上距原点最近的点
  (az, el):     轴方向的球坐标（方位角、仰角）
  theta:        累积旋转角（弧度）

预测性 ICP:
  1. 用 EKF 预测的 delta_theta 计算预测增量变换 T_pred
  2. 将上一帧动态点云用 T_pred 变换到预测位置
  3. 对 (当前动态点云 → 预测位置点云) 运行 ICP，得到修正变换 T_corr
  4. 累积总变换 T_acc = T_corr @ T_pred @ T_acc_prev
  5. 对 T_acc 做 Screw 分解，直接观测转轴参数

使用方法:
  /home/wby/active_vision/openpi/.venv/bin/python vlm_based/test_ekf_for_axis.py
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
MIN_AREA       = 3000      # 动态区域最小面积 (px)
MIN_DYN_PTS    = 3000       # 启动 EKF 需要的最小动态点数量
INIT_FRAMES    = 2        # 初始化需要的最少连续有效帧数
MIN_INIT_THETA_DEG = 5.0  # 初始化所需的最小累积旋转角 (度)，小于此角 screw 分解得 p 不可靠
ICP_MAX_ITER   = 30       # ICP 最大迭代次数
ICP_MAX_DIST   = 0.08     # ICP 最大匹配距离 (m)
ICP_INIT_MAX_DIST = 0.04  # 初始化 ICP 最大匹配距离 (更小, 局部 ICP)
ICP_INIT_RMSE_THR = 0.015 # 初始化 ICP RMSE 阈值，超过则视为失败
ICP_CORR_ITER  = 10       # EKF 更新时 ICP 最大迭代次数

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

# EKF 噪声 [px, py, pz, az, el, theta]
Q_DIAG = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])  # 过程噪声
R_DIAG = np.array([2e-2, 2e-2, 2e-2, 5e-2, 5e-2, 1e-2])  # 观测噪声

# =============================================================================
# 数据加载
# =============================================================================

def load_camera_info(root: Path):
    """返回 3x3 内参矩阵 K 及图像尺寸 (W, H)."""
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
    """返回 {frame_id: c2w_4x4}. 约定: Z-forward (OpenCV)."""
    poses = {}
    with open(root / "camera_pose.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            fid = int(parts[0])
            t = np.array([float(x) for x in parts[1:4]])
            q = np.array([float(x) for x in parts[4:8]])  # xyzw
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R_scipy.from_quat(q).as_matrix()
            c2w[:3, 3] = t
            poses[fid] = c2w
    return poses


def load_link_poses(root: Path):
    """返回 {frame_id: {link_name: pos_3d}}."""
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
    """返回 {frame_id: pos_3d}."""
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

def depth_to_world(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray):
    """(H,W) depth -> (H,W,3) world points. 无效像素 = [0,0,0]."""
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
    """pts: (N,3). 返回 bool mask，True = 在某段圆柱内。"""
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

# =============================================================================
# 动态点云提取
# =============================================================================

def extract_dynamic_pc(prev_depth: np.ndarray,
                        curr_depth: np.ndarray,
                        K: np.ndarray,
                        c2w: np.ndarray,
                        robot_segs,
                        diff_thresh: float = DIFF_THRESH,
                        min_area:    int   = MIN_AREA):
    """
    返回 (dyn_pts_world, dyn_mask_2d).
    若无有效动态点则返回 (None, None).
    """
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
    dyn_mask = result.astype(bool)

    if not np.any(dyn_mask):
        return None, None

    # 反投影动态像素到世界坐标
    world_pts_map = depth_to_world(curr_depth, K, c2w)   # (H, W, 3)
    dyn_pts       = world_pts_map[dyn_mask]              # (M, 3)
    valid_mask    = np.linalg.norm(dyn_pts, axis=1) > 0
    dyn_pts       = dyn_pts[valid_mask]

    if len(dyn_pts) == 0:
        return None, dyn_mask

    # 去除机械臂点云
    if robot_segs:
        robot_mask = dist_to_segments(dyn_pts, robot_segs)
        dyn_pts    = dyn_pts[~robot_mask]
        if len(dyn_pts) < 5:
            return None, dyn_mask

    return dyn_pts, dyn_mask

# =============================================================================
# ICP (Point-to-Point, SVD)
# =============================================================================

def icp(src: np.ndarray, tgt: np.ndarray,
        max_iter: int = ICP_MAX_ITER,
        max_dist: float = ICP_MAX_DIST):
    """
    src, tgt: (N,3). 将 src 配准到 tgt.
    返回 (T_4x4, rmse).
    """
    T_total = np.eye(4, dtype=np.float64)
    src_cur = src.copy().astype(np.float64)
    prev_rmse = 1e9

    for _ in range(max_iter):
        tree = KDTree(tgt)
        dists, idx = tree.query(src_cur, k=1, workers=-1)
        inliers = dists < max_dist

        if np.sum(inliers) < 6:
            break

        s_in = src_cur[inliers]
        t_in = tgt[idx[inliers]].astype(np.float64)

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
        if abs(prev_rmse - rmse) < 1e-5:
            break
        prev_rmse = rmse

    final_rmse = prev_rmse if prev_rmse < 1e9 else 1e6
    return T_total.astype(np.float32), final_rmse


def icp_local(src: np.ndarray, tgt: np.ndarray,
             max_iter: int = ICP_MAX_ITER,
             max_dist: float = ICP_INIT_MAX_DIST):
    """
    局部 ICP：先以质心对齐作为初值，再用小 max_dist 配准。
    适用于帧间小运动的初始化场景，防止大距离误匹配。
    返回 (T_4x4, rmse, inlier_ratio).
    """
    src_d = src.astype(np.float64)
    tgt_d = tgt.astype(np.float64)

    # 步骤 1: 质心对齐（平移初始猜测）
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

        inlier_ratio = np.mean(inliers)
        if np.sum(inliers) < 6 or inlier_ratio < 0.1:
            break

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
        if abs(prev_rmse - rmse) < 1e-6:
            break
        prev_rmse = rmse

    # 最终评估
    tree = KDTree(tgt_d)
    dists_final, _ = tree.query(src_cur, k=1)
    inliers_final = dists_final < max_dist
    rmse_final    = float(np.sqrt(np.mean(dists_final[inliers_final]**2))) \
                    if np.any(inliers_final) else 1e6
    inlier_ratio  = float(np.mean(inliers_final))

    return T_total.astype(np.float32), rmse_final, inlier_ratio

# =============================================================================
# Screw Theory
# =============================================================================

def rot_matrix_np(n: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues 公式. n 应为单位向量."""
    n = n / (np.linalg.norm(n) + 1e-9)
    K = np.array([[    0, -n[2],  n[1]],
                  [ n[2],     0, -n[0]],
                  [-n[1],  n[0],     0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def screw_transform_np(p: np.ndarray, n: np.ndarray, theta: float) -> np.ndarray:
    """绕通过 p 点且方向为 n 的轴旋转 theta 弧度的 4x4 变换矩阵."""
    R_mat = rot_matrix_np(n, theta)
    t_vec = (np.eye(3) - R_mat) @ p
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3]  = t_vec
    return T.astype(np.float32)


def screw_from_transform(T: np.ndarray, p_prior: np.ndarray = None):
    """
    将 4x4 变换矩阵分解为 (pivot_p, axis_n, theta).

    p_prior: 点云质心（用于正则化）。当旋转角小时，(I-R)p=t_perp
              近于奇异，正则化可防止 p 飞到远处。
    """
    R_mat = T[:3, :3].astype(np.float64)
    t_vec = T[:3, 3].astype(np.float64)

    trace = np.clip((np.trace(R_mat) - 1) / 2, -1., 1.)
    theta = float(np.arccos(trace))

    if abs(theta) < 1e-4:
        # 近乎纯平移
        n = t_vec / (np.linalg.norm(t_vec) + 1e-9)
        # 把转轴置于极远处（沿与 t 垂直的方向）
        perp = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
        perp -= np.dot(perp, n) * n
        perp /= np.linalg.norm(perp)
        p = perp * 5.0  # 5m 之外
        return p, n, theta

    # 提取轴方向
    n = np.array([R_mat[2,1] - R_mat[1,2],
                  R_mat[0,2] - R_mat[2,0],
                  R_mat[1,0] - R_mat[0,1]]) / (2 * np.sin(theta))
    n /= (np.linalg.norm(n) + 1e-9)

    # 求轴上离原点最近的点 (I-R)p = t_perp
    t_perp = t_vec - np.dot(t_vec, n) * n
    A = np.eye(3) - R_mat

    if p_prior is None:
        # 无先验：直接最小二乘
        p, _, _, _ = np.linalg.lstsq(A, t_perp, rcond=None)
    else:
        # 正则化最小二乘: min ||Ap - t_perp||^2 + lam * ||p - p_prior||^2
        # lam: theta 越小正则化越强，防止 p 飞向远处
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
    """
    状态: x = [px, py, pz, az, el, theta]
    观测: 通过 ICP 累积变换的 Screw 分解得到 (p_obs, n_obs, theta_obs)
    """

    def __init__(self, p0: np.ndarray, n0: np.ndarray, theta0: float):
        n0   = n0 / (np.linalg.norm(n0) + 1e-9)
        az0  = float(np.arctan2(n0[1], n0[0]))
        el0  = float(np.arcsin(np.clip(n0[2], -1., 1.)))
        self.x = np.array([p0[0], p0[1], p0[2], az0, el0, theta0],
                           dtype=np.float64)
        # 初始协方差
        self.P = np.diag([0.05, 0.05, 0.05, 0.2, 0.2, 0.05])
        self.Q = np.diag(Q_DIAG)
        self.R = np.diag(R_DIAG)

    def predict(self, delta_theta: float):
        """
        状态转移: 轴不变，theta += delta_theta.
        F = I (轴参数的偏导数均为 1).
        """
        self.x[5] += delta_theta
        self.P = self.P + self.Q   # F=I, so P = P + Q

    def update(self, T_accumulated: np.ndarray):
        """
        T_accumulated: 从初始帧到当前帧的累积变换 (4x4).
        对其做 Screw 分解，得到 (p_obs, n_obs, theta_obs) 作为观测 z.
        観测方程 h(x) ≈ x (直接状态观测), H = I.
        """
        p_obs, n_obs, theta_obs = screw_from_transform(T_accumulated)

        # 将 n_obs 转为球坐标
        n_obs = n_obs / (np.linalg.norm(n_obs) + 1e-9)
        az_obs = float(np.arctan2(n_obs[1], n_obs[0]))
        el_obs = float(np.arcsin(np.clip(n_obs[2], -1., 1.)))

        # 处理方位角符号歧义 (轴可以有两个方向)
        az_cur, el_cur = self.x[3], self.x[4]
        n_cur = az_el_to_n(az_cur, el_cur)
        if np.dot(n_obs, n_cur) < 0:
            n_obs  = -n_obs
            az_obs = float(np.arctan2(n_obs[1], n_obs[0]))
            el_obs = float(np.arcsin(np.clip(n_obs[2], -1., 1.)))
            theta_obs = -theta_obs

        z = np.array([p_obs[0], p_obs[1], p_obs[2], az_obs, el_obs, theta_obs])

        # H = I (direct state observation)
        H = np.eye(6)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        innovation = z - self.x
        # 角度环绕校正
        for idx in [3, 4, 5]:
            while innovation[idx] >  np.pi: innovation[idx] -= 2 * np.pi
            while innovation[idx] < -np.pi: innovation[idx] += 2 * np.pi

        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P

    def get_axis(self):
        """返回 (pivot_p, n, theta)."""
        px, py, pz, az, el, theta = self.x
        n = az_el_to_n(az, el)
        return np.array([px, py, pz]), n, float(theta)

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

    def update(self, frame_id: int, dyn_pts, pivot, n_dir, theta: float,
               rgb_img=None, rmse: float = 0.):
        # --- 3D Axes ---
        self.ax3d.cla()
        ax = self.ax3d
        if dyn_pts is not None and len(dyn_pts) > 0:
            ax.scatter(dyn_pts[:,0], dyn_pts[:,1], dyn_pts[:,2],
                       s=3, c='orange', alpha=0.7, label=f'Dyn PC ({len(dyn_pts)})')
        if pivot is not None:
            ext = 0.5
            p0  = pivot - n_dir * ext
            p1  = pivot + n_dir * ext
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    'c-', linewidth=3, label='Est. Axis')
            ax.scatter(*pivot, s=120, c='red', zorder=6, label='Pivot')
        # 世界原点坐标轴
        ax.quiver(0,0,0, 0.3,0,0, color='r', arrow_length_ratio=0.3)
        ax.quiver(0,0,0, 0,0.3,0, color='g', arrow_length_ratio=0.3)
        ax.quiver(0,0,0, 0,0,0.3, color='b', arrow_length_ratio=0.3)
        ax.set_xlim(0.2, 1.6); ax.set_ylim(0.0, 1.5); ax.set_zlim(0.2, 1.4)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_id}\n'
                     f'θ={np.rad2deg(theta):.1f}°  ICP_rmse={rmse:.4f}')
        if pivot is not None:
            ax.legend(fontsize=7, loc='upper left')

        # --- θ over time ---
        self.theta_hist.append(np.rad2deg(theta))
        self.frame_hist.append(frame_id)
        self.ax2d.cla()
        self.ax2d.plot(self.frame_hist, self.theta_hist, 'b-o', markersize=3)
        self.ax2d.set_xlabel('Frame'); self.ax2d.set_ylabel('θ (°)')
        self.ax2d.set_title('Cumulative θ')
        self.ax2d.grid(True)

        # --- RGB overlay ---
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
    K, W, H       = load_camera_info(DATA_ROOT)
    cam_poses     = load_camera_poses(DATA_ROOT)
    link_poses_all = load_link_poses(DATA_ROOT)
    ee_poses      = load_ee_poses(DATA_ROOT)

    depth_files = sorted(DATA_ROOT.glob("depth/depth_*.npy"))
    rgb_dir     = DATA_ROOT / "rgb"
    total       = len(depth_files)
    print(f"[Main] 共 {total} 帧")

    vis = Visualizer()

    prev_depth    = None
    prev_dyn_pts  = None    # 上一帧有效动态点云（世界坐标）
    prev_frame_id = None
    dyn_buf       = []      # 初始化用缓冲：存 [(frame_id, dyn_pts)]

    ekf           = None
    initialized   = False
    T_accumulated = np.eye(4, dtype=np.float32)  # 物体从参考帧到当前帧的累积变换

    for depth_file in depth_files:
        frame_id = int(depth_file.stem.split("_")[-1])

        if frame_id not in cam_poses or frame_id not in link_poses_all:
            continue

        c2w       = cam_poses[frame_id]
        lp_dict   = link_poses_all[frame_id]
        robot_segs = get_robot_segments(lp_dict)

        curr_depth = np.load(str(depth_file)).astype(np.float32)

        # 加载 RGB
        rgb_path = rgb_dir / f"rgb_{frame_id:06d}.png"
        rgb_img  = None
        if rgb_path.exists():
            bgr = cv2.imread(str(rgb_path))
            if bgr is not None:
                rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if prev_depth is None:
            prev_depth    = curr_depth
            prev_frame_id = frame_id
            continue

        # --- 提取动态点云 ---
        dyn_pts, _ = extract_dynamic_pc(prev_depth, curr_depth, K, c2w, robot_segs)
        prev_depth = curr_depth

        n_dyn = 0 if dyn_pts is None else len(dyn_pts)
        if n_dyn < MIN_DYN_PTS:
            print(f"[Frame {frame_id}] 动态点不足 ({n_dyn}), 等待...")
            vis.update(frame_id, dyn_pts, None, None, 0., rgb_img)
            prev_frame_id = frame_id
            continue

        print(f"[Frame {frame_id}] 动态点: {n_dyn}")

        # =====================================================================
        # 阶段一: 几何初始化（局部 ICP 多帧累积直到转角足够大）
        # =====================================================================
        if not initialized:
            if prev_dyn_pts is None:
                prev_dyn_pts  = dyn_pts
                prev_frame_id = frame_id
                vis.update(frame_id, dyn_pts, None, None, 0., rgb_img)
                continue

            # 局部 ICP：当前帧 → 上一帧
            T_step, rmse, inlier_ratio = icp_local(dyn_pts, prev_dyn_pts)
            print(f"[Init] Frame {frame_id}: ICP RMSE={rmse:.4f}  inlier={inlier_ratio:.2f}")

            if rmse > ICP_INIT_RMSE_THR or inlier_ratio < 0.2:
                print(f"[Init]   ICP 质量不足，放弃本帧，重置积累...")
                T_accumulated = np.eye(4, dtype=np.float32)
                prev_dyn_pts  = dyn_pts
                prev_frame_id = frame_id
                vis.update(frame_id, dyn_pts, None, None, 0., rgb_img)
                continue

            # 连续累积：T_acc = T_step @ T_acc_prev
            T_accumulated = T_step.astype(np.float32) @ T_accumulated

            # 检查累积旋转角
            trace_acc = np.clip((np.trace(T_accumulated[:3, :3]) - 1) / 2, -1., 1.)
            theta_acc  = float(np.degrees(np.arccos(trace_acc)))
            print(f"[Init]   累积旋转角: {theta_acc:.2f}° (目标={MIN_INIT_THETA_DEG}°)")

            if theta_acc >= MIN_INIT_THETA_DEG:
                # 虚转足够大，进行 screw 分解
                print(f"[Init] 累积旋转角已达 {theta_acc:.2f}° ≥ {MIN_INIT_THETA_DEG}°，开始 Screw 分解...")
                p0, n0, theta0 = screw_from_transform(T_accumulated)
                print(f"[Init] p0={np.round(p0,3)}  n0={np.round(n0,3)}  θ0={np.rad2deg(theta0):.2f}°")

                if np.linalg.norm(p0) > 4.0:
                    print("[Init] 检测到近似平移模式 (|p| 较大)")

                ekf           = AxisEKF(p0, n0, float(theta0))
                # T_accumulated 就是初始化时的累积变换，直接用于跟踪
                initialized   = True
                prev_dyn_pts  = dyn_pts
                print("[Init] EKF 已初始化!")
            else:
                prev_dyn_pts  = dyn_pts

            prev_frame_id = frame_id
            vis.update(frame_id, dyn_pts, None, None, np.rad2deg(theta_acc), rgb_img)
            continue

        # =====================================================================
        # 阶段二: EKF 闭环
        # =====================================================================

        # 2a. 用夹爪位移估计 delta_theta
        delta_theta_pred = 0.0
        if prev_frame_id in ee_poses and frame_id in ee_poses:
            ee_delta = ee_poses[frame_id] - ee_poses[prev_frame_id]
            pivot_cur, n_cur, theta_cur = ekf.get_axis()
            perp = ee_delta - np.dot(ee_delta, n_cur) * n_cur
            r_vec = ee_poses[frame_id] - pivot_cur
            r_vec -= np.dot(r_vec, n_cur) * n_cur
            radius = np.linalg.norm(r_vec)
            if radius > 0.02:
                delta_theta_pred = np.linalg.norm(perp) / radius
                # 判断方向
                cross = np.cross(r_vec / (radius + 1e-9), perp / (np.linalg.norm(perp) + 1e-9))
                if np.dot(cross, n_cur) < 0:
                    delta_theta_pred = -delta_theta_pred

        # 2b. EKF Predict
        ekf.predict(delta_theta_pred)
        pivot, n_dir, theta = ekf.get_axis()

        # 2c. 预测性 ICP
        rmse = 0.
        if prev_dyn_pts is not None and len(prev_dyn_pts) >= 6:
            # 用预测增量变换将上一帧动态点云变换到预测的当前位置
            T_pred      = screw_transform_np(pivot, n_dir, delta_theta_pred)
            prev_warped = (T_pred[:3,:3] @ prev_dyn_pts.T).T + T_pred[:3,3]

            # ICP: 当前动态点 → 预测的参考位置
            T_corr, rmse = icp(dyn_pts, prev_warped, max_iter=ICP_CORR_ITER,
                               max_dist=ICP_MAX_DIST * 1.5)
            print(f"[Frame {frame_id}] ICP 修正 RMSE = {rmse:.4f}")

            # 累积总变换: T_acc_new = T_corr @ T_pred @ T_acc_old
            T_accumulated = T_corr.astype(np.float32) @ \
                            T_pred.astype(np.float32) @ \
                            T_accumulated

            # 2d. EKF Update
            ekf.update(T_accumulated)
            pivot, n_dir, theta = ekf.get_axis()

        prev_dyn_pts  = dyn_pts
        prev_frame_id = frame_id

        print(f"[Frame {frame_id}] p={np.round(pivot,3)}  n={np.round(n_dir,3)}  θ={np.rad2deg(theta):.2f}°")

        vis.update(frame_id, dyn_pts, pivot, n_dir, theta, rgb_img, rmse)

    print("\n全部帧处理完毕。关闭图窗退出。")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
