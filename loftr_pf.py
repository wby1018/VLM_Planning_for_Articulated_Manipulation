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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =============================================================================
# Config
# =============================================================================
DATA_ROOT = Path("record_pull_arc_with_joint_0")

DIFF_THRESH       = 0.001
MIN_AREA          = 3000
MIN_DYN_MATCHES   = 15
LOFTR_CONF_THRESH = 0.5
LOFTR_SIZE        = (640, 480)
MIN_INIT_DISP_M   = 0.08

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

# PF 超参数
N_PARTICLES    = 500
SIGMA_KIN      = 0.003   # 夹爪 3D 重投影误差标准差 (m)
SIGMA_RAD      = 0.001  # 半径一致性误差标准差 (m)，强约束
SIGMA_VIS      = 0.001  # 视觉点云误差标准差 (m) (原 0.01，调小以增加权重)
PROC_NOISE_W   = 1e-8   # 极小值，锁定转轴方向 (几乎不动)
PROC_NOISE_V   = 2e-5   # 较大值，允许转轴位置快速平移收敛 (快速变)
PROC_NOISE_T   = 5e-4   # theta 噪声
ESS_RATIO      = 0.5    # 有效粒子数比例阈值，低于此重采样

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
    resized = cv2.resize(bgr_img, size)
    t = KN.image_to_tensor(resized, keepdim=False).float() / 255.
    return KN.color.rgb_to_grayscale(t).to(device)


def match_loftr(t0: torch.Tensor, t1: torch.Tensor, matcher,
                conf_thresh: float = LOFTR_CONF_THRESH):
    with torch.no_grad():
        corr = matcher({"image0": t0, "image1": t1})
    mkpts0 = corr['keypoints0'].cpu().numpy()
    mkpts1 = corr['keypoints1'].cpu().numpy()
    conf   = corr['confidence'].cpu().numpy()
    mask   = conf > conf_thresh
    return mkpts0[mask], mkpts1[mask]

# =============================================================================
# 3D 对应点提取
# =============================================================================

def get_3d_correspondences(mkpts0: np.ndarray, mkpts1: np.ndarray,
                            depth0: np.ndarray, depth1: np.ndarray,
                            K: np.ndarray,
                            c2w0: np.ndarray, c2w1: np.ndarray,
                            dyn_mask_curr: np.ndarray,
                            robot_segs):
    H, W  = depth0.shape
    lw, lh = LOFTR_SIZE
    sx = W / lw
    sy = H / lh

    px0 = mkpts0[:, 0] * sx;  py0 = mkpts0[:, 1] * sy
    px1 = mkpts1[:, 0] * sx;  py1 = mkpts1[:, 1] * sy

    ix0 = np.clip(np.round(px0).astype(int), 0, W - 1)
    iy0 = np.clip(np.round(py0).astype(int), 0, H - 1)
    ix1 = np.clip(np.round(px1).astype(int), 0, W - 1)
    iy1 = np.clip(np.round(py1).astype(int), 0, H - 1)

    z0 = depth0[iy0, ix0].astype(np.float64)
    z1 = depth1[iy1, ix1].astype(np.float64)

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

    if robot_segs:
        bad = dist_to_segments(src_w, robot_segs) | dist_to_segments(tgt_w, robot_segs)
        src_w = src_w[~bad]
        tgt_w = tgt_w[~bad]

    if len(src_w) < MIN_DYN_MATCHES:
        return None, None

    return src_w.astype(np.float32), tgt_w.astype(np.float32)

# =============================================================================
# Kabsch SVD 刚体变换
# =============================================================================

def kabsch_transform(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
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
# Screw Theory 工具
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


def p_to_v(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """轴上一点 p 和方向 ω → 螺旋线速度分量 v = -ω × p."""
    return -np.cross(omega, p)


def v_to_p(v: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """螺旋参数 (ω, v) → 轴上距原点最近的点 p = ω × v."""
    return np.cross(omega, v)

# =============================================================================
# 粒子滤波器
# =============================================================================

class AxisPF:
    """
    粒子螺旋参数 ξ_i = (ω_i, v_i, θ_i)，共 7 维：
      ω  (omega): 旋转轴方向（单位向量）
      v:          线速度分量，满足 v·ω=0，隐含轴位置 p = ω×v
      θ  (theta): 累积旋转角 (rad)

    转换关系：
      p → v :  v = p_to_v(p, ω) = -ω×p
      v → p :  p = v_to_p(v, ω) =  ω×v
    """

    def __init__(self, omega0: np.ndarray, v0: np.ndarray, theta0: float,
                 n_particles: int = N_PARTICLES):
        """
        直接以螺旋参数初始化。
        omega0: 旋转轴方向（会被归一化）
        v0:     线速度分量（会被投影至 ω⊥）
        theta0: 初始累积旋转角
        """
        self.N  = n_particles
        omega0  = omega0 / (np.linalg.norm(omega0) + 1e-9)
        # 确保 v0 ⊥ omega0
        v0 = v0 - np.dot(v0, omega0) * omega0

        # 以先验 ξ=(omega0, v0, theta0) 为中心高斯采样
        omega_noise = np.random.randn(self.N, 3) * 0.05
        v_noise     = np.random.randn(self.N, 3) * 0.05
        theta_noise = np.random.randn(self.N)    * 0.05

        omegas = omega0 + omega_noise
        norms  = np.linalg.norm(omegas, axis=1, keepdims=True)
        self.omegas = omegas / (norms + 1e-9)               # (N, 3)

        vs_raw = v0 + v_noise
        dot    = np.sum(vs_raw * self.omegas, axis=1, keepdims=True)
        self.vs = vs_raw - dot * self.omegas                 # (N, 3) 保证 v·ω=0

        self.thetas  = np.full(self.N, theta0) + theta_noise # (N,)
        self.weights = np.ones(self.N) / self.N

    # ------------------------------------------------------------------
    # 预测步：加小噪声，保持粒子多样性
    # ------------------------------------------------------------------
    def predict(self, delta_theta: float = 0.0):
        omega_noise = np.random.randn(self.N, 3) * PROC_NOISE_W
        v_noise     = np.random.randn(self.N, 3) * PROC_NOISE_V
        theta_noise = np.random.randn(self.N)    * PROC_NOISE_T

        self.omegas = self.omegas + omega_noise
        norms = np.linalg.norm(self.omegas, axis=1, keepdims=True)
        self.omegas = self.omegas / (norms + 1e-9)

        vs_raw = self.vs + v_noise
        dot    = np.sum(vs_raw * self.omegas, axis=1, keepdims=True)
        self.vs = vs_raw - dot * self.omegas

        self.thetas += delta_theta + theta_noise

    # ------------------------------------------------------------------
    # 每粒子独立解算 Δθ₁
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_delta_theta(p: np.ndarray, omega: np.ndarray,
                              c_prev: np.ndarray, c_curr: np.ndarray) -> float:
        """
        在粒子轴 (ω, p) 定义的圆面上，从 EE 位移解算最优旋转角 Δθ。
        """
        v_prev = c_prev - p
        v_curr = c_curr - p
        # 投影到 ω 的正交补
        v_prev_proj = v_prev - np.dot(v_prev, omega) * omega
        v_curr_proj = v_curr - np.dot(v_curr, omega) * omega
        norm_p = np.linalg.norm(v_prev_proj)
        norm_c = np.linalg.norm(v_curr_proj)
        if norm_p < 1e-6 or norm_c < 1e-6:
            return 0.0
        cross     = np.cross(v_prev_proj, v_curr_proj)
        sin_theta = np.dot(cross, omega) / (norm_p * norm_c)
        cos_theta = np.dot(v_prev_proj, v_curr_proj) / (norm_p * norm_c)
        return float(np.arctan2(sin_theta, cos_theta))

    # ------------------------------------------------------------------
    # 权重更新：基于每粒子独立几何验证
    # ------------------------------------------------------------------
    def update(self, src_pts: np.ndarray, tgt_pts: np.ndarray,
               ee_pos_prev: np.ndarray, ee_pos_curr: np.ndarray):
        """
        src_pts, tgt_pts : (M,3) 当前帧间 LoFTR 3D 对应点对
        ee_pos_prev/curr : (3,)  夹爪末端位置（世界系）

        每个粒子独立进行三项评分：
          A. 利用 EE 位移解算该粒子下的专属 Δθᵢ
          B. 运动学误差: ||EE_curr - R(ωᵢ,Δθᵢ)(EE_prev-pᵢ) - pᵢ||
          C. 半径一致性: ||r_curr - r_prev|| (刺性约束)
          D. 视觉点云误差: 用 Δθᵢ 旋转 src_pts 后与 tgt_pts 的匹配误差
        """
        log_w = np.zeros(self.N)
        has_ee  = (ee_pos_prev is not None) and (ee_pos_curr is not None)
        has_vis = (src_pts is not None) and (len(src_pts) > 0)

        for i in range(self.N):
            omega_i = self.omegas[i]
            p_i     = v_to_p(self.vs[i], omega_i)

            # --- A. 独立解算 Δθᵢ ---
            if has_ee:
                dtheta_i = self._estimate_delta_theta(p_i, omega_i, ee_pos_prev, ee_pos_curr)
            else:
                dtheta_i = self.thetas[i]   # fallback

            R_i = rot_matrix_np(omega_i, dtheta_i)

            # --- B. 运动学误差 ---
            if has_ee:
                ee_pred  = R_i @ (ee_pos_prev - p_i) + p_i
                r_kin    = float(np.linalg.norm(ee_pos_curr - ee_pred))
                log_w[i] += -0.5 * (r_kin / SIGMA_KIN) ** 2

                # --- C. 半径一致性约束 ---
                r_prev = np.linalg.norm(np.cross(omega_i, ee_pos_prev - p_i))
                r_curr = np.linalg.norm(np.cross(omega_i, ee_pos_curr - p_i))
                r_rad  = abs(r_curr - r_prev)
                log_w[i] += -0.5 * (r_rad / SIGMA_RAD) ** 2

            # --- D. 视觉点云误差 ---
            if has_vis:
                pts_pred = (R_i @ (src_pts - p_i).T).T + p_i
                r_vis    = float(np.mean(np.linalg.norm(tgt_pts - pts_pred, axis=1)))
                log_w[i] += -0.5 * (r_vis / SIGMA_VIS) ** 2

        # 数值稳定
        log_w -= log_w.max()
        w_new  = self.weights * np.exp(log_w)
        w_sum  = w_new.sum()
        self.weights = w_new / w_sum if w_sum > 1e-300 else np.ones(self.N) / self.N

        # 有效粒子数检测 & 重采样
        ess = 1.0 / (np.sum(self.weights ** 2) + 1e-300)
        if ess < ESS_RATIO * self.N:
            self._resample()

    # ------------------------------------------------------------------
    # 系统重采样 (Systematic Resampling)
    # ------------------------------------------------------------------
    def _resample(self):
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0
        positions = (np.arange(self.N) + np.random.uniform()) / self.N
        indices   = np.searchsorted(cumsum, positions)
        self.omegas  = self.omegas[indices]
        self.vs      = self.vs[indices]
        self.thetas  = self.thetas[indices]
        self.weights = np.ones(self.N) / self.N

    # ------------------------------------------------------------------
    # 加权均值估计
    # ------------------------------------------------------------------
    def get_axis(self):
        w = self.weights

        # 加权方向均值（用符号修正保持朝向一致）
        omega_mean = self.omegas[0].copy()
        signs = np.sign(self.omegas @ omega_mean)
        signs[signs == 0] = 1.0
        omega_w = (w[:, None] * signs[:, None] * self.omegas).sum(0)
        omega_w /= (np.linalg.norm(omega_w) + 1e-9)

        v_w     = (w[:, None] * self.vs).sum(0)
        # 重新投影，确保 v·ω=0
        v_w    -= np.dot(v_w, omega_w) * omega_w

        theta_w = float(np.dot(w, self.thetas))
        p_w     = v_to_p(v_w, omega_w)
        return p_w, omega_w, theta_w

    def get_uncertainty(self):
        """返回 (sigma_p, sigma_theta) 加权标准差。"""
        w = self.weights
        _, omega_mean, theta_mean = self.get_axis()

        ps = np.array([v_to_p(self.vs[i], self.omegas[i]) for i in range(self.N)])
        p_mean = (w[:, None] * ps).sum(0)
        sigma_p = float(np.sqrt((w * np.sum((ps - p_mean)**2, axis=1)).sum()))
        sigma_t = float(np.sqrt((w * (self.thetas - theta_mean)**2).sum()))
        return sigma_p, sigma_t

# =============================================================================
# Matplotlib Visualization for Initialization
# =============================================================================

def visualize_init_matplotlib(src_pts, tgt_pts, pivot, n_dir):
    print(f"\n[Visualizer] src={len(src_pts)} pts, tgt={len(tgt_pts)} pts")
    print("[Visualizer] 显示初始化可视化... 关闭窗口继续。")
    was_interactive = plt.isinteractive()
    plt.ioff()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    skip = max(1, len(src_pts) // 500)
    ax.scatter(src_pts[::skip, 0], src_pts[::skip, 1], src_pts[::skip, 2],
               c='red', s=2, alpha=0.5, label='Start (Ref)')
    ax.scatter(tgt_pts[::skip, 0], tgt_pts[::skip, 1], tgt_pts[::skip, 2],
               c='green', s=2, alpha=0.5, label='Current (Init)')

    ext = 1.0
    p0 = pivot - n_dir * ext
    p1 = pivot + n_dir * ext
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            'b-', linewidth=3, label='Est. Axis')
    ax.scatter(pivot[0], pivot[1], pivot[2], c='blue', s=100, marker='X')

    ax.quiver(0,0,0, 0.2,0,0, color='r')
    ax.quiver(0,0,0, 0,0.2,0, color='g')
    ax.quiver(0,0,0, 0,0,0.2, color='b')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("PF Initialization: Start(Red) vs End(Green)")
    ax.legend()

    all_pts = np.vstack([src_pts, tgt_pts])
    center  = all_pts.mean(0)
    max_r   = np.ptp(all_pts, axis=0).max() / 2.0
    ax.set_xlim(center[0]-max_r, center[0]+max_r)
    ax.set_ylim(center[1]-max_r, center[1]+max_r)
    ax.set_zlim(center[2]-max_r, center[2]+max_r)

    # 使用阻塞式显示，但只针对当前窗口
    fig.canvas.draw()
    plt.show(block=False)
    
    # 循环等待直到当前窗口被手动关闭
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
    
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
               rgb_img=None, n_matches: int = 0, init_disp_cm: float = None,
               sigma_p: float = None):
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
            sp_str = f'  σp={sigma_p*100:.1f}cm' if sigma_p is not None else ''
            # title = f'Frame {frame_id}\nθ={np.rad2deg(theta):.1f}°  matches={n_matches}{sp_str}'
        ax.set_title(title)
        if pivot is not None:
            ax.legend(fontsize=7, loc='upper left')

        self.val_hist.append(init_disp_cm if init_disp_cm is not None else np.rad2deg(theta))
        self.frame_hist.append(frame_id)
        self.ax2d.cla()
        self.ax2d.plot(self.frame_hist, self.val_hist, 'b-o', markersize=3)
        self.ax2d.set_xlabel('Frame')
        self.ax2d.set_ylabel('avg_disp (cm) / θ (°)')
        self.ax2d.set_title('Init disp / PF θ')
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
# LoFTRAxisEstimator — 可调用的封装类
# =============================================================================

class LoFTRAxisEstimator:
    """
    基于 LoFTR 特征匹配的旋转轴粒子滤波器（可调用封装）。

    用法示例
    --------
    estimator = LoFTRAxisEstimator(
        K       = camera_intrinsic_matrix,
        omega0  = np.array([0, 0, 1]),
        p0      = np.array([0.979, 0.267, 0.0]),
        theta0  = 0.0,
        visualize = True,
    )
    for each_frame:
        result = estimator.step(frame_id, curr_depth, curr_rgb_bgr,
                                c2w, lp_dict, ee_poses_dict)
        if result is not None:
            pivot, n_dir, theta, sigma_p, sigma_t = result
    """

    def __init__(self,
                 K: np.ndarray,
                 omega0: np.ndarray,
                 p0: np.ndarray,
                 theta0: float = 0.0,
                 n_particles: int = N_PARTICLES,
                 visualize: bool = True,
                 device=None):
        """
        Parameters
        ----------
        K           : (3,3) 相机内参矩阵
        omega0      : (3,)  转轴方向初始值（会被归一化）
        p0          : (3,)  转轴上一点（世界系，米）
        theta0      : float 初始累积旋转角 (rad)，默认 0
        n_particles : int   粒子数
        visualize   : bool  是否开启实时可视化窗口，默认 True
        device      : torch.device or None，None 则自动选择 cuda/cpu
        """
        self.K = K.astype(np.float32)

        # LoFTR 模型
        self.device  = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.matcher = LoFTR(pretrained='outdoor').to(self.device).eval()
        print(f"[LoFTRAxisEstimator] LoFTR 加载完成, 设备: {self.device}")

        # 螺旋参数初始化
        omega0 = np.array(omega0, dtype=np.float64)
        omega0 = omega0 / (np.linalg.norm(omega0) + 1e-9)
        v0     = p_to_v(p0, omega0)
        self.pf = AxisPF(omega0, v0, float(theta0), n_particles=n_particles)
        self.T_accumulated = np.eye(4, dtype=np.float32)
        print(f"[LoFTRAxisEstimator] PF 初始化: ω={np.round(omega0,3)}, "
              f"p0={np.round(p0,3)}, N={n_particles}")

        # 帧间状态
        self._prev_depth    = None
        self._prev_rgb_bgr  = None
        self._prev_c2w      = None
        self._prev_frame_id = None

        # 可视化
        self.visualize = visualize
        self._vis = Visualizer() if visualize else None

    # ------------------------------------------------------------------
    # 核心接口：处理单帧
    # ------------------------------------------------------------------
    def step(self,
             frame_id: int,
             curr_depth: np.ndarray,
             curr_rgb_bgr: np.ndarray,
             c2w: np.ndarray,
             lp_dict: dict,
             ee_poses_dict: dict):
        """
        处理当前帧，更新粒子滤波器并返回估计结果。

        Parameters
        ----------
        frame_id      : 当前帧编号
        curr_depth    : (H,W) float32 深度图（米）
        curr_rgb_bgr  : (H,W,3) uint8 BGR 图像（可为 None）
        c2w           : (4,4) float32 相机→世界变换矩阵
        lp_dict       : {link_name: pos(3,)} 机械臂关节位置字典
        ee_poses_dict : {frame_id: pos(3,)} 末端执行器位置字典

        Returns
        -------
        (pivot, n_dir, theta, sigma_p, sigma_t) 或 None（数据不足时）
        """
        robot_segs = get_robot_segments(lp_dict)
        rgb_img    = cv2.cvtColor(curr_rgb_bgr, cv2.COLOR_BGR2RGB) \
                     if curr_rgb_bgr is not None else None

        # 第一帧——记录状态后直接返回
        if self._prev_depth is None:
            self._prev_depth    = curr_depth
            self._prev_rgb_bgr  = curr_rgb_bgr
            self._prev_c2w      = c2w
            self._prev_frame_id = frame_id
            return None

        # 动态掩码
        dyn_mask = get_dynamic_mask(self._prev_depth, curr_depth)

        if np.sum(dyn_mask) < MIN_AREA:
            print(f"[Frame {frame_id}] 动态区域不足 ({np.sum(dyn_mask)} px), 跳过")
            pivot, n_dir, theta = self.pf.get_axis()
            if self._vis:
                self._vis.update(frame_id, None, pivot, n_dir, theta, rgb_img)
            self._advance(curr_depth, curr_rgb_bgr, c2w, frame_id)
            return pivot, n_dir, theta, *self.pf.get_uncertainty()

        if curr_rgb_bgr is None or self._prev_rgb_bgr is None:
            self._advance(curr_depth, curr_rgb_bgr, c2w, frame_id)
            return None

        # LoFTR 特征匹配
        t0 = img_to_loftr_tensor(self._prev_rgb_bgr, self.device)
        t1 = img_to_loftr_tensor(curr_rgb_bgr,       self.device)
        mkpts0, mkpts1 = match_loftr(t0, t1, self.matcher)

        src_w, tgt_w = get_3d_correspondences(
            mkpts0, mkpts1,
            self._prev_depth, curr_depth,
            self.K, self._prev_c2w, c2w,
            dyn_mask, robot_segs
        )

        if src_w is None:
            # print(f"[Frame {frame_id}] LoFTR 动态匹配点不足, 跳过")
            self._advance(curr_depth, curr_rgb_bgr, c2w, frame_id)
            return None

        T_step = kabsch_transform(src_w, tgt_w)

        # --- PF 预测步：从 EE 估计 delta_theta ---
        delta_theta_pred = 0.0
        prev_fid = self._prev_frame_id
        if prev_fid in ee_poses_dict and frame_id in ee_poses_dict:
            ee_delta = ee_poses_dict[frame_id] - ee_poses_dict[prev_fid]
            pivot_cur, n_cur, _ = self.pf.get_axis()
            perp  = ee_delta - np.dot(ee_delta, n_cur) * n_cur
            r_vec = ee_poses_dict[frame_id] - pivot_cur
            r_vec -= np.dot(r_vec, n_cur) * n_cur
            radius = np.linalg.norm(r_vec)
            if radius > 0.02:
                delta_theta_pred = np.linalg.norm(perp) / radius
                cross = np.cross(r_vec / (radius + 1e-9),
                                 perp  / (np.linalg.norm(perp) + 1e-9))
                if np.dot(cross, n_cur) < 0:
                    delta_theta_pred = -delta_theta_pred

        self.pf.predict(delta_theta_pred)
        self.T_accumulated = T_step @ self.T_accumulated

        # --- PF 更新步 ---
        ee_prev = ee_poses_dict.get(prev_fid,   None)
        ee_curr = ee_poses_dict.get(frame_id,   None)
        self.pf.update(src_w, tgt_w, ee_prev, ee_curr)

        pivot, n_dir, theta = self.pf.get_axis()
        sigma_p, sigma_t    = self.pf.get_uncertainty()

        print(f"[Frame {frame_id}] p={np.round(pivot,3)}  n={np.round(n_dir,3)}  "
              f"θ={np.rad2deg(theta):.2f}°  σp={sigma_p*100:.1f}cm  "
              f"σθ={np.rad2deg(sigma_t):.1f}°  matches={len(src_w)}")

        if self._vis:
            self._vis.update(frame_id, tgt_w, pivot, n_dir, theta, rgb_img,
                             n_matches=len(src_w), sigma_p=sigma_p)

        self._advance(curr_depth, curr_rgb_bgr, c2w, frame_id)
        return pivot, n_dir, theta, sigma_p, sigma_t

    # ------------------------------------------------------------------
    # 查询当前估计
    # ------------------------------------------------------------------
    def get_axis(self):
        """返回 (pivot, n_dir, theta)。"""
        return self.pf.get_axis()

    def get_uncertainty(self):
        """返回 (sigma_p, sigma_theta)。"""
        return self.pf.get_uncertainty()

    def finish(self):
        """所有帧处理完毕后调用，阻塞直到可视化窗口关闭。"""
        if self._vis:
            print("\n全部帧处理完毕。关闭图窗退出。")
            plt.ioff()
            plt.show()

    # ------------------------------------------------------------------
    # 内部：推进帧状态
    # ------------------------------------------------------------------
    def _advance(self, depth, rgb_bgr, c2w, frame_id):
        self._prev_depth    = depth
        self._prev_rgb_bgr  = rgb_bgr
        self._prev_c2w      = c2w.copy()
        self._prev_frame_id = frame_id


# =============================================================================
# Demo main（与之前逻辑完全等价）
# =============================================================================

def main(visualize: bool = True, step: int = 1):
    """
    从 DATA_ROOT 加载录制数据，演示 LoFTRAxisEstimator 的调用方式。

    Parameters
    ----------
    visualize : bool  是否显示实时可视化窗口，默认 True
    """
    print(f"[Main] 加载数据集: {DATA_ROOT}")
    K, _, _        = load_camera_info(DATA_ROOT)
    cam_poses      = load_camera_poses(DATA_ROOT)
    link_poses_all = load_link_poses(DATA_ROOT)
    ee_poses       = load_ee_poses(DATA_ROOT)

    depth_files = sorted(DATA_ROOT.glob("depth/depth_*.npy"))
    rgb_dir     = DATA_ROOT / "rgb"
    print(f"[Main] 共 {len(depth_files)} 帧")

    # 硬编码初始转轴参数
    p0     = np.array([0.9, 0.4, 0.0])
    omega0 = np.array([0.0, 0.0, 1.0])

    estimator = LoFTRAxisEstimator(
        K         = K,
        omega0    = omega0,
        p0        = p0,
        theta0    = 0.0,
        n_particles = N_PARTICLES,
        visualize = visualize,
    )

    for depth_file in depth_files[::step]:
        frame_id = int(depth_file.stem.split("_")[-1])
        if frame_id not in cam_poses or frame_id not in link_poses_all:
            continue

        c2w     = cam_poses[frame_id]
        lp_dict = link_poses_all[frame_id]

        curr_depth   = np.load(str(depth_file)).astype(np.float32)
        rgb_path     = rgb_dir / f"rgb_{frame_id:06d}.png"
        curr_rgb_bgr = cv2.imread(str(rgb_path)) if rgb_path.exists() else None

        estimator.step(
            frame_id     = frame_id,
            curr_depth   = curr_depth,
            curr_rgb_bgr = curr_rgb_bgr,
            c2w          = c2w,
            lp_dict      = lp_dict,
            ee_poses_dict = ee_poses,
        )

    estimator.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-vis', action='store_true', help='禁用可视化窗口')
    parser.add_argument('--step', type=int, default=1, help='处理数据集时的步长 (默认 1)')
    args = parser.parse_args()
    main(visualize=not args.no_vis, step=args.step)
