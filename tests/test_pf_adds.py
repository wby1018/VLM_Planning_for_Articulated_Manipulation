#!/usr/bin/env python3
"""ADD-S based PF for rotation axis estimation (no LoFTR)."""
import os
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DATA_ROOT = Path("record_pull_arc_with_joint_0")

DIFF_THRESH   = 0.001
MIN_AREA      = 3000
ADDS_N_SAMPLE = 300   # 每帧采样点数

R_ARM, R_FOREARM, R_HAND, R_FINGER = 0.15, 0.13, 0.07, 0.03
CONNECTIONS = [
    ('panda_link0','panda_link1',R_ARM),('panda_link1','panda_link2',R_ARM),
    ('panda_link2','panda_link3',R_ARM),('panda_link3','panda_link4',R_ARM),
    ('panda_link4','panda_link5',R_FOREARM),('panda_link5','panda_link6',R_FOREARM),
    ('panda_link6','panda_link7',R_FOREARM),('panda_link7','panda_link8',R_FOREARM),
    ('panda_hand','panda_leftfinger',R_HAND),('panda_hand','panda_rightfinger',R_HAND),
    ('panda_leftfinger','panda_leftfinger',R_FINGER),
    ('panda_rightfinger','panda_rightfinger',R_FINGER),
]

# PF 超参数
N_PARTICLES  = 500
SIGMA_KIN    = 0.003
SIGMA_RAD    = 0.05
SIGMA_ADDS   = 0.02   # ADD-S 残差标准差 (m)
PROC_NOISE_W = 1e-7
PROC_NOISE_V = 1e-2
PROC_NOISE_T = 5e-3
ESS_RATIO    = 0.9

# =============================================================================
# 数据加载
# =============================================================================
def load_camera_info(root):
    info = {}
    with open(root / "camera_info.txt") as f:
        for line in f:
            k, _, v = line.strip().partition(":")
            info[k.strip()] = float(v.strip())
    K = np.array([[info['fx'],0,info['cx']],[0,info['fy'],info['cy']],[0,0,1]], dtype=np.float32)
    return K, int(info['width']), int(info['height'])

def load_camera_poses(root):
    poses = {}
    with open(root / "camera_pose.txt") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 8: continue
            fid = int(p[0])
            t = np.array([float(x) for x in p[1:4]])
            q = np.array([float(x) for x in p[4:8]])
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3,:3] = R_scipy.from_quat(q).as_matrix()
            c2w[:3,3] = t
            poses[fid] = c2w
    return poses

def load_link_poses(root):
    all_lp = {}
    with open(root / "link_poses.txt") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9: continue
            fid, name = int(p[0]), p[1]
            if fid not in all_lp: all_lp[fid] = {}
            all_lp[fid][name] = np.array([float(x) for x in p[2:5]])
    return all_lp

def load_ee_poses(root):
    poses = {}
    with open(root / "ee_pose.txt") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 4: continue
            poses[int(p[0])] = np.array([float(x) for x in p[1:4]])
    return poses

# =============================================================================
# 点云工具
# =============================================================================
def dist_to_segments(pts, segments):
    inside = np.zeros(len(pts), dtype=bool)
    for A, B, radius in segments:
        AB = B - A; mag_sq = float(np.dot(AB, AB))
        if mag_sq < 1e-6:
            dist = np.linalg.norm(pts - A, axis=1)
        else:
            t = np.clip(np.sum((pts-A)*AB, axis=1)/mag_sq, 0., 1.)
            dist = np.linalg.norm(pts - (A + t[:,None]*AB), axis=1)
        inside |= (dist < radius)
    return inside

def get_robot_segments(lp_dict):
    segs = []
    for s, e, r in CONNECTIONS:
        if s in lp_dict and e in lp_dict:
            segs.append((lp_dict[s], lp_dict[e], r))
    return segs

def get_dynamic_mask(prev_depth, curr_depth):
    valid = (prev_depth > 0) & (curr_depth > 0)
    diff  = np.abs(curr_depth.astype(np.float32) - prev_depth.astype(np.float32))
    raw   = valid & (diff > DIFF_THRESH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    result = np.zeros_like(cleaned)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= MIN_AREA:
            result[labels == lbl] = 1
    return result.astype(bool)

def get_dynamic_pointcloud(depth, K, c2w, dyn_mask, robot_segs, n_sample=ADDS_N_SAMPLE):
    """从动态掩码中提取世界坐标系 3D 点云，随机采样 n_sample 点。"""
    H, W = depth.shape
    ys, xs = np.where(dyn_mask & (depth > 0))
    if len(ys) == 0:
        return None
    z = depth[ys, xs].astype(np.float64)
    xc = (xs - K[0,2]) / K[0,0] * z
    yc = (ys - K[1,2]) / K[1,1] * z
    pts_c = np.stack([xc, yc, z], axis=1)
    ones  = np.ones((len(pts_c),1))
    pts_w = (c2w.astype(np.float64) @ np.hstack([pts_c, ones]).T).T[:,:3].astype(np.float32)
    if robot_segs:
        pts_w = pts_w[~dist_to_segments(pts_w, robot_segs)]
    if len(pts_w) == 0:
        return None
    idx = np.random.choice(len(pts_w), min(n_sample, len(pts_w)), replace=False)
    return pts_w[idx]

# =============================================================================
# Screw Theory
# =============================================================================
def rot_matrix_np(n, theta):
    n = n / (np.linalg.norm(n) + 1e-9)
    K = np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

def p_to_v(p, omega): return -np.cross(omega, p)
def v_to_p(v, omega): return np.cross(omega, v)

# =============================================================================
# ADD-S 计算
# =============================================================================
def compute_adds(ref_pts, curr_pts):
    """ADD-S: 对 ref_pts 中每点找 curr_pts 中最近邻，返回平均最近邻距离。"""
    tree = cKDTree(curr_pts)
    dists, _ = tree.query(ref_pts, k=1)
    return float(np.mean(dists))

# =============================================================================
# 粒子滤波器
# =============================================================================
class AxisPF:
    """粒子状态 ξ_i = (ω_i, v_i, θ_i)，ADD-S 替换 LoFTR 视觉项。"""

    def __init__(self, omega0, v0, theta0, n_particles=N_PARTICLES):
        self.N = n_particles
        omega0 = omega0 / (np.linalg.norm(omega0) + 1e-9)
        v0 = v0 - np.dot(v0, omega0) * omega0
        omegas = omega0 + np.random.randn(self.N, 3) * 0.05
        norms  = np.linalg.norm(omegas, axis=1, keepdims=True)
        self.omegas = omegas / (norms + 1e-9)
        vs_raw = v0 + np.random.randn(self.N, 3) * 0.05
        dot    = np.sum(vs_raw * self.omegas, axis=1, keepdims=True)
        self.vs      = vs_raw - dot * self.omegas
        self.thetas  = np.full(self.N, theta0) + np.random.randn(self.N) * 0.05
        self.weights = np.ones(self.N) / self.N

    def predict(self, delta_theta=0.0):
        self.omegas += np.random.randn(self.N, 3) * PROC_NOISE_W
        norms = np.linalg.norm(self.omegas, axis=1, keepdims=True)
        self.omegas /= (norms + 1e-9)
        vs_raw = self.vs + np.random.randn(self.N, 3) * PROC_NOISE_V
        dot    = np.sum(vs_raw * self.omegas, axis=1, keepdims=True)
        self.vs = vs_raw - dot * self.omegas
        self.thetas += delta_theta + np.random.randn(self.N) * PROC_NOISE_T

    @staticmethod
    def _estimate_delta_theta(p, omega, c_prev, c_curr):
        v_prev = c_prev - p; v_curr = c_curr - p
        vpp = v_prev - np.dot(v_prev, omega)*omega
        vpc = v_curr - np.dot(v_curr, omega)*omega
        np_, nc = np.linalg.norm(vpp), np.linalg.norm(vpc)
        if np_ < 1e-6 or nc < 1e-6: return 0.0
        cross = np.cross(vpp, vpc)
        return float(np.arctan2(np.dot(cross, omega)/(np_*nc),
                                np.dot(vpp, vpc)/(np_*nc)))

    def update(self, ref_pts, curr_pts, ee_pos_prev, ee_pos_curr):
        """
        ref_pts  : (M,3) 参考帧动态点云（世界系）
        curr_pts : (M,3) 当前帧动态点云（世界系）
        ee_pos_prev/curr : (3,) 夹爪末端位置

        每粒子评分：
          A. EE 位移解算专属 Δθᵢ（帧间增量）
          B. 运动学 3D 重投影误差
          C. 半径一致性约束
          D. ADD-S: ref_pts 绕粒子轴旋转总角度 θᵢ 后与 curr_pts 最近邻平均距离
        """
        log_w   = np.zeros(self.N)
        has_ee  = (ee_pos_prev is not None) and (ee_pos_curr is not None)
        has_vis = (ref_pts is not None) and (curr_pts is not None) and \
                  (len(ref_pts) > 0) and (len(curr_pts) > 0)
        curr_tree = cKDTree(curr_pts) if has_vis else None

        for i in range(self.N):
            omega_i = self.omegas[i]
            p_i     = v_to_p(self.vs[i], omega_i)

            # A. 独立解算 Δθᵢ（帧间）
            dtheta_i = self._estimate_delta_theta(p_i, omega_i, ee_pos_prev, ee_pos_curr) \
                       if has_ee else self.thetas[i]
            R_inc = rot_matrix_np(omega_i, dtheta_i)

            # B. 运动学误差
            if has_ee:
                ee_pred = R_inc @ (ee_pos_prev - p_i) + p_i
                r_kin   = float(np.linalg.norm(ee_pos_curr - ee_pred))
                log_w[i] += -0.5 * (r_kin / SIGMA_KIN) ** 2

                # C. 半径一致性
                r_prev = np.linalg.norm(np.cross(omega_i, ee_pos_prev - p_i))
                r_curr = np.linalg.norm(np.cross(omega_i, ee_pos_curr - p_i))
                log_w[i] += -0.5 * (abs(r_curr - r_prev) / SIGMA_RAD) ** 2

            # D. ADD-S: 用总累积角 θᵢ 旋转参考帧点云
            if has_vis:
                R_total  = rot_matrix_np(omega_i, self.thetas[i])
                rot_ref  = (R_total @ (ref_pts - p_i).T).T + p_i
                dists, _ = curr_tree.query(rot_ref, k=1)
                r_adds   = float(np.mean(dists))
                log_w[i] += -0.5 * (r_adds / SIGMA_ADDS) ** 2

        log_w -= log_w.max()
        w_new  = self.weights * np.exp(log_w)
        w_sum  = w_new.sum()
        self.weights = w_new/w_sum if w_sum > 1e-300 else np.ones(self.N)/self.N
        ess = 1.0 / (np.sum(self.weights**2) + 1e-300)
        if ess < ESS_RATIO * self.N:
            self._resample()

    def _resample(self):
        cumsum = np.cumsum(self.weights); cumsum[-1] = 1.0
        pos = (np.arange(self.N) + np.random.uniform()) / self.N
        idx = np.searchsorted(cumsum, pos)
        self.omegas  = self.omegas[idx]; self.vs = self.vs[idx]
        self.thetas  = self.thetas[idx]; self.weights = np.ones(self.N)/self.N

    def get_axis(self):
        w = self.weights
        ref = self.omegas[0].copy()
        signs = np.sign(self.omegas @ ref); signs[signs==0] = 1.0
        omega_w = (w[:,None] * signs[:,None] * self.omegas).sum(0)
        omega_w /= (np.linalg.norm(omega_w) + 1e-9)
        v_w = (w[:,None] * self.vs).sum(0)
        v_w -= np.dot(v_w, omega_w) * omega_w
        return v_to_p(v_w, omega_w), omega_w, float(np.dot(w, self.thetas))

    def get_uncertainty(self):
        w = self.weights
        _, _, theta_mean = self.get_axis()
        ps = np.array([v_to_p(self.vs[i], self.omegas[i]) for i in range(self.N)])
        p_mean = (w[:,None]*ps).sum(0)
        sigma_p = float(np.sqrt((w * np.sum((ps-p_mean)**2, axis=1)).sum()))
        sigma_t = float(np.sqrt((w * (self.thetas-theta_mean)**2).sum()))
        return sigma_p, sigma_t

# =============================================================================
# Visualizer
# =============================================================================
class Visualizer:
    def __init__(self):
        plt.ion()
        self.fig  = plt.figure(figsize=(14,6))
        self.ax3d = self.fig.add_subplot(131, projection='3d')
        self.ax2d = self.fig.add_subplot(132)
        self.axim = self.fig.add_subplot(133)
        self.val_hist = []; self.frame_hist = []

    def update(self, frame_id, dyn_pts, pivot, n_dir, theta, rgb_img=None,
               n_pts=0, sigma_p=None):
        self.ax3d.cla(); ax = self.ax3d
        if dyn_pts is not None and len(dyn_pts) > 0:
            ax.scatter(dyn_pts[:,0],dyn_pts[:,1],dyn_pts[:,2],s=3,c='orange',alpha=0.7)
        if pivot is not None:
            ext=0.5; p0=pivot-n_dir*ext; p1=pivot+n_dir*ext
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],'c-',lw=3,label='Axis')
            ax.scatter(*pivot,s=120,c='red',zorder=6,label='Pivot')
        ax.quiver(0,0,0,.3,0,0,color='r',arrow_length_ratio=.3)
        ax.quiver(0,0,0,0,.3,0,color='g',arrow_length_ratio=.3)
        ax.quiver(0,0,0,0,0,.3,color='b',arrow_length_ratio=.3)
        ax.set_xlim(.2,1.6); ax.set_ylim(0,1.5); ax.set_zlim(.2,1.4)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        sp = f'  σp={sigma_p*100:.1f}cm' if sigma_p is not None else ''
        ax.set_title(f'Frame {frame_id}\nθ={np.rad2deg(theta):.1f}°  pts={n_pts}{sp}')
        if pivot is not None: ax.legend(fontsize=7)

        self.val_hist.append(np.rad2deg(theta)); self.frame_hist.append(frame_id)
        self.ax2d.cla()
        self.ax2d.plot(self.frame_hist, self.val_hist,'b-o',markersize=3)
        self.ax2d.set_xlabel('Frame'); self.ax2d.set_ylabel('θ (°)')
        self.ax2d.set_title('PF θ estimate'); self.ax2d.grid(True)

        self.axim.cla()
        if rgb_img is not None:
            self.axim.imshow(rgb_img); self.axim.set_title(f'RGB {frame_id}')
            self.axim.axis('off')
        plt.tight_layout()
        try: plt.pause(0.001)
        except Exception: pass

# =============================================================================
# Main
# =============================================================================
def main():
    print(f"[Main] 加载数据集: {DATA_ROOT}")
    K, img_W, img_H = load_camera_info(DATA_ROOT)
    cam_poses      = load_camera_poses(DATA_ROOT)
    link_poses_all = load_link_poses(DATA_ROOT)
    ee_poses       = load_ee_poses(DATA_ROOT)

    depth_files = sorted(DATA_ROOT.glob("depth/depth_*.npy"))
    rgb_dir     = DATA_ROOT / "rgb"
    print(f"[Main] 共 {len(depth_files)} 帧")

    vis = Visualizer()

    # =========================================================================
    # 硬编码初始化 ξ=(ω, v)
    # =========================================================================
    p0     = np.array([0.979, 0.267, 0.0])
    omega0 = np.array([0.0, 0.0, 1.0])
    omega0 = omega0 / (np.linalg.norm(omega0) + 1e-9)
    v0     = p_to_v(p0, omega0)
    theta0 = 0.0

    pf = AxisPF(omega0, v0, theta0, n_particles=N_PARTICLES)
    print(f"[Main] PF 初始化: ω={omega0}, p={p0}, N={N_PARTICLES}")

    prev_depth    = None
    prev_c2w      = None
    prev_frame_id = None
    ref_pts       = None   # 参考帧动态点云（ADD-S 用）

    for depth_file in depth_files:
        frame_id = int(depth_file.stem.split("_")[-1])
        if frame_id not in cam_poses or frame_id not in link_poses_all:
            continue

        c2w     = cam_poses[frame_id]
        lp_dict = link_poses_all[frame_id]
        robot_segs = get_robot_segments(lp_dict)

        curr_depth = np.load(str(depth_file)).astype(np.float32)
        rgb_path   = rgb_dir / f"rgb_{frame_id:06d}.png"
        curr_rgb   = cv2.imread(str(rgb_path)) if rgb_path.exists() else None
        rgb_img    = cv2.cvtColor(curr_rgb, cv2.COLOR_BGR2RGB) if curr_rgb is not None else None

        if prev_depth is None:
            prev_depth = curr_depth; prev_c2w = c2w; prev_frame_id = frame_id
            continue

        dyn_mask = get_dynamic_mask(prev_depth, curr_depth)

        if np.sum(dyn_mask) < MIN_AREA:
            print(f"[Frame {frame_id}] 动态区域不足 ({np.sum(dyn_mask)} px)")
            pivot, n_dir, theta = pf.get_axis()
            vis.update(frame_id, None, pivot, n_dir, theta, rgb_img)
            prev_depth = curr_depth; prev_c2w = c2w; prev_frame_id = frame_id
            continue

        # 当前帧动态点云
        curr_pts = get_dynamic_pointcloud(curr_depth, K, c2w, dyn_mask, robot_segs)
        if curr_pts is None or len(curr_pts) < 5:
            prev_depth = curr_depth; prev_c2w = c2w; prev_frame_id = frame_id
            continue

        # 首次检测到动态区域时记录参考帧点云
        if ref_pts is None:
            ref_pts = get_dynamic_pointcloud(curr_depth, K, c2w, dyn_mask, robot_segs)
            print(f"[Main] 记录参考帧点云 (Frame {frame_id}), {len(ref_pts)} pts")
            prev_depth = curr_depth; prev_c2w = c2w; prev_frame_id = frame_id
            continue

        # =====================================================================
        # PF 更新
        # =====================================================================
        # 2a. 估计 delta_theta（EE 帧间位移 → 轴角速度）
        delta_theta_pred = 0.0
        if prev_frame_id in ee_poses and frame_id in ee_poses:
            ee_delta = ee_poses[frame_id] - ee_poses[prev_frame_id]
            pivot_cur, n_cur, _ = pf.get_axis()
            perp  = ee_delta - np.dot(ee_delta, n_cur) * n_cur
            r_vec = ee_poses[frame_id] - pivot_cur
            r_vec -= np.dot(r_vec, n_cur) * n_cur
            radius = np.linalg.norm(r_vec)
            if radius > 0.02:
                delta_theta_pred = np.linalg.norm(perp) / radius
                cross = np.cross(r_vec/(radius+1e-9), perp/(np.linalg.norm(perp)+1e-9))
                if np.dot(cross, n_cur) < 0:
                    delta_theta_pred = -delta_theta_pred

        # 2b. Predict
        pf.predict(delta_theta_pred)

        # 2c. Update (ADD-S 视觉 + EE 运动学)
        ee_prev = ee_poses.get(prev_frame_id, None)
        ee_curr = ee_poses.get(frame_id, None)
        pf.update(ref_pts, curr_pts, ee_prev, ee_curr)

        pivot, n_dir, theta = pf.get_axis()
        sigma_p, sigma_t    = pf.get_uncertainty()
        print(f"[Frame {frame_id}] p={np.round(pivot,3)}  n={np.round(n_dir,3)}  "
              f"θ={np.rad2deg(theta):.2f}°  σp={sigma_p*100:.1f}cm  "
              f"σθ={np.rad2deg(sigma_t):.1f}°  curr_pts={len(curr_pts)}")

        prev_depth = curr_depth; prev_c2w = c2w; prev_frame_id = frame_id
        vis.update(frame_id, curr_pts, pivot, n_dir, theta, rgb_img,
                   n_pts=len(curr_pts), sigma_p=sigma_p)

    print("\n全部帧处理完毕。关闭图窗退出。")
    plt.ioff(); plt.show()


if __name__ == "__main__":
    main()
