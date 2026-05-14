#!/usr/bin/env python3
"""
loftr_fg.py — LoFTR + Factor Graph screw-axis estimator.
Same data loading / observation pipeline as loftr_pf.py.
Backend replaces particle filter with a sliding-window least-squares
factor graph over screw axis ξ=(ω,v) and per-frame angles θ_i.
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import kornia as KN
from kornia.feature import LoFTR
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy

# =============================================================================
# Config  (data / observation params identical to loftr_pf)
# =============================================================================
DATA_ROOT         = Path("record_pull_arc_with_joint_0")

DIFF_THRESH       = 0.001
MIN_AREA          = 3000
MIN_DYN_MATCHES   = 15
LOFTR_CONF_THRESH = 0.5
LOFTR_SIZE        = (640, 480)

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

# FG hyperparameters
SIGMA_GRIP        = 0.005   # gripper trajectory factor noise (m)
SIGMA_VIS         = 0.010    # adjacent-frame vis noise (m) — weakened, auxiliary only
SIGMA_DTHETA      = 0.03    # angle continuity factor noise (rad)
SIGMA_PRIOR_OMEGA = 0.01    # prior noise on ω direction
SIGMA_PRIOR_V     = 0.01    # prior noise on v (axis position)
SIGMA_MONO        = 0.01    # monotone angle factor noise (rad)
W_NORM            = 50.0    # soft unit-norm penalty weight for ω
MAX_VIS_PTS       = 100      # max LoFTR points subsampled per observation
WINDOW_SIZE       = 25      # sliding-window frame count
MAX_NFEV          = 300     # optimizer function evaluations per step

# Keyframe hyperparameters
SIGMA_VIS_KF_RECENT = 0.003  # recent keyframe vis factor noise (m) — stronger
SIGMA_VIS_KF_ANCHOR = 0.005  # anchor (first) keyframe vis factor noise (m) — strongest
KF_DTHETA_MIN       = 0.17   # min |Δθ| from last KF to trigger new KF (rad ≈ 9.7°)
KF_MIN_MATCHES      = 8      # min LoFTR matches required to add a keyframe
MAX_KEYFRAMES       = 5      # max keyframes in buffer (keeps anchor + N-1 recent)

# Adjacent-frame visual factor angle gating
ADJ_VIS_MIN_DTHETA = np.deg2rad(2.0)   # skip adjacent factor if |Δθ| below this
ADJ_W_MIN          = 0.05              # min weight for adjacent factor
ADJ_W_MAX          = 0.50             # max weight for adjacent factor
ADJ_W_FULL_DEG     = 10.0             # |Δθ| (deg) at which adj weight reaches ADJ_W_MAX

# =============================================================================
# Data loading  (identical to loftr_pf)
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
            t   = np.array([float(x) for x in parts[1:4]])
            q   = np.array([float(x) for x in parts[4:8]])
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R_scipy.from_quat(q).as_matrix()
            c2w[:3,  3] = t
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
# Point cloud utilities  (identical to loftr_pf)
# =============================================================================

def dist_to_segments(pts: np.ndarray, segments) -> np.ndarray:
    inside = np.zeros(len(pts), dtype=bool)
    for A, B, radius in segments:
        AB     = B - A
        mag_sq = float(np.dot(AB, AB))
        if mag_sq < 1e-6:
            dist = np.linalg.norm(pts - A, axis=1)
        else:
            AP      = pts - A
            t       = np.clip(np.sum(AP * AB, axis=1) / mag_sq, 0., 1.)
            closest = A + t[:, None] * AB
            dist    = np.linalg.norm(pts - closest, axis=1)
        inside |= (dist < radius)
    return inside


def get_robot_segments(lp_dict: dict):
    segs = []
    for s, e, r in CONNECTIONS:
        if s in lp_dict and e in lp_dict:
            segs.append((lp_dict[s], lp_dict[e], r))
    return segs

# =============================================================================
# Dynamic mask  (identical to loftr_pf)
# =============================================================================

def get_dynamic_mask(prev_depth: np.ndarray, curr_depth: np.ndarray,
                     diff_thresh: float = DIFF_THRESH,
                     min_area: int = MIN_AREA) -> np.ndarray:
    valid    = (prev_depth > 0) & (curr_depth > 0)
    diff     = np.abs(curr_depth.astype(np.float32) - prev_depth.astype(np.float32))
    raw_mask = valid & (diff > diff_thresh)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    result = np.zeros_like(cleaned)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            result[labels == lbl] = 1
    return result.astype(bool)

# =============================================================================
# LoFTR matching  (identical to loftr_pf)
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
# 3D correspondence extraction  (identical to loftr_pf)
# =============================================================================

def get_3d_correspondences(mkpts0: np.ndarray, mkpts1: np.ndarray,
                            depth0: np.ndarray, depth1: np.ndarray,
                            K: np.ndarray,
                            c2w0: np.ndarray, c2w1: np.ndarray,
                            dyn_mask_curr: np.ndarray,
                            robot_segs):
    H, W   = depth0.shape
    lw, lh = LOFTR_SIZE
    sx = W / lw;  sy = H / lh

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
        bad   = dist_to_segments(src_w, robot_segs) | dist_to_segments(tgt_w, robot_segs)
        src_w = src_w[~bad]
        tgt_w = tgt_w[~bad]

    if len(src_w) < MIN_DYN_MATCHES:
        return None, None

    return src_w.astype(np.float32), tgt_w.astype(np.float32)

# =============================================================================
# Screw theory utilities  (identical to loftr_pf)
# =============================================================================

def p_to_v(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return -np.cross(omega, p)


def v_to_p(v: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.cross(omega, v)

# =============================================================================
# SE(3) exponential map for screw axis
# =============================================================================

def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]], dtype=np.float64)


def _exp_screw(xi: np.ndarray, theta: float) -> np.ndarray:
    """
    SE(3) exponential map for screw axis ξ=(ω, v) at angle θ.
    ω is normalised internally; v is projected to ω⊥.
    Returns 4×4 homogeneous transform.
    """
    omega = xi[:3].astype(np.float64)
    v     = xi[3:].astype(np.float64)
    omega = omega / (np.linalg.norm(omega) + 1e-9)
    v     = v - np.dot(v, omega) * omega          # enforce v ⊥ ω
    K     = _skew(omega)
    ct    = np.cos(theta)
    st    = np.sin(theta)
    R     = np.eye(3) + st * K + (1.0 - ct) * (K @ K)
    t     = (np.eye(3) * theta + (1.0 - ct) * K + (theta - st) * (K @ K)) @ v
    T     = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def _apply_screw(xi: np.ndarray, theta: float, pts: np.ndarray) -> np.ndarray:
    """Apply exp(ξ θ) to (N,3) points, return (N,3)."""
    T = _exp_screw(xi, theta)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]

# =============================================================================
# Factor Graph  —  AxisFG
# =============================================================================

class AxisFG:
    """
    Sliding-window least-squares factor graph for screw-axis estimation.

    State variables (flat parameter vector for optimizer):
      params = [ξ(6),  θ_0, θ_1, ..., θ_{N-1}]

    where ξ = [ω(3); v(3)] encodes the screw axis and θ_i is the
    cumulative door-opening angle at frame i in the current window.

    Factors
    -------
    1. TypeCheck prior        — ξ ≈ ξ₀  (ω very tight, v relaxed)
    2. Gripper trajectory     — exp(ξ θ_i) g_ref ≈ g_i
    3. Adjacent LoFTR         — exp(ξ (θ_i − θ_k)) x_j^k ≈ x_j^i  (huber)
    4. Keyframe LoFTR         — exp(ξ · Δθ_frozen) x_j^kf ≈ x_j^curr  (huber)
    5. Angle continuity       — θ_i − θ_{i-1} ≈ dtheta_init_i
    6. Soft unit-norm         — ||ω|| ≈ 1
    """

    def __init__(self, omega0: np.ndarray, v0: np.ndarray, theta0: float = 0.0):
        omega0 = np.array(omega0, dtype=np.float64)
        omega0 /= np.linalg.norm(omega0) + 1e-9
        v0 = np.array(v0, dtype=np.float64)
        v0 -= np.dot(v0, omega0) * omega0           # enforce v ⊥ ω

        self.xi0     = np.r_[omega0, v0]            # TypeCheck prior (fixed)
        self.xi_est  = self.xi0.copy()              # current ξ estimate

        self._thetas       = [float(theta0)]       # window theta list
        self._dtheta_inits = [0.0]                  # kinematic dtheta_init per frame
        self._push_count   = 0                      # total push_frame calls (monotone)
        self._offset       = 0                      # frames trimmed from front

        self._ee_ref      = None                    # EE position at θ=0 frame
        self._ee_obs      = []                      # [(local_idx, pos_3d), ...]
        self._vis_obs     = []                      # [(local_k, local_i, src, tgt), ...]
        # Keyframe visual factors: dtheta frozen outside window
        # items: (dtheta_frozen, local_i, src, tgt, sigma)
        self._kf_vis_obs  = []

        self._last_sol = None                       # saved for uncertainty query

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_ref_ee(self, ee_pos: np.ndarray):
        """Call once with the EE position at the reference frame (θ=0)."""
        self._ee_ref = np.array(ee_pos, dtype=np.float64)

    def push_frame(self,
                   ee_pos: np.ndarray = None,
                   src_pts: np.ndarray = None,
                   tgt_pts: np.ndarray = None,
                   kf_vis_list: list = None,
                   dtheta_init: float = 0.0,
                   optimize: bool = True):
        """
        Register a new frame and re-optimise the window.

        Parameters
        ----------
        ee_pos       : (3,) gripper world position, or None
        src_pts      : (M,3) LoFTR source 3D points (prev frame), or None
        tgt_pts      : (M,3) LoFTR target 3D points (curr frame), or None
        kf_vis_list  : list of (src_kf, tgt_curr, sigma, dtheta_frozen) tuples
                       for keyframe → current visual factors
        dtheta_init  : kinematic initial guess for Δθ
        """
        self._push_count += 1
        self._thetas.append(self._thetas[-1] + dtheta_init)
        self._dtheta_inits.append(float(dtheta_init))
        curr_local = len(self._thetas) - 1
        prev_local = curr_local - 1

        if ee_pos is not None:
            self._ee_obs.append((curr_local, np.array(ee_pos, dtype=np.float64)))

        # Adjacent-frame visual factor
        # Skip if angle step too small (axis unobservable, noise dominated)
        if (src_pts is not None and tgt_pts is not None
                and len(src_pts) >= MIN_DYN_MATCHES
                and abs(dtheta_init) >= ADJ_VIS_MIN_DTHETA):
            M = len(src_pts)
            if M > MAX_VIS_PTS:
                idx = np.random.choice(M, MAX_VIS_PTS, replace=False)
                src_pts, tgt_pts = src_pts[idx], tgt_pts[idx]
            # Angle-adaptive sigma: larger angle → smaller sigma → stronger constraint
            dtheta_deg = abs(np.rad2deg(dtheta_init))
            t = min((dtheta_deg - np.rad2deg(ADJ_VIS_MIN_DTHETA))
                    / (ADJ_W_FULL_DEG - np.rad2deg(ADJ_VIS_MIN_DTHETA)), 1.0)
            w_adj = ADJ_W_MIN + t * (ADJ_W_MAX - ADJ_W_MIN)  # in [ADJ_W_MIN, ADJ_W_MAX]
            sigma_adj = SIGMA_VIS / w_adj
            self._vis_obs.append((prev_local, curr_local,
                                  src_pts.astype(np.float64),
                                  tgt_pts.astype(np.float64),
                                  sigma_adj))

        # Keyframe visual factors (frozen Δθ, constraint on ξ only)
        if kf_vis_list:
            for src_kf, tgt_curr, sigma_kf, dtheta_frozen in kf_vis_list:
                if src_kf is None or len(src_kf) < MIN_DYN_MATCHES:
                    continue
                M = len(src_kf)
                if M > MAX_VIS_PTS:
                    idx = np.random.choice(M, MAX_VIS_PTS, replace=False)
                    src_kf  = src_kf[idx]
                    tgt_curr = tgt_curr[idx]
                self._kf_vis_obs.append((
                    float(dtheta_frozen),
                    curr_local,
                    src_kf.astype(np.float64),
                    tgt_curr.astype(np.float64),
                    float(sigma_kf),
                ))

        self._trim()
        if optimize:
            self._optimize()

    def get_axis(self):
        """Return (pivot_3d, omega_unit_3d, theta_current)."""
        omega = self.xi_est[:3]
        omega = omega / (np.linalg.norm(omega) + 1e-9)
        v     = self.xi_est[3:]
        v     = v - np.dot(v, omega) * omega
        p     = v_to_p(v, omega)
        return p, omega, float(self._thetas[-1])

    def get_uncertainty(self):
        """Return (sigma_p, sigma_theta) estimated from optimizer Jacobian."""
        if self._last_sol is None:
            return 0.05, np.deg2rad(10.0)
        sol = self._last_sol
        if sol.jac is None:
            return 0.02, np.deg2rad(5.0)
        try:
            JtJ = sol.jac.T @ sol.jac
            diag = np.linalg.pinv(JtJ).diagonal()
            n = len(self._thetas)
            # sigma for v (indices 3:6) → proxy for axis position uncertainty
            sigma_p = float(np.sqrt(np.mean(np.abs(diag[3:6]))))
            # sigma for last theta (index 6+n-1)
            theta_var_idx = 6 + n - 1
            sigma_t = float(np.sqrt(abs(diag[theta_var_idx]))) \
                      if theta_var_idx < len(diag) else np.deg2rad(5.0)
            return sigma_p, sigma_t
        except Exception:
            return 0.02, np.deg2rad(5.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim(self):
        """Drop oldest frames when window exceeds WINDOW_SIZE."""
        n = len(self._thetas)
        if n <= WINDOW_SIZE:
            return
        cut = n - WINDOW_SIZE
        self._offset       += cut
        self._thetas        = self._thetas[cut:]
        self._dtheta_inits  = self._dtheta_inits[cut:]
        self._ee_obs       = [(i - cut, p) for i, p in self._ee_obs
                              if i - cut >= 0]
        self._vis_obs      = [(k - cut, i - cut, s, t, sg)
                              for k, i, s, t, sg in self._vis_obs
                              if k - cut >= 0 and i - cut >= 0]
        # kf_vis_obs only needs curr_local (i) to be in window
        self._kf_vis_obs   = [(dt, i - cut, s, t, sg)
                              for dt, i, s, t, sg in self._kf_vis_obs
                              if i - cut >= 0]

    def _residuals(self, params: np.ndarray) -> np.ndarray:
        n      = len(self._thetas)
        xi     = params[:6]
        thetas = params[6: 6 + n]
        res    = []

        # 1. TypeCheck prior on ξ  (ω tightly pinned, v loosely constrained)
        res.append((xi[:3] - self.xi0[:3]) / SIGMA_PRIOR_OMEGA)   # axis direction
        res.append((xi[3:] - self.xi0[3:]) / SIGMA_PRIOR_V)       # axis position

        # 2. Soft unit-norm on ω
        res.append(np.array([W_NORM * (float(np.dot(xi[:3], xi[:3])) - 1.0)]))

        # 3. Gripper trajectory: exp(ξ θ_i) g_ref ≈ g_i
        if self._ee_ref is not None:
            for idx, g_i in self._ee_obs:
                if idx < n:
                    g_pred = _apply_screw(xi, thetas[idx], self._ee_ref[None])[0]
                    res.append((g_pred - g_i) / SIGMA_GRIP)

        # 4. Adjacent-frame LoFTR: exp(ξ (θ_i − θ_k)) x_j^k ≈ x_j^i
        #    sigma_adj stored per-observation (angle-adaptive, in [SIGMA_VIS/ADJ_W_MAX, SIGMA_VIS/ADJ_W_MIN])
        for ik, ii, src, tgt, sigma_adj in self._vis_obs:
            if ik < n and ii < n:
                dth  = float(thetas[ii]) - float(thetas[ik])
                pred = _apply_screw(xi, dth, src)
                res.append((pred - tgt).ravel() / sigma_adj)

        # 5. Keyframe LoFTR: exp(ξ · Δθ_frozen) x_j^kf ≈ x_j^curr
        #    Δθ_frozen is fixed (not a variable); constrains ξ via long baseline
        for dtheta_frozen, ii, src, tgt, sigma_kf in self._kf_vis_obs:
            if ii < n:
                pred = _apply_screw(xi, dtheta_frozen, src)
                res.append((pred - tgt).ravel() / sigma_kf)

        # 6. Angle continuity: θ_i − θ_{i-1} ≈ dtheta_init_i
        for i in range(1, n):
            expected = self._dtheta_inits[i] if i < len(self._dtheta_inits) else 0.0
            res.append(np.array(
                [((thetas[i] - thetas[i - 1]) - expected) / SIGMA_DTHETA]
            ))

        return np.concatenate(res)

    def _optimize(self):
        n  = len(self._thetas)
        x0 = np.r_[self.xi_est, self._thetas]
        try:
            sol = least_squares(
                self._residuals,
                x0,
                method='trf',
                loss='huber',
                f_scale=2.0,
                max_nfev=MAX_NFEV,
                ftol=1e-5,
                xtol=1e-5,
                gtol=1e-8,
            )
            self.xi_est   = sol.x[:6]
            self._thetas  = list(sol.x[6: 6 + n])
            self._last_sol = sol
        except Exception as exc:
            print(f"[FG] optimize error: {exc}")

    # ------------------------------------------------------------------
    # Keyframe theta query helpers
    # ------------------------------------------------------------------

    @property
    def push_count(self) -> int:
        """Total number of push_frame() calls (monotonically increasing)."""
        return self._push_count

    @property
    def offset(self) -> int:
        """Number of frames trimmed from the front of the window."""
        return self._offset

    def get_theta_at_push_idx(self, push_idx: int):
        """Return theta for a given push_idx if still in window, else None."""
        local = push_idx - self._offset
        if 0 <= local < len(self._thetas):
            return float(self._thetas[local])
        return None

# =============================================================================
# Visualization  (identical to loftr_pf, labels updated for FG)
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
               rgb_img=None, n_matches: int = 0, sigma_p: float = None):
        self.ax3d.cla()
        ax = self.ax3d
        if dyn_pts is not None and len(dyn_pts) > 0:
            ax.scatter(dyn_pts[:, 0], dyn_pts[:, 1], dyn_pts[:, 2],
                       s=3, c='orange', alpha=0.7, label=f'Matched ({len(dyn_pts)})')
        if pivot is not None:
            ext = 0.5
            p0  = pivot - n_dir * ext
            p1  = pivot + n_dir * ext
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    'c-', linewidth=3, label='Est. Axis')
            ax.scatter(*pivot, s=120, c='red', zorder=6, label='Pivot')
        ax.quiver(0, 0, 0, 0.3, 0,   0,   color='r', arrow_length_ratio=0.3)
        ax.quiver(0, 0, 0, 0,   0.3, 0,   color='g', arrow_length_ratio=0.3)
        ax.quiver(0, 0, 0, 0,   0,   0.3, color='b', arrow_length_ratio=0.3)
        ax.set_xlim(0.2, 1.6); ax.set_ylim(0.0, 1.5); ax.set_zlim(0.2, 1.4)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        sp_str = f'  σp={sigma_p * 100:.1f}cm' if sigma_p is not None else ''
        ax.set_title(f'Frame {frame_id}\nθ={np.rad2deg(theta):.1f}°  '
                     f'matches={n_matches}{sp_str}')
        if pivot is not None:
            ax.legend(fontsize=7, loc='upper left')

        self.val_hist.append(np.rad2deg(theta))
        self.frame_hist.append(frame_id)
        self.ax2d.cla()
        self.ax2d.plot(self.frame_hist, self.val_hist, 'b-o', markersize=3)
        self.ax2d.set_xlabel('Frame')
        self.ax2d.set_ylabel('θ (°)')
        self.ax2d.set_title('FG door angle θ')
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
# LoFTRAxisEstimatorFG — drop-in replacement for LoFTRAxisEstimator
# =============================================================================

class LoFTRAxisEstimatorFG:
    """
    Factor-graph–based rotating-door axis estimator using LoFTR visual matches.

    Usage
    -----
    estimator = LoFTRAxisEstimatorFG(K=K, omega0=..., p0=..., theta0=0.0)
    for each frame:
        result = estimator.step(frame_id, curr_depth, curr_rgb_bgr,
                                c2w, lp_dict, ee_poses_dict)
        if result is not None:
            pivot, n_dir, theta, sigma_p, sigma_t = result
    estimator.finish()
    """

    def __init__(self,
                 K: np.ndarray,
                 omega0: np.ndarray,
                 p0: np.ndarray,
                 theta0: float = 0.0,
                 visualize: bool = True,
                 device=None):
        """
        Parameters
        ----------
        K         : (3,3) camera intrinsics
        omega0    : (3,)  Hard-coded initial axis direction (normalised internally).
                          FG is NOT created until the first valid dynamic point cloud
                          is observed — this guess is used at that moment.
        p0        : (3,)  Hard-coded initial point on axis (world frame, metres).
        theta0    : float initial cumulative angle (rad), default 0
        visualize : bool  show live plot window
        device    : torch.device or None (auto-select cuda/cpu)
        """
        self.K = K.astype(np.float32)

        self.device  = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.matcher = LoFTR(pretrained='outdoor').to(self.device).eval()
        print(f"[LoFTRAxisEstimatorFG] LoFTR loaded on {self.device}")

        # --- Hard-coded initial guess (used at FG init time) ---
        self._omega0_init = np.array(omega0, dtype=np.float64)
        self._omega0_init /= np.linalg.norm(self._omega0_init) + 1e-9
        self._p0_init     = np.array(p0,     dtype=np.float64)
        self._theta0_init = float(theta0)

        # FG is created lazily on first valid dynamic-point-cloud frame
        self.fg               = None
        self._fg_initialized  = False

        self._prev_depth    = None
        self._prev_rgb_bgr  = None
        self._prev_c2w      = None
        self._prev_frame_id = None

        # Keyframe buffer — each entry is a dict:
        # { frame_id, depth, rgb_bgr, c2w, theta_abs, fg_push_idx }
        self._keyframes: list = []
        self._last_opt_gate_kf_count = -1

        self.visualize = visualize
        self._vis = Visualizer() if visualize else None

    # ------------------------------------------------------------------

    def step(self,
             frame_id: int,
             curr_depth: np.ndarray,
             curr_rgb_bgr: np.ndarray,
             c2w: np.ndarray,
             lp_dict: dict,
             ee_poses_dict: dict):
        """
        Process one frame and return the current screw-axis estimate.

        The Factor Graph is initialised lazily: it is created only when the
        first frame that contains a valid dynamic point cloud (src_w is not
        None) is encountered.  Before that moment every frame is silently
        buffered and None is returned.

        Returns
        -------
        (pivot, n_dir, theta, sigma_p, sigma_t) or None when FG not yet
        initialised or data is insufficient.
        """
        robot_segs = get_robot_segments(lp_dict)
        rgb_img    = cv2.cvtColor(curr_rgb_bgr, cv2.COLOR_BGR2RGB) \
                     if curr_rgb_bgr is not None else None

        # ----------------------------------------------------------------
        # Very first frame — just cache and return
        # ----------------------------------------------------------------
        if self._prev_depth is None:
            self._advance(curr_depth, curr_rgb_bgr, c2w, frame_id)
            print(f"[Frame {frame_id}] Reference frame cached. "
                  "Waiting for first dynamic point cloud to init FG.")
            return None

        # ----------------------------------------------------------------
        # Dynamic mask check
        # ----------------------------------------------------------------
        dyn_mask = get_dynamic_mask(self._prev_depth, curr_depth)
        dyn_px   = int(np.sum(dyn_mask))

        if dyn_px < MIN_AREA:
            print(f"[Frame {frame_id}] dynamic region too small "
                  f"({dyn_px} px)" +
                  (" — FG not yet initialised, skip" if not self._fg_initialized
                   else ", skip"))
            if self._fg_initialized:
                pivot, n_dir, theta = self.fg.get_axis()
                if self._vis:
                    self._vis.update(frame_id, None, pivot, n_dir, theta, rgb_img)
                self._advance(curr_depth, curr_rgb_bgr, c2w, frame_id)
                return pivot, n_dir, theta, *self.fg.get_uncertainty()
            self._advance(curr_depth, curr_rgb_bgr, c2w, frame_id)
            return None

        if curr_rgb_bgr is None or self._prev_rgb_bgr is None:
            self._advance(curr_depth, curr_rgb_bgr, c2w, frame_id)
            return None

        # ----------------------------------------------------------------
        # LoFTR feature matching
        # ----------------------------------------------------------------
        t0 = img_to_loftr_tensor(self._prev_rgb_bgr, self.device)
        t1 = img_to_loftr_tensor(curr_rgb_bgr,       self.device)
        mkpts0, mkpts1 = match_loftr(t0, t1, self.matcher)

        src_w, tgt_w = get_3d_correspondences(
            mkpts0, mkpts1,
            self._prev_depth, curr_depth,
            self.K, self._prev_c2w, c2w,
            dyn_mask, robot_segs,
        )

        # ----------------------------------------------------------------
        # Lazy FG initialisation — triggered by first valid dynamic matches
        # ----------------------------------------------------------------
        if not self._fg_initialized:
            if src_w is None:
                print(f"[Frame {frame_id}] Dynamic mask OK but no valid LoFTR "
                      "matches yet — waiting to init FG.")
                self._advance(curr_depth, curr_rgb_bgr, c2w, frame_id)
                return None

            # === First valid dynamic point cloud — init FG now ===
            v0 = p_to_v(self._p0_init, self._omega0_init)
            self.fg = AxisFG(self._omega0_init, v0, self._theta0_init)
            # Use the *previous* frame's EE position as the rotation reference
            ref_fid = self._prev_frame_id
            if ref_fid in ee_poses_dict:
                self.fg.set_ref_ee(ee_poses_dict[ref_fid])
            self._fg_initialized = True
            print(f"[Frame {frame_id}] *** FG initialised *** "
                  f"ω={np.round(self._omega0_init, 3)}, "
                  f"p0={np.round(self._p0_init, 3)}, "
                  f"θ₀={self._theta0_init:.3f}")

        # ----------------------------------------------------------------
        # EE-based kinematic Δθ guess
        # ----------------------------------------------------------------
        dtheta_init  = 0.0
        prev_fid     = self._prev_frame_id
        ee_prev      = ee_poses_dict.get(prev_fid, None)
        ee_curr      = ee_poses_dict.get(frame_id, None)

        if ee_prev is not None and ee_curr is not None:
            pivot_cur, n_cur, _ = self.fg.get_axis()
            ee_delta = ee_curr - ee_prev
            perp     = ee_delta - np.dot(ee_delta, n_cur) * n_cur
            r_vec    = ee_curr - pivot_cur
            r_vec   -= np.dot(r_vec, n_cur) * n_cur
            radius   = np.linalg.norm(r_vec)
            if radius > 0.02 and np.linalg.norm(perp) > 1e-4:
                dtheta_init = np.linalg.norm(perp) / radius
                cross = np.cross(r_vec / (radius + 1e-9),
                                 perp  / (np.linalg.norm(perp) + 1e-9))
                if np.dot(cross, n_cur) < 0:
                    dtheta_init = -dtheta_init

        # ----------------------------------------------------------------
        # Keyframe matching: anchor + most-recent (if FG already running)
        # ----------------------------------------------------------------
        kf_vis_list = []
        if self._fg_initialized and len(self._keyframes) > 0:
            _, _, theta_pre = self.fg.get_axis()  # theta before push

            # Anchor keyframe (index 0) — long baseline
            kf_anc = self._keyframes[0]
            src_anc, tgt_anc = self._match_pair(
                kf_anc['depth'], kf_anc['rgb_bgr'], kf_anc['c2w'],
                curr_depth, curr_rgb_bgr, c2w, dyn_mask, robot_segs,
            )
            if src_anc is not None:
                theta_kf = self.fg.get_theta_at_push_idx(kf_anc['fg_push_idx'])
                if theta_kf is None:
                    theta_kf = kf_anc['theta_abs']
                dtheta_frozen = (theta_pre + dtheta_init) - theta_kf
                kf_vis_list.append((src_anc, tgt_anc,
                                    SIGMA_VIS_KF_ANCHOR, dtheta_frozen))
                # print(f"[Frame {frame_id}] anchor KF {kf_anc['frame_id']}: "
                #       f"{len(src_anc)} matches, Δθ={np.rad2deg(dtheta_frozen):.1f}°")

            # Most-recent keyframe (last) — medium baseline (skip if same as anchor)
            if len(self._keyframes) > 1:
                kf_rec = self._keyframes[-1]
                src_rec, tgt_rec = self._match_pair(
                    kf_rec['depth'], kf_rec['rgb_bgr'], kf_rec['c2w'],
                    curr_depth, curr_rgb_bgr, c2w, dyn_mask, robot_segs,
                )
                if src_rec is not None:
                    theta_kf = self.fg.get_theta_at_push_idx(kf_rec['fg_push_idx'])
                    if theta_kf is None:
                        theta_kf = kf_rec['theta_abs']
                    dtheta_frozen = (theta_pre + dtheta_init) - theta_kf
                    kf_vis_list.append((src_rec, tgt_rec,
                                        SIGMA_VIS_KF_RECENT, dtheta_frozen))
                    # print(f"[Frame {frame_id}] recent KF {kf_rec['frame_id']}: "
                    #       f"{len(src_rec)} matches, Δθ={np.rad2deg(dtheta_frozen):.1f}°")

        # ----------------------------------------------------------------
        # Push frame into FG. Keep the axis fixed until the first non-anchor
        # keyframe exists; one anchor alone is too short-baseline for stable axis optimisation.
        # ----------------------------------------------------------------
        allow_axis_opt = len(self._keyframes) >= 2
        if (not allow_axis_opt
                and len(self._keyframes) != self._last_opt_gate_kf_count):
            print(f"[Frame {frame_id}] FG optimize gated: KFs={len(self._keyframes)}/2; "
                  "accumulating frames without axis optimisation.")
            self._last_opt_gate_kf_count = len(self._keyframes)

        self.fg.push_frame(
            ee_pos=ee_curr,
            src_pts=src_w,
            tgt_pts=tgt_w,
            kf_vis_list=kf_vis_list if kf_vis_list else None,
            dtheta_init=dtheta_init,
            optimize=allow_axis_opt,
        )

        pivot, n_dir, theta = self.fg.get_axis()
        sigma_p, sigma_t    = self.fg.get_uncertainty()

        # ----------------------------------------------------------------
        # Keyframe update
        # ----------------------------------------------------------------
        n_matches = len(src_w) if src_w is not None else 0
        self._maybe_add_keyframe(frame_id, curr_depth, curr_rgb_bgr, c2w,
                                 n_matches, theta)

        print(f"[Frame {frame_id}] p={np.round(pivot, 3)}  n={np.round(n_dir, 3)}  "
              f"θ={np.rad2deg(theta):.2f}°  σp={sigma_p * 100:.1f}cm  "
              f"σθ={np.rad2deg(sigma_t):.1f}°  adj={n_matches}  "
              f"KFs={len(self._keyframes)}")

        if self._vis:
            self._vis.update(frame_id, tgt_w, pivot, n_dir, theta, rgb_img,
                             n_matches=n_matches, sigma_p=sigma_p)

        self._advance(curr_depth, curr_rgb_bgr, c2w, frame_id)
        return pivot, n_dir, theta, sigma_p, sigma_t

    # ------------------------------------------------------------------

    def get_axis(self):
        """Return (pivot, n_dir, theta), or None if FG not yet initialised."""
        if not self._fg_initialized:
            return None
        return self.fg.get_axis()

    def get_uncertainty(self):
        """Return (sigma_p, sigma_theta), or (None, None) if not initialised."""
        if not self._fg_initialized:
            return None, None
        return self.fg.get_uncertainty()

    def get_keyframe_count(self) -> int:
        """Return the number of keyframes currently stored."""
        return len(self._keyframes)

    # ------------------------------------------------------------------
    # Keyframe management
    # ------------------------------------------------------------------

    def _match_pair(self, depth0, rgb0, c2w0, depth1, rgb1, c2w1,
                    dyn_mask, robot_segs):
        """Run LoFTR between frame0 and frame1, return (src_w, tgt_w)."""
        if rgb0 is None or rgb1 is None:
            return None, None
        t0 = img_to_loftr_tensor(rgb0, self.device)
        t1 = img_to_loftr_tensor(rgb1, self.device)
        mkpts0, mkpts1 = match_loftr(t0, t1, self.matcher)
        return get_3d_correspondences(
            mkpts0, mkpts1, depth0, depth1,
            self.K, c2w0, c2w1, dyn_mask, robot_segs,
        )

    def _add_keyframe(self, frame_id, depth, rgb_bgr, c2w, theta_abs):
        """Store a new keyframe entry."""
        entry = {
            'frame_id':    frame_id,
            'depth':       depth.copy(),
            'rgb_bgr':     rgb_bgr.copy() if rgb_bgr is not None else None,
            'c2w':         c2w.copy(),
            'theta_abs':   float(theta_abs),
            'fg_push_idx': self.fg.push_count,   # index for theta lookup
        }
        self._keyframes.append(entry)
        # Keep anchor (index 0) + at most MAX_KEYFRAMES-1 recent ones
        if len(self._keyframes) > MAX_KEYFRAMES:
            self._keyframes = [self._keyframes[0]] + self._keyframes[2:]
        # print(f"[KF] Added keyframe {frame_id}  θ={np.rad2deg(theta_abs):.1f}°  "
        #       f"total KFs={len(self._keyframes)}")

    def _maybe_add_keyframe(self, frame_id, depth, rgb_bgr, c2w,
                            n_matches, theta_curr):
        """Decide whether to promote current frame to a keyframe."""
        if not self._fg_initialized:
            return
        # Always add the very first keyframe right after FG init
        if len(self._keyframes) == 0:
            self._add_keyframe(frame_id, depth, rgb_bgr, c2w, theta_curr)
            return
        # Add new keyframe when angle change and match quality are sufficient
        last_theta = self._keyframes[-1]['theta_abs']
        if (abs(theta_curr - last_theta) >= KF_DTHETA_MIN
                and n_matches >= KF_MIN_MATCHES):
            self._add_keyframe(frame_id, depth, rgb_bgr, c2w, theta_curr)

    def finish(self):
        """Block until the visualisation window is closed."""
        if self._vis:
            print("\nAll frames processed. Close the plot window to exit.")
            plt.ioff()
            plt.show()

    def _advance(self, depth, rgb_bgr, c2w, frame_id):
        self._prev_depth    = depth
        self._prev_rgb_bgr  = rgb_bgr
        self._prev_c2w      = c2w.copy()
        self._prev_frame_id = frame_id

# =============================================================================
# Demo main
# =============================================================================

def main(visualize: bool = True, step: int = 1):
    print(f"[Main] Loading dataset: {DATA_ROOT}")
    K, _, _        = load_camera_info(DATA_ROOT)
    cam_poses      = load_camera_poses(DATA_ROOT)
    link_poses_all = load_link_poses(DATA_ROOT)
    ee_poses       = load_ee_poses(DATA_ROOT)

    depth_files = sorted(DATA_ROOT.glob("depth/depth_*.npy"))
    print(f"[Main] {len(depth_files)} frames total")

    # ---------------------------------------------------------------
    # Hard-coded initial guess — edit these two lines to tune.
    # FG will only be created when the first valid dynamic point
    # cloud appears; these values serve as the prior at that moment.
    # ---------------------------------------------------------------
    p0     = np.array([0.9, 0.4, 0.0])   # <-- point on axis (world, metres)
    omega0 = np.array([0.0, 0.0, 1.0])   # <-- axis direction (normalised auto)

    estimator = LoFTRAxisEstimatorFG(
        K         = K,
        omega0    = omega0,
        p0        = p0,
        theta0    = 0.0,
        visualize = visualize,
    )

    for depth_file in depth_files[::step]:
        frame_id = int(depth_file.stem.split("_")[-1])
        if frame_id not in cam_poses or frame_id not in link_poses_all:
            continue

        c2w     = cam_poses[frame_id]
        lp_dict = link_poses_all[frame_id]

        curr_depth   = np.load(str(depth_file)).astype(np.float32)
        rgb_path     = DATA_ROOT / "rgb" / f"rgb_{frame_id:06d}.png"
        curr_rgb_bgr = cv2.imread(str(rgb_path)) if rgb_path.exists() else None

        estimator.step(
            frame_id      = frame_id,
            curr_depth    = curr_depth,
            curr_rgb_bgr  = curr_rgb_bgr,
            c2w           = c2w,
            lp_dict       = lp_dict,
            ee_poses_dict = ee_poses,
        )

    estimator.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-vis', action='store_true')
    parser.add_argument('--step',   type=int, default=1)
    args = parser.parse_args()
    main(visualize=not args.no_vis, step=args.step)
