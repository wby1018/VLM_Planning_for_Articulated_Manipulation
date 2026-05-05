#!/usr/bin/env python3
"""
action_server.py — VLM-based Action Server for Articulated Object Manipulation

Architecture
------------
  client_mujoco.py  <--ZMQ REQ-REP-->  action_server.py
                                              |
                                    det_pipeline (HTTP :8000)
                                              |
                                       VLM API

Pipeline
--------
  VLM initial plan
  → MoveTo → Approach → Grasp
  → [auto] ProbePull  (small pull along panel_normal, saves P0/P1)
  → [auto] TypeCheck  (5-hypothesis point cloud matching, no LoFTR)
      Translation → Pull_Linear
      Rotation    → Pull_Arc  (LoFTR-PF refines axis online)
      Unknown     → keep VLM plan
  → Release

Wire protocol (ZMQ port 5555)
-----------------------------
Client → Server  compressed pickle of obs_dict:
  'rgb'        : {'shape':(H,W,3),'dtype':'uint8',   'data':bytes}  — RGB from rgbd_camera
  'depth'      : {'shape':(H,W),  'dtype':'float32', 'data':bytes}  — depth in metres
  'cam_pos'    : {'shape':(3,),   'dtype':'float64', 'data':bytes}  — world pos of camera
  'cam_mat'    : {'shape':(3,3),  'dtype':'float64', 'data':bytes}  — camera rotation matrix
  'fovy'       : {'shape':(1,),   'dtype':'float64', 'data':bytes}  — vertical FoV (deg)
  'agent_pos'  : {'shape':(1,T,10),'dtype':'float32','data':bytes}  — eef[3]+rot6d[6]+g[1]
  'point_cloud': {'shape':(1,T,1280,3),'dtype':'float32','data':bytes}

Server → Client  compressed pickle of action_dict:
  {'shape':(10,), 'dtype':'float32', 'data':bytes}
  layout: [target_pos(3), target_rot_6d(6), target_gripper(1)]
"""

import base64
import json
import os
import pickle
import queue
import threading
import traceback
import zlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import zmq
import matplotlib.pyplot as plt
from openai import OpenAI
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

from loftr_pf import LoFTRAxisEstimator

# ─────────────────────────── Configuration ────────────────────────────────────

ZMQ_PORT         = 5555
DET_SERVER_URL   = "http://127.0.0.1:8000/detect"
DET_QUERIES      = ["handle", "drawer", "cabinet door"]

VLM_MODEL        = "gpt-5.4"   # model name used by the Responses API
DET_THRESHOLD    = 0.2

USE_VISUALIZER   = True

USER_INSTRUCTION = "open the cabinet door"

# Motion tuning
APPROACH_DIST   = 0.19   # m — standoff before grasping
STAGE_POS_TOL   = 0.02   # m — position tolerance to mark stage done (tightened from 0.038)
STAGE_ANG_TOL   = 0.08   # rad
STAGE_GRIP_TOL  = 0.004  # m gripper width
GRASP_TIMEOUT_STEPS = 15 # steps (approx 1.5s) to wait for gripper

MAX_DELTA_POS   = 0.018  # m per step (unused by server, handled by client)
MAX_DELTA_ANG   = np.deg2rad(2.5)
GRIPPER_DELTA   = 0.003  # m per step
GRIPPER_OPEN    = 0.04
GRIPPER_CLOSED  = 0.000
GRASP_OFFSET    = 0.07  # m — distance from Panda 'hand' frame to finger tips
GRASP_DEPTH_OFFSET = 0.015  # m — pull back the target pose to avoid hitting the door
TARGET_Z_OFFSET = 0.01   # m — upward shift of the target grasp point
NORMAL_SCAN_RADIUS = 0.15 # m — radius around handle to sample panel points

PULL_LINEAR_DIST = 0.38  # m total drawer pull
PULL_STEP        = 0.005  # m per step increment
PULL_ARC_ANGLE   = np.deg2rad(90)  # total door sweep
ARC_STEP         = np.deg2rad(0.3)  # rad per step

# ProbePull / TypeCheck
PROBE_DISTANCE       = 0.05   # m — small pull for type identification
TYPECHECK_MARGIN     = 0.002  # m — margin for Translation vs Rotation decision
EDGE_PERCENTILE      = 10     # % — edge band width for axis fitting
CHAMFER_TRIM_RATIO   = 0.9    # top 90% nearest-neighbour distances used
MIN_PROBE_PCD_POINTS = 20     # minimum points in P0/P1 for TypeCheck
MIN_TYPECHECK_THETA  = 0.01   # rad — minimum rotation angle to consider
ARC_SMOOTH_ALPHA     = 0.8    # smoothing factor for LoFTR axis update

# ─────────────────────────── Rotation helpers ─────────────────────────────────

def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """6D rotation representation → 3×3 rotation matrix."""
    r1 = r6d[:3];  r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    tmp = r6d[3:]
    r2 = tmp - np.dot(tmp, r1) * r1;  r2 = r2 / (np.linalg.norm(r2) + 1e-8)
    r3 = np.cross(r1, r2)
    return np.stack([r1, r2, r3], axis=1)


def matrix_to_rot6d(mat: np.ndarray) -> np.ndarray:
    return np.concatenate([mat[:, 0], mat[:, 1]])


def wxyz_to_scipy(wxyz: np.ndarray) -> R:
    """MuJoCo w,x,y,z → scipy Rotation."""
    return R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


def scipy_to_wxyz(rot: R) -> np.ndarray:
    q = rot.as_quat()  # scipy: x,y,z,w
    return np.array([q[3], q[0], q[1], q[2]])


def look_at_rotation(forward: np.ndarray,
                     up_hint: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Build a 3×3 rotation matrix whose Z-axis points along *forward*."""
    z = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(up_hint, z)
    if np.linalg.norm(right) < 1e-6:
        up_hint = np.array([0.0, 1.0, 0.0])
        right = np.cross(up_hint, z)
    x = right / (np.linalg.norm(right) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)

# ─────────────────────────── Deserialisation helper ───────────────────────────

def deser(d: Dict) -> np.ndarray:
    return np.frombuffer(d['data'], dtype=np.dtype(d['dtype'])).reshape(d['shape'])


def decode_mask(b64_str: str, h: int, w: int) -> Optional[np.ndarray]:
    """Decode base64 PNG mask to boolean array."""
    try:
        mask_bytes = base64.b64decode(b64_str)
        mask_arr = np.frombuffer(mask_bytes, np.uint8)
        mask_gray = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None: return None
        if mask_gray.shape != (h, w):
            mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask_gray > 128
    except Exception as e:
        print(f"[Utils] Mask decode failed: {e}")
        return None

# ─────────────────────────── 3-D geometry ─────────────────────────────────────

def backproject_pixel(u: float, v: float, depth_val: float,
                      cam_pos: np.ndarray, cam_mat: np.ndarray,
                      fovy_deg: float, h: int, w: int) -> np.ndarray:
    """
    Back-project image pixel (u, v) with given metric depth to world 3-D.

    MuJoCo camera convention:
      • camera looks along its local -Z axis
      • Y is up in camera space
    """
    f = (h / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    x_cam =  (u - w / 2.0) * depth_val / f
    y_cam = -(v - h / 2.0) * depth_val / f   # image-Y down → camera-Y up
    z_cam = -depth_val                         # depth along -Z
    return cam_pos + cam_mat @ np.array([x_cam, y_cam, z_cam])


def backproject_mask_to_pcd(depth: np.ndarray, mask: np.ndarray,
                             cam_pos: np.ndarray, cam_mat: np.ndarray,
                             fovy_deg: float,
                             max_points: int = 2000) -> np.ndarray:
    """Back-project all masked depth pixels to world-space point cloud (Nx3)."""
    h, w = depth.shape
    f = (h / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)

    v_idx, u_idx = np.where(mask)
    zs = depth[v_idx, u_idx]
    valid = zs > 0
    v_idx, u_idx, zs = v_idx[valid], u_idx[valid], zs[valid]

    if len(v_idx) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if len(v_idx) > max_points:
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(v_idx), max_points, replace=False)
        v_idx, u_idx, zs = v_idx[idxs], u_idx[idxs], zs[idxs]

    # Vectorised back-projection (MuJoCo: camera looks along -Z, Y up)
    x_c = (u_idx - w / 2.0) * zs / f
    y_c = -(v_idx - h / 2.0) * zs / f
    z_c = -zs
    local_pts = np.stack([x_c, y_c, z_c], axis=1)   # (N,3)
    world_pts = cam_pos + (cam_mat @ local_pts.T).T  # (N,3)
    return world_pts


def rotate_around_axis(point: np.ndarray, axis_point: np.ndarray,
                        axis_dir: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a single 3-D point around an axis (Rodrigues)."""
    k = axis_dir / (np.linalg.norm(axis_dir) + 1e-8)
    v = point - axis_point
    rotated = (v * np.cos(theta)
               + np.cross(k, v) * np.sin(theta)
               + k * np.dot(k, v) * (1.0 - np.cos(theta)))
    return rotated + axis_point


def rotate_pcd_around_axis(pts: np.ndarray, axis_point: np.ndarray,
                            axis_dir: np.ndarray, theta: float) -> np.ndarray:
    """Rotate Nx3 point cloud around a 3-D axis (vectorised Rodrigues)."""
    k = axis_dir / (np.linalg.norm(axis_dir) + 1e-8)
    v = pts - axis_point                               # (N,3)
    k_dot_v = (v @ k)[:, None]                         # (N,1)
    rotated = (v * np.cos(theta)
               + np.cross(k, v) * np.sin(theta)
               + k * k_dot_v * (1.0 - np.cos(theta)))
    return rotated + axis_point


def signed_angle_around_axis(v1: np.ndarray, v2: np.ndarray,
                              axis: np.ndarray) -> float:
    """Signed angle from v1 to v2 about *axis* (right-hand rule)."""
    k = axis / (np.linalg.norm(axis) + 1e-8)
    v1p = v1 - np.dot(v1, k) * k
    v2p = v2 - np.dot(v2, k) * k
    n1, n2 = np.linalg.norm(v1p), np.linalg.norm(v2p)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    v1p, v2p = v1p / n1, v2p / n2
    cos_a = float(np.clip(np.dot(v1p, v2p), -1.0, 1.0))
    sin_a = float(np.dot(np.cross(v1p, v2p), k))
    return float(np.arctan2(sin_a, cos_a))


def compute_theta_from_gripper(g0: np.ndarray, g1: np.ndarray,
                                axis_point: np.ndarray,
                                axis_dir: np.ndarray) -> float:
    """Infer rotation angle for a given axis hypothesis from gripper displacement."""
    v0 = g0 - axis_point
    v1 = g1 - axis_point
    return signed_angle_around_axis(v0, v1, axis_dir)


def trimmed_chamfer(A: np.ndarray, B: np.ndarray,
                    trim: float = CHAMFER_TRIM_RATIO) -> float:
    """
    Symmetric trimmed Chamfer distance between point clouds A and B.
    Uses the lower *trim* quantile of nearest-neighbour distances to be robust
    against occlusion changes and mask errors.
    """
    if len(A) == 0 or len(B) == 0:
        return float("inf")

    tree_B = KDTree(B)
    dists_AB, _ = tree_B.query(A)
    thresh_AB = float(np.quantile(dists_AB, trim))
    E_AB = float(dists_AB[dists_AB <= thresh_AB].mean())

    tree_A = KDTree(A)
    dists_BA, _ = tree_A.query(B)
    thresh_BA = float(np.quantile(dists_BA, trim))
    E_BA = float(dists_BA[dists_BA <= thresh_BA].mean())

    return (E_AB + E_BA) / 2.0


def fit_3d_line_ransac(pts: np.ndarray,
                        n_iter: int = 50,
                        inlier_thresh: float = 0.02,
                        min_inliers: int = 5
                        ) -> Optional[Dict[str, Any]]:
    """
    Fit a 3-D line to *pts* using RANSAC + PCA refinement.

    Returns dict with keys: point, dir, fit_error, n_inliers.
    Returns None when there are insufficient inliers.
    """
    if len(pts) < min_inliers:
        return None

    best_inlier_mask = np.zeros(len(pts), dtype=bool)
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(len(pts), 2, replace=False)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        d = p2 - p1
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            continue
        d = d / nd

        diff = pts - p1
        proj = (diff @ d)[:, None] * d
        dists = np.linalg.norm(diff - proj, axis=1)
        inlier_mask = dists < inlier_thresh

        if inlier_mask.sum() > best_inlier_mask.sum():
            best_inlier_mask = inlier_mask

    if best_inlier_mask.sum() < min_inliers:
        # PCA fallback
        best_inlier_mask = np.ones(len(pts), dtype=bool)

    inlier_pts = pts[best_inlier_mask]
    ctr = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - ctr, full_matrices=False)
    direction = Vt[0]  # axis of maximum variance

    # Compute residual error
    diff = inlier_pts - ctr
    proj = (diff @ direction)[:, None] * direction
    fit_error = float(np.linalg.norm(diff - proj, axis=1).mean())

    return {
        "point": ctr,
        "dir": direction,
        "fit_error": fit_error,
        "n_inliers": int(best_inlier_mask.sum()),
    }


def estimate_edge_axes(P0: np.ndarray,
                        percentile: int = EDGE_PERCENTILE
                        ) -> List[Dict[str, Any]]:
    """
    Fit 3-D lines to the four edge bands (left, right, bottom, top) of P0.

    Each returned dict contains: side, point, dir, fit_error, n_inliers.
    """
    if len(P0) < 30:
        return []

    # Determine horizontal direction by XY-plane PCA
    ctr = P0.mean(axis=0)
    pts_xy = P0[:, :2] - ctr[:2]
    _, _, Vt_2d = np.linalg.svd(pts_xy, full_matrices=False)
    horiz_2d = Vt_2d[0]
    horizontal_dir = np.array([horiz_2d[0], horiz_2d[1], 0.0])
    hn = np.linalg.norm(horizontal_dir)
    if hn < 1e-6:
        return []
    horizontal_dir /= hn

    # Side direction: perpendicular to horizontal in XY plane
    side_dir = np.array([-horizontal_dir[1], horizontal_dir[0], 0.0])

    side_proj = P0 @ side_dir  # projection onto side direction
    z_proj    = P0[:, 2]        # world height

    axes = []
    configs = [
        ("left",   side_proj, False),
        ("right",  side_proj, True),
        ("bottom", z_proj,    False),
        ("top",    z_proj,    True),
    ]

    for side, proj, use_upper in configs:
        if use_upper:
            thresh = float(np.percentile(proj, 100 - percentile))
            edge_mask = proj >= thresh
        else:
            thresh = float(np.percentile(proj, percentile))
            edge_mask = proj <= thresh

        edge_pts = P0[edge_mask]
        if len(edge_pts) < 5:
            continue

        line_res = fit_3d_line_ransac(edge_pts)
        if line_res is None:
            continue

        axes.append({
            "side":      side,
            "point":     line_res["point"],
            "dir":       line_res["dir"],
            "fit_error": line_res["fit_error"],
            "n_inliers": line_res["n_inliers"],
        })

    return axes


def type_check(P0: np.ndarray, P1: np.ndarray,
               g0: np.ndarray, g1: np.ndarray,
               panel_normal: np.ndarray) -> Dict[str, Any]:
    """
    Compare 5 motion hypotheses (1 Translation + 4 Rotation candidates) using
    trimmed Chamfer distance between the probe start/end point clouds.

    Returns a dict with:
      motion_type   : "Translation" | "Rotation" | "Unknown"
      confidence    : float [0, 1]
      E_translation : float
      E_rotation    : float (best among all axes)
      best_axis_side: str
      best_axis_point: ndarray(3,) | None
      best_axis_dir  : ndarray(3,) | None
      theta_axis     : float (rad)
      axis_fit_error : float
    """
    result: Dict[str, Any] = {
        "motion_type":    "Unknown",
        "confidence":     0.0,
        "E_translation":  float("nan"),
        "E_rotation":     float("nan"),
        "best_axis_side": None,
        "best_axis_point": None,
        "best_axis_dir":  None,
        "theta_axis":     0.0,
        "axis_fit_error": float("nan"),
    }

    if len(P0) < MIN_PROBE_PCD_POINTS or len(P1) < MIN_PROBE_PCD_POINTS:
        print(f"[TypeCheck] Insufficient points: P0={len(P0)}, P1={len(P1)} — returning Unknown")
        return result

    dg = g1 - g0  # actual gripper displacement

    # ── H0: Translation ──────────────────────────────────────────────────────
    P0_pred_T = P0 + dg
    E_T = trimmed_chamfer(P0_pred_T, P1)
    result["E_translation"] = E_T
    print(f"[TypeCheck] H0 (Translation): E={E_T:.5f}")

    # ── H1–H4: Rotation around each candidate edge axis ──────────────────────
    candidate_axes = estimate_edge_axes(P0)
    if not candidate_axes:
        print("[TypeCheck] No valid edge axes found — defaulting to Translation")
        result["motion_type"] = "Translation"
        result["confidence"] = 0.5
        return result

    best_E_rot    = float("inf")
    best_axis_rec = None

    for ax in candidate_axes:
        theta = compute_theta_from_gripper(g0, g1, ax["point"], ax["dir"])
        if abs(theta) < MIN_TYPECHECK_THETA:
            print(f"[TypeCheck] {ax['side']}: θ={np.rad2deg(theta):.2f}° too small, skip")
            continue

        P0_pred_R = rotate_pcd_around_axis(P0, ax["point"], ax["dir"], theta)
        E = trimmed_chamfer(P0_pred_R, P1)

        print(f"[TypeCheck] {ax['side']}: E={E:.5f}  θ={np.rad2deg(theta):.1f}°  "
              f"fit_err={ax['fit_error']:.4f}  n={ax['n_inliers']}")

        if E < best_E_rot:
            best_E_rot = E
            best_axis_rec = {
                "side":       ax["side"],
                "point":      ax["point"].copy(),
                "dir":        ax["dir"].copy(),
                "theta":      theta,
                "fit_error":  ax["fit_error"],
            }

    result["E_rotation"] = best_E_rot

    if best_axis_rec is not None:
        result["best_axis_side"]  = best_axis_rec["side"]
        result["best_axis_point"] = best_axis_rec["point"]
        result["best_axis_dir"]   = best_axis_rec["dir"]
        result["theta_axis"]      = best_axis_rec["theta"]
        result["axis_fit_error"]  = best_axis_rec["fit_error"]

    # ── Decision ─────────────────────────────────────────────────────────────
    E_T_ok = not (np.isnan(E_T) or np.isinf(E_T))
    E_R_ok = not (np.isnan(best_E_rot) or np.isinf(best_E_rot)) and best_axis_rec is not None

    if E_T_ok and E_R_ok:
        denom = E_T + best_E_rot + 1e-8
        if E_T + TYPECHECK_MARGIN < best_E_rot:
            result["motion_type"] = "Translation"
            result["confidence"]  = (best_E_rot - E_T) / denom
        elif best_E_rot + TYPECHECK_MARGIN < E_T:
            if abs(best_axis_rec["theta"]) < np.deg2rad(2.0):
                result["motion_type"] = "Unknown"
                result["confidence"]  = 0.1
            else:
                result["motion_type"] = "Rotation"
                result["confidence"]  = (E_T - best_E_rot) / denom
        # else: stays Unknown
    elif E_T_ok:
        result["motion_type"] = "Translation"
        result["confidence"]  = 0.5
    elif E_R_ok:
        result["motion_type"] = "Rotation"
        result["confidence"]  = 0.5

    print(f"[TypeCheck] Decision: {result['motion_type']} "
          f"(conf={result['confidence']:.3f}, E_T={E_T:.5f}, E_R={best_E_rot:.5f})")
    return result


def estimate_panel_normal(depth: np.ndarray, handle_bbox: List[int],
                           handle_3d: np.ndarray,
                           cam_pos: np.ndarray, cam_mat: np.ndarray,
                           fovy_deg: float,
                           parent_mask: Optional[np.ndarray] = None,
                           scan_radius: float = NORMAL_SCAN_RADIUS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the outward face normal of the panel around the handle.
    Uses a 3D radius search + 2D handle bbox exclusion + SAM mask filtering.
    """
    h, w = depth.shape
    hx1, hy1, hx2, hy2 = [int(v) for v in handle_bbox]

    # 1. Determine 2D scan range.
    if parent_mask is not None:
        v_idx, u_idx = np.where(parent_mask)
        if len(v_idx) == 0:
            u1, v1, u2, v2 = max(0, hx1-100), max(0, hy1-100), min(w, hx2+100), min(h, hy2+100)
        else:
            u1, v1, u2, v2 = u_idx.min(), v_idx.min(), u_idx.max(), v_idx.max()
    else:
        u1, v1, u2, v2 = max(0, hx1-120), max(0, hy1-120), min(w, hx2+120), min(h, hy2+120)

    f = (h / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)

    # Grid sampling (step 2 for speed)
    vs, us = np.meshgrid(range(v1, v2, 2), range(u1, u2, 2), indexing='ij')
    vs = vs.flatten();  us = us.flatten()
    zs = depth[vs, us]
    valid_depth = zs > 0
    vs, us, zs = vs[valid_depth], us[valid_depth], zs[valid_depth]

    final_pts = []
    for i in range(len(zs)):
        u, v, z = us[i], vs[i], zs[i]

        if parent_mask is not None and not parent_mask[v, u]:
            continue
        if (hx1 - 3 <= u <= hx2 + 3) and (hy1 - 3 <= v <= hy2 + 3):
            continue

        x_c = (u - w/2.0) * z / f
        y_c = -(v - h/2.0) * z / f
        p_world = cam_pos + cam_mat @ np.array([x_c, y_c, -z])

        if np.linalg.norm(p_world - handle_3d) < scan_radius:
            final_pts.append(p_world)

    if len(final_pts) < 10:
        print(f"[Planner] [ERROR] normal estimation failed: only {len(final_pts)} points. Using fallback.")
        return -cam_mat[:, 2], np.array(final_pts) if final_pts else np.zeros((0, 3))

    pts_world = np.stack(final_pts)
    ctr = pts_world.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts_world - ctr)
    normal = Vt[-1]

    # Force horizontal (z=0)
    normal[2] = 0
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    if np.dot(normal, cam_pos - ctr) < 0:
        normal = -normal
    return normal, pts_world


def estimate_hinge_params(depth: np.ndarray,
                           parent_mask: np.ndarray,
                           handle_bbox: List[int],
                           cam_pos: np.ndarray, cam_mat: np.ndarray,
                           fovy_deg: float) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Estimate hinge world position by looking at the extreme horizontal
    edge of the door mask opposite to the handle.
    """
    h, w = depth.shape
    v_idx, u_idx = np.where(parent_mask)
    if len(v_idx) == 0:
        return np.array([0, 0, 0]), 0.3

    u_min, u_max = u_idx.min(), u_idx.max()
    hx1, hy1, hx2, hy2 = handle_bbox
    hu = (hx1 + hx2) / 2.0

    mask_width = u_max - u_min
    if hu > (u_min + u_max) / 2.0:
        edge_pixels = (parent_mask[:, u_min : u_min + max(1, int(mask_width * 0.01))])
        edge_v_local, edge_u_local = np.where(edge_pixels)
        target_u = edge_u_local + u_min
        target_v = edge_v_local
    else:
        edge_pixels = (parent_mask[:, u_max - max(1, int(mask_width * 0.05)) : u_max + 1])
        edge_v_local, edge_u_local = np.where(edge_pixels)
        target_u = edge_u_local + (u_max - max(1, int(mask_width * 0.05)))
        target_v = edge_v_local

    if len(target_u) == 0:
        return np.array([0,0,0]), 0.3

    f = (h / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)

    step = max(1, len(target_u) // 200)
    target_u, target_v = target_u[::step], target_v[::step]

    edge_pts_3d = []
    for u, v in zip(target_u, target_v):
        z = depth[v, u]
        if z <= 0: continue
        x_c = (u - w/2.0) * z / f
        y_c = -(v - h/2.0) * z / f
        p_world = cam_pos + cam_mat @ np.array([x_c, y_c, -z])
        edge_pts_3d.append(p_world)

    if len(edge_pts_3d) < 5:
        return np.array([0,0,0]), 0.3

    pts = np.stack(edge_pts_3d)

    z_med = np.median(pts[:, 2])
    z_std = np.std(pts[:, 2])
    valid_mask = np.abs(pts[:, 2] - z_med) < max(0.1, z_std * 2.0)
    pts = pts[valid_mask]

    if len(pts) == 0:
        return np.array([0,0,0]), 0.3

    hinge_xy = pts[:, :2].mean(axis=0)
    hinge_pos = np.array([hinge_xy[0], hinge_xy[1], z_med])

    h_uc, h_vc = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
    h_z = depth[int(h_vc), int(h_uc)]
    if h_z <= 0:
        h_z = z_med
    h_xc = (h_uc - w/2.0) * h_z / f
    h_yc = -(h_vc - h/2.0) * h_z / f
    handle_3d_proj = cam_pos + cam_mat @ np.array([h_xc, h_yc, -h_z])

    radius = np.linalg.norm(handle_3d_proj[:2] - hinge_xy)
    print(f"[Planner] Hinge: Side={'Left' if hu > (u_min+u_max)/2.0 else 'Right'}, "
          f"Pos={np.round(hinge_pos, 3)}, Radius={radius:.3f}m")

    return hinge_pos, radius, pts

# ─────────────────────────── Detection & annotation ───────────────────────────

def call_detection(rgb_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """POST to det_pipeline FastAPI server and return detection list."""
    _, buf = cv2.imencode('.jpg', rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    try:
        resp = requests.post(
            DET_SERVER_URL,
            files={"file": ("obs.jpg", buf.tobytes(), "image/jpeg")},
            data={"text_queries": json.dumps(DET_QUERIES),
                  "score_threshold": DET_THRESHOLD},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("detections", [])
    except Exception as e:
        print(f"[Det] Detection call failed: {e}")
        return []


def draw_annotated_image(rgb_bgr: np.ndarray,
                          detections: List[Dict]) -> np.ndarray:
    """
    Cleaner visualization:
    1. Draw only high-contrast bboxes and small IDs on the main image.
    2. Add a structured legend in the corner showing [ID]: Label (Score).
    """
    H, W = rgb_bgr.shape[:2]
    out = rgb_bgr.copy()

    PALETTE = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 165, 255), (203, 192, 255),
        (128, 0, 128), (128, 128, 0), (0, 128, 0), (0, 0, 128),
    ]

    legend_items = []

    for det in detections:
        idx = det.get("index", 0)
        color = PALETTE[int(idx) % len(PALETTE)]

        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = det["detection"][0]["label"]
        score = det["detection"][0]["score"]

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        id_text = f"{idx}"
        (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (x1, y1), (x1 + tw + 4, y1 + th + 4), color, -1)

        text_color = (255 - color[0], 255 - color[1], 255 - color[2])
        cv2.putText(out, id_text, (x1 + 2, y1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

        legend_items.append({"id": idx, "label": label, "score": score, "color": color})

    if legend_items:
        margin = 10
        item_h = 24
        header_h = 25
        leg_w  = 200
        leg_h  = header_h + len(legend_items) * item_h + margin

        leg_x1 = W - leg_w - margin
        leg_y1 = margin
        overlay = out.copy()
        cv2.rectangle(overlay, (leg_x1, leg_y1), (leg_x1 + leg_w, leg_y1 + leg_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

        cv2.putText(out, "LEGEND", (leg_x1 + 5, leg_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        for i, item in enumerate(legend_items):
            iy = leg_y1 + header_h + i * item_h + 15
            cv2.rectangle(out, (leg_x1 + 8, iy - 10), (leg_x1 + 20, iy + 2), item["color"], -1)
            text = f"[{item['id']}] {item['label']} ({item['score']:.2f})"
            cv2.putText(out, text, (leg_x1 + 28, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return out

# ─────────────────────────── Visualization ────────────────────────────────────

class Visualizer3D:
    def __init__(self, window_name="Action Server Visualizer"):
        plt.ion()
        self.fig = plt.figure(window_name, figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.view_init(elev=25, azim=-135)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')

        self.ax.set_xlim(0.5, 1.5)
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_zlim(0.0, 1.2)
        self.ax.set_box_aspect((1, 1, 1))

        self.points_scat = self.ax.scatter([], [], [], s=2, c='gray', alpha=0.3, label='Background PCD')
        self.normal_scat = self.ax.scatter([], [], [], s=10, c='red', alpha=0.8, label='Normal Sample Pts')
        self.hinge_pts_scat = self.ax.scatter([], [], [], s=15, c='blue', alpha=0.9, label='Hinge Edge Pts')
        self.hinge_scat  = self.ax.scatter([], [], [], s=150, c='magenta', edgecolors='black', alpha=1.0, label='Hinge Axis', marker='*')
        self.hinge_line, = self.ax.plot([], [], [], color='magenta', linewidth=3, alpha=0.8, label='Hinge Line')
        self.curr_quivers = None
        self.tgt_quivers  = None

    def update(self, points=None, normal_pts=None, hinge_pos=None,
               hinge_pts=None, curr_pos=None, curr_mat=None,
               tgt_pos=None, tgt_mat=None):
        updated = False

        if points is not None and len(points) > 0:
            self.points_scat._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            z = points[:, 2]
            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
            self.points_scat.set_array(z_norm)
            updated = True

        if normal_pts is not None and len(normal_pts) > 0:
            self.normal_scat._offsets3d = (normal_pts[:, 0], normal_pts[:, 1], normal_pts[:, 2])
            updated = True

        if hinge_pts is not None and len(hinge_pts) > 0:
            self.hinge_pts_scat._offsets3d = (hinge_pts[:, 0], hinge_pts[:, 1], hinge_pts[:, 2])
            updated = True

        if hinge_pos is not None:
            self.hinge_scat._offsets3d = (np.array([hinge_pos[0]]), np.array([hinge_pos[1]]), np.array([hinge_pos[2]]))
            self.hinge_line.set_data_3d([hinge_pos[0], hinge_pos[0]],
                                        [hinge_pos[1], hinge_pos[1]],
                                        [0.0, 1.2])
            updated = True

        if curr_pos is not None and curr_mat is not None:
            if self.curr_quivers:
                for q in self.curr_quivers: q.remove()
            self.curr_quivers = self._draw_axes(curr_pos, curr_mat, scale=0.15, alpha=1.0)
            updated = True

        if tgt_pos is not None and tgt_mat is not None:
            if self.tgt_quivers:
                for q in self.tgt_quivers: q.remove()
            self.tgt_quivers = self._draw_axes(tgt_pos, tgt_mat, scale=0.1, alpha=0.4)
            updated = True

        if updated:
            self.fig.canvas.draw_idle()

        plt.pause(0.01)

    def _draw_axes(self, pos, mat, scale=0.1, alpha=1.0):
        qs = []
        colors = ['r', 'g', 'b']
        for i in range(3):
            q = self.ax.quiver(pos[0], pos[1], pos[2],
                               mat[0, i], mat[1, i], mat[2, i],
                               length=scale, color=colors[i], alpha=alpha,
                               linewidth=2, normalize=True)
            qs.append(q)
        return qs

# ─────────────────────────── VLM API ──────────────────────────────────────────

VLM_SYSTEM_PROMPT = """\
**Role**: You are a high-level Robotic Task Planner for articulated object manipulation.

**Task**:
Analyze the input image containing ID-labeled bounding boxes. Based on the user's natural
language instruction, generate a structured JSON plan to operate drawers or cabinet doors.

**Object Categories & Logic**:
1. **Drawer**: A sliding object.
   - `motion_type`: `Translation`
   - `valid_action`: `Pull_Linear`
2. **Cabinet Door**: A hinged swinging object.
   - `motion_type`: `Rotation`
   - `valid_action`: `Pull_Arc`

**Gripper Orientation Logic**:
- `Vertical`: Select if the handle's width is significantly greater than its height.
- `Horizontal`: Select if the handle's height >= width, or if it is a knob/square shape.

**Strict Constraints**:
1. **Valid Stages**: ONLY use these: ["MoveTo", "Grasp", "Pull_Linear", "Pull_Arc", "Release"]
   (The server will auto-insert Approach, ProbePull, TypeCheck — do NOT include them.)
2. **Output Format**: Return ONLY a valid JSON object. No preamble, no post-analysis.

**JSON Schema**:
{
  "target_handle_id": int,
  "parent_object_id": int,
  "motion_type": "Translation" | "Rotation",
  "gripper_orientation": "Horizontal" | "Vertical",
  "plan": ["Stage_1", "Stage_2", ...]
}

---
**User Instruction**: """


def _image_bgr_to_data_url(bgr: np.ndarray) -> str:
    """Encode a BGR numpy image as a base64 JPEG data URL for the Responses API."""
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def call_vlm(annotated_bgr: np.ndarray, user_instruction: str,
             detections: List[Dict],
             use_api: bool = True) -> Dict[str, Any]:
    """
    Call the VLM with the annotated image and user instruction.

    Parameters
    ----------
    use_api : bool
        If True, call the real OpenAI Responses API (reads OPENAI_API_KEY from env).
        If False, fall back to the hardcoded default plan.
    """
    if use_api:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[VLM] WARNING: OPENAI_API_KEY not set — falling back to hardcoded plan.")
        else:
            try:
                client = OpenAI(api_key=api_key)
                image_data_url = _image_bgr_to_data_url(annotated_bgr)
                prompt = VLM_SYSTEM_PROMPT + user_instruction

                response = client.responses.create(
                    model=VLM_MODEL,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": image_data_url},
                            ],
                        }
                    ],
                )
                raw_text = response.output_text.strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                plan = json.loads(raw_text)
                print(f"[VLM] API response: {plan}")
                return plan
            except Exception as e:
                print(f"[VLM] API call failed: {e} — falling back to hardcoded plan.")

    print("[VLM] Using hardcoded default plan.")
    # return {
    #     'target_handle_id': 8, 'parent_object_id': 7,
    #     'motion_type': 'Translation', 'gripper_orientation': 'Vertical',
    #     'plan': ['MoveTo', 'Grasp', 'Pull_Linear', 'Release']
    # }
    return {
        'target_handle_id': 3, 'parent_object_id': 2,
        'motion_type': 'Rotation', 'gripper_orientation': 'Horizontal',
        'plan': ['MoveTo', 'Grasp', 'Pull_Arc', 'Release']
    }


# ─────────────────────────── LoFTR Axis Estimator Thread ─────────────────────

class LoFTREstimatorThread:
    """
    Background thread that continuously refines the rotation axis using LoFTR-PF.

    Started only after TypeCheck determines Rotation.
    Frames are pushed via push(); the latest axis estimate is readable via
    get_axis() at any time.  Drops frames when the particle filter is slower
    than the control loop so the queue never grows unbounded.
    """

    def __init__(self, K: np.ndarray, omega0: np.ndarray, p0: np.ndarray):
        self._estimator = LoFTRAxisEstimator(K, omega0, p0, theta0=0.0, visualize=False)
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._pivot    = np.array(p0,     dtype=np.float64).copy()
        self._axis_dir = np.array(omega0,  dtype=np.float64).copy()
        # Normalise stored direction
        nd = np.linalg.norm(self._axis_dir)
        if nd > 1e-6:
            self._axis_dir /= nd
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, frame_id: int, depth: np.ndarray, rgb_bgr: np.ndarray,
             c2w: np.ndarray, ee_poses_dict: Dict[int, np.ndarray]) -> None:
        try:
            self._queue.put_nowait((frame_id, depth, rgb_bgr, c2w, ee_poses_dict))
        except queue.Full:
            pass  # drop frame when thread is still processing the previous one

    def get_axis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (pivot_point, axis_direction) of latest estimate."""
        with self._lock:
            return self._pivot.copy(), self._axis_dir.copy()

    def get_pivot(self) -> np.ndarray:
        """Backward-compatible pivot accessor."""
        with self._lock:
            return self._pivot.copy()

    def stop(self) -> None:
        self._queue.put(None)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            frame_id, depth, rgb_bgr, c2w, ee_poses_dict = item
            try:
                result = self._estimator.step(
                    frame_id, depth, rgb_bgr, c2w,
                    lp_dict={}, ee_poses_dict=ee_poses_dict
                )
                if result is not None:
                    pivot, n_dir, theta, sigma_p, sigma_t = result
                    n_dir = np.array(n_dir, dtype=np.float64)
                    nd = np.linalg.norm(n_dir)
                    if nd > 1e-6:
                        n_dir /= nd
                    with self._lock:
                        self._pivot    = pivot.copy()
                        self._axis_dir = n_dir
                    print(f"[LoFTR] pivot={np.round(pivot[:3], 3)}  "
                          f"dir={np.round(n_dir, 3)}  "
                          f"θ={np.rad2deg(theta):.1f}°  σp={sigma_p * 100:.1f}cm")
            except Exception as exc:
                print(f"[LoFTR Thread] frame {frame_id}: {exc}")


# ─────────────────────────── Action Planner ───────────────────────────────────

class ActionPlanner:
    """
    Stateful planner that:
      1. On first step: runs detection → VLM → builds geometric targets.
      2. Every step: drives gripper toward the current stage target, advances
         stage when tolerance is met.

    Auto-inserted stages (after VLM plan):
      Approach   — inserted between MoveTo and Grasp
      ProbePull  — inserted after Grasp (small probe pull to collect P0/P1)
      TypeCheck  — inserted after ProbePull (decides Translation vs Rotation)

    Stage targets
    -------------
    MoveTo       : pre-grasp position (handle + panel_normal * APPROACH_DIST),  gripper open
    Approach     : handle 3-D position (standoff), gripper open
    Grasp        : handle 3-D position, gripper closing to GRIPPER_CLOSED
    ProbePull    : grasp_pos + panel_normal * PROBE_DISTANCE, gripper closed
    TypeCheck    : hold position, compute motion type from P0/P1 Chamfer matching
    Pull_Linear  : incremental target advancing along panel_normal; gripper closed
    Pull_Arc     : incremental arc around TypeCheck-derived 3-D axis; gripper closed
                   (LoFTR-PF refines the axis online)
    Release      : stay in place, gripper opens
    """

    def __init__(self, user_instruction: str = USER_INSTRUCTION,
                 use_visualizer: bool = USE_VISUALIZER,
                 use_api: bool = True):
        self.user_instruction = user_instruction
        self.use_visualizer = use_visualizer
        self.use_api = use_api
        self.visualizer = Visualizer3D() if use_visualizer else None
        self._reset()

    # ── public ──────────────────────────────────────────────────────────────

    def reset(self):
        self._reset()

    def process(self, obs: Dict) -> np.ndarray:
        """
        Consume one observation dict, return action (10,) float32:
        [target_pos(3), target_rot_6d(6), target_gripper(1)]
        """
        rgb_arr  = deser(obs['rgb'])          # (H,W,3)
        depth    = deser(obs['depth'])        # (H,W) metres
        cam_pos  = deser(obs['cam_pos'])      # (3,)
        cam_mat  = deser(obs['cam_mat'])      # (3,3)
        fovy     = float(deser(obs['fovy'])[0])
        ap       = deser(obs['agent_pos'])    # (1,T,10)
        pc       = deser(obs['point_cloud'])  # (1,T,N,3)

        agent_last  = ap[0, -1, :]            # (10,)
        curr_pos    = agent_last[:3].copy()
        curr_rot    = rot6d_to_matrix(agent_last[3:9])
        curr_grip   = float(agent_last[9])
        pcloud_last = pc[0, -1, :, :]         # (1280,3)

        rgb_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        # Cache current observation for _on_complete access
        self._curr_depth   = depth
        self._curr_rgb_bgr = rgb_bgr
        self._curr_cam_pos = cam_pos
        self._curr_cam_mat = cam_mat
        self._curr_fovy    = fovy

        # Track EE history for LoFTR PF (frame_id → world position)
        self._frame_id += 1
        self._ee_history[self._frame_id] = curr_pos.copy()
        if len(self._ee_history) > 200:
            del self._ee_history[min(self._ee_history)]

        if self.state == "INIT":
            print("[Planner] Entering INIT state, calling backend services...")
            self._initialize(rgb_bgr, depth, cam_pos, cam_mat, fovy,
                             curr_pos, pcloud_last)
            print(f"[Planner] INIT done. Stages: {self.stages}")

        if self.state in ("ERROR", "DONE"):
            return np.concatenate([curr_pos, matrix_to_rot6d(curr_rot), [curr_grip]]).astype(np.float32)

        # ── Feed LoFTR thread and update arc axis during Pull_Arc ─────────
        current_stage = (self.stages[self.stage_idx]
                         if self.stage_idx < len(self.stages) else "")
        if (self.state == "EXECUTING"
                and current_stage == "Pull_Arc"
                and self._loftr_thread is not None):
            R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = cam_mat.astype(np.float32) @ R_flip
            c2w[:3, 3]  = cam_pos.astype(np.float32)
            self._loftr_thread.push(
                self._frame_id, depth.copy(), rgb_bgr.copy(),
                c2w, self._ee_history.copy()
            )
            if self.arc_initialized and self.arc_axis_point is not None:
                new_pivot, new_dir = self._loftr_thread.get_axis()
                self._update_arc_axis(new_pivot, new_dir)

        action = self._step(curr_pos, curr_rot, curr_grip)

        if self.visualizer:
            vis_hinge = (self.arc_axis_point if self.arc_axis_point is not None
                         else self.arc_hinge)
            self.visualizer.update(
                points=pcloud_last, normal_pts=self.normal_pts,
                hinge_pos=vis_hinge, hinge_pts=self.hinge_edge_pts,
                curr_pos=curr_pos, curr_mat=curr_rot
            )

        return action

    # ── private: initialisation ─────────────────────────────────────────────

    def _reset(self):
        self.state      : str            = "INIT"
        self.plan_data  : Optional[Dict] = None
        self.stages     : List[str]      = []
        self.stage_idx  : int            = 0
        self.stage_step_count : int      = 0
        self.detections : List[Dict]     = []

        self.handle_3d         : Optional[np.ndarray] = None
        self.grasp_target_pos  : Optional[np.ndarray] = None
        self.grasp_pos         : Optional[np.ndarray] = None
        self.grasp_quat        : Optional[np.ndarray] = None
        self.post_pull_pos     : Optional[np.ndarray] = None
        self.approach_pos      : Optional[np.ndarray] = None
        self.approach_quat     : Optional[np.ndarray] = None
        self.panel_normal      : Optional[np.ndarray] = None
        self.normal_pts        : Optional[np.ndarray] = None
        self.hinge_edge_pts    : np.ndarray            = np.zeros((0, 3))
        self.parent_mask       : Optional[np.ndarray] = None

        # Pull_Linear state
        self.pull_target : Optional[np.ndarray] = None

        # Legacy arc state (kept for fallback / visualizer)
        self.arc_hinge        : Optional[np.ndarray] = None
        self.arc_radius       : float = 0.3
        self.arc_current_angle: float = 0.0
        self.arc_swept        : float = 0.0
        self.arc_direction    : float = 1.0
        self.arc_initialized  : bool  = False
        self.arc_init_theta   : float = 0.0   # angle already swept by ProbePull
        self.arc_accumulated  : float = 0.0   # angle accumulated since Pull_Arc started
        # 0° reference: gripper pose at START of ProbePull (= end of Grasp)
        self.arc_ref_pos  : Optional[np.ndarray] = None
        self.arc_ref_quat : Optional[np.ndarray] = None

        # Generalised 3-D arc axis (set by TypeCheck)
        self.arc_axis_point   : Optional[np.ndarray] = None  # point on axis
        self.arc_axis_dir     : Optional[np.ndarray] = None  # unit direction
        self.arc_probe_theta  : float = 0.0                  # angle from probe

        # Probe observations (saved at Grasp completion and ProbePull completion)
        self.probe_start_depth        : Optional[np.ndarray]         = None
        self.probe_start_rgb          : Optional[np.ndarray]         = None
        self.probe_start_cam_pose     : Optional[Tuple]              = None
        self.probe_start_gripper_pos  : Optional[np.ndarray]         = None
        self.probe_start_object_pcd   : Optional[np.ndarray]         = None

        self.probe_end_depth          : Optional[np.ndarray]         = None
        self.probe_end_rgb            : Optional[np.ndarray]         = None
        self.probe_end_cam_pose       : Optional[Tuple]              = None
        self.probe_end_gripper_pos    : Optional[np.ndarray]         = None
        self.probe_end_object_pcd     : Optional[np.ndarray]         = None

        # TypeCheck result
        self.typecheck_result : Optional[Dict] = None

        # LoFTR axis estimator thread (Rotation motion only, started post-TypeCheck)
        self._loftr_thread: Optional[LoFTREstimatorThread] = None
        self._frame_id    : int  = 0
        self._ee_history  : Dict[int, np.ndarray] = {}

        # Current observation cache (set in process(), used in _on_complete())
        self._curr_depth   : Optional[np.ndarray] = None
        self._curr_rgb_bgr : Optional[np.ndarray] = None
        self._curr_cam_pos : Optional[np.ndarray] = None
        self._curr_cam_mat : Optional[np.ndarray] = None
        self._curr_fovy    : float                 = 60.0

    def _initialize(self, rgb_bgr, depth, cam_pos, cam_mat, fovy,
                    curr_pos, point_cloud):
        H, W = depth.shape

        # ── 1. Detection ───────────────────────────────────────────────────
        print("[Planner] [Step 1/7] Running detection...")
        self.detections = call_detection(rgb_bgr)
        print(f"[Planner] [Step 1/7] {len(self.detections)} objects detected.")

        if not self.detections:
            print("[Planner] [Step 1/7] No detections — entering ERROR state.")
            self.state = "ERROR"
            return

        annotated = draw_annotated_image(rgb_bgr, self.detections)
        cv2.imwrite("action_server_annotated.jpg", annotated)
        print("[Planner] [Step 2/7] Annotated image saved.")

        # ── 2. VLM ────────────────────────────────────────────────────────
        print("[Planner] [Step 3/7] Calling VLM reasoning...")
        plan = call_vlm(annotated, self.user_instruction, self.detections,
                        use_api=self.use_api)
        print(f"[Planner] [Step 3/7] VLM Plan resolved: {plan}")
        self.plan_data = plan

        # Build stage list with auto-insertions
        stages = list(plan["plan"])
        # Auto-insert Approach between MoveTo and Grasp
        if "MoveTo" in stages:
            m_idx = stages.index("MoveTo")
            if m_idx + 1 < len(stages) and stages[m_idx + 1] == "Grasp":
                print("[Planner] Auto-inserting Approach stage...")
                stages.insert(m_idx + 1, "Approach")
        # Auto-insert ProbePull and TypeCheck after Grasp
        if "Grasp" in stages:
            g_idx = stages.index("Grasp")
            stages.insert(g_idx + 1, "TypeCheck")
            stages.insert(g_idx + 1, "ProbePull")
            print("[Planner] Auto-inserting ProbePull and TypeCheck stages...")
        self.stages = stages

        # ── 3. Resolve handle and parent detections ────────────────────────
        print("[Planner] [Step 4/7] Resolving 3D positions...")
        handle_det = next(
            (d for d in self.detections if d["index"] == plan["target_handle_id"]),
            self.detections[0],
        )

        # ── 4. Handle 3-D position via depth back-projection ───────────────
        bx1, by1, bx2, by2 = handle_det["box"]
        uc, vc = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        d_val = self._sample_valid_depth(depth, vc, uc, H, W)

        if d_val > 0:
            self.handle_3d = backproject_pixel(uc, vc, d_val, cam_pos, cam_mat, fovy, H, W)
        else:
            self.handle_3d = point_cloud.mean(axis=0)

        self.handle_3d[2] += TARGET_Z_OFFSET
        print(f"[Planner] [Step 4/7] handle_3d = {np.round(self.handle_3d, 3)}")

        # ── 5. Panel normal ────────────────────────────────────────────────
        print("[Planner] [Step 5/7] Estimating panel normal...")
        parent_det = next(
            (d for d in self.detections if d["index"] == plan.get("parent_object_id")), None
        )
        parent_mask = (decode_mask(parent_det["mask"], H, W)
                       if (parent_det and "mask" in parent_det) else None)
        self.parent_mask = parent_mask

        self.panel_normal, self.normal_pts = estimate_panel_normal(
            depth, handle_det["box"], self.handle_3d,
            cam_pos, cam_mat, fovy,
            parent_mask=parent_mask
        )
        print(f"[Planner] Final normal = {np.round(self.panel_normal, 3)}")

        # ── 6. Approach orientation ────────────────────────────────────────
        print("[Planner] [Step 6/7] Planning approach trajectories...")
        approach_dir = -self.panel_normal
        up_hint = (np.array([1.0, 0.0, 0.0])
                   if plan.get("gripper_orientation") == "Horizontal"
                   else np.array([0.0, 0.0, 1.0]))
        approach_rot = look_at_rotation(approach_dir, up_hint)
        self.approach_quat    = scipy_to_wxyz(R.from_matrix(approach_rot))
        self.grasp_target_pos = self.handle_3d - GRASP_OFFSET * approach_dir
        self.approach_pos     = self.grasp_target_pos + self.panel_normal * APPROACH_DIST

        # ── 7. Arc fallback parameters (hinged door only) ──────────────────
        if plan["motion_type"] == "Rotation":
            print("[Planner] [Step 7/7] Estimating initial hinge (fallback for TypeCheck Unknown)...")
            if parent_mask is not None:
                self.arc_hinge, self.arc_radius, self.hinge_edge_pts = estimate_hinge_params(
                    depth, parent_mask, handle_det["box"],
                    cam_pos, cam_mat, fovy
                )
            else:
                print("[Planner] [WARN] No parent mask for hinge estimation!")
                self.arc_hinge = np.array([0., 0., 0.], dtype=np.float64)
                self.arc_radius = 0.3

            # NOTE: LoFTR thread is NOT started here.
            # It is deferred to _on_complete("TypeCheck") if TypeCheck → Rotation.
        else:
            print("[Planner] [Step 7/7] Translation plan — no hinge estimation needed.")

        print("[Planner] All INIT steps finished successfully.")
        self.state     = "EXECUTING"
        self.stage_idx = 0

    @staticmethod
    def _sample_valid_depth(depth, vc, uc, H, W, radius=3):
        """Return first non-zero depth near pixel (uc, vc)."""
        for dv, du in [(0,0),(0,2),(0,-2),(2,0),(-2,0),(2,2),(-2,-2)]:
            vv = int(vc + dv);  uu = int(uc + du)
            if 0 <= vv < H and 0 <= uu < W:
                d = float(depth[vv, uu])
                if d > 0:
                    return d
        return 0.0

    def _update_arc_axis(self, new_pivot: np.ndarray, new_dir: np.ndarray) -> None:
        """Apply a LoFTR axis update with exponential smoothing."""
        if self.arc_axis_point is None or self.arc_axis_dir is None:
            return

        # Plausibility checks
        jump = np.linalg.norm(new_pivot - self.arc_axis_point)
        dot = float(np.dot(new_dir, self.arc_axis_dir))

        # if jump > 1.00:
        #     if self._frame_id % 20 == 0:
        #         print(f"[Planner] Axis update rejected: jump={jump:.3f}m > 1.0m")
        #     return
        
        # if abs(dot) < 0.70:
        #     if self._frame_id % 20 == 0:
        #         print(f"[Planner] Axis update rejected: abs(dot)={abs(dot):.3f} < 0.7")
        #     return
        
        if dot < 0:
            new_dir = -new_dir # Handle flipped axis direction

        alpha = ARC_SMOOTH_ALPHA
        self.arc_axis_point = ((1 - alpha) * self.arc_axis_point + alpha * new_pivot)
        blended = (1 - alpha) * self.arc_axis_dir + alpha * new_dir
        n = np.linalg.norm(blended)
        if n > 1e-6:
            self.arc_axis_dir = blended / n

        print(f"[Planner] arc_axis refined: point={np.round(self.arc_axis_point, 3)}, "
              f"dir={np.round(self.arc_axis_dir, 3)}")

    def _run_type_check(self) -> None:
        """Run TypeCheck and store the result in self.typecheck_result."""
        P0 = self.probe_start_object_pcd
        P1 = self.probe_end_object_pcd
        g0 = self.probe_start_gripper_pos
        g1 = self.probe_end_gripper_pos

        if (P0 is None or P1 is None or g0 is None or g1 is None
                or len(P0) == 0 or len(P1) == 0):
            print("[TypeCheck] Missing probe data — falling back to Unknown")
            self.typecheck_result = {
                "motion_type": "Unknown", "confidence": 0.0,
                "E_translation": float("nan"), "E_rotation": float("nan"),
                "best_axis_side": None, "best_axis_point": None,
                "best_axis_dir": None, "theta_axis": 0.0, "axis_fit_error": float("nan"),
            }
            return

        print(f"[TypeCheck] Running — P0={len(P0)} pts, P1={len(P1)} pts, "
              f"Δg={np.round(g1 - g0, 4)}")
        self.typecheck_result = type_check(P0, P1, g0, g1, self.panel_normal)

    # ── private: per-step execution ────────────────────────────────────────

    def _step(self, curr_pos, curr_rot, curr_grip) -> np.ndarray:
        if self.stage_idx >= len(self.stages):
            self.state = "DONE"
            print("[Planner] All stages complete.")
            return np.concatenate([curr_pos, matrix_to_rot6d(curr_rot), [curr_grip]]).astype(np.float32)

        stage = self.stages[self.stage_idx]
        self.stage_step_count += 1
        tgt_pos, tgt_quat, tgt_grip = self._target(stage, curr_pos, curr_rot, curr_grip)

        pos_err  = np.linalg.norm(tgt_pos - curr_pos)
        tgt_rot   = wxyz_to_scipy(tgt_quat)
        cur_rot_s = R.from_matrix(curr_rot)
        ang_err  = np.linalg.norm((tgt_rot * cur_rot_s.inv()).as_rotvec())
        grip_err = abs(tgt_grip - curr_grip)

        if self._is_stage_done(stage, pos_err, ang_err, grip_err, curr_pos):
            print(f"[Planner] Stage '{stage}' done "
                  f"(pos_err={pos_err:.4f}m, grip_err={grip_err:.4f})")
            self._on_complete(stage, curr_pos, curr_rot)
            self.stage_idx += 1
            if self.stage_idx < len(self.stages):
                print(f"[Planner] → '{self.stages[self.stage_idx]}'")

        if self.visualizer:
            self.visualizer.update(tgt_pos=tgt_pos, tgt_mat=tgt_rot.as_matrix())

        return np.concatenate([tgt_pos, matrix_to_rot6d(tgt_rot.as_matrix()), [tgt_grip]]).astype(np.float32)

    # ── stage target computation ───────────────────────────────────────────

    def _target(self, stage: str, curr_pos, curr_rot,
                curr_grip) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (target_pos, target_quat_wxyz, target_gripper_width)."""

        # ── MoveTo ────────────────────────────────────────────────────────
        if stage == "MoveTo":
            return self.approach_pos, self.approach_quat, GRIPPER_OPEN

        # ── Approach ──────────────────────────────────────────────────────
        elif stage == "Approach":
            return self.grasp_target_pos, self.approach_quat, GRIPPER_OPEN

        # ── Grasp ─────────────────────────────────────────────────────────
        elif stage == "Grasp":
            return self.grasp_target_pos, self.approach_quat, GRIPPER_CLOSED

        # ── ProbePull ─────────────────────────────────────────────────────
        elif stage == "ProbePull":
            start = self.grasp_pos if self.grasp_pos is not None else curr_pos
            probe_target = start + self.panel_normal * PROBE_DISTANCE
            q = (self.grasp_quat if self.grasp_quat is not None
                 else self.approach_quat)
            return probe_target, q, GRIPPER_CLOSED

        # ── TypeCheck ─────────────────────────────────────────────────────
        elif stage == "TypeCheck":
            # Trigger computation on first visit
            if self.typecheck_result is None:
                self._run_type_check()
            # Hold current position while computing
            q = (self.grasp_quat if self.grasp_quat is not None
                 else self.approach_quat)
            return curr_pos.copy(), q, GRIPPER_CLOSED

        # ── Pull_Linear ───────────────────────────────────────────────────
        elif stage == "Pull_Linear":
            start = self.grasp_pos if self.grasp_pos is not None else curr_pos
            final = start + self.panel_normal * PULL_LINEAR_DIST

            if self.pull_target is None:
                self.pull_target = start.copy()
            remaining = final - self.pull_target
            d = np.linalg.norm(remaining)
            if d > PULL_STEP:
                self.pull_target = self.pull_target + remaining / d * PULL_STEP
            else:
                self.pull_target = final.copy()

            return (self.pull_target,
                    self.grasp_quat if self.grasp_quat is not None else self.approach_quat,
                    GRIPPER_CLOSED)

        # ── Pull_Arc ──────────────────────────────────────────────────────
        elif stage == "Pull_Arc":
            # Resolve axis: prefer TypeCheck/LoFTR result, fallback to legacy arc_hinge + Z
            if self.arc_axis_point is not None and self.arc_axis_dir is not None:
                axis_point = self.arc_axis_point
                axis_dir   = self.arc_axis_dir
            else:
                axis_point = (self.arc_hinge if self.arc_hinge is not None
                              else np.zeros(3, dtype=np.float64))
                axis_dir   = np.array([0.0, 0.0, 1.0])

            # 0° reference pose: gripper position/orientation at START of ProbePull (= end of Grasp)
            # arc_ref_pos is saved in _on_complete('Grasp') and is NEVER overwritten by ProbePull.
            ref_pos  = self.arc_ref_pos  if self.arc_ref_pos  is not None else curr_pos
            ref_quat = self.arc_ref_quat if self.arc_ref_quat is not None else self.approach_quat

            if not self.arc_initialized:
                # Determine rotation direction once at init
                v = ref_pos - axis_point
                v_perp = v - np.dot(v, axis_dir) * axis_dir
                r = float(np.linalg.norm(v_perp))
                self.arc_radius = r if r > 1e-4 else self.arc_radius

                tangent_pos = np.cross(axis_dir, v_perp)
                tn = np.linalg.norm(tangent_pos)
                if tn > 1e-6:
                    tangent_pos /= tn
                    self.arc_direction = 1.0 if np.dot(tangent_pos, self.panel_normal) > 0 else -1.0
                else:
                    self.arc_direction = 1.0

                # Offset from ProbePull: abs value of angle already swept
                self.arc_init_theta  = abs(self.arc_probe_theta)
                self.arc_accumulated = 0.0
                self.arc_swept       = 0.0
                self.arc_initialized = True
                print(f"[Planner] Arc init: axis_point={np.round(axis_point, 3)}, "
                      f"axis_dir={np.round(axis_dir, 3)}, "
                      f"dir={self.arc_direction}, radius={self.arc_radius:.3f}m, "
                      f"init_offset={np.rad2deg(self.arc_init_theta):.1f}°")

            # Time-step driven: increment by ARC_STEP each call, independent of curr_pos
            self.arc_accumulated += ARC_STEP
            self.arc_swept        = self.arc_accumulated  # for _is_stage_done

            # Target angle from arc_ref_pos (0° reference = start of ProbePull)
            target_theta = self.arc_direction * (self.arc_init_theta + self.arc_accumulated)

            if self.stage_step_count % 10 == 0:
                print(f"[Planner] Pull_Arc step {self.stage_step_count}: "
                      f"accumulated={np.rad2deg(self.arc_accumulated):.1f}°, "
                      f"target_theta={np.rad2deg(target_theta):.1f}°")

            # ── Adaptive radius: use curr_pos → latest axis distance ────────
            # When LoFTR updates axis_point, the perpendicular distance from
            # arc_ref_pos to the new axis may differ from the actual gripper
            # distance. We always use curr_pos's actual distance to the latest
            # axis so the target stays on the physically correct arc.
            v_curr      = curr_pos - axis_point
            curr_perp   = v_curr - np.dot(v_curr, axis_dir) * axis_dir
            curr_r      = float(np.linalg.norm(curr_perp))
            if curr_r > 1e-3:
                self.arc_radius = curr_r   # track actual radius

            # 0° reference direction in the axis-perpendicular plane
            v_ref     = ref_pos - axis_point
            ref_perp  = v_ref - np.dot(v_ref, axis_dir) * axis_dir
            n_ref     = float(np.linalg.norm(ref_perp))
            ref_dir   = ref_perp / n_ref if n_ref > 1e-4 else np.array([1., 0., 0.])

            # Rotate the reference direction by target_theta around axis
            target_dir = R.from_rotvec(axis_dir * target_theta).apply(ref_dir)

            # Target position: axis_point + along-axis offset from ref + adaptive radius
            along_ref = float(np.dot(v_ref, axis_dir))   # height along axis (from ref)
            arc_pos   = axis_point + along_ref * axis_dir + self.arc_radius * target_dir

            # Orientation: rotate arc_ref_quat by target_theta (fully absolute, zero drift)
            r_ref    = wxyz_to_scipy(ref_quat)
            r_offset = R.from_rotvec(axis_dir * target_theta)
            arc_quat = scipy_to_wxyz(r_offset * r_ref)

            return arc_pos, arc_quat, GRIPPER_CLOSED

        # ── Release ───────────────────────────────────────────────────────
        elif stage == "Release":
            stay = (self.post_pull_pos if self.post_pull_pos is not None
                    else curr_pos).copy()
            q = (self.grasp_quat if self.grasp_quat is not None
                 else self.approach_quat)
            return stay, q, GRIPPER_OPEN

        else:
            print(f"[Planner] Unknown stage '{stage}' — holding position.")
            return (curr_pos.copy(),
                    scipy_to_wxyz(R.from_matrix(curr_rot)),
                    curr_grip)

    def _is_stage_done(self, stage: str,
                        pos_err: float, ang_err: float, grip_err: float,
                        curr_pos: np.ndarray) -> bool:
        if stage in ("MoveTo", "Approach"):
            return (pos_err < STAGE_POS_TOL) and (ang_err < STAGE_ANG_TOL)

        elif stage == "Grasp":
            return (grip_err < STAGE_GRIP_TOL) or (self.stage_step_count > GRASP_TIMEOUT_STEPS)

        elif stage == "ProbePull":
            if self.grasp_pos is None:
                return False
            pulled = np.linalg.norm(curr_pos - self.grasp_pos)
            return pulled >= PROBE_DISTANCE - 0.005

        elif stage == "TypeCheck":
            # Completes as soon as the result is available (computed in _target)
            return self.typecheck_result is not None

        elif stage == "Pull_Linear":
            if self.grasp_pos is None:
                return False
            pulled = np.linalg.norm(curr_pos - self.grasp_pos)
            return pulled >= PULL_LINEAR_DIST - STAGE_POS_TOL

        elif stage == "Pull_Arc":
            return PULL_ARC_ANGLE - self.arc_swept < np.deg2rad(1.5)

        elif stage == "Release":
            return (grip_err < STAGE_GRIP_TOL) or (self.stage_step_count > GRASP_TIMEOUT_STEPS)

        return False

    def _on_complete(self, stage: str, curr_pos: np.ndarray, curr_rot: np.ndarray):
        """Save state that later stages depend on."""
        self.stage_step_count = 0

        if stage == "Grasp":
            self.grasp_pos  = curr_pos.copy()
            self.grasp_quat = scipy_to_wxyz(R.from_matrix(curr_rot))
            # Save arc 0° reference: this position/orientation is BEFORE ProbePull moves things
            self.arc_ref_pos  = curr_pos.copy()
            self.arc_ref_quat = scipy_to_wxyz(R.from_matrix(curr_rot))
            print(f"[Planner] Grasp recorded at {np.round(self.grasp_pos, 3)}")

            # Save probe start observations
            self.probe_start_gripper_pos = curr_pos.copy()
            if self._curr_depth is not None:
                self.probe_start_depth    = self._curr_depth.copy()
                self.probe_start_rgb      = self._curr_rgb_bgr.copy()
                self.probe_start_cam_pose = (self._curr_cam_pos.copy(),
                                             self._curr_cam_mat.copy(),
                                             self._curr_fovy)
                if self.parent_mask is not None:
                    self.probe_start_object_pcd = backproject_mask_to_pcd(
                        self.probe_start_depth, self.parent_mask,
                        self._curr_cam_pos, self._curr_cam_mat, self._curr_fovy
                    )
                    print(f"[Planner] Probe start PCD: {len(self.probe_start_object_pcd)} pts")

        elif stage == "ProbePull":
            # Save probe end observations
            self.probe_end_gripper_pos = curr_pos.copy()
            if self._curr_depth is not None:
                self.probe_end_depth    = self._curr_depth.copy()
                self.probe_end_rgb      = self._curr_rgb_bgr.copy()
                self.probe_end_cam_pose = (self._curr_cam_pos.copy(),
                                           self._curr_cam_mat.copy(),
                                           self._curr_fovy)
                if self.parent_mask is not None:
                    # Method A: reuse probe_start parent_mask for speed
                    self.probe_end_object_pcd = backproject_mask_to_pcd(
                        self.probe_end_depth, self.parent_mask,
                        self._curr_cam_pos, self._curr_cam_mat, self._curr_fovy
                    )
                    print(f"[Planner] Probe end PCD: {len(self.probe_end_object_pcd)} pts")

            # Update grasp reference to current position so Pull_* starts from here
            self.grasp_pos  = curr_pos.copy()
            self.grasp_quat = scipy_to_wxyz(R.from_matrix(curr_rot))
            print(f"[Planner] ProbePull done. Updated grasp_pos={np.round(self.grasp_pos, 3)}")

        elif stage == "TypeCheck":
            result = self.typecheck_result
            motion_type = result.get("motion_type", "Unknown")
            E_T = result.get("E_translation", float("nan"))
            E_R = result.get("E_rotation",    float("nan"))
            side = result.get("best_axis_side", "N/A")
            print(f"[Planner] TypeCheck Result: Type={motion_type}, Direction/Side={side}")
            print(f"[Planner] TypeCheck Scores: E_T={E_T:.5f}, E_R={E_R:.5f}")

            # Modify remaining stages based on TypeCheck result
            remaining = list(self.stages[self.stage_idx + 1:])
            new_remaining = []
            for s in remaining:
                if motion_type == "Translation" and s == "Pull_Arc":
                    new_remaining.append("Pull_Linear")
                elif motion_type == "Rotation" and s == "Pull_Linear":
                    new_remaining.append("Pull_Arc")
                else:
                    new_remaining.append(s)
            self.stages[self.stage_idx + 1:] = new_remaining

            if motion_type == "Translation":
                # Clear any arc state
                self.arc_axis_point = None
                self.arc_axis_dir   = None
                print(f"[Planner] TypeCheck → Translation. Stages: {self.stages}")

            elif motion_type == "Rotation":
                # Set arc axis from TypeCheck result
                self.arc_axis_point  = result.get("best_axis_point")
                self.arc_axis_dir    = result.get("best_axis_dir")
                self.arc_probe_theta = result.get("theta_axis", 0.0)
                self.arc_direction   = (1.0 if self.arc_probe_theta >= 0 else -1.0)

                # Correct grasp_quat: the gripper should be perpendicular to the door.
                # Instead of the misaligned curr_rot from ProbePull end, we rotate the 
                # initial perfect approach_quat by the probe rotation angle.
                r_init = wxyz_to_scipy(self.approach_quat)
                r_corrected = R.from_rotvec(self.arc_axis_dir * self.arc_probe_theta) * r_init
                self.grasp_quat = scipy_to_wxyz(r_corrected)
                print(f"[Planner] Grasp orientation corrected by {np.rad2deg(self.arc_probe_theta):.1f}°")

                # Start LoFTR-PF thread with TypeCheck axis as initial estimate
                if (self._loftr_thread is None
                        and self.arc_axis_point is not None
                        and self.arc_axis_dir is not None
                        and self._curr_depth is not None):
                    H, W = self._curr_depth.shape
                    f_cam = (H / 2.0) / np.tan(np.deg2rad(self._curr_fovy) / 2.0)
                    K = np.array([[f_cam, 0, W / 2.0],
                                  [0, f_cam, H / 2.0],
                                  [0,     0,     1.0]], dtype=np.float32)
                    print("[Planner] Starting LoFTR-PF thread with TypeCheck axis...")
                    self._loftr_thread = LoFTREstimatorThread(
                        K, self.arc_axis_dir, self.arc_axis_point
                    )

                print(f"[Planner] TypeCheck → Rotation. "
                      f"side={result.get('best_axis_side')}, "
                      f"θ={np.rad2deg(self.arc_probe_theta):.1f}°, "
                      f"Stages: {self.stages}")

            else:  # Unknown — keep VLM initial plan
                print("[Planner] TypeCheck → Unknown. Keeping VLM initial plan.")
                # If VLM plan included Pull_Arc and we have a hinge estimate, use it
                if "Pull_Arc" in self.stages[self.stage_idx + 1:]:
                    if self.arc_hinge is not None:
                        self.arc_axis_point = self.arc_hinge.copy()
                        self.arc_axis_dir   = np.array([0.0, 0.0, 1.0])
                        print("[Planner] Fallback: using initial hinge + Z axis for Pull_Arc")

        elif stage in ("Pull_Linear", "Pull_Arc"):
            self.post_pull_pos = curr_pos.copy()
            self.pull_target = None
            if stage == "Pull_Arc" and self._loftr_thread is not None:
                self._loftr_thread.stop()
                self._loftr_thread = None

# ─────────────────────────── ZMQ server main loop ─────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='VLM Action Server')
    parser.add_argument('--use_api', type=lambda x: x.lower() != 'false',
                        default=True, metavar='BOOL',
                        help='Use real VLM API (default: true). Pass false to use hardcoded reply.')
    args = parser.parse_args()

    context = zmq.Context()
    socket  = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"[Server] VLM Action Server listening on tcp://*:{ZMQ_PORT}")
    print(f"[Server] Detection endpoint: {DET_SERVER_URL}")
    print(f"[Server] User instruction: '{USER_INSTRUCTION}'")
    print(f"[Server] use_api={args.use_api}")

    planner = ActionPlanner(user_instruction=USER_INSTRUCTION, use_api=args.use_api)

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    while True:
        socks = dict(poller.poll(timeout=10))
        if socket in socks and socks[socket] == zmq.POLLIN:
            raw = socket.recv()
            try:
                obs = pickle.loads(zlib.decompress(raw))
                missing = [k for k in ['rgb', 'depth', 'agent_pos', 'point_cloud'] if k not in obs]
                if missing:
                    print(f"[Server] Error: Missing keys in obs: {missing}")
                    socket.send(b"ERROR")
                    continue
            except Exception as e:
                print(f"[Server] Deserialise error: {e}")
                socket.send(b"ERROR")
                continue

            try:
                action = planner.process(obs)
            except Exception:
                traceback.print_exc()
                action = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32)

            reply = {
                'shape': action.shape,
                'dtype': str(action.dtype),
                'data' : action.tobytes(),
            }
            socket.send(zlib.compress(pickle.dumps(reply, protocol=pickle.HIGHEST_PROTOCOL)))
        else:
            if planner.visualizer:
                planner.visualizer.update()


if __name__ == "__main__":
    main()
