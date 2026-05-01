#!/usr/bin/env python3
"""
action_server.py — VLM-based Action Server for Articulated Object Manipulation

Architecture
------------
  client_mujoco.py  <--ZMQ REQ-REP-->  action_server.py
                                              |
                                    det_pipeline (HTTP :8000)
                                              |
                                       VLM API (TODO)

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
APPROACH_DIST   = 0.20   # m — standoff before grasping
STAGE_POS_TOL   = 0.032  # m — position tolerance to mark stage done
STAGE_ANG_TOL   = 0.08   # rad
STAGE_GRIP_TOL  = 0.004  # m gripper width
GRASP_TIMEOUT_STEPS = 15 # steps (approx 1.5s) to wait for gripper

MAX_DELTA_POS   = 0.018  # m per step (unused by server, handled by client)
MAX_DELTA_ANG   = np.deg2rad(2.5)
GRIPPER_DELTA   = 0.003  # m per step
GRIPPER_OPEN    = 0.04
GRIPPER_CLOSED  = 0.000
GRASP_OFFSET    = 0.085  # m — distance from Panda 'hand' frame to finger tips
TARGET_Z_OFFSET = 0.01   # m — upward shift of the target grasp point
NORMAL_SCAN_RADIUS = 0.15 # m — radius around handle to sample panel points

PULL_LINEAR_DIST = 0.38  # m total drawer pull
PULL_STEP        = 0.005  # m per step increment
PULL_ARC_ANGLE   = np.deg2rad(90)  # total door sweep
ARC_STEP         = np.deg2rad(0.6)  # rad per step

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
    
    # 1. Determine 2D scan range. If we have a mask, use its bounds. 
    # Otherwise fallback to a large window around handle.
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
        
        # A. Mask filter (if provided)
        if parent_mask is not None and not parent_mask[v, u]:
            continue
            
        # B. Handle exclusion filter (with 3px margin)
        if (hx1 - 3 <= u <= hx2 + 3) and (hy1 - 3 <= v <= hy2 + 3):
            continue
            
        # C. Back-project and check 3D radius
        # (Internal back-projection for speed inside loop)
        x_c = (u - w/2.0) * z / f
        y_c = -(v - h/2.0) * z / f
        p_world = cam_pos + cam_mat @ np.array([x_c, y_c, -z])
        
        if np.linalg.norm(p_world - handle_3d) < scan_radius:
            final_pts.append(p_world)

    if len(final_pts) < 10:
        print(f"[Planner] [ERROR] normalize estimation failed: only {len(final_pts)} points found. Using fallback.")
        return -cam_mat[:, 2], np.array(final_pts) if final_pts else np.zeros((0, 3))

    pts_world = np.stack(final_pts)
    ctr = pts_world.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts_world - ctr)
    normal = Vt[-1]

    # Force horizontal (z=0)
    normal[2] = 0
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # Normal must point toward camera
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
    
    # Identify hinge side: opposite to handle
    mask_width = u_max - u_min
    if hu > (u_min + u_max) / 2.0:
        # Handle is on the right -> hinge is Left
        edge_u = u_min
        edge_pixels = (parent_mask[:, u_min : u_min + max(1, int(mask_width * 0.01))])
        edge_v_local, edge_u_local = np.where(edge_pixels)
        target_u = edge_u_local + u_min
        target_v = edge_v_local
    else:
        # Handle is on the left -> hinge is Right
        edge_u = u_max
        edge_pixels = (parent_mask[:, u_max - max(1, int(mask_width * 0.05)) : u_max + 1])
        edge_v_local, edge_u_local = np.where(edge_pixels)
        target_u = edge_u_local + (u_max - max(1, int(mask_width * 0.05)))
        target_v = edge_v_local

    if len(target_u) == 0:
        return np.array([0,0,0]), 0.3

    f = (h / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    
    # Sample subset for speed
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
    
    # Outlier removal: filter by Z-height consistency
    z_med = np.median(pts[:, 2])
    z_std = np.std(pts[:, 2])
    valid_mask = np.abs(pts[:, 2] - z_med) < max(0.1, z_std * 2.0)
    pts = pts[valid_mask]
    
    if len(pts) == 0:
        return np.array([0,0,0]), 0.3

    # The hinge axis is the average XY of the edge points
    hinge_xy = pts[:, :2].mean(axis=0)
    hinge_pos = np.array([hinge_xy[0], hinge_xy[1], z_med])
    
    # Sample handle 3D again to ensure radius is consistent
    h_uc, h_vc = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
    h_z = depth[int(h_vc), int(h_uc)]
    if h_z <= 0: # handle center might be a hole, sample nearby
        h_z = z_med 
    h_xc = (h_uc - w/2.0) * h_z / f
    h_yc = -(h_vc - h/2.0) * h_z / f
    handle_3d_proj = cam_pos + cam_mat @ np.array([h_xc, h_yc, -h_z])
    
    radius = np.linalg.norm(handle_3d_proj[:2] - hinge_xy)
    print(f"[Planner] Hinge optimized: Side={'Left' if hu > (u_min+u_max)/2.0 else 'Right'}, Pos={np.round(hinge_pos, 3)}, Radius={radius:.3f}m")
    
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
    
    # 高对比度色板 (BGR 格式)
    PALETTE = [
        (0, 0, 255),      # 红
        (0, 255, 0),      # 绿
        (255, 0, 0),      # 蓝
        (0, 255, 255),    # 黄
        (255, 0, 255),    # 品红
        (255, 255, 0),    # 青
        (0, 165, 255),    # 橘
        (203, 192, 255),  # 粉
        (128, 0, 128),    # 紫
        (128, 128, 0),    # 深青
        (0, 128, 0),      # 暗绿
        (0, 0, 128),      # 深红
    ]

    legend_items = []
    
    for det in detections:
        idx = det.get("index", 0)
        color = PALETTE[int(idx) % len(PALETTE)]
        
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = det["detection"][0]["label"]
        score = det["detection"][0]["score"]
        
        # 1. 绘制极简边界框
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        
        # 2. 在框左上角画一个小 ID 标识 (背景色块 + 负色文字)
        id_text = f"{idx}"
        (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (x1, y1), (x1 + tw + 4, y1 + th + 4), color, -1)
        
        # 使用负色计算文字颜色以确保高对比度
        text_color = (255 - color[0], 255 - color[1], 255 - color[2])
        cv2.putText(out, id_text, (x1 + 2, y1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
        
        legend_items.append({
            "id": idx,
            "label": label,
            "score": score,
            "color": color
        })

    # 3. 绘制角落图例 (Legend)
    if legend_items:
        margin = 10
        item_h = 24
        header_h = 25
        leg_w  = 200
        leg_h  = header_h + len(legend_items) * item_h + margin
        
        # 图例底色 (半透明黑)
        leg_x1 = W - leg_w - margin
        leg_y1 = margin
        overlay = out.copy()
        cv2.rectangle(overlay, (leg_x1, leg_y1), (leg_x1 + leg_w, leg_y1 + leg_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
        
        # 标题
        cv2.putText(out, "LEGEND", (leg_x1 + 5, leg_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 列表条目
        for i, item in enumerate(legend_items):
            iy = leg_y1 + header_h + i * item_h + 15
            # 画一个颜色小方块
            cv2.rectangle(out, (leg_x1 + 8, iy - 10), (leg_x1 + 20, iy + 2), item["color"], -1)
            # 画文字
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
        
        # Initial view
        self.ax.view_init(elev=25, azim=-135)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        
        # Fixed limits for stability
        self.ax.set_xlim(0.5, 1.5)
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_zlim(0.0, 1.2)
        
        # KEY FIX: Force equal aspect ratio so axes don't look skewed/slanted
        self.ax.set_box_aspect((1, 1, 1)) 

        # Plot components
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
            # Update point cloud
            self.points_scat._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            # Color by height
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
            # Draw a vertical line from floor to ceiling
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
        
        # Standard matplotlib event processing
        plt.pause(0.01)

    def _draw_axes(self, pos, mat, scale=0.1, alpha=1.0):
        """Draw RGB vectors for orientation."""
        qs = []
        colors = ['r', 'g', 'b'] # X, Y, Z
        for i in range(3):
            q = self.ax.quiver(pos[0], pos[1], pos[2], 
                               mat[0, i], mat[1, i], mat[2, i],
                               length=scale, color=colors[i], alpha=alpha, 
                               linewidth=2, normalize=True)
            qs.append(q)
        return qs

# ─────────────────────────── VLM API (TODO stub) ──────────────────────────────

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
                # Strip optional markdown code fences
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                plan = json.loads(raw_text)
                print(f"[VLM] API response: {plan}")
                return plan
            except Exception as e:
                print(f"[VLM] API call failed: {e} — falling back to hardcoded plan.")

    # ── Hardcoded fallback ────────────────────────────────────────────────────
    print("[VLM] Using hardcoded default plan.")
    return {'target_handle_id': 8, 'parent_object_id': 7, 'motion_type': 'Translation', 'gripper_orientation': 'Vertical', 'plan': ['MoveTo', 'Grasp', 'Pull_Linear', 'Release']}


# ─────────────────────────── LoFTR Axis Estimator Thread ─────────────────────

class LoFTREstimatorThread:
    """
    Background thread that continuously refines the rotation axis using LoFTR-PF.

    Frames are pushed via push(); the latest pivot estimate is readable via
    get_pivot() at any time.  Drops frames when the particle filter is slower
    than the control loop so the queue never grows unbounded.
    """

    def __init__(self, K: np.ndarray, omega0: np.ndarray, p0: np.ndarray):
        self._estimator = LoFTRAxisEstimator(K, omega0, p0, theta0=0.0, visualize=False)
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._pivot = np.array(p0, dtype=np.float64).copy()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, frame_id: int, depth: np.ndarray, rgb_bgr: np.ndarray,
             c2w: np.ndarray, ee_poses_dict: Dict[int, np.ndarray]) -> None:
        try:
            self._queue.put_nowait((frame_id, depth, rgb_bgr, c2w, ee_poses_dict))
        except queue.Full:
            pass  # drop frame when thread is still processing the previous one

    def get_pivot(self) -> np.ndarray:
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
                    with self._lock:
                        self._pivot = pivot.copy()
                    print(f"[LoFTR] pivot={np.round(pivot[:2], 3)}  "
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

    Stage targets
    -------------
    MoveTo       : pre-grasp position (handle + panel_normal * APPROACH_DIST)
                   gripper open
    Grasp        : handle 3-D position, gripper closing to GRIPPER_CLOSED
    Pull_Linear  : incremental target advancing along panel_normal by PULL_STEP
                   per step until PULL_LINEAR_DIST is reached; gripper closed
    Pull_Arc     : incremental arc around estimated hinge; gripper closed
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
        [delta_pos(3), delta_rot_6d(6), gripper_cmd(1)]
        """
        rgb_arr  = deser(obs['rgb'])          # (H,W,3)
        depth    = deser(obs['depth'])        # (H,W) metres
        cam_pos  = deser(obs['cam_pos'])      # (3,)
        cam_mat  = deser(obs['cam_mat'])      # (3,3)
        fovy     = float(deser(obs['fovy'])[0])
        ap       = deser(obs['agent_pos'])    # (1,T,10)
        pc       = deser(obs['point_cloud'])  # (1,T,N,3)
        
        # print(f"[Server] Recv obs: RGB={rgb_arr.shape}, PC={pc.shape}, AP={ap.shape}")

        agent_last  = ap[0, -1, :]            # (10,)
        curr_pos    = agent_last[:3].copy()
        curr_rot    = rot6d_to_matrix(agent_last[3:9])
        curr_grip   = float(agent_last[9])
        pcloud_last = pc[0, -1, :, :]         # (1280,3)

        rgb_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        # Track EE history for LoFTR PF (frame_id → world position)
        self._frame_id += 1
        self._ee_history[self._frame_id] = curr_pos.copy()
        if len(self._ee_history) > 200:
            del self._ee_history[min(self._ee_history)]

        if self.state == "INIT":
            print("[Planner] Entering INIT state, calling backend services...")
            self._initialize(rgb_bgr, depth, cam_pos, cam_mat, fovy,
                             curr_pos, pcloud_last)
            print(f"[Planner] INIT done. State: {self.state}")

        if self.state in ("ERROR", "DONE"):
            # Return current pose as target to stop: [curr_pos, curr_rot6d, curr_grip]
            return np.concatenate([curr_pos, matrix_to_rot6d(curr_rot), [curr_grip]]).astype(np.float32)

        # ── Feed LoFTR thread and update arc hinge during Pull_Arc ────────
        if (self.state == "EXECUTING"
                and self.stage_idx < len(self.stages)
                and self.stages[self.stage_idx] == "Pull_Arc"
                and self._loftr_thread is not None):
            # MuJoCo: camera looks along -Z, Y up  →  CV: looks along +Z, Y down
            R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = cam_mat.astype(np.float32) @ R_flip
            c2w[:3, 3]  = cam_pos.astype(np.float32)
            self._loftr_thread.push(
                self._frame_id, depth.copy(), rgb_bgr.copy(),
                c2w, self._ee_history.copy()
            )
            if self.arc_initialized:
                self._update_arc_hinge(self._loftr_thread.get_pivot(), curr_pos)

        action = self._step(curr_pos, curr_rot, curr_grip)
        
        # Update visualization
        if self.visualizer:
            # We need target pos/mat for visualization. 
            # Note: self._step actually computes these internally. 
            # Let's peek at them if we stored them, or just rely on _step's side effects
            # Actually, I'll pass the latest target info to visualizer in _step or here.
            self.visualizer.update(points=pcloud_last, normal_pts=self.normal_pts,
                                   hinge_pos=self.arc_hinge, hinge_pts=self.hinge_edge_pts,
                                   curr_pos=curr_pos, curr_mat=curr_rot)
            
        return action

    # ── private: initialisation ─────────────────────────────────────────────

    def _reset(self):
        self.state      : str            = "INIT"
        self.plan_data  : Optional[Dict] = None
        self.stages     : List[str]      = []
        self.stage_idx  : int            = 0
        self.stage_step_count : int      = 0
        self.detections : List[Dict]     = []

        self.handle_3d    : Optional[np.ndarray] = None
        self.grasp_target_pos : Optional[np.ndarray] = None # 'hand' pose to reach handle
        self.grasp_pos    : Optional[np.ndarray] = None
        self.grasp_quat   : Optional[np.ndarray] = None
        self.post_pull_pos: Optional[np.ndarray] = None
        self.approach_pos : Optional[np.ndarray] = None
        self.approach_quat: Optional[np.ndarray] = None  # wxyz
        self.panel_normal : Optional[np.ndarray] = None
        self.normal_pts   : Optional[np.ndarray] = None
        self.hinge_edge_pts : np.ndarray         = np.zeros((0, 3))

        # Pull_Linear state
        self.pull_target : Optional[np.ndarray] = None

        # Pull_Arc state
        self.arc_hinge        : Optional[np.ndarray] = None
        self.arc_radius       : float = 0.3
        self.arc_current_angle: float = 0.0
        self.arc_swept        : float = 0.0   # accumulated sweep (0 → PULL_ARC_ANGLE)
        self.arc_direction    : float = 1.0   # +1 for ccw, -1 for cw
        self.arc_initialized  : bool  = False

        # LoFTR axis estimator thread (Rotation motion only)
        self._loftr_thread: Optional[LoFTREstimatorThread] = None
        self._frame_id    : int  = 0
        self._ee_history  : Dict[int, np.ndarray] = {}

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
        print(f"[Planner] [Step 3/7] VLM Plan resolved: {plan.get('target_handle_id')}")
        self.plan_data = plan
        
        # 自动补全策略：如果在 MoveTo 之后直接接 Grasp，自动插入 Approach 阶段
        stages = list(plan["plan"])
        if "MoveTo" in stages:
            m_idx = stages.index("MoveTo")
            # 如果 MoveTo 后面是 Grasp，或者 MoveTo 是最后一项且后面该接 Grasp
            if m_idx + 1 < len(stages) and stages[m_idx + 1] == "Grasp":
                print("[Planner] 检测到缺失 Approach 阶段，自动插入...")
                stages.insert(m_idx + 1, "Approach")
        
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
        
        # Apply upward offset
        self.handle_3d[2] += TARGET_Z_OFFSET
        
        print(f"[Planner] [Step 4/7] handle_3d resolved: {np.round(self.handle_3d, 3)} (with +{TARGET_Z_OFFSET}m Z-offset)")

        # ── 5. Panel normal ────────────────────────────────────────────────
        print("[Planner] [Step 5/7] Estimating panel normal...")
        # Resolve parent object mask for precise filtering
        parent_det = next((d for d in self.detections if d["index"] == plan.get("parent_object_id")), None)
        parent_mask = decode_mask(parent_det["mask"], H, W) if (parent_det and "mask" in parent_det) else None
        
        self.panel_normal, self.normal_pts = estimate_panel_normal(
            depth, handle_det["box"], self.handle_3d, 
            cam_pos, cam_mat, fovy, 
            parent_mask=parent_mask
        )
        print(f"[Planner] Final normal = {np.round(self.panel_normal, 3)}")
        
        # ── 6. Approach orientation ────────────────────────────────────────
        print("[Planner] [Step 6/7] Planning approach trajectories...")
        approach_dir = -self.panel_normal
        up_hint = (np.array([1.0, 0.0, 0.0]) if plan.get("gripper_orientation") == "Horizontal" else np.array([0.0, 0.0, 1.0]))
        approach_rot = look_at_rotation(approach_dir, up_hint)
        self.approach_quat = scipy_to_wxyz(R.from_matrix(approach_rot))
        self.grasp_target_pos = self.handle_3d - GRASP_OFFSET * approach_dir
        self.approach_pos = self.grasp_target_pos + self.panel_normal * APPROACH_DIST

        # ── 7. Arc parameters (hinged door only) ──────────────────────────
        if plan["motion_type"] == "Rotation":
            print("[Planner] [Step 7/7] Estimating hinge for Rotation...")
            if parent_mask is not None:
                self.arc_hinge, self.arc_radius, self.hinge_edge_pts = estimate_hinge_params(
                    depth, parent_mask, handle_det["box"],
                    cam_pos, cam_mat, fovy
                )
            else:
                print("[Planner] [WARN] No parent mask for hinge estimation!")
                self.arc_hinge, self.arc_radius = np.array([0,0,0], dtype=np.float64), 0.3

            # Build K from MuJoCo camera params and start LoFTR thread
            H, W = depth.shape
            f_cam = (H / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
            K = np.array([[f_cam, 0, W / 2.0],
                          [0, f_cam, H / 2.0],
                          [0,     0,      1.0]], dtype=np.float32)
            omega0 = np.array([0.0, 0.0, 1.0])  # vertical door hinge
            print("[Planner] [Step 7/7] Starting LoFTR axis estimator thread...")
            self._loftr_thread = LoFTREstimatorThread(K, omega0, self.arc_hinge)

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

    def _update_arc_hinge(self, new_pivot: np.ndarray, curr_pos: np.ndarray) -> None:
        """Apply a LoFTR pivot update: refresh arc_hinge XY, radius, current_angle.

        Only XY is updated — with omega=[0,0,1] the PF pivot always has Z≈0,
        while the hinge Z from initial edge detection is correct.
        """
        if np.linalg.norm(new_pivot[:2] - self.arc_hinge[:2]) < 0.005:
            return  # < 5 mm change — not worth adjusting
        diff  = curr_pos[:2] - new_pivot[:2]
        new_r = float(np.linalg.norm(diff))
        if not (0.05 < new_r < 1.0):
            return  # outside plausible door-radius range
        print(f"[Planner] arc_hinge XY {np.round(self.arc_hinge[:2], 3)} "
              f"→ {np.round(new_pivot[:2], 3)}  r={new_r:.3f}m")
        self.arc_hinge[:2]     = new_pivot[:2]
        self.arc_radius        = new_r
        self.arc_current_angle = np.arctan2(diff[1], diff[0])

    # ── private: per-step execution ────────────────────────────────────────

    def _step(self, curr_pos, curr_rot, curr_grip) -> np.ndarray:
        if self.stage_idx >= len(self.stages):
            self.state = "DONE"
            print("[Planner] All stages complete.")
            # Return current pose as target to stop
            d_rot_6d = matrix_to_rot6d(curr_rot)
            return np.concatenate([curr_pos, d_rot_6d, [curr_grip]]).astype(np.float32)

        stage = self.stages[self.stage_idx]
        self.stage_step_count += 1
        tgt_pos, tgt_quat, tgt_grip = self._target(stage, curr_pos, curr_rot, curr_grip)

        # ── completion check ─────────────────────────────────────────
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

        # Update visualization with target pose
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

        # ── Approach [NEW] ────────────────────────────────────────────────
        if stage == "Approach":
            return self.grasp_target_pos, self.approach_quat, GRIPPER_OPEN

        # ── Grasp ─────────────────────────────────────────────────────────
        elif stage == "Grasp":
            return self.grasp_target_pos, self.approach_quat, GRIPPER_CLOSED

        # ── Pull_Linear ───────────────────────────────────────────────────
        elif stage == "Pull_Linear":
            start = self.grasp_pos if self.grasp_pos is not None else curr_pos
            final = start + self.panel_normal * PULL_LINEAR_DIST

            # Initialise or advance incremental target
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
            if not self.arc_initialized:
                start = self.grasp_pos if self.grasp_pos is not None else curr_pos
                diff  = start[:2] - self.arc_hinge[:2]
                self.arc_current_angle = np.arctan2(diff[1], diff[0])
                self.arc_radius = float(np.linalg.norm(diff)) or self.arc_radius
                self.arc_swept  = 0.0

                radius_vec  = diff
                tangent_ccw = np.array([-radius_vec[1], radius_vec[0]])
                dot = np.dot(tangent_ccw, self.panel_normal[:2])
                self.arc_direction = 1.0 if dot > 0 else -1.0
                self.arc_initialized = True

                print(f"[Planner] Arc init: angle={np.rad2deg(self.arc_current_angle):.1f}°, "
                      f"dir={self.arc_direction}, radius={self.arc_radius:.3f}m")

            # Advance arc by one step
            step = min(ARC_STEP, PULL_ARC_ANGLE - self.arc_swept)
            self.arc_current_angle += self.arc_direction * step
            self.arc_swept += step

            handle_z = (self.grasp_pos[2] if self.grasp_pos is not None
                        else curr_pos[2])
            arc_pos = np.array([
                self.arc_hinge[0] + self.arc_radius * np.cos(self.arc_current_angle),
                self.arc_hinge[1] + self.arc_radius * np.sin(self.arc_current_angle),
                handle_z,
            ])

            # Rotate gripper by the total swept angle around the global Z axis
            delta_theta = self.arc_direction * self.arc_swept
            r_grasp  = wxyz_to_scipy(self.grasp_quat)
            r_offset = R.from_rotvec([0, 0, delta_theta])
            arc_quat = scipy_to_wxyz(r_offset * r_grasp)

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
            # Done when closed OR timeout (due to object collision)
            return (grip_err < STAGE_GRIP_TOL) or (self.stage_step_count > GRASP_TIMEOUT_STEPS)

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
            print(f"[Planner] Grasp recorded at {np.round(self.grasp_pos, 3)}")
        elif stage in ("Pull_Linear", "Pull_Arc"):
            self.post_pull_pos = curr_pos.copy()
            self.pull_target = None   # reset for potential reuse
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
        # Poll with timeout to keep visualizer responsive
        socks = dict(poller.poll(timeout=10))
        if socket in socks and socks[socket] == zmq.POLLIN:
            raw = socket.recv()
            try:
                obs = pickle.loads(zlib.decompress(raw))
                # 验证必要 key
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
            # Keep UI responsive if no data is coming
            if planner.visualizer:
                planner.visualizer.update()


if __name__ == "__main__":
    main()
