import os
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import rerun as rr
    HAVE_RERUN = True
except ImportError:
    HAVE_RERUN = False
    print("[WARN] rerun-sdk 未安装，跳过 3D 可视化。")

def load_camera_info(path):
    info = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.split(':')
                info[key.strip()] = float(val.strip())
    return info

def load_poses(path):
    poses = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            frame_id = int(parts[0])
            if frame_id not in poses: poses[frame_id] = []
            # frame_id name tx ty tz qx qy qz qw
            name = parts[1]
            p = np.array([float(x) for x in parts[2:5]])
            q = np.array([float(x) for x in parts[5:9]]) # xyzw
            poses[frame_id].append({'name': name, 'p': p, 'q': q})
    return poses

def load_camera_poses(path):
    poses = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            frame_id = int(parts[0])
            p = np.array([float(x) for x in parts[1:4]])
            q = np.array([float(x) for x in parts[4:8]]) # xyzw
            poses[frame_id] = {'p': p, 'q': q}
    return poses

def reconstruct_pc(depth, info, cam_pose):
    H, W = depth.shape
    fx, fy = info['fx'], info['fy']
    cx, cy = info['cx'], info['cy']
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    d = depth.flatten()
    
    # Mask out zero depth
    mask = d > 0
    u, v, d = u[mask], v[mask], d[mask]
    
    # Standard OpenCV Camera Frame (Z-forward, X-right, Y-down)
    # Note: camera_pose.txt now contains the optical pose, so we can project directly.
    z_cam = d
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(d)], axis=1) # (N, 4)
    
    # World Transform (Optical to World)
    p, q = cam_pose['p'], cam_pose['q'] # xyzw
    rot = R.from_quat(q).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = p
    
    pts_world = (T @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float32)

def dist_to_segments(pts, segments):
    """
    pts: (N, 3)
    segments: list of (A, B, radius)
    Returns: mask (N,) where True means point is INSIDE any cylinder
    """
    inside_mask = np.zeros(len(pts), dtype=bool)
    for A, B, radius in segments:
        AB = B - A
        mag_sq = np.dot(AB, AB)
        if mag_sq < 1e-6: # Sphere if segment is zero-length
            dist = np.linalg.norm(pts - A, axis=1)
        else:
            AP = pts - A
            t = np.sum(AP * AB, axis=1) / mag_sq
            t = np.clip(t, 0, 1)
            closest = A + t[:, None] * AB
            dist = np.linalg.norm(pts - closest, axis=1)
        
        inside_mask |= (dist < radius)
    return inside_mask

def main():
    root = "record_pull_arc_with_joint_0"
    frame_id = 130
    
    print(f"--- Loading data for frame {frame_id} ---")
    info = load_camera_info(os.path.join(root, "camera_info.txt"))
    cam_poses = load_camera_poses(os.path.join(root, "camera_pose.txt"))
    link_poses_all = load_poses(os.path.join(root, "link_poses.txt"))
    depth = np.load(os.path.join(root, f"depth/depth_{frame_id:06d}.npy"))
    
    if frame_id not in cam_poses or frame_id not in link_poses_all:
        print("Error: frame_id not found in files")
        return

    # --- Rerun ─────────────────────────────────────────────────────────────────
    if HAVE_RERUN:
        rr.init("RobotFilter", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
        # Log World Axes
        # X: Red, Y: Green, Z: Blue
        rr.log("world/axes", rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["X", "Y", "Z"]
        ))

    # 1. Reconstruct Point Cloud
    points = reconstruct_pc(depth, info, cam_poses[frame_id])
    print(f"Original points: {len(points)}")
    
    # 2. Rerun Camera & Image Logging
    if HAVE_RERUN:
        p, q = cam_poses[frame_id]['p'], cam_poses[frame_id]['q']
        
        # Log Camera Transform (Directly from Optical-to-World pose in file)
        rr.log("world/camera", rr.Transform3D(
            translation=p,
            rotation=rr.Quaternion(xyzw=q)
        ))
        
        # Log Pinhole
        rr.log("world/camera", rr.Pinhole(
            focal_length=[info['fx'], info['fy']],
            principal_point=[info['cx'], info['cy']],
            width=info['width'],
            height=info['height']
        ))
        
        # Log RGB if available (loading it now)
        rgb_path = os.path.join(root, f"rgb/rgb_{frame_id:06d}.png")
        if os.path.exists(rgb_path):
            import cv2
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            rr.log("world/camera/rgb", rr.Image(rgb))
    
    # 2. Prepare Segments
    # We map link names to positions
    lp_dict = {l['name']: l['p'] for l in link_poses_all[frame_id]}
    
    # Radius dictionary
    R_ARM = 0.15
    R_FOREARM = 0.13
    R_HAND = 0.07
    R_FINGER = 0.03
    
    # Connection list: (start_link, end_link, radius)
    # Note: the links are named panda_link0 ... panda_link8, panda_hand, panda_leftfinger, etc.
    connections = [
        ('panda_link0', 'panda_link1', R_ARM),
        ('panda_link1', 'panda_link2', R_ARM),
        ('panda_link2', 'panda_link3', R_ARM),
        ('panda_link3', 'panda_link4', R_ARM),
        ('panda_link4', 'panda_link5', R_FOREARM),
        ('panda_link5', 'panda_link6', R_FOREARM),
        ('panda_link6', 'panda_link7', R_FOREARM),
        ('panda_link7', 'panda_link8', R_FOREARM),
        ('panda_hand', 'panda_leftfinger', R_HAND),
        ('panda_hand', 'panda_rightfinger', R_HAND),
        ('panda_leftfinger', 'panda_leftfinger', R_FINGER),  # Sphere at tip
        ('panda_rightfinger', 'panda_rightfinger', R_FINGER), # Sphere at tip
    ]
    
    segments = []
    for s, e, r in connections:
        if s in lp_dict and e in lp_dict:
            segments.append((lp_dict[s], lp_dict[e], r))
        else:
            print(f"Warning: Link {s} or {e} not found")

    # 3. Filter Points
    inside = dist_to_segments(points, segments)
    filtered_points = points[~inside]
    removed_count = np.sum(inside)
    
    print(f"Removed points: {removed_count} ({removed_count/len(points)*100:.1f}%)")
    print(f"Filtered points: {len(filtered_points)}")
    
    # 4. Visualization & Save
    if HAVE_RERUN:
        # Original points (gray, small)
        rr.log("world/original_pc", rr.Points3D(points, colors=[150, 150, 150], radii=0.002))
        
        # Filtered points (green, default size)
        rr.log("world/filtered_pc", rr.Points3D(filtered_points, radii=0.004))
        
        # Removed points (red, slightly larger to highlight)
        removed_points = points[inside]
        if len(removed_points) > 0:
            rr.log("world/removed_robot_pc", rr.Points3D(removed_points, colors=[255, 50, 50], radii=0.006))
            
        # Visualize cylinders as lines for reference
        for i, (A, B, radius) in enumerate(segments):
            rr.log(f"world/robot_links/link_{i}", rr.LineStrips3D([np.stack([A, B])], radii=radius, colors=[0, 255, 255]))

    np.save("original_pc.npy", points)
    np.save("filtered_pc.npy", filtered_points)
    print("Saved original_pc.npy and filtered_pc.npy")
    if HAVE_RERUN:
        print("Rerun visualization active. Keep the rerun window open to view results.")

if __name__ == "__main__":
    main()
