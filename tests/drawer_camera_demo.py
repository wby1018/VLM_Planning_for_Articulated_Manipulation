import os
import xml.etree.ElementTree as ET
import time
import math
import numpy as np
import cv2
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

def save_ply(filename, points):
    """Saves a list of 3D points to a PLY file."""
    header = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt='%.6f %.6f %.6f')

def get_camera_intrinsics(fovy, h, w):
    """Calculates camera intrinsics from fovy and resolution."""
    f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return f, f, w / 2.0, h / 2.0

def _mjcf_children_as_xml(parent_xml_element):
    return "".join(ET.tostring(child, encoding="unicode") for child in list(parent_xml_element))

def _scale_vec_attr(elem, attr_name, scale):
    raw = elem.get(attr_name)
    if not raw:
        return
    try:
        vals = [float(x) for x in raw.replace(",", " ").split() if x]
        elem.set(attr_name, " ".join(str(v * scale) for v in vals))
    except Exception:
        pass

def _scale_cabinet_tree(cabinet_worldbody, scale):
    for elem in cabinet_worldbody.iter():
        _scale_vec_attr(elem, "pos", scale)
        if elem.tag == "geom":
            _scale_vec_attr(elem, "size", scale)
        elif elem.tag == "joint":
            axis = elem.get("type", "hinge")
            if axis == "slide":
                _scale_vec_attr(elem, "range", scale)

def _vec_from_attr(elem, attr_name, default):
    raw = elem.get(attr_name)
    if not raw:
        return list(default)
    try:
        vals = [float(x) for x in raw.replace(",", " ").split() if x]
        if len(vals) == len(default):
            return vals
    except Exception:
        pass
    return list(default)

def _compute_cabinet_local_min_z(cabinet_worldbody):
    min_z = float("inf")
    def walk(node, parent_pos):
        nonlocal min_z
        node_pos = _vec_from_attr(node, "pos", [0.0, 0.0, 0.0]) if node.tag == "body" else [0.0, 0.0, 0.0]
        cur_pos = [parent_pos[i] + node_pos[i] for i in range(3)]
        if node.tag == "geom":
            gpos = _vec_from_attr(node, "pos", [0.0, 0.0, 0.0])
            gtype = node.get("type", "sphere")
            size = _vec_from_attr(node, "size", [])
            z_center = cur_pos[2] + gpos[2]
            if gtype == "box" and len(size) >= 3:
                z_low = z_center - size[2]
            elif gtype in ("sphere", "capsule", "cylinder") and len(size) >= 1:
                z_low = z_center - size[0]
            else:
                z_low = z_center
            min_z = min(min_z, z_low)
        for child in list(node):
            walk(child, cur_pos)
    for child in list(cabinet_worldbody):
        walk(child, [0.0, 0.0, 0.0])
    return 0.0 if min_z == float("inf") else min_z

def get_look_at_quat(eye, target):
    """
    Computes a quaternion that rotates the camera to look at target from eye.
    MuJoCo cameras look along the -Z axis by default, with Y up.
    """
    eye = np.array(eye)
    target = np.array(target)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    # Standard look-at for MuJoCo (camera points at -Z)
    # Target direction for camera Z axis is -forward
    z_axis = -forward
    # Choose an 'up' vector
    up = np.array([0, 0, 1])
    if abs(np.dot(up, z_axis)) > 0.99:
        up = np.array([0, 1, 0])
    
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    r = R.from_matrix(mat)
    return r.as_quat()[[3, 0, 1, 2]] # w, x, y, z

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cabinet_xml_path = os.path.join(base_dir, "46230", "cabinet_fixed.xml")
    
    cabinet_root = ET.parse(cabinet_xml_path).getroot()
    cabinet_worldbody = cabinet_root.find("worldbody")
    
    # Remove sites (grasp targets, etc.) to keep the scene clean
    def _remove_sites_recursive(node):
        for child in list(node):
            if child.tag == 'site':
                node.remove(child)
            else:
                _remove_sites_recursive(child)
    _remove_sites_recursive(cabinet_worldbody)
    
    # Scale and find min_z
    cabinet_scale = 1.0
    _scale_cabinet_tree(cabinet_worldbody, cabinet_scale)
    cabinet_min_z_local = _compute_cabinet_local_min_z(cabinet_worldbody)
    cabinet_world_pos = [1.0, 0.0, -cabinet_min_z_local]
    cabinet_worldbody_xml = _mjcf_children_as_xml(cabinet_worldbody)

    scene_xml = f"""
<mujoco>
    <option integrator="implicitfast" timestep="0.002"/>
    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="true"/>
        <geom type="plane" pos="0 0 0" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
        
        <body name="target_cabinet" pos="{" ".join(map(str, cabinet_world_pos))}">
            {cabinet_worldbody_xml}
        </body>

        <body name="camera_mocap" mocap="true" pos="0 0 2">
            <camera name="rgbd_camera" mode="fixed" fovy="60"/>
        </body>
    </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    
    height, width = 480, 640
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    # ── Data Saving Setup ──
    cwd = os.getcwd()
    rgb_dir = os.path.join(cwd, "rgb")
    depth_dir = os.path.join(cwd, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    pose_file = open(os.path.join(cwd, "camera_pose.txt"), "w")
    
    fovy = model.cam("rgbd_camera").fovy[0]
    fx, fy, cx, cy = get_camera_intrinsics(fovy, height, width)
    with open(os.path.join(cwd, "camera_info.txt"), "w") as f:
        f.write(f"width: {width}\nheight: {height}\nfov: {fovy}\nfx: {fx}\nfy: {fy}\ncx: {cx}\ncy: {cy}\n")
    
    # Find the bottom drawer joint
    # We will look for slide joints and find the one with the lowest body position
    lowest_z = float('inf')
    bottom_drawer_joint_name = None
    
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE:
            body_id = model.jnt_bodyid[i]
            # Since mj_forward hasn't run, xpos might be zero, but we can look at the geom/body pos in XML or wait
            # Let's run forward once to get initial positions
            pass
    
    mujoco.mj_forward(model, data)
    
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE:
            body_id = model.jnt_bodyid[i]
            z_pos = data.xpos[body_id][2]
            if z_pos < lowest_z:
                lowest_z = z_pos
                bottom_drawer_joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    
    print(f"Detected bottom drawer joint: {bottom_drawer_joint_name}")
    
    # Define triangle for camera movement
    # Triangle vertices above and around the cabinet
    v1 = np.array([-1.8, -2.5, 2.0])
    v2 = np.array([ 3.5,  0.0, 2.0])
    v3 = np.array([-1.8,  2.5, 2.0])
    triangle_points = [v1, v2, v3]
    
    cabinet_center = np.array(cabinet_world_pos) + np.array([0, 0, 0.5]) # Approximate center
    
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera_mocap")
    mocap_idx = model.body_mocapid[mocap_id]
    
    print("\n" + "="*50)
    print("🚀 Drawer & Camera Demo Started!")
    print(f"  - Cabinet Pos : {cabinet_world_pos}")
    print(f"  - Active Joint: {bottom_drawer_joint_name}")
    print("  - Camera Mode : Triangle Movement + Look-At")
    print("  - Visualization: Separate OpenCV Window ('Camera Feed')")
    print("  - Controls    : Press 'q' in CV window or close MuJoCo viewer to stop.")
    print("="*50 + "\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        last_save_time = -1.0
        frame_id = 0
        
        while viewer.is_running():
            elapsed = time.time() - start_time
            
            # 1. Animate drawer (Open and Close loop)
            # Use a sine wave to go from 0 to 0.4 (range is usually 0 to 0.8 according to XML)
            joint_range = 0.4 
            joint_val = (math.sin(elapsed * 1.0) + 1.0) / 2.0 * joint_range
            if bottom_drawer_joint_name:
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, bottom_drawer_joint_name)
                data.qpos[model.jnt_qposadr[jnt_id]] = joint_val
            
            # 2. Animate camera position in a triangle
            # T is period for one lap
            lap_period = 10.0
            t = (elapsed % lap_period) / lap_period
            # Interpolate between vertices
            if t < 1/3:
                p = (t * 3)
                cam_pos = v1 * (1-p) + v2 * p
            elif t < 2/3:
                p = (t - 1/3) * 3
                cam_pos = v2 * (1-p) + v3 * p
            else:
                p = (t - 2/3) * 3
                cam_pos = v3 * (1-p) + v1 * p
            
            data.mocap_pos[mocap_idx] = cam_pos
            
            # 3. Always look at the cabinet
            cam_quat = get_look_at_quat(cam_pos, cabinet_center)
            data.mocap_quat[mocap_idx] = cam_quat
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            
            # 4. Render and show in separate window
            renderer.update_scene(data, camera="rgbd_camera")
            rgb_frame = renderer.render()
            
            # Render depth for saving
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera="rgbd_camera")
            depth_array = renderer.render()
            renderer.disable_depth_rendering()
            
            # --- Save Data Every 0.2 Seconds ---
            if elapsed - last_save_time >= 0.2:
                last_save_time = elapsed
                
                # Save RGB
                rgb_filename = os.path.join(rgb_dir, f"rgb_{frame_id:06d}.png")
                cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                
                # Save Depth PLY
                # Project depth to world-space point cloud
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgbd_camera")
                cam_pos = data.cam_xpos[cam_id]
                cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
                
                # Create grid of u, v coordinates
                v, u = np.indices((height, width))
                z = depth_array
                mask = z > 0
                u, v, z = u[mask], v[mask], z[mask]
                
                x_cam = (u - cx) * z / fx
                y_cam = -(v - cy) * z / fy  # MuJoCo camera Y is up, image Y is down
                z_cam = -z                  # MuJoCo camera looks along -Z
                
                cam_space_pts = np.stack([x_cam, y_cam, z_cam], axis=1)
                world_pts = cam_pos + cam_space_pts @ cam_mat.T
                
                depth_filename = os.path.join(depth_dir, f"depth_{frame_id:06d}.ply")
                save_ply(depth_filename, world_pts)
                
                # Save Pose
                cur_cam_pos = data.mocap_pos[mocap_idx]
                cur_cam_quat = data.mocap_quat[mocap_idx] # MuJoCo format (w, x, y, z)
                # Reorder to (x, y, z, w) as requested
                xyzw_quat = [cur_cam_quat[1], cur_cam_quat[2], cur_cam_quat[3], cur_cam_quat[0]]
                
                pose_line = f"{frame_id:06d} {' '.join(map(str, cur_cam_pos))} {' '.join(map(str, xyzw_quat))}\n"
                pose_file.write(pose_line)
                pose_file.flush()
                
                frame_id += 1
            
            # Convert to BGR for OpenCV
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Camera Feed", bgr_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Maintain real-time (approx)
            time.sleep(0.01)

    pose_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
