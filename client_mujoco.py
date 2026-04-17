import os
import xml.etree.ElementTree as ET
import time
import math
import numpy as np
import cv2
import zmq
import zlib
import pickle
import copy
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer
import fpsample

def compute_ik(model, data_ik, body_id, target_pos, target_quat, max_iter=50, tol=1e-4, lr=0.5):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    
    jnt_params = []
    for i in range(1, 8):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        if jnt_id != -1:
            jnt_params.append((model.jnt_dofadr[jnt_id], model.jnt_qposadr[jnt_id]))
            
    qpos_ik = data_ik.qpos.copy()
    
    for _ in range(max_iter):
        mujoco.mj_kinematics(model, data_ik)
        mujoco.mj_comPos(model, data_ik)
        
        curr_pos = data_ik.xpos[body_id]
        curr_quat = data_ik.xquat[body_id]
        
        err_pos = target_pos - curr_pos
        
        # Orient error
        R_curr = R.from_quat([curr_quat[1], curr_quat[2], curr_quat[3], curr_quat[0]])
        R_targ = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
        err_rot = (R_targ * R_curr.inv()).as_rotvec()
        
        err = np.concatenate([err_pos, err_rot])
        if np.linalg.norm(err) < tol:
            break
            
        mujoco.mj_jacBody(model, data_ik, jacp, jacr, body_id)
        
        jnt_v_indices = [v for v, q in jnt_params]
        J = np.vstack([jacp[:, jnt_v_indices], jacr[:, jnt_v_indices]])
        
        lambd = 0.05
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambd * np.eye(6))
        delta_q = J_pinv @ err
        
        for idx, (v_idx, q_idx) in enumerate(jnt_params):
            qpos_ik[q_idx] += delta_q[idx] * lr
            
        data_ik.qpos[:] = qpos_ik[:]
        
    return data_ik.qpos.copy()

# -----------------
# 数学库替代 (不依赖PyTorch/Articubot)
# -----------------
def rotation_transfer_6D_to_matrix(rotation):
    r1 = rotation[:3]
    r2 = rotation[3:]
    
    # Normalize r1 with epsilon to avoid division by zero
    norm1 = np.linalg.norm(r1)
    if norm1 < 1e-9:
        # If r1 is zero, return identity as a safe fallback
        return np.eye(3)
    x1 = r1 / norm1
    
    # Orthogonalize and normalize r2
    dot = np.sum(r2 * x1)
    temp = r2 - dot * x1
    norm2 = np.linalg.norm(temp)
    if norm2 < 1e-9:
        # Fallback to something orthogonal to x1 if r2 is parallel to r1 or zero
        if abs(x1[0]) < 0.9:
            ortho = np.array([1, 0, 0], dtype=np.float32)
        else:
            ortho = np.array([0, 1, 0], dtype=np.float32)
        temp = ortho - np.sum(ortho * x1) * x1
        x2 = temp / (np.linalg.norm(temp) + 1e-9)
    else:
        x2 = temp / norm2
        
    x3 = np.cross(x1, x2)
    matrix = np.stack((x1, x2, x3), axis=1)
    return matrix

def rotation_transfer_matrix_to_6D(matrix):
    r1 = matrix[:, 0]
    r2 = matrix[:, 1]
    return np.concatenate((r1, r2))

# -----------------
# 沿用原有的 MuJoCo 解析与辅助函数
# -----------------
def _mjcf_children_as_xml(parent_xml_element):
    return "".join(ET.tostring(child, encoding="unicode") for child in list(parent_xml_element))

def _find_direct_child_body(parent, body_name):
    for child in list(parent):
        if child.tag == "body" and child.get("name") == body_name:
            return child
    return None

def _strip_names_recursive(elem, keep_root=False):
    if (not keep_root) and ("name" in elem.attrib):
        elem.attrib.pop("name", None)
    for child in list(elem):
        _strip_names_recursive(child, keep_root=False)

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

# -----------------
# 图像处理与观测组装
# -----------------
def get_point_cloud_from_mujoco(renderer_depth, model, data, cabinet_body_id):
    # 递归查找所有归属于 Target Cabinet 的 Body 及其对应的 Geom
    descendant_bodies = [cabinet_body_id]
    for i in range(model.nbody):
        curr = i
        while curr != 0:
            if curr == cabinet_body_id:
                if i not in descendant_bodies:
                    descendant_bodies.append(i)
                break
            curr = model.body_parentid[curr]
            
    cabinet_geom_ids = set()
    for b_id in descendant_bodies:
        adr = model.body_geomadr[b_id]
        num = model.body_geomnum[b_id]
        if adr >= 0:
            for i in range(num):
                cabinet_geom_ids.add(adr + i)
    
    # 核心修复：防止机械臂遮挡抽出点云，我们需要让渲染器无视其他全部无关 Geometry
    orig_groups = model.geom_group.copy()
    model.geom_group[:] = 3 # 全部归到视觉组3
    model.geom_group[list(cabinet_geom_ids)] = 1 # 抽屉归到视觉组1
    
    vopt = mujoco.MjvOption()
    vopt.geomgroup[0] = 0
    vopt.geomgroup[1] = 1  # 仅开启视觉组1
    vopt.geomgroup[2] = 0
    vopt.geomgroup[3] = 0  # 关闭视觉组3 (即所有干扰物)
    vopt.geomgroup[4] = 0
    vopt.geomgroup[5] = 0
    
    renderer_depth.enable_depth_rendering()
    renderer_depth.update_scene(data, camera="rgbd_camera", scene_option=vopt)
    depth_array = renderer_depth.render()
    renderer_depth.disable_depth_rendering()
    
    renderer_depth.enable_segmentation_rendering()
    renderer_depth.update_scene(data, camera="rgbd_camera", scene_option=vopt)
    seg_array = renderer_depth.render()
    renderer_depth.disable_segmentation_rendering()
    
    # 恢复物理环境视觉组
    model.geom_group[:] = orig_groups

    geom_ids = seg_array[:, :, 0]
    h, w = depth_array.shape
    fovy = model.cam("rgbd_camera").fovy[0] 
    f_y = (h / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    f_x = f_y 
    
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgbd_camera")
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3) 
    
    vs, us = np.where(depth_array > 0)
    # Masking manually explicitly
    mask = np.isin(geom_ids[vs, us], list(cabinet_geom_ids))
    v_filt = vs[mask]
    u_filt = us[mask]
    z_filt = depth_array[v_filt, u_filt]
    
    cx, cy = w / 2.0, h / 2.0
    x_filt = (u_filt - cx) * z_filt / f_x
    y_filt = (v_filt - cy) * z_filt / f_y
    
    cam_space_p = np.stack([x_filt, -y_filt, -z_filt], axis=0)
    world_p = cam_pos[:, None] + cam_mat @ cam_space_p
    point_cloud = world_p.T
    
    num_points = 4280
    if len(point_cloud) == 0:
        return np.zeros((1, num_points, 3), dtype=np.float32)
        
    if point_cloud.shape[0] < num_points:
        to_add = num_points - point_cloud.shape[0]
        rand_idx = np.random.choice(point_cloud.shape[0], to_add, replace=True)
        point_cloud = np.concatenate([point_cloud, point_cloud[rand_idx]], axis=0)
        
    try:
        h_val = min(9, np.log2(num_points))
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(point_cloud[:, :3], num_points, h=h_val)
    except:
        kdline_fps_samples_idx = fpsample.fps_npdu_kdtree_sampling(point_cloud[:, :3], num_points)
        
    point_cloud = point_cloud[np.array(sorted(kdline_fps_samples_idx))]
    return point_cloud.astype(np.float32)[None, :, :]  

def get_gripper_pcd(model, data, eef_report_offset=0.0):
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    lf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    rf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    
    hand_mat = data.xmat[hand_id].reshape(3, 3)
    hand_z_axis = hand_mat[:, 2]
    
    # eef_report_offset > 0：将上报的所有 keypoint 沿 hand 朝向后退 offset 米
    # 让 policy 以为 gripper 比实际位置靠后，从而命令继续往前走
    shift = -eef_report_offset * hand_z_axis   # 负号：沿 Z 轴向后偏

    hand_pos = data.xpos[hand_id] + shift
    lf_pos   = data.xpos[lf_id]   + shift
    rf_pos   = data.xpos[rf_id]   + shift
    eef_pos  = data.xpos[hand_id] + shift
    
    # 顺序匹配 PyBullet: [right_hand, right_finger, left_finger, eef/grasptarget]
    gripper_pc = np.array([hand_pos, rf_pos, lf_pos, eef_pos])
    return gripper_pc.astype(np.float32)[None, :, :]

def get_raw_rgb_depth(renderer_rgb, renderer_depth, model, data):
    """
    Render raw RGB (H×W×3 uint8) and depth (H×W float32, metres) from
    rgbd_camera with *all* scene objects visible (no geom-group filtering).
    """
    renderer_rgb.update_scene(data, camera="rgbd_camera")
    rgb = renderer_rgb.render().copy()           # (H,W,3) uint8

    renderer_depth.enable_depth_rendering()
    renderer_depth.update_scene(data, camera="rgbd_camera")
    depth = renderer_depth.render().copy()       # (H,W) float32
    renderer_depth.disable_depth_rendering()

    return rgb, depth


def get_camera_params(model, data):
    """Return (cam_pos, cam_mat, fovy) for 'rgbd_camera'."""
    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgbd_camera")
    cam_pos = data.cam_xpos[cam_id].copy()              # (3,)  float64
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()# (3,3) float64
    fovy    = float(model.cam("rgbd_camera").fovy[0])
    return cam_pos, cam_mat, fovy


def get_agent_pos(model, data, eef_report_offset=0.0):
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    
    hand_mat = data.xmat[hand_id].reshape(3, 3)
    hand_z_axis = hand_mat[:, 2]
    # eef_report_offset > 0：上报位置沿 hand 朝向后退，让 policy 以为 arm 比实际靠后
    eef_pos = data.xpos[hand_id] - eef_report_offset * hand_z_axis
    
    orient_6d = rotation_transfer_matrix_to_6D(hand_mat)

    # 真实 gripper 开合量（匹配 PyBullet 的 cur_joint_angle）
    f1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    f2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    if f1_id >= 0 and f2_id >= 0:
        g = 0.5 * (data.qpos[model.jnt_qposadr[f1_id]] +
                   data.qpos[model.jnt_qposadr[f2_id]])
    else:
        g = 0.04  # fallback: fully open

    pos_ori = eef_pos.tolist() + orient_6d.tolist() + [g]
    return np.array(pos_ori, dtype=np.float32)[None, :]

def main():
    # ZMQ Client 初始化
    context = zmq.Context()
    print("🔌 正在连接 ZMQ Server (tcp://localhost:5555) ...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    
    print("搭建带 Mocap 节点的增强型 MuJoCo 场景...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cabinet_xml = os.path.join(base_dir, "46230", "cabinet_fixed.xml")
    menagerie_root = os.path.abspath(os.path.join(base_dir, "mujoco_menagerie"))
    panda_xml_path = os.path.join(menagerie_root, "franka_emika_panda", "panda.xml")

    cabinet_root = ET.parse(cabinet_xml).getroot()
    cabinet_worldbody = cabinet_root.find("worldbody")
    
    # 删除柜子 XML 中的 site 标记点（grasp_target/pre_grasp_target/inner_target）
    # 这些点会被深度相机拍到并混入点云造成干扰
    def _remove_sites_recursive(node):
        for child in list(node):
            if child.tag == 'site':
                node.remove(child)
            else:
                _remove_sites_recursive(child)
    _remove_sites_recursive(cabinet_worldbody)
    
    panda_root = ET.parse(panda_xml_path).getroot()
    panda_worldbody = panda_root.find("worldbody")

    allowed_non_worldbody_tags = {"compiler", "option", "default", "asset"}
    panda_non_worldbody_xml = "".join(
        ET.tostring(child, encoding="unicode")
        for child in list(panda_root) if child.tag in allowed_non_worldbody_tags
    )

    cabinet_scale = 1
    _scale_cabinet_tree(cabinet_worldbody, cabinet_scale)
    cabinet_worldbody_xml = _mjcf_children_as_xml(cabinet_worldbody)
    cabinet_min_z_local = _compute_cabinet_local_min_z(cabinet_worldbody)
    cabinet_world_pos = f"1.367079 0.197510 {0.0 - cabinet_min_z_local}"

    panda_root_body = _find_direct_child_body(panda_worldbody, "link0")
    left_arm = copy.deepcopy(panda_root_body)
    right_arm = copy.deepcopy(panda_root_body)
    left_arm.set("name", "left_panda_root")
    right_arm.set("name", "right_panda_root")
    _strip_names_recursive(right_arm, keep_root=True)

    left_arm.set("pos", "0.2 0.6 0.0")
    right_arm.set("pos", "0.0 -1.0 0.0")
    left_arm.set("quat", "1 0 0 0")
    right_arm.set("quat", "1 0 0 0")
    
    panda_extra_nodes = []
    for child in list(panda_worldbody):
        if child.tag == "body" and child.get("name") == "link0": continue
        if child.tag == "light": continue
        panda_extra_nodes.append(child)

    bimanual_nodes_xml = "".join(ET.tostring(node, encoding="unicode") for node in [left_arm, right_arm, *panda_extra_nodes])

    scene_xml = f"""
<mujoco>
    {panda_non_worldbody_xml}
    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="true"/>
        <geom type="plane" pos="0 0 0" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
        <camera name="rgbd_camera" mode="targetbody" target="target_cabinet" pos="-1.14 -0.09 1.2" fovy="60"/>
        
        <body name="target_cabinet" pos="{cabinet_world_pos}">
            {cabinet_worldbody_xml}
        </body>
        {bimanual_nodes_xml}
    </worldbody>
</mujoco>
"""
    old_cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(panda_xml_path))
        model = mujoco.MjModel.from_xml_string(scene_xml)
        data = mujoco.MjData(model)
    finally:
        os.chdir(old_cwd)

    # ── 物理仿真稳定性设置 ──
    # 1. 隐式积分：对刚性 PD 控制和接触碰撞天然稳定（不发散）
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # 2. 缩短步长：finer dt → 更稳
    model.opt.timestep = 0.001
    # 3. 接触参数
    # solimp=[dmin, dmax, width, mid, power]  solref=[timeconst, dampratio]
    model.opt.o_solref[0] = 0.005  # 更短的接触时间常数 → 更硬的接触，法向力建立更快
    model.opt.o_solimp[0] = 0.99   # 接触阻抗下限
    model.opt.o_solimp[1] = 0.999  # 接触阻抗上限（越高=接触越硬）
    # 4. 【关键】开启 No-slip 修正迭代：强制执行库仑摩擦约束，消除贴面滑动
    #    这是解决"摩擦系数高但仍然滑动"的根本手段，默认值为 0（不修正）
    model.opt.noslip_iterations = 10


    # ── 物理碰撞与摩擦力设置 ──
    # 只给机械臂手部（hand/fingers）设置高摩擦力，防止抽屉把手滑脱
    # 注意：body 名是 hand / left_finger / right_finger，不是关节名
    _finger_body_names = ["hand", "left_finger", "right_finger"]
    _finger_body_ids = []
    for _name in _finger_body_names:
        _bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _name)
        if _bid != -1:
            _finger_body_ids.append(_bid)
        else:
            print(f"[WARN] body '{_name}' not found in model")

    for _gid in range(model.ngeom):
        _bid = model.geom_bodyid[_gid]
        # 检查该 geom 是否属于手部或其子节点（递归向上）
        _is_hand_part = False
        _test_bid = _bid
        while _test_bid > 0:
            if _test_bid in _finger_body_ids:
                _is_hand_part = True
                break
            _test_bid = model.body_parentid[_test_bid]
        
        if _is_hand_part:
            # 跳过纯视觉 geom（contype=0 表示不参与碰撞）
            if model.geom_contype[_gid] == 0:
                continue
            # 夹爪摩擦力设为 5.0，与把手摩擦力匹配，防止拉抽屉时滑脱
            model.geom_friction[_gid, 0] = 5.0   # 切向摩擦 (Slide)
            model.geom_friction[_gid, 1] = 0.1   # 扭转摩擦 (Torsional)
            model.geom_friction[_gid, 2] = 0.01  # 滚动摩擦 (Rolling)
            # condim=4：各向异性摩擦，能同时抵抗滑移和扭转
            model.geom_condim[_gid] = 4
            # 【关键】硬化夹爪接触面：MuJoCo 取两接触体中更软者作为接触刚度
            # 把手已设 solref="0.005 1" solimp="0.95 0.99 0.001"，手指默认更软
            # 将手指设为同等或更硬，确保接触法向力能迅速建立，从而产生有效摩擦
            model.geom_solref[_gid, 0] = 0.004   # 比把手更快（0.005→0.004）
            model.geom_solref[_gid, 1] = 1.0
            model.geom_solimp[_gid, 0] = 0.99    # dmin: 高阻抗下限
            model.geom_solimp[_gid, 1] = 0.999   # dmax: 极高阻抗上限
            model.geom_solimp[_gid, 2] = 0.001   # width: 窄过渡区 → 接触快速变硬
        else:
            model.geom_friction[_gid, 0] = 1.0   # 其他物体（如抽屉滑轨）保持默认低摩擦

    # 设置合理的初始姿态 (匹配 PyBullet [0, 0, 0, -0.4, 0, 0.4, 0])
    # 从 PyBullet 成功运行的 DEBUG 输出提取的真实训练分布初始关节角
    # 对应 Initial GRASPTARGET≈[0.615, 0.381, 0.658]，手臂已接近柜子把手位置
    init_qpos = [-1.3046, -0.5386, 1.5432, -1.5671, -0.1289, 2.8682, -1.3343]
    for i in range(1, 8):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        if jnt_id != -1:
            q_idx = model.jnt_qposadr[jnt_id]
            data.qpos[q_idx] = init_qpos[i-1]

    # 初始化手指关节为 0.04m（全开），与 PyBullet init 状态一致
    for fname in ["finger_joint1", "finger_joint2"]:
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = 0.04

    mujoco.mj_kinematics(model, data)
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    
    # 初始化独立的 IK 计算用的 data_ik
    data_ik = mujoco.MjData(model)
    data_ik.qpos[:] = data.qpos[:]
    mujoco.mj_kinematics(model, data_ik)
    mujoco.mj_forward(model, data)

    print("启动带有 DP3 （基于 ZMQ Client) 的 MuJoCo Viewer... ")

    renderer_depth = mujoco.Renderer(model, height=480, width=640)
    renderer_rgb   = mujoco.Renderer(model, height=480, width=640)
    cabinet_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cabinet")

    # 监控用：打印抽屉关节名
    _drawer_joints = ["joint_0", "joint_1", "joint_2"]

    obs_history = {
        'point_cloud': [],
        'gripper_pcd': [],
        'agent_pos': []
    }
    
    # target_pos 跟踪 grasptarget 位置（与 get_agent_pos 上报的 eef_pos 语义一致）
    hand_mat_init = data.xmat[hand_id].reshape(3, 3)
    hand_z_init = hand_mat_init[:, 2]
    target_pos = data.xpos[hand_id].copy() 
    target_quat = data.xquat[hand_id].copy()
    
    render_fps = 10
    # ── 运动控制参数与初始状态 ──
    VELOCITY_FACTOR = 1   # 0.0 ~ 1.0
    MAX_POS_SPEED   = 0.3   # m/s
    MAX_ROT_SPEED   = 1.0   # rad/s
    MAX_GRIP_SPEED  = 0.05  # m/s
    
    # 指令目标 (从 Server 接收)
    goal_pos  = data.xpos[hand_id].copy()
    goal_quat = data.xquat[hand_id].copy() # wxyz
    goal_grip = 0.04

    # 当前平滑控制点 (Setpoint)
    ctrl_target_pos  = data.xpos[hand_id].copy()
    ctrl_target_quat = data.xquat[hand_id].copy() # wxyz
    current_finger_pos = 0.04

    step_idx = 0
    # 排查用：EEF_REPORT_OFFSET=0 是真实观测基准，确认语义正确后再调整
    EEF_REPORT_OFFSET = 0.0   # 单位：米
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            cur_pc = get_point_cloud_from_mujoco(renderer_depth, model, data, cabinet_body_id)
            cur_gp = get_gripper_pcd(model, data, eef_report_offset=EEF_REPORT_OFFSET)
            cur_ap = get_agent_pos(model, data, eef_report_offset=EEF_REPORT_OFFSET)

            # RGB + raw depth + camera params (needed by VLM action server)
            cur_rgb, cur_depth = get_raw_rgb_depth(renderer_rgb, renderer_depth, model, data)
            cam_pos, cam_mat, fovy = get_camera_params(model, data)

            # === 排查打印：step=0 时输出各关键点，用于对应 PyBullet debug 参考值 ===
            if step_idx == 0:
                _gp = cur_gp[0]  # (4, 3)
                _hand, _rf, _lf, _eef = _gp[0], _gp[1], _gp[2], _gp[3]
                print("[DEBUG step=0]")
                print(f"  hand : {np.round(_hand, 4)}")
                print(f"  rf   : {np.round(_rf, 4)}")
                print(f"  lf   : {np.round(_lf, 4)}")
                print(f"  eef  : {np.round(_eef, 4)}")
                print(f"  eef-hand dist : {np.linalg.norm(_eef - _hand):.4f} m  (PyBullet 入 ~0.0648)")
                print(f"  rf-lf dist    : {np.linalg.norm(_rf - _lf):.4f} m  (PyBullet 入 ~0.08)")
                print(f"  agent_pos eef : {np.round(cur_ap[0, :3], 4)}")
                print(f"  agent_pos g   : {cur_ap[0, -1]:.4f}  (PyBullet 入初始 ~0.04)")
            
            obs_history['point_cloud'].append(cur_pc)
            obs_history['gripper_pcd'].append(cur_gp)
            obs_history['agent_pos'].append(cur_ap)
            if len(obs_history['point_cloud']) > 2:
                obs_history['point_cloud'].pop(0)
                obs_history['gripper_pcd'].pop(0)
                obs_history['agent_pos'].pop(0)
                
            batch_pc = np.stack([x[0] for x in obs_history['point_cloud']], axis=0)[None, ...]
            batch_gp = np.stack([x[0] for x in obs_history['gripper_pcd']], axis=0)[None, ...]
            batch_ap = np.stack([x[0] for x in obs_history['agent_pos']], axis=0)[None, ...]
            if batch_pc.shape[1] < 2:
                batch_pc = np.repeat(batch_pc, 2, axis=1)
                batch_gp = np.repeat(batch_gp, 2, axis=1)
                batch_ap = np.repeat(batch_ap, 2, axis=1)
            
            obs_dict = {
                'point_cloud': batch_pc,
                'gripper_pcd': batch_gp,
                'agent_pos': batch_ap
            }

            # 序列化处理以规避两端 NumPy 版本差异
            serialized_dict = {}
            for k, v in obs_dict.items():
                serialized_dict[k] = {'shape': v.shape, 'dtype': str(v.dtype), 'data': v.tobytes()}
            # VLM action server 额外需要的字段
            for k, v in [('rgb',     cur_rgb),
                         ('depth',   cur_depth),
                         ('cam_pos', cam_pos),
                         ('cam_mat', cam_mat),
                         ('fovy',    np.array([fovy]))]:
                serialized_dict[k] = {'shape': v.shape, 'dtype': str(v.dtype), 'data': v.tobytes()}
                
            # ----------- 发送网络请求 -----------
            req_msg = zlib.compress(pickle.dumps(serialized_dict, protocol=pickle.HIGHEST_PROTOCOL))
            socket.send(req_msg)
            
            # 等待 Server 传回指令
            rep_msg = socket.recv()
            if rep_msg == b"ERROR":
                print("Server 遇到异常报错，客户端已停止。")
                break
            
            action_dict = pickle.loads(zlib.decompress(rep_msg))
            np_action = np.frombuffer(action_dict['data'], dtype=np.dtype(action_dict['dtype'])).reshape(action_dict['shape'])
            
            # Server 发来的是绝对位姿目标
            goal_pos  = np_action[:3]
            goal_rot_6d = np_action[3:9]
            goal_grip = float(np_action[9]) if len(np_action) > 9 else 0.04
            
            # 转换 6D 旋转到 wxyz 四元数
            goal_mat = rotation_transfer_6D_to_matrix(goal_rot_6d)
            _g_quat = R.from_matrix(goal_mat).as_quat() # x,y,z,w
            goal_quat = np.array([_g_quat[3], _g_quat[0], _g_quat[1], _g_quat[2]]) # w,x,y,z

            # ── 预计算 PD 控制相关的索引 ──
            arm_jnt_idxs = []
            for i in range(1, 8):
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                if jnt_id != -1:
                    arm_jnt_idxs.append((model.jnt_qposadr[jnt_id], model.jnt_dofadr[jnt_id]))
            finger_jnt_idxs = []
            for fname in ["finger_joint1", "finger_joint2"]:
                fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
                if fid >= 0:
                    finger_jnt_idxs.append((model.jnt_qposadr[fid], model.jnt_dofadr[fid]))

            # ── 物理执行阶段 (包含平滑插值) ──
            dt = model.opt.timestep 
            for sub_i in range(80):
                # 1. 线性位移插值
                diff_pos = goal_pos - ctrl_target_pos
                dist = np.linalg.norm(diff_pos)
                step_pos = VELOCITY_FACTOR * MAX_POS_SPEED * dt
                if dist > step_pos:
                    ctrl_target_pos += (diff_pos / dist) * step_pos
                else:
                    ctrl_target_pos = goal_pos.copy()
                
                # 2. 旋转插值 (Axis-Angle)
                r_goal = R.from_quat([goal_quat[1], goal_quat[2], goal_quat[3], goal_quat[0]])
                r_curr = R.from_quat([ctrl_target_quat[1], ctrl_target_quat[2], ctrl_target_quat[3], ctrl_target_quat[0]])
                r_diff = r_goal * r_curr.inv()
                rot_vec = r_diff.as_rotvec()
                angle = np.linalg.norm(rot_vec)
                step_ang = VELOCITY_FACTOR * MAX_ROT_SPEED * dt
                if angle > step_ang:
                    ctrl_target_quat_scipy = R.from_rotvec(rot_vec / angle * step_ang) * r_curr
                    _q = ctrl_target_quat_scipy.as_quat()
                    ctrl_target_quat = np.array([_q[3], _q[0], _q[1], _q[2]])
                else:
                    ctrl_target_quat = goal_quat.copy()
                
                # 3. 夹爪宽度插值
                diff_g = goal_grip - current_finger_pos
                step_g = VELOCITY_FACTOR * MAX_GRIP_SPEED * dt
                if abs(diff_g) > step_g:
                    current_finger_pos += np.sign(diff_g) * step_g
                else:
                    current_finger_pos = goal_grip

                # 4. 每 5 个 substep 更新一次 IK 目标
                if sub_i % 5 == 0:
                    data_ik.qpos[:] = data.qpos[:]
                    target_qpos = compute_ik(model, data_ik, hand_id, ctrl_target_pos, ctrl_target_quat)

                # 5. 实时 PD 驱动 (调优参数以克服阻力并防打滑)
                KP_ARM, KD_ARM = 800.0, 100.0
                # 夹爪 KP=5000：拉抽屉时把手的侧向力会把夹爪撑开，需要极强的恢复力
                # 夹持力 ∝ KP × 位置误差，法向力越大，摩擦力越大（f=μN）
                KP_FINGER, KD_FINGER = 5000.0, 200.0
                for (q_adr, v_adr) in arm_jnt_idxs:
                    err = target_qpos[q_adr] - data.qpos[q_adr]
                    err_dot = -data.qvel[v_adr]
                    data.qfrc_applied[v_adr] = data.qfrc_bias[v_adr] + KP_ARM * err + KD_ARM * err_dot
                for (q_adr, v_adr) in finger_jnt_idxs:
                    err = current_finger_pos - data.qpos[q_adr]
                    err_dot = -data.qvel[v_adr]
                    data.qfrc_applied[v_adr] = data.qfrc_bias[v_adr] + KP_FINGER * err + KD_FINGER * err_dot
                
                mujoco.mj_step(model, data)

            if step_idx % 5 == 0:
                jnt_angles = []
                for jname in ["joint_0", "joint_1", "joint_2"]:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    if jid >= 0:
                        jnt_angles.append(data.qpos[model.jnt_qposadr[jid]])
                cur_eef_pos  = data.xpos[hand_id]
                print(f"[step={step_idx}] finger={current_finger_pos:.3f}  drawer_joints={np.round(jnt_angles, 4)}  EEF={np.round(cur_eef_pos, 3)}")
            
            viewer.sync()
            step_idx += 1
            
            time_until_next_step = 0.1 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
