import time
import numpy as np
import zmq
import zlib
import pickle
import threading
import queue
from scipy.spatial.transform import Rotation as R
import fpsample

import sapien
import sapien.render

# -----------------
# 数学工具
# -----------------
def rotation_transfer_6D_to_matrix(rotation):
    r1 = rotation[:3]
    x1 = r1 / (np.linalg.norm(r1) + 1e-9)
    r2 = rotation[3:]
    x2 = r2 - np.sum(r2 * x1) * x1
    x2 /= np.linalg.norm(x2) + 1e-9
    x3 = np.cross(x1, x2)
    return np.stack((x1, x2, x3), axis=1)

def rotation_transfer_matrix_to_6D(matrix):
    return np.concatenate((matrix[:, 0], matrix[:, 1]))

# -----------------
# SAPIEN 相机方向
# -----------------
def sapien_look_at(eye, target, up=np.array([0., 0., 1.])):
    """
    SAPIEN 相机约定：entity 的 +x 轴在世界坐标中为前向（identity 姿态）。
    look_at 公式：R 的列 = [fwd, -right, up2]，使 cam_z_entity 指向 fwd 方向。
    """
    eye, target = np.array(eye, float), np.array(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    mat = np.column_stack([fwd, -right, up2])
    q = R.from_matrix(mat).as_quat()  # xyzw
    return sapien.Pose(p=eye, q=[q[3], q[0], q[1], q[2]])  # wxyz

# -----------------
# 观测提取
# -----------------
def take_picture_once(cam):
    """一次渲染，返回所有图像缓冲区（避免重复拍照）。"""
    cam.take_picture()
    pos_buf = cam.get_picture('Position')      # (H,W,4) 相机空间，z<0=前方
    seg_buf = cam.get_picture('Segmentation')  # (H,W,4) uint32，ch1=per_scene_id
    col_buf = cam.get_picture('Color')         # (H,W,4) float32 [0,1]
    return pos_buf, seg_buf, col_buf

def get_point_cloud_from_buffers(pos_buf, seg_buf, cam, cabinet_seg_ids, num_points=1280):
    """从已拍照的缓冲区提取柜子点云，返回 (1, num_points, 3)。"""
    M = cam.get_model_matrix()  # 4×4，相机→世界
    H, W = pos_buf.shape[:2]

    # 用 model_matrix 将相机空间坐标转换到世界坐标
    pts_cam = pos_buf[:, :, :3].reshape(-1, 3).astype(np.float64)
    ones = np.ones((H * W, 1), dtype=np.float64)
    pts_h = np.hstack([pts_cam, ones])           # (N,4)
    pts_world = (M @ pts_h.T).T[:, :3].reshape(H, W, 3).astype(np.float32)

    # 分割过滤：ch1 = per_scene_id（对应物理连杆 ID）
    seg_id = seg_buf[:, :, 1].astype(np.int32)
    mask = np.isin(seg_id, list(cabinet_seg_ids)) & (pos_buf[:, :, 2] != 0)
    points = pts_world[mask]  # (K, 3)

    if len(points) == 0:
        return np.zeros((1, num_points, 3), dtype=np.float32)

    if points.shape[0] < num_points:
        rand_idx = np.random.choice(points.shape[0], num_points, replace=True)
        points = points[rand_idx]
    else:
        try:
            h_val = min(9, np.log2(num_points))
            idx = fpsample.bucket_fps_kdline_sampling(points[:, :3], num_points, h=h_val)
        except Exception:
            idx = fpsample.fps_npdu_kdtree_sampling(points[:, :3], num_points)
        points = points[np.array(sorted(idx))]

    return points.astype(np.float32)[None, :, :]  # (1, num_points, 3)

def get_raw_rgb_depth_from_buffers(pos_buf, col_buf):
    """
    RGB (H,W,3 uint8) 和 depth (H,W float32 正值，米)。
    Position z<0 为前方（OpenGL 约定），depth = -z。
    """
    rgb = (col_buf[:, :, :3] * 255).astype(np.uint8)
    depth = (-pos_buf[:, :, 2]).astype(np.float32)  # 取反得正值深度
    return rgb, depth

def get_camera_params(cam):
    """
    返回 (cam_pos, cam_mat, fovy_deg)，与 client_mujoco.py 格式一致。
    cam_mat = model_matrix[:3,:3]（列 = [cam_x, cam_y, cam_z_backward]）。
    """
    pose = cam.entity.get_pose()
    cam_pos = np.array(pose.p, dtype=np.float64)
    cam_mat = cam.get_model_matrix()[:3, :3].astype(np.float64)
    fovy_deg = float(np.rad2deg(cam.fovy))
    return cam_pos, cam_mat, fovy_deg

def get_gripper_pcd(hand_link, lf_link, rf_link):
    """返回 (1, 4, 3)，顺序匹配训练数据：[hand, rf, lf, eef=hand]。"""
    hand_pos = np.array(hand_link.entity.get_pose().p, dtype=np.float32)
    lf_pos   = np.array(lf_link.entity.get_pose().p,   dtype=np.float32)
    rf_pos   = np.array(rf_link.entity.get_pose().p,   dtype=np.float32)
    eef_pos  = hand_pos.copy()
    return np.array([[hand_pos, rf_pos, lf_pos, eef_pos]], dtype=np.float32)

def get_agent_pos(hand_link, panda):
    """返回 (1, 10)：eef_pos(3) + rot_6d(6) + gripper(1)。"""
    pose = hand_link.entity.get_pose()
    eef_pos = np.array(pose.p, dtype=np.float32)
    quat_xyzw = [pose.q[1], pose.q[2], pose.q[3], pose.q[0]]
    hand_rot = R.from_quat(quat_xyzw).as_matrix()
    orient_6d = rotation_transfer_matrix_to_6D(hand_rot)
    qpos = panda.get_qpos()
    g = float(np.mean(qpos[7:9]))  # 两指关节平均值
    res = np.concatenate([eef_pos, orient_6d, [g]])
    return res.astype(np.float32)[None, :]  # (1, 10)

# -----------------
# 数值雅可比 IK
# -----------------
def compute_ik(panda, hand_link, arm_indices, target_pos, target_quat_wxyz,
               max_iter=50, tol=1e-4, lr=0.5):
    """数值雅可比 IK，计算后恢复原始 qpos，返回目标 qpos。"""
    orig_qpos = panda.get_qpos().copy()
    qpos = orig_qpos.copy()
    n_arm = len(arm_indices)

    for _ in range(max_iter):
        panda.set_qpos(qpos)
        pose = hand_link.entity.get_pose()
        curr_pos = np.array(pose.p)
        curr_quat_xyzw = [pose.q[1], pose.q[2], pose.q[3], pose.q[0]]

        err_pos = target_pos - curr_pos
        R_curr = R.from_quat(curr_quat_xyzw)
        R_targ = R.from_quat([target_quat_wxyz[1], target_quat_wxyz[2],
                               target_quat_wxyz[3], target_quat_wxyz[0]])
        err_rot = (R_targ * R_curr.inv()).as_rotvec()
        err = np.concatenate([err_pos, err_rot])

        if np.linalg.norm(err) < tol:
            break

        J = np.zeros((6, n_arm))
        eps = 1e-4
        for col, ji in enumerate(arm_indices):
            qp = qpos.copy(); qp[ji] += eps
            panda.set_qpos(qp)
            p_plus = np.array(hand_link.entity.get_pose().p)
            q_plus = hand_link.entity.get_pose().q
            R_plus = R.from_quat([q_plus[1], q_plus[2], q_plus[3], q_plus[0]])

            qm = qpos.copy(); qm[ji] -= eps
            panda.set_qpos(qm)
            p_minus = np.array(hand_link.entity.get_pose().p)
            q_minus = hand_link.entity.get_pose().q
            R_minus = R.from_quat([q_minus[1], q_minus[2], q_minus[3], q_minus[0]])

            J[:3, col] = (p_plus - p_minus) / (2 * eps)
            J[3:, col] = (R_plus * R_minus.inv()).as_rotvec() / (2 * eps)

        lambd = 0.05
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambd * np.eye(6))
        qpos[arm_indices] += (J_pinv @ err) * lr

    panda.set_qpos(orig_qpos)  # 恢复
    return qpos

# -----------------
# 异步通讯线程
# -----------------
class ZMQCommunicationThread(threading.Thread):
    def __init__(self, url="tcp://localhost:5555"):
        super().__init__(daemon=True)
        self.url = url
        self.obs_queue = queue.Queue(maxsize=1)
        self.action_queue = queue.Queue(maxsize=1)
        self.running = True

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(self.url)
        print(f"[Thread] 已连接 VLM Action Server ({self.url})...")

        while self.running:
            try:
                # 获取观测数据（如果没有则阻塞等待）
                obs_msg = self.obs_queue.get(timeout=1.0)
                
                # 序列化并压缩
                ser = {k: {'shape': v.shape, 'dtype': str(v.dtype), 'data': v.tobytes()}
                       for k, v in obs_msg.items()}
                socket.send(zlib.compress(pickle.dumps(ser, protocol=pickle.HIGHEST_PROTOCOL)))

                # 接收回复
                rep = socket.recv()
                if rep == b"ERROR":
                    print("[Thread] Server 遇到异常。")
                    break

                action_raw = pickle.loads(zlib.decompress(rep))
                np_action = np.frombuffer(action_raw['data'],
                                          dtype=action_raw['dtype']).reshape(action_raw['shape'])

                # 存入结果队列（覆盖旧的，只保留最新的）
                if not self.action_queue.empty():
                    try: self.action_queue.get_nowait()
                    except: pass
                self.action_queue.put(np_action)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Thread] 通讯发生错误: {e}")
                import traceback
                traceback.print_exc()
                break
        print("[Thread] 通讯线程退出。")

# -----------------
# 主循环
# -----------------
def main():
    # 初始化通讯线程
    comm_thread = ZMQCommunicationThread("tcp://localhost:5555")
    comm_thread.start()

    # 初始化 SAPIEN 场景
    scene = sapien.Scene()
    scene.set_timestep(1 / 1000.0)  # 1ms，与 MuJoCo 一致
    scene.add_ground(altitude=0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [1, 1, 1], shadow=True)
    scene.add_point_light([0, 0, 3], [1, 1, 1], shadow=False)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True

    # 加载柜子（46230）
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cabinet_urdf = os.path.join(base_dir, "46230", "mobility.urdf")
    cabinet = loader.load(cabinet_urdf)
    cabinet.set_pose(sapien.Pose(p=[1.367079, 0.197510, 0.82]))

    # 柜子分割 ID
    cabinet_seg_ids = set(l.entity.per_scene_id for l in cabinet.get_links())

    # 加载 Panda 机械臂
    panda_urdf = os.path.join(base_dir, "panda", "panda_v2.urdf")
    # if not os.path.exists(panda_urdf):
    #      panda_urdf = "/home/wby/active_vision/vlm_based/panda/panda_v2.urdf"
    
    panda = loader.load(panda_urdf)
    panda.set_pose(sapien.Pose(p=[0.2, 0.6, 0.0]))

    panda_links = panda.get_links()
    hand_link = next(l for l in panda_links if l.name == 'panda_hand')
    lf_link   = next(l for l in panda_links if l.name == 'panda_leftfinger')
    rf_link   = next(l for l in panda_links if l.name == 'panda_rightfinger')

    active_joints = panda.get_active_joints()
    arm_joints    = [j for j in active_joints if j.name.startswith('panda_joint')]
    gripper_joints = [j for j in active_joints if 'finger' in j.name]
    arm_indices    = [active_joints.index(j) for j in arm_joints]
    gripper_indices = [active_joints.index(j) for j in gripper_joints]

    for j in arm_joints:
        j.set_drive_property(stiffness=1200, damping=80)
    for j in gripper_joints:
        # force_limit 控制夹爪实际能输出的最大关节力矩（N），提高此值可增强夹力
        j.set_drive_property(stiffness=5000, damping=2000, force_limit=2000000.0)

    # # 给抽屉的关节加内部阻尼和摩擦，模拟真实的轨道
    # for joint in cabinet.get_active_joints():
    #     joint.set_friction(2.0) # 真实的抽屉轨道起步是需要力气的
    #     joint.set_armature(0.5) # 增加转子惯量，防抖动
    #     joint.set_drive_property(stiffness=0, damping=5) 

    # 给机械臂的手指增加转子惯量，使其被撑开时显得“极重”
    for j in gripper_joints:
        j.set_armature(np.ones(j.dof, dtype=np.float32) * 1.0) # 这是一个黑魔法，能让夹爪变得像石头一样硬

    init_qpos = np.array([-1.3046, -0.5386, 1.5432, -1.5671, -0.1289, 2.8682, -1.3343, 0.04, 0.04])
    panda.set_qpos(init_qpos)
    for i, ji in enumerate(arm_indices): arm_joints[i].set_drive_target(float(init_qpos[ji]))
    for i, ji in enumerate(gripper_indices): gripper_joints[i].set_drive_target(float(init_qpos[ji]))

    for _ in range(200): scene.step()

    for joint in cabinet.get_active_joints():
        joint.set_friction(0.01)
        joint.set_drive_property(stiffness=10, damping=3)
    
    # 把手没有独立 link，其碰撞面是所属抽屉 link 的一部分（link_0/1/2...）
    # 对所有柜子 link 设置高摩擦材质，以增强与夹爪的接触力
    handle_mat = scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    for link in cabinet.get_links():
        for comp in link.entity.get_components():
            if isinstance(comp, sapien.physx.PhysxCollisionShape):
                comp.set_material(handle_mat)

    # 手指摩擦力：摩擦系数是夹住物体的主要力来源，需足够高
    ps_mat = scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    for link in [lf_link, rf_link]:
        for comp in link.entity.get_components():
            if isinstance(comp, sapien.physx.PhysxCollisionShape): comp.set_material(ps_mat)

    cam = scene.add_camera('rgbd_camera', 640, 480, np.deg2rad(60), 0.01, 10.0)
    cam_eye, cam_target = np.array([-1.14, -0.09, 1.2]), np.array([1.367079, 0.197510, 0.82])
    cam.entity.set_pose(sapien_look_at(cam_eye, cam_target))

    viewer = None
    try:
        from sapien.utils import Viewer
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.set_camera_xyz(x=-1.14, y=-0.09, z=1.2)
        viewer.set_camera_rpy(r=0, p=-0.3, y=0.1)
        viewer.paused = False
    except Exception as e: print(f"[WARN] 无法创建 Viewer: {e}")

    # 控制与同步参数
    VELOCITY_FACTOR = 0.2
    MAX_POS_SPEED, MAX_ROT_SPEED, MAX_GRIP_SPEED = 0.4, 1.5, 0.1
    DT = 0.001

    # 运行时状态
    init_pose = hand_link.entity.get_pose()
    goal_pos, goal_quat_wxyz, goal_grip = np.array(init_pose.p), np.array(init_pose.q), 0.04
    ctrl_pos, ctrl_quat, current_grip = goal_pos.copy(), goal_quat_wxyz.copy(), 0.04
    target_qpos = init_qpos.copy()

    obs_history = {'pc': [], 'gp': [], 'ap': []}
    step_idx = 0
    
    # 预览模式
    started = False
    while not started:
        if viewer is not None:
            if viewer.closed: return
            if viewer.window.key_down('space'): started = True
            viewer.render()
        scene.step()
        time.sleep(0.01)

    print("🚀 异步控制已启动...")
    
    # 时钟对齐
    start_wall_time = time.perf_counter()
    accumulated_sim_time = 0.0

    try:
        while True:
            if viewer is not None and viewer.closed: break

            current_wall_time = time.perf_counter() - start_wall_time
            
            # --- 1. 检查是否有新 Action 到达 ---
            if not comm_thread.action_queue.empty():
                np_action = comm_thread.action_queue.get()
                goal_pos     = np_action[:3].astype(np.float64)
                goal_rot_6d  = np_action[3:9]
                goal_grip    = float(np_action[9]) if len(np_action) > 9 else 0.04
                
                goal_mat = rotation_transfer_6D_to_matrix(goal_rot_6d)
                q_xyzw = R.from_matrix(goal_mat).as_quat()
                goal_quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
                
                step_idx += 1
                if step_idx % 5 == 0:
                    print(f"\r[Step {step_idx}] 已接收新 Action. Pos: {np.round(goal_pos, 3)}", end="")

            # --- 2. 仿真步进与控制插值 (追赶真实时间) ---
            # 这里的 loop 保证仿真时间不落后于现实时间
            while accumulated_sim_time < current_wall_time:
                # 插值计算
                # Position
                diff_pos = goal_pos - ctrl_pos
                dist = np.linalg.norm(diff_pos)
                step_pos = VELOCITY_FACTOR * MAX_POS_SPEED * DT
                if dist > step_pos: ctrl_pos += (diff_pos / dist) * step_pos
                else: ctrl_pos = goal_pos.copy()

                # Rotation
                r_goal = R.from_quat([goal_quat_wxyz[1], goal_quat_wxyz[2], goal_quat_wxyz[3], goal_quat_wxyz[0]])
                r_curr = R.from_quat([ctrl_quat[1], ctrl_quat[2], ctrl_quat[3], ctrl_quat[0]])
                r_diff = r_goal * r_curr.inv()
                rot_vec = r_diff.as_rotvec()
                angle = np.linalg.norm(rot_vec)
                step_ang = VELOCITY_FACTOR * MAX_ROT_SPEED * DT
                if angle > step_ang:
                    r_new = R.from_rotvec(rot_vec / angle * step_ang) * r_curr
                    q = r_new.as_quat()
                    ctrl_quat = np.array([q[3], q[0], q[1], q[2]])
                else: ctrl_quat = goal_quat_wxyz.copy()

                # Gripper
                diff_g = goal_grip - current_grip
                step_g = VELOCITY_FACTOR * MAX_GRIP_SPEED * DT
                if abs(diff_g) > step_g: current_grip += np.sign(diff_g) * step_g
                else: current_grip = goal_grip

                # IK 更新 (为了性能，降低频率到 100Hz)
                if int(accumulated_sim_time * 1000) % 10 == 0:
                    target_qpos = compute_ik(panda, hand_link, arm_indices, ctrl_pos, ctrl_quat, max_iter=20, tol=1e-3, lr=0.5)

                for i, ji in enumerate(arm_indices): arm_joints[i].set_drive_target(float(target_qpos[ji]))
                for i, ji in enumerate(gripper_indices): gripper_joints[i].set_drive_target(float(current_grip))

                scene.step()
                accumulated_sim_time += DT

            # --- 3. 观测采集与异步发送任务推送 ---
            # 如果通讯线程空闲，则采集当前观测并推送
            if comm_thread.obs_queue.empty():
                scene.update_render()
                pos_buf, seg_buf, col_buf = take_picture_once(cam)
                
                cur_pc = get_point_cloud_from_buffers(pos_buf, seg_buf, cam, cabinet_seg_ids)
                cur_gp = get_gripper_pcd(hand_link, lf_link, rf_link)
                cur_ap = get_agent_pos(hand_link, panda)
                cur_rgb, cur_depth = get_raw_rgb_depth_from_buffers(pos_buf, col_buf)
                cam_pos, cam_mat, fovy = get_camera_params(cam)
                
                obs_history['pc'].append(cur_pc); obs_history['gp'].append(cur_gp); obs_history['ap'].append(cur_ap)
                if len(obs_history['pc']) > 2:
                    for k in obs_history: obs_history[k].pop(0)

                batch_pc = np.stack([x[0] for x in obs_history['pc']], axis=0)[None, ...]
                batch_gp = np.stack([x[0] for x in obs_history['gp']], axis=0)[None, ...]
                batch_ap = np.stack([x[0] for x in obs_history['ap']], axis=0)[None, ...]
                
                if batch_pc.shape[1] < 2:
                    batch_pc = np.repeat(batch_pc, 2, axis=1)
                    batch_gp = np.repeat(batch_gp, 2, axis=1)
                    batch_ap = np.repeat(batch_ap, 2, axis=1)

                obs_msg = {
                    'point_cloud': batch_pc, 'gripper_pcd': batch_gp, 'agent_pos': batch_ap,
                    'rgb': cur_rgb, 'depth': cur_depth, 'cam_pos': cam_pos, 'cam_mat': cam_mat, 'fovy': np.array([fovy]),
                }
                comm_thread.obs_queue.put(obs_msg)

            # --- 4. 渲染 ---
            if viewer is not None:
                viewer.render()
            
            # 略微休眠防 CPU 占用过高
            time.sleep(0.001)

    except KeyboardInterrupt: print("\n用户中断。")
    finally:
        comm_thread.running = False
        if viewer is not None:
            try: viewer.close()
            except: pass
        print("仿真已关闭。")

if __name__ == "__main__":
    main()
