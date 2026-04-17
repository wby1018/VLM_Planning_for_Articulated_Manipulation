"""
柜体共三层抽屉（link_0 最下、link_1 中间、link_2 最上）。
第一层：最上抽屉 joint_2 + inner_target_2；推合后右臂去第二层绿点，
拉中间抽屉 joint_1，左臂对 inner_target_1 环绕拍照（第二层 = 中间层，不是最下层）。
"""

import os
import struct
import time
import zlib
from datetime import datetime

import mujoco
import mujoco.viewer
import numpy as np
from ikpy.chain import Chain


_ROBO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENE = os.path.join(_ROBO_DIR, "bimanual_env", "scene.xml")
DEFAULT_URDF = os.path.join(_ROBO_DIR, "panda.urdf")

R_EE = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def _write_png_rgb(path, rgb):
    """RGB uint8 HxWx3（或 HxWx4 则丢弃 A）写 PNG，仅依赖标准库 + numpy。"""
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 image array")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3].copy()
    h, w = int(arr.shape[0]), int(arr.shape[1])

    def _chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw_rows = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    zdata = zlib.compress(raw_rows, 9)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", zdata))
        f.write(_chunk(b"IEND", b""))


class DualArmDrawerRetract:
    def __init__(self, xml_path, urdf_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.arm_chain = Chain.from_urdf_file(
            urdf_path,
            base_elements=["link0"],
            active_links_mask=[False, True, True, True, True, True, True, True, False],
        )

        self.right_actuators = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"panda/actuator{i}")
            for i in range(1, 9)
        ]
        self.left_actuators = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"panda_1/actuator{i}")
            for i in range(1, 9)
        ]

        self.right_hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda/hand")
        self.left_hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_1/hand")
        self.left_link0 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_1/link0")
        self.left_camera_rig = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_1/camera_rig"
        )
        self.left_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_hand_camera"
        )

        self.pre_grasp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pre_grasp_target_2")
        self.grasp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_target_2")
        self.inner_inspect = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "inner_target_2")
        # 第二层：中间抽屉 joint_1（自上而下第二层）；绿/红/蓝 = *_1
        self.pre_grasp_layer2 = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pre_grasp_target_1"
        )
        self.grasp_layer2 = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_target_1"
        )
        self.inner_inspect_layer2 = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "inner_target_1"
        )

        self.right_mount = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_arm_mount")

        self.drawer_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_2"
        )
        self.drawer_dof_adr = int(self.model.jnt_dofadr[self.drawer_joint_id])
        self.drawer_qpos_adr = int(self.model.jnt_qposadr[self.drawer_joint_id])
        # 第二层：中间抽屉 joint_1
        self.drawer_l2_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_1"
        )
        self.drawer_l2_dof_adr = int(self.model.jnt_dofadr[self.drawer_l2_joint_id])
        self.drawer_l2_qpos_adr = int(self.model.jnt_qposadr[self.drawer_l2_joint_id])

        self._left_joint7_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "panda_1/joint7"
        )
        _jr = np.asarray(self.model.jnt_range, dtype=float).reshape(-1, 2)
        self._left_j7_lo = float(_jr[self._left_joint7_id, 0])
        self._left_j7_hi = float(_jr[self._left_joint7_id, 1])
        self._drawer_top_q_lo = float(_jr[self.drawer_joint_id, 0])
        self._drawer_top_q_hi = float(_jr[self.drawer_joint_id, 1])
        self._drawer_mid_q_lo = float(_jr[self.drawer_l2_joint_id, 0])
        self._drawer_mid_q_hi = float(_jr[self.drawer_l2_joint_id, 1])

        self.start_time = time.time()
        self.last_q_right = None
        self.last_q_left = None
        self._t_left_motion_begin = None

        # 蓝点上方悬停 + 水平公转拍照（世界 XY 平面，Z 为蓝点高度 + 悬停高度）
        self._photo_orbit_steps = 10
        self._photo_orbit_step_deg = 36
        self._cam_hover_above_blue_z = 0.18
        self._cam_orbit_radius_xy = 0.1
        self._photo_capture_idx = None
        self._photo_pose_t0 = None
        self._photo_all_done = False
        self._photo_settle_s = 0.9
        self._photo_after_orbit_lock_s = 0.65
        self._orbit_locked_at = None
        self._photo_renderer = None
        self._photo_dir = None
        # 左臂：先竖直抬高（over）→ 再绕蓝点上方公转（orbit）
        self._left_vert_phase = "over"
        self._left_over_lift_z = 0.32
        # 柜体顶板约 z≈0.73；第二层蓝点较低，须强制从更高处越过柜顶再下降，否则会 IK 穿侧板
        self._cabinet_safe_z = 0.84
        # 第二层 over：先相机到蓝点正上方（高于柜顶），再夹爪到蓝点；orbit 仍用相机目标
        self._left_over_tol = 0.1
        self._left_over_timeout_s = 24.0
        # 第二层：到达 over 须先对准中间抽屉蓝点（inner_target_1），禁止仅靠超时进入 orbit/拍照
        self._left_over_tol_l2 = 0.055
        self._left_over_min_time_l2_s = 1.8
        # orbit：先 IK 至起点再锁定 1–6；公转步进仅改 joint7
        self._orbit_q_locked = None
        self._orbit_q7_base = None
        self._orbit_lock_tol = 0.1
        self._orbit_lock_timeout_s = 24.0
        self._l2_open_dir_w = None
        self._l2_orbit_along_open_m = 0.05
        # 第二层 over：先「相机在蓝点正上方」越柜顶，再「夹爪到蓝点上方一点」（不必贴蓝点，减少擦柜与旋转回弹）
        self._left_l2_over_stage = None  # None | "cam_high" | "hand_blue"
        self._l2_over_peak_extra_z = 0.12  # 在 _cabinet_safe_z 上再加一段，确保明显高过柜顶
        self._left_l2_peak_tol = 0.10  # 第一段「相机到柜顶上方」略放宽，避免卡死
        # 夹爪目标 = 蓝点与「门侧」插值（grasp_target_1 在把手/前缘），再略 +Z；0.5=二者中点
        self._l2_inspect_lerp_to_door = 0.5
        self._l2_hand_above_blue_m = 0.022  # 在插值点上再加一点高度，防蹭抽屉底
        self._left_l2_hand_tol = 0.085  # 第二段到位容差略宽，避免反复 IK 抬起回缩
        # 第二层 orbit：光心只在参考点（抽屉内插值点）上方小幅抬高；勿用 _cabinet_safe_z 否则会把手臂整段拉出抽屉
        self._l2_orbit_cam_hover_z = 0.07
        self._l2_orbit_cam_min_above_ref_z = 0.035
        self._l2_orbit_radius_xy = 0.08  # 第二层公转水平半径略小，少横向甩动
        # 第二层拍照后沿轨后撤（替代固定 _post_*_delta 世界向量）
        self._post_retract_along_open_m = 0.08
        self._post_over_along_open_m = 0.04

        # 拍照完成后：左臂归位、右臂推抽屉；True=关 joint_1（中间），False=关 joint_2（顶层）
        self._post_close_joint_1 = False
        self._post_photo = False
        self._left_post_sub = None
        self._t_left_post_begin = None
        self._left_retract_blue = None
        self._left_q_home_start = None
        self._t_home_blend_begin = None
        self._left_home_done = False
        self._t_close_begin = None
        self._left_home_blend_s = 7.0
        self._left_lift_tol = 0.12
        self._left_lift_timeout_s = 12.0
        # 推合 joint_2：提高力、缩短缓升，加快关抽屉
        self._drawer_close_force = 78.0
        self._drawer_close_ramp_s = 5.5
        self._drawer_close_fine_force = 52.0
        self._drawer_fully_closed_eps = 0.002
        self._drawer_fine_push_below_q = 0.012
        self._drawer_closed_gripper_open_delay_s = 0.9
        self._t_drawer_appears_closed = None
        # 右臂：顶层合拢并张开后 → 后撤 → 第二层绿点
        self._right_after_close = None
        self._t_right_r2_begin = None
        self._right_retreat_grasp_ref = None
        self._right_retreat_s = 4.0
        self._right_retreat_dx = -0.26
        self._right_retreat_dz = 0.18
        # 第二层右臂：pre/接近路径沿 joint_1 拉开方向外移一段，减少贴柜碰撞卡在手柄口外（夹把手仍用真实 grasp 点）
        self._r_l2_pre_along_open_m = 0.14
        self._r_l2_approach_blend = 0.45
        # 第二层：到达绿点后开始「与第一层相同」的拉门 + 左臂拍照
        self._layer2_started = False
        # 一旦为 True：左臂已开始探入条件满足过，slide_still 等短暂变 False 时不拉回 home（避免抬↔降循环）
        self._l2_left_inspect_armed = False
        self._t_l2_begin = None
        self._t_to_green_begin = None
        self._photo_file_tag = "l1"

        self.t_approach_a_end = 4.0
        self.t_approach_b_end = 10.0
        self.t_extend_end = 14.0
        self.t_gripper_close_end = 17.0
        self.t_settle_end = 20.0
        self.t_pull_end = 55.0
        # 左臂启动：须等抽屉 joint_2 接近拉满（相对 0.8 行程）；避免半开时左臂已探入
        self.after_pull_grace_s = 1.0
        # joint_2 ∈ [0,0.8]；≈0.68 视为「够开」再伸入；兜底仅当已明显拉开仍略低于主阈值
        self.min_drawer_open_for_left = 0.68
        self.left_fallback_after_s = 45.0
        self.min_drawer_open_fallback = 0.58
        # 第二层 joint_1：略降阈值，避免读数边界导致左臂不启动
        self.min_drawer_open_for_left_l2 = 0.65
        # 中间抽屉已拉开超过此 qpos 时，左臂蓝点目标必须是 inner_target_1（与 FSM 标志解耦，避免仍对准顶层 inner_target_2）
        self._mid_drawer_open_blue_dot_q = 0.12
        # 相对关节行程比例（joint_1 / joint_2 绝对 qpos 不可直接比）
        self._frac_left_ready_l1 = 0.78
        self._frac_left_ready_l2 = 0.58
        self._frac_strong_open = 0.88
        self._frac_post_pull_l2 = 0.35
        self.GRIPPER_OPEN = 255.0
        self.GRIPPER_CLOSED = 0.0
        # 拉抽屉：更大广义力 + 更短 ramp，加快拉开；维持阶段比例见下
        self.pull_ramp_duration = 11.0
        self.drawer_pull_force = 62.0
        self._pull_maintain_strong = 0.88
        self._pull_maintain_weak = 0.38
        # 第二层：抽屉滑动已停（用于左臂启动）；维持力直到接近 jnt_range 上限
        self._drawer_vel_stop_eps = 0.004
        self._drawer_still_hold_s = 0.12
        self._l2_q_at_pull_begin = None
        self._l2_still_since = None

        # 左臂等待姿态（与 keyframe panda_1/home 一致）：腕部带相机，夹爪张开
        self._left_hold_ctrl = np.array(
            [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0], dtype=float
        )

    def _drawer_travel_frac(self, q, lo, hi):
        span = hi - lo
        if span <= 1e-9:
            return 0.0
        return float(np.clip((q - lo) / span, 0.0, 1.0))

    def _clip_chain(self, vec):
        return [
            np.clip(
                q,
                self.arm_chain.links[i].bounds[0] + 1e-4,
                self.arm_chain.links[i].bounds[1] - 1e-4,
            )
            for i, q in enumerate(vec)
        ]

    def _left_seed_from_sim(self):
        q = [0.0] * len(self.arm_chain.links)
        for i in range(7):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"panda_1/joint{i + 1}"
            )
            adr = self.model.jnt_qposadr[jid]
            q[i + 1] = float(self.data.qpos[adr])
        return q

    def _site_world(self, site_id):
        """任意 site 的世界坐标；挂在 link_2 上的点会随 joint_2 实时变化。"""
        return np.asarray(self.data.site_xpos[site_id], dtype=float).copy()

    def _drawer_blue_dot_world(self):
        """当前任务对应的内部蓝点：第一层 inner_target_2（最上抽屉 joint_2），第二层 inner_target_1（中间抽屉 joint_1）。"""
        q_mid = float(self.data.qpos[self.drawer_l2_qpos_adr])
        mid_drawer_open = q_mid >= float(self._mid_drawer_open_blue_dot_q)
        # 逻辑标志 + 物理开度：中间抽屉已拉开则必须对准其内蓝点，不能仍用顶层 inner_target_2
        use_layer2_blue = (
            self._layer2_started
            or self._photo_file_tag == "l2"
            or mid_drawer_open
        )
        sid = (
            self.inner_inspect_layer2 if use_layer2_blue else self.inner_inspect
        )
        return self._site_world(sid)

    def _drawer_joint1_open_dir_world(self):
        """joint_1 拉开方向：由 inner_target_1（蓝点）对应该 dof 的瞬时速度方向归一化，与滑轨一致。"""
        nv = int(self.model.nv)
        jacp = np.zeros((3, nv), dtype=float)
        jacr = np.zeros((3, nv), dtype=float)
        mujoco.mj_jacSite(
            self.model, self.data, jacp, jacr, int(self.inner_inspect_layer2)
        )
        dof = int(self.drawer_l2_dof_adr)
        d = np.asarray(jacp[:, dof], dtype=float).reshape(3)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return (d / n).astype(float)

    def _joint1_open_dir_world(self):
        """右臂路径等：与蓝点处同向（刚性抽屉）。"""
        return self._drawer_joint1_open_dir_world()

    def _layer2_slide_targets_active(self):
        q_mid = float(self.data.qpos[self.drawer_l2_qpos_adr])
        return (
            self._layer2_started
            or self._photo_file_tag == "l2"
            or q_mid >= float(self._mid_drawer_open_blue_dot_q)
        )

    def _norm3(self, v, fallback=None):
        if fallback is None:
            fallback = np.array([0.0, 0.0, 1.0], dtype=float)
        d = np.asarray(v, dtype=float).reshape(3)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return fallback.astype(float)
        return (d / n).astype(float)

    def _left_ik_blue_world(self):
        """左臂 IK 末端跟踪的世界点：第二层为蓝点与抽屉门侧（grasp_target_1）之间；第一层按任务选蓝点。"""
        if self._layer2_slide_targets_active():
            p_blue = self._site_world(self.inner_inspect_layer2)
            p_door = self._site_world(self.grasp_layer2)
            t = float(np.clip(self._l2_inspect_lerp_to_door, 0.0, 1.0))
            out = (1.0 - t) * p_blue + t * p_door
            out[2] += float(self._l2_hand_above_blue_m)
            return np.asarray(out, dtype=float)
        return self._drawer_blue_dot_world()

    def _left_over_cam_target_world(self, blue_w):
        """over：相机光心在蓝点正上方 + 不低于柜体安全高度（第一层与第二层第一段共用）。"""
        b = np.asarray(blue_w, dtype=float).reshape(3)
        lift_z = float(self._left_over_lift_z)
        z_safe = float(self._cabinet_safe_z)
        z_up = max(b[2] + lift_z, z_safe)
        return np.array([b[0], b[1], z_up], dtype=float)

    def _left_l2_cam_over_peak_world(self, blue_w):
        """第二层第一段：相机目标在蓝点正上方，Z 强制明显高于柜顶，便于从上方再下降。"""
        b = np.asarray(blue_w, dtype=float).reshape(3)
        lift_z = float(self._left_over_lift_z)
        z_safe = float(self._cabinet_safe_z)
        z_up = max(
            b[2] + lift_z,
            z_safe + float(self._l2_over_peak_extra_z),
        )
        return np.array([b[0], b[1], z_up], dtype=float)

    def _camera_orbit_target_world(self, blue_w, orbit_index):
        """相机光心目标：蓝点上方悬停 + 水平面内偏移；仅用于 orbit 起点（orbit_index=0）的 IK。
        拍照时的「每 36°」由关节 panda_1/joint7 增量实现，不再用不同 orbit_index 做全身 IK。"""
        b = np.asarray(blue_w, dtype=float).reshape(3)
        ang = np.deg2rad(self._photo_orbit_step_deg * int(orbit_index))
        if self._layer2_slide_targets_active():
            r = float(self._l2_orbit_radius_xy)
            hz = float(self._l2_orbit_cam_hover_z)
            d = self._l2_open_dir_w
            if d is None:
                d = self._drawer_joint1_open_dir_world()
            d = self._norm3(d)
            base = b + d * float(self._l2_orbit_along_open_m)
        else:
            r = float(self._cam_orbit_radius_xy)
            hz = float(self._cam_hover_above_blue_z)
            base = b
        out = base + np.array([r * np.cos(ang), r * np.sin(ang), hz], dtype=float)
        if self._layer2_slide_targets_active():
            # 第二层在抽屉内俯视：只保证略高于参考点，禁止 max(..., _cabinet_safe_z)，否则会强制抬到柜顶以上
            out[2] = max(
                out[2],
                float(b[2]) + float(self._l2_orbit_cam_min_above_ref_z),
            )
        return out

    def _solve_ik(self, target_local, seed, use_ori):
        safe = self._clip_chain(seed)
        tp = np.asarray(target_local, dtype=float).reshape(3)
        if use_ori:
            try:
                return self.arm_chain.inverse_kinematics(
                    target_position=tp,
                    target_orientation=R_EE,
                    orientation_mode="all",
                    initial_position=safe,
                )
            except TypeError:
                try:
                    return self.arm_chain.inverse_kinematics(
                        tp, initial_position=safe, target_orientation=R_EE
                    )
                except Exception:
                    pass
            except Exception:
                pass
        return self.arm_chain.inverse_kinematics(
            target_position=tp,
            orientation_mode=None,
            initial_position=safe,
        )

    def _ensure_photo_dir(self):
        if self._photo_dir is None:
            self._photo_dir = os.path.join(
                _ROBO_DIR,
                "drawer_captures",
                datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            os.makedirs(self._photo_dir, exist_ok=True)

    def _save_cam_png(self, index):
        self._ensure_photo_dir()
        if self._photo_renderer is None:
            self._photo_renderer = mujoco.Renderer(
                self.model, height=720, width=960
            )
        self._photo_renderer.update_scene(self.data, camera="left_hand_camera")
        rgb = self._photo_renderer.render()
        if self._photo_file_tag == "l1":
            fname = f"drawer_inner_{index:02d}.png"
        else:
            fname = f"drawer_inner_{self._photo_file_tag}_{index:02d}.png"
        path = os.path.join(self._photo_dir, fname)
        try:
            from PIL import Image

            Image.fromarray(rgb).save(path)
        except Exception:
            _write_png_rgb(path, rgb)
        print(f"已保存相机图: {path}")

    def _left_step_towards_cam_world(self, cam_pos_world, blend_al):
        seed_l = self._left_seed_from_sim()
        if self.last_q_left is not None:
            seed_l = [
                0.5 * a + 0.5 * b for a, b in zip(seed_l, self.last_q_left)
            ]
        # 手→相机：当前帧 world 向量（刚性固连），使期望相机位置对应到 ikpy 的 hand 原点目标
        if self.left_cam_id >= 0:
            delta_w = self.data.cam_xpos[self.left_cam_id] - self.data.xpos[
                self.left_hand
            ]
        else:
            delta_w = self.data.xpos[self.left_camera_rig] - self.data.xpos[
                self.left_hand
            ]
        p_hand_des_w = np.asarray(cam_pos_world, dtype=float).reshape(
            3
        ) - np.asarray(delta_w, dtype=float).reshape(3)
        p0 = self.data.xpos[self.left_link0]
        R0 = self.data.xmat[self.left_link0].reshape(3, 3)
        tpl = (R0.T @ (p_hand_des_w - p0)).reshape(3)
        raw_l = self._solve_ik(tpl, seed_l, False)
        self.last_q_left = (
            raw_l
            if self.last_q_left is None
            else blend_al * np.asarray(raw_l)
            + (1.0 - blend_al) * np.asarray(self.last_q_left)
        )
        for i, aid in enumerate(self.left_actuators[:7]):
            self.data.ctrl[aid] = float(self.last_q_left[i + 1])
        self.data.ctrl[self.left_actuators[7]] = self._left_hold_ctrl[7]

    def _left_step_towards_hand_world(self, hand_pos_world, blend_al):
        """末端执行器（ikpy 与 MuJoCo hand 原点一致）直接去世界坐标，不做手→相机偏移。"""
        seed_l = self._left_seed_from_sim()
        if self.last_q_left is not None:
            seed_l = [
                0.5 * a + 0.5 * b for a, b in zip(seed_l, self.last_q_left)
            ]
        p_hand_des_w = np.asarray(hand_pos_world, dtype=float).reshape(3)
        p0 = self.data.xpos[self.left_link0]
        R0 = self.data.xmat[self.left_link0].reshape(3, 3)
        tpl = (R0.T @ (p_hand_des_w - p0)).reshape(3)
        raw_l = self._solve_ik(tpl, seed_l, False)
        self.last_q_left = (
            raw_l
            if self.last_q_left is None
            else blend_al * np.asarray(raw_l)
            + (1.0 - blend_al) * np.asarray(self.last_q_left)
        )
        for i, aid in enumerate(self.left_actuators[:7]):
            self.data.ctrl[aid] = float(self.last_q_left[i + 1])
        self.data.ctrl[self.left_actuators[7]] = self._left_hold_ctrl[7]

    def _left_arm_qpos7(self):
        q = np.zeros(7, dtype=float)
        for i in range(7):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"panda_1/joint{i + 1}"
            )
            q[i] = float(self.data.qpos[self.model.jnt_qposadr[jid]])
        return q

    def _apply_orbit_joint7_sweep(self, blue_w, t_ins):
        """orbit：未锁定则 IK 至公转起点（角度 0）；锁定后肩肘固定，仅扫 panda_1/joint7。"""
        if self._orbit_q_locked is None:
            cam_target_w = self._camera_orbit_target_world(blue_w, 0)
            if self._photo_capture_idx is not None and not self._photo_all_done:
                al = 0.9
            else:
                al = 0.32 + 0.55 * t_ins
            self._left_step_towards_cam_world(cam_target_w, al)
            return

        q = self._orbit_q_locked.copy()
        oidx = (
            int(self._photo_capture_idx)
            if self._photo_capture_idx is not None
            else 0
        )
        dq7 = np.deg2rad(self._photo_orbit_step_deg * oidx)
        q[6] = np.clip(
            self._orbit_q7_base + dq7, self._left_j7_lo, self._left_j7_hi
        )
        for i, aid in enumerate(self.left_actuators[:7]):
            self.data.ctrl[aid] = float(q[i])
        self.data.ctrl[self.left_actuators[7]] = self._left_hold_ctrl[7]

    def run(self):
        self.data.ctrl[self.right_actuators[-1]] = self.GRIPPER_OPEN
        self.data.ctrl[self.left_actuators[-1]] = self.GRIPPER_OPEN
        for i, aid in enumerate(self.left_actuators[:7]):
            self.data.ctrl[aid] = self._left_hold_ctrl[i]
        self.data.ctrl[self.left_actuators[7]] = self._left_hold_ctrl[7]
        mujoco.mj_forward(self.model, self.data)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                t0 = time.time()
                now = time.time() - self.start_time

                rb = self.data.site_xpos[self.right_mount]
                pre_w = self._site_world(self.pre_grasp)
                grasp_w = self.data.site_xpos[self.grasp].copy()
                drawer_q_top = float(self.data.qpos[self.drawer_qpos_adr])
                drawer_q_l2 = float(self.data.qpos[self.drawer_l2_qpos_adr])
                if self._layer2_slide_targets_active():
                    self._l2_open_dir_w = self._drawer_joint1_open_dir_world()
                final_pre = pre_w.copy()
                if abs(final_pre[0] - rb[0]) > 0.8:
                    final_pre[0] = rb[0] + 0.80

                if (
                    self._post_photo
                    and self._right_after_close == "to_green"
                    and not self._layer2_started
                    and self._t_to_green_begin is None
                ):
                    self._t_to_green_begin = now

                drawer_f_top = 0.0
                drawer_f_mid = 0.0
                phase_r = "approach"

                if self._layer2_started and not self._post_photo:
                    tl2 = now - self._t_l2_begin
                    pre_w2 = self._site_world(self.pre_grasp_layer2)
                    grasp_w2 = self._site_world(self.grasp_layer2).copy()
                    d_open = self._joint1_open_dir_world()
                    final_pre = pre_w2 + d_open * float(self._r_l2_pre_along_open_m)
                    if abs(final_pre[0] - rb[0]) > 0.8:
                        final_pre[0] = rb[0] + 0.80
                    phase_r = "approach"
                    if tl2 < self.t_approach_a_end:
                        tw = final_pre.copy()
                        tw[0] -= 0.25
                        self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_OPEN
                    elif tl2 < self.t_approach_b_end:
                        tb = np.clip(
                            (tl2 - self.t_approach_a_end)
                            / (self.t_approach_b_end - self.t_approach_a_end),
                            0.0,
                            1.0,
                        )
                        tw = final_pre.copy()
                        tw[0] -= (1.0 - tb) * 0.25
                        self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_OPEN
                    elif tl2 < self.t_extend_end:
                        te = np.clip(
                            (tl2 - self.t_approach_b_end)
                            / (self.t_extend_end - self.t_approach_b_end),
                            0.0,
                            1.0,
                        )
                        tw = (1.0 - te) * final_pre + te * grasp_w2
                        self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_OPEN
                    elif tl2 < self.t_gripper_close_end:
                        tw = grasp_w2.copy()
                        g = np.clip(
                            (tl2 - self.t_extend_end)
                            / (self.t_gripper_close_end - self.t_extend_end),
                            0.0,
                            1.0,
                        )
                        self.data.ctrl[self.right_actuators[7]] = (
                            self.GRIPPER_OPEN * (1.0 - g) + self.GRIPPER_CLOSED * g
                        )
                        phase_r = "grasp"
                    elif tl2 < self.t_settle_end:
                        tw = grasp_w2.copy()
                        self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_CLOSED
                        phase_r = "grasp"
                    elif tl2 < self.t_pull_end:
                        tw = grasp_w2.copy()
                        self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_CLOSED
                        phase_r = "pull"
                        s = np.clip(
                            (tl2 - self.t_settle_end) / self.pull_ramp_duration,
                            0.0,
                            1.0,
                        )
                        drawer_f_mid = (s * s) * self.drawer_pull_force
                    else:
                        tw = grasp_w2.copy()
                        self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_CLOSED
                        phase_r = "pull"
                        if tl2 < self.t_pull_end + 95.0:
                            q_hi_m = self._drawer_mid_q_hi
                            if drawer_q_l2 < self.min_drawer_open_for_left:
                                drawer_f_mid = (
                                    self._pull_maintain_strong * self.drawer_pull_force
                                )
                            elif drawer_q_l2 < q_hi_m - 0.005:
                                drawer_f_mid = (
                                    self._pull_maintain_weak * self.drawer_pull_force
                                )
                            else:
                                drawer_f_mid = 0.0
                        else:
                            drawer_f_mid = 0.0
                elif self._post_photo:
                    eps = self._drawer_fully_closed_eps
                    grasp_w2_close = self._site_world(self.grasp_layer2).copy()
                    if self._post_close_joint_1:
                        if self._right_after_close is None:
                            tw = grasp_w2_close.copy()
                            phase_r = "close"
                            s = np.clip(
                                (now - self._t_close_begin)
                                / self._drawer_close_ramp_s,
                                0.0,
                                1.0,
                            )
                            if drawer_q_l2 > eps:
                                primary = -(s * s) * self._drawer_close_force
                                if s >= 1.0:
                                    primary = -self._drawer_close_force
                                drawer_f_mid = primary
                                if drawer_q_l2 < self._drawer_fine_push_below_q:
                                    drawer_f_mid = min(
                                        drawer_f_mid,
                                        -self._drawer_close_fine_force,
                                    )
                                self._t_drawer_appears_closed = None
                            else:
                                drawer_f_mid = 0.0
                                if self._t_drawer_appears_closed is None:
                                    self._t_drawer_appears_closed = now
                            grip_open = (
                                self._t_drawer_appears_closed is not None
                                and (now - self._t_drawer_appears_closed)
                                >= self._drawer_closed_gripper_open_delay_s
                            )
                            self.data.ctrl[self.right_actuators[7]] = (
                                self.GRIPPER_OPEN
                                if grip_open
                                else self.GRIPPER_CLOSED
                            )
                            if drawer_q_l2 <= eps and grip_open:
                                self._post_photo = False
                                self._post_close_joint_1 = False
                                self._layer2_started = False
                                self._photo_file_tag = "l1"
                                self._right_after_close = None
                                self._t_drawer_appears_closed = None
                                print(
                                    "中间抽屉 joint_1 已推合，双臂任务结束"
                                )
                    elif self._right_after_close is None:
                        tw = grasp_w.copy()
                        phase_r = "close"
                        s = np.clip(
                            (now - self._t_close_begin)
                            / self._drawer_close_ramp_s,
                            0.0,
                            1.0,
                        )
                        if drawer_q_top > eps:
                            primary = -(s * s) * self._drawer_close_force
                            if s >= 1.0:
                                primary = -self._drawer_close_force
                            drawer_f_top = primary
                            if drawer_q_top < self._drawer_fine_push_below_q:
                                drawer_f_top = min(
                                    drawer_f_top,
                                    -self._drawer_close_fine_force,
                                )
                            self._t_drawer_appears_closed = None
                        else:
                            drawer_f_top = 0.0
                            if self._t_drawer_appears_closed is None:
                                self._t_drawer_appears_closed = now
                        grip_open = (
                            self._t_drawer_appears_closed is not None
                            and (now - self._t_drawer_appears_closed)
                            >= self._drawer_closed_gripper_open_delay_s
                        )
                        self.data.ctrl[self.right_actuators[7]] = (
                            self.GRIPPER_OPEN if grip_open else self.GRIPPER_CLOSED
                        )
                        if drawer_q_top <= eps and grip_open:
                            self._right_after_close = "retreat"
                            self._t_right_r2_begin = now
                            self._right_retreat_grasp_ref = grasp_w.copy()
                            print(
                                "右臂后撤，随后前往第二层（中间抽屉）绿点 pre_grasp_target_1"
                            )
                    elif self._right_after_close == "retreat":
                        drawer_f_top = 0.0
                        phase_r = "retreat"
                        u = np.clip(
                            (now - self._t_right_r2_begin) / self._right_retreat_s,
                            0.0,
                            1.0,
                        )
                        w = u * u * (3.0 - 2.0 * u)
                        r_off = np.array(
                            [
                                self._right_retreat_dx,
                                0.0,
                                self._right_retreat_dz,
                            ],
                            dtype=float,
                        )
                        base = self._right_retreat_grasp_ref.reshape(3)
                        tw = (1.0 - w) * base + w * (base + r_off)
                        self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_OPEN
                        if w >= 1.0 - 1e-6:
                            self._right_after_close = "to_green"
                    elif self._right_after_close == "to_green":
                        drawer_f_top = 0.0
                        phase_r = "to_green"
                        # 右臂去中间层绿点起即视为第二层任务，左臂目标蓝点切到 inner_target_1（与 _layer2_started 同步稍早）
                        self._photo_file_tag = "l2"
                        tw = self._site_world(self.pre_grasp_layer2).copy()
                        self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_OPEN
                elif now < self.t_approach_a_end:
                    tw = final_pre.copy()
                    tw[0] -= 0.25
                    self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_OPEN
                elif now < self.t_approach_b_end:
                    tb = np.clip(
                        (now - self.t_approach_a_end)
                        / (self.t_approach_b_end - self.t_approach_a_end),
                        0.0,
                        1.0,
                    )
                    tw = final_pre.copy()
                    tw[0] -= (1.0 - tb) * 0.25
                    self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_OPEN
                elif now < self.t_extend_end:
                    te = np.clip(
                        (now - self.t_approach_b_end)
                        / (self.t_extend_end - self.t_approach_b_end),
                        0.0,
                        1.0,
                    )
                    tw = (1.0 - te) * final_pre + te * grasp_w
                    self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_OPEN
                elif now < self.t_gripper_close_end:
                    tw = grasp_w.copy()
                    g = np.clip(
                        (now - self.t_extend_end)
                        / (self.t_gripper_close_end - self.t_extend_end),
                        0.0,
                        1.0,
                    )
                    self.data.ctrl[self.right_actuators[7]] = (
                        self.GRIPPER_OPEN * (1.0 - g) + self.GRIPPER_CLOSED * g
                    )
                    phase_r = "grasp"
                elif now < self.t_settle_end:
                    tw = grasp_w.copy()
                    self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_CLOSED
                    phase_r = "grasp"
                elif now < self.t_pull_end:
                    tw = grasp_w.copy()
                    self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_CLOSED
                    phase_r = "pull"
                    s = np.clip(
                        (now - self.t_settle_end) / self.pull_ramp_duration, 0.0, 1.0
                    )
                    drawer_f_top = (s * s) * self.drawer_pull_force
                else:
                    tw = grasp_w.copy()
                    self.data.ctrl[self.right_actuators[7]] = self.GRIPPER_CLOSED
                    phase_r = "pull"
                    # 名义拉抽屉结束后继续推滑轨，直到开度够或超时（避免卡在半开）
                    if now < self.t_pull_end + 95.0:
                        q_hi_t = self._drawer_top_q_hi
                        if drawer_q_top < self.min_drawer_open_for_left:
                            drawer_f_top = (
                                self._pull_maintain_strong * self.drawer_pull_force
                            )
                        elif drawer_q_top < q_hi_t - 0.005:
                            drawer_f_top = (
                                self._pull_maintain_weak * self.drawer_pull_force
                            )
                        else:
                            drawer_f_top = 0.0
                    else:
                        drawer_f_top = 0.0

                target_pos_local_r = np.asarray(tw - rb, dtype=float).reshape(3)
                hm = self.data.xmat[self.right_hand].reshape(3, 3)
                target_pos_local_r = target_pos_local_r - hm @ np.array([0, 0, 0.105])

                if self.last_q_right is None:
                    seed_r = [0.0, 0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785, 0.0]
                else:
                    seed_r = self.last_q_right

                try:
                    raw_r = self._solve_ik(target_pos_local_r, seed_r, True)
                    if phase_r in ("pull", "close"):
                        ar = 0.88
                    elif phase_r == "to_green":
                        ar = 0.78
                    elif phase_r == "retreat":
                        ar = 0.72
                    elif phase_r == "grasp":
                        ar = 0.28
                    elif (
                        self._layer2_started
                        and not self._post_photo
                        and phase_r == "approach"
                    ):
                        ar = float(self._r_l2_approach_blend)
                    else:
                        ar = 0.2
                    self.last_q_right = (
                        raw_r
                        if self.last_q_right is None
                        else ar * np.asarray(raw_r) + (1 - ar) * np.asarray(self.last_q_right)
                    )
                    for i, aid in enumerate(self.right_actuators[:7]):
                        self.data.ctrl[aid] = float(self.last_q_right[i + 1])
                except Exception:
                    pass

                # —— 左臂：第一层 joint_2（顶层抽屉）仅 drawer_ok + 时间/行程；第二层 joint_1 另加 full_photo ——
                frac_top = self._drawer_travel_frac(
                    drawer_q_top, self._drawer_top_q_lo, self._drawer_top_q_hi
                )
                gripper_settled_l1 = now >= self.t_gripper_close_end + 0.35
                settle_ok_l1 = gripper_settled_l1 or (
                    frac_top >= self._frac_strong_open * 0.92
                )
                drawer_ok_l1 = (
                    frac_top >= self._frac_left_ready_l1
                    or drawer_q_top >= self.min_drawer_open_for_left
                    or (
                        now
                        >= self.t_pull_end
                        + self.after_pull_grace_s
                        + self.left_fallback_after_s
                        and drawer_q_top >= self.min_drawer_open_fallback
                    )
                )
                pull_nominal_done_l1 = now >= self.t_pull_end + self.after_pull_grace_s
                drawer_strong_l1 = (
                    frac_top >= self._frac_strong_open or drawer_q_top >= 0.72
                )
                left_ready_l1 = (
                    not self._post_photo
                    and not self._layer2_started
                    and settle_ok_l1
                    and drawer_ok_l1
                    and (pull_nominal_done_l1 or drawer_strong_l1)
                )
                # 第二层：中间抽屉拉开后再探入；full_photo 仅约束本层
                left_ready_l2 = False
                if self._layer2_started and self._t_l2_begin is not None:
                    tl2 = now - self._t_l2_begin
                    frac_mid = self._drawer_travel_frac(
                        drawer_q_l2, self._drawer_mid_q_lo, self._drawer_mid_q_hi
                    )
                    gripper_settled_l2 = tl2 >= self.t_gripper_close_end + 0.35
                    settle_ok_l2 = gripper_settled_l2 or (
                        frac_mid >= self._frac_strong_open * 0.92
                    )
                    pull_nominal_done_l2 = tl2 >= self.t_pull_end + self.after_pull_grace_s
                    post_pull_open_l2 = pull_nominal_done_l2 and (
                        frac_mid >= self._frac_post_pull_l2
                    )
                    t_fb_l2 = (
                        self.t_pull_end
                        + self.after_pull_grace_s
                        + self.left_fallback_after_s
                    )
                    drawer_ok_l2 = (
                        frac_mid >= self._frac_left_ready_l2
                        or drawer_q_l2 >= self.min_drawer_open_for_left_l2
                        or post_pull_open_l2
                        or (tl2 >= t_fb_l2 and drawer_q_l2 >= self.min_drawer_open_fallback)
                    )
                    if tl2 >= self.t_settle_end and self._l2_q_at_pull_begin is None:
                        self._l2_q_at_pull_begin = float(drawer_q_l2)
                    vq_l2 = float(self.data.qvel[self.drawer_l2_dof_adr])
                    if abs(vq_l2) < self._drawer_vel_stop_eps:
                        if self._l2_still_since is None:
                            self._l2_still_since = now
                        slide_still_l2 = (
                            now - self._l2_still_since
                        ) >= self._drawer_still_hold_s
                    else:
                        self._l2_still_since = None
                        slide_still_l2 = False
                    pull_progress_l2 = (
                        self._l2_q_at_pull_begin is not None
                        and (drawer_q_l2 - self._l2_q_at_pull_begin) > 0.02
                    )
                    left_ready_l2 = (
                        not self._post_photo
                        and settle_ok_l2
                        and drawer_ok_l2
                        and tl2 >= self.t_settle_end + 0.15
                        and pull_progress_l2
                        and slide_still_l2
                    )
                    if left_ready_l2:
                        self._l2_left_inspect_armed = True
                left_ready_inspect = left_ready_l1 or left_ready_l2 or (
                    self._layer2_started
                    and not self._post_photo
                    and self._l2_left_inspect_armed
                )

                if (
                    not left_ready_l1
                    and not self._post_photo
                    and not self._layer2_started
                ):
                    self._t_left_motion_begin = None
                    self._left_vert_phase = "over"
                    self._left_l2_over_stage = None
                    self._orbit_q_locked = None
                    self._orbit_q7_base = None
                    self._orbit_locked_at = None
                    self._photo_capture_idx = None
                    self._photo_pose_t0 = None
                    self._photo_all_done = False
                    self._right_after_close = None
                    self._right_retreat_grasp_ref = None
                    for i, aid in enumerate(self.left_actuators[:7]):
                        self.data.ctrl[aid] = self._left_hold_ctrl[i]
                    self.data.ctrl[self.left_actuators[7]] = self._left_hold_ctrl[7]
                elif (
                    self._layer2_started
                    and not self._post_photo
                    and not left_ready_l2
                    and not self._l2_left_inspect_armed
                ):
                    self._t_left_motion_begin = None
                    self._left_vert_phase = "over"
                    self._left_l2_over_stage = None
                    self._orbit_q_locked = None
                    self._orbit_q7_base = None
                    self._orbit_locked_at = None
                    self._photo_capture_idx = None
                    self._photo_pose_t0 = None
                    self._photo_all_done = False
                    for i, aid in enumerate(self.left_actuators[:7]):
                        self.data.ctrl[aid] = self._left_hold_ctrl[i]
                    self.data.ctrl[self.left_actuators[7]] = self._left_hold_ctrl[7]
                elif self._post_photo and not self._left_home_done:
                    try:
                        if self._left_post_sub == "lift":
                            lb = self._left_retract_blue
                            if (
                                self._post_close_joint_1
                                and self._l2_open_dir_w is not None
                            ):
                                d = self._norm3(self._l2_open_dir_w)
                                pr = float(self._post_retract_along_open_m)
                                po = float(self._post_over_along_open_m)
                                z_lift = max(
                                    float(lb[2]) + float(self._left_over_lift_z),
                                    float(self._cabinet_safe_z),
                                )
                                lift_tgt = np.array(
                                    [
                                        lb[0] + d[0] * (pr + po),
                                        lb[1] + d[1] * (pr + po),
                                        z_lift,
                                    ],
                                    dtype=float,
                                )
                            else:
                                lift_tgt = lb + np.array(
                                    [0.0, 0.0, self._left_over_lift_z],
                                    dtype=float,
                                )
                            self._left_step_towards_cam_world(lift_tgt, 0.85)
                        elif self._left_post_sub == "home":
                            if self._t_home_blend_begin is None:
                                self._t_home_blend_begin = now
                                self._left_q_home_start = self._left_arm_qpos7().copy()
                            t_h = np.clip(
                                (now - self._t_home_blend_begin)
                                / self._left_home_blend_s,
                                0.0,
                                1.0,
                            )
                            w = t_h * t_h * (3.0 - 2.0 * t_h)
                            for i in range(7):
                                self.data.ctrl[self.left_actuators[i]] = (
                                    (1.0 - w) * self._left_q_home_start[i]
                                    + w * self._left_hold_ctrl[i]
                                )
                            self.data.ctrl[self.left_actuators[7]] = (
                                self._left_hold_ctrl[7]
                            )
                    except Exception:
                        pass
                elif self._post_photo and self._left_home_done:
                    for i, aid in enumerate(self.left_actuators[:7]):
                        self.data.ctrl[aid] = self._left_hold_ctrl[i]
                    self.data.ctrl[self.left_actuators[7]] = self._left_hold_ctrl[7]
                elif left_ready_inspect:
                    if self._t_left_motion_begin is None:
                        self._t_left_motion_begin = now
                    t_ins = np.clip(
                        (now - self._t_left_motion_begin) / 5.0, 0.0, 1.0
                    )
                    if (
                        self._left_vert_phase == "orbit"
                        and self._photo_capture_idx is None
                        and not self._photo_all_done
                        and t_ins >= 1.0
                        and self._orbit_q_locked is not None
                        and self._orbit_locked_at is not None
                        and (now - self._orbit_locked_at)
                        >= self._photo_after_orbit_lock_s
                    ):
                        self._photo_capture_idx = 0
                        self._photo_pose_t0 = now

                    try:
                        if self._left_vert_phase == "over":
                            b_ins = self._left_ik_blue_world()
                            if self._layer2_slide_targets_active():
                                if self._left_l2_over_stage is None:
                                    self._left_l2_over_stage = "cam_high"
                                if self._left_l2_over_stage == "cam_high":
                                    cam_tgt = self._left_l2_cam_over_peak_world(
                                        b_ins
                                    )
                                    self._left_step_towards_cam_world(
                                        cam_tgt, 0.82
                                    )
                                else:
                                    self._left_step_towards_hand_world(
                                        b_ins, 0.88
                                    )
                            else:
                                cam_target_w = self._left_over_cam_target_world(
                                    b_ins
                                )
                                self._left_step_towards_cam_world(
                                    cam_target_w, 0.88
                                )
                        else:
                            b_ins = self._left_ik_blue_world()
                            self._apply_orbit_joint7_sweep(b_ins, t_ins)
                    except Exception:
                        pass

                self.data.qfrc_applied[self.drawer_dof_adr] = drawer_f_top
                self.data.qfrc_applied[self.drawer_l2_dof_adr] = drawer_f_mid

                mujoco.mj_step(self.model, self.data)

                if (
                    self._post_photo
                    and self._right_after_close == "to_green"
                    and not self._layer2_started
                ):
                    g = self._site_world(self.pre_grasp_layer2)
                    h = self.data.xpos[self.right_hand]
                    if self._t_to_green_begin is None:
                        self._t_to_green_begin = now
                    if np.linalg.norm(h - g) < 0.11 or (
                        now - self._t_to_green_begin
                    ) > 6.0:
                        self._layer2_started = True
                        self._t_l2_begin = now
                        self._post_photo = False
                        self._right_after_close = None
                        self._t_to_green_begin = None
                        self._photo_file_tag = "l2"
                        self._l2_left_inspect_armed = False
                        self._left_vert_phase = "over"
                        self._left_l2_over_stage = "cam_high"
                        self._orbit_q_locked = None
                        self._orbit_q7_base = None
                        self._orbit_locked_at = None
                        self._photo_capture_idx = None
                        self._photo_pose_t0 = None
                        self._photo_all_done = False
                        self._left_home_done = False
                        self._t_left_motion_begin = None
                        self._l2_q_at_pull_begin = None
                        self._l2_still_since = None
                        print(
                            "第二层：右臂拉 joint_1（中间抽屉），左臂对 inner_target_1 环绕拍照"
                        )

                if (
                    left_ready_inspect
                    and self._left_vert_phase == "over"
                    and self.left_cam_id >= 0
                ):
                    blue_now = self._left_ik_blue_world()
                    if self._layer2_slide_targets_active():
                        if self._left_l2_over_stage is None:
                            self._left_l2_over_stage = "cam_high"
                        if self._left_l2_over_stage == "cam_high":
                            peak_tgt = self._left_l2_cam_over_peak_world(blue_now)
                            cpos = self.data.cam_xpos[self.left_cam_id]
                            dist_peak = float(
                                np.linalg.norm(cpos - peak_tgt)
                            )
                            if (
                                dist_peak < self._left_l2_peak_tol
                                and self._t_left_motion_begin is not None
                                and (now - self._t_left_motion_begin)
                                >= self._left_over_min_time_l2_s
                            ):
                                self._left_l2_over_stage = "hand_blue"
                                self._t_left_motion_begin = now
                            hover_ok = False
                        else:
                            hpos = self.data.xpos[self.left_hand]
                            dist_over = float(
                                np.linalg.norm(hpos - blue_now)
                            )
                            hover_ok = (
                                dist_over < self._left_l2_hand_tol
                                and self._t_left_motion_begin is not None
                                and (now - self._t_left_motion_begin)
                                >= self._left_over_min_time_l2_s
                            )
                    else:
                        over_tgt = self._left_over_cam_target_world(blue_now)
                        cpos = self.data.cam_xpos[self.left_cam_id]
                        dist_over = float(np.linalg.norm(cpos - over_tgt))
                        hover_ok = (
                            dist_over < self._left_over_tol
                            or (
                                self._t_left_motion_begin is not None
                                and (now - self._t_left_motion_begin)
                                > self._left_over_timeout_s
                            )
                        )
                    if hover_ok:
                        self._left_vert_phase = "orbit"
                        self._t_left_motion_begin = now
                        self._orbit_q_locked = None
                        self._orbit_q7_base = None
                        self._orbit_locked_at = None

                if (
                    left_ready_inspect
                    and self._left_vert_phase == "orbit"
                    and self._orbit_q_locked is None
                    and self.left_cam_id >= 0
                ):
                    blue_now = self._left_ik_blue_world()
                    orbit_cam0 = self._camera_orbit_target_world(blue_now, 0)
                    cpos = self.data.cam_xpos[self.left_cam_id]
                    t_orbit = (
                        0.0
                        if self._t_left_motion_begin is None
                        else (now - self._t_left_motion_begin)
                    )
                    if (
                        np.linalg.norm(cpos - orbit_cam0) < self._orbit_lock_tol
                        or t_orbit > self._orbit_lock_timeout_s
                    ):
                        self._orbit_q_locked = self._left_arm_qpos7()
                        self._orbit_q7_base = float(self._orbit_q_locked[6])
                        self._orbit_locked_at = now

                if (
                    self._post_photo
                    and not self._left_home_done
                    and self._left_post_sub == "lift"
                    and self._left_retract_blue is not None
                    and self.left_cam_id >= 0
                ):
                    lb = self._left_retract_blue
                    if (
                        self._post_close_joint_1
                        and self._l2_open_dir_w is not None
                    ):
                        d = self._norm3(self._l2_open_dir_w)
                        pr = float(self._post_retract_along_open_m)
                        po = float(self._post_over_along_open_m)
                        z_lift = max(
                            float(lb[2]) + float(self._left_over_lift_z),
                            float(self._cabinet_safe_z),
                        )
                        lift_tgt = np.array(
                            [
                                lb[0] + d[0] * (pr + po),
                                lb[1] + d[1] * (pr + po),
                                z_lift,
                            ],
                            dtype=float,
                        )
                    else:
                        lift_tgt = lb + np.array(
                            [0.0, 0.0, self._left_over_lift_z], dtype=float
                        )
                    cpos = self.data.cam_xpos[self.left_cam_id]
                    tlf = (
                        0.0
                        if self._t_left_post_begin is None
                        else (now - self._t_left_post_begin)
                    )
                    if (
                        np.linalg.norm(cpos - lift_tgt) < self._left_lift_tol
                        or tlf > self._left_lift_timeout_s
                    ):
                        self._left_post_sub = "home"
                        self._t_home_blend_begin = None

                if (
                    left_ready_inspect
                    and self._left_vert_phase == "orbit"
                    and self._photo_capture_idx is not None
                    and not self._photo_all_done
                    and self._photo_pose_t0 is not None
                    and self._photo_capture_idx < self._photo_orbit_steps
                    and (now - self._photo_pose_t0) >= self._photo_settle_s
                ):
                    self._save_cam_png(self._photo_capture_idx)
                    self._photo_capture_idx += 1
                    self._photo_pose_t0 = now
                    if self._photo_capture_idx >= self._photo_orbit_steps:
                        self._photo_all_done = True
                        self._ensure_photo_dir()
                        print(
                            f"环绕拍摄完成，共 {self._photo_orbit_steps} 张（每 {self._photo_orbit_step_deg}°），目录: {self._photo_dir}"
                        )
                        if self._layer2_started:
                            print("第二层抽屉 inner_target_1 拍摄结束")
                            self._post_photo = True
                            self._post_close_joint_1 = True
                            self._t_close_begin = now
                            self._left_post_sub = "lift"
                            self._t_left_post_begin = now
                            self._left_retract_blue = (
                                self._left_ik_blue_world().copy()
                            )
                            self._t_drawer_appears_closed = None
                            self._left_home_done = False
                            self._t_home_blend_begin = None
                            print(
                                "左臂复位，随后右臂推合中间抽屉 joint_1"
                            )
                        elif not self._post_photo:
                            self._post_photo = True
                            self._post_close_joint_1 = False
                            self._t_close_begin = now
                            self._left_post_sub = "lift"
                            self._t_left_post_begin = now
                            self._left_retract_blue = (
                                self._left_ik_blue_world().copy()
                            )
                            self._t_drawer_appears_closed = None
                            print(
                                "左臂从柜子上方归位，右臂推合顶层抽屉 joint_2"
                            )

                if (
                    self._post_photo
                    and not self._left_home_done
                    and self._left_post_sub == "home"
                    and self._t_home_blend_begin is not None
                    and (now - self._t_home_blend_begin) >= self._left_home_blend_s
                ):
                    self._left_home_done = True

                viewer.sync()

                elapsed = time.time() - t0
                if self.model.opt.timestep > elapsed:
                    time.sleep(self.model.opt.timestep - elapsed)


if __name__ == "__main__":
    try:
        DualArmDrawerRetract(DEFAULT_SCENE, DEFAULT_URDF).run()
    except Exception as e:
        print(f"运行失败: {e}")
