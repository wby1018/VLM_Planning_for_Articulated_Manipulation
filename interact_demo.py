"""Headless scripted interaction: Panda arm approaches drawer handle of
cabinet 40147, closes its gripper, and pulls the drawer open. Saves one PNG
per keyframe + intermediate frames under out/interact/ so you can inspect
the interaction, and stitches them into an MP4.

Reuses compute_ik from client_sapien_40147.py. No ZMQ / Viewer needed.
"""
import os, time, subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R

import sapien
import sapien.render

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client_sapien_40147 import compute_ik, sapien_look_at  # noqa


def save_rgb(cam, path):
    cam.take_picture()
    col = cam.get_picture('Color')
    img = (np.clip(col[..., :3], 0, 1) * 255).astype(np.uint8)
    import cv2
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "out", "interact")
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # --- build scene ---
    scene = sapien.Scene()
    scene.set_timestep(1 / 1000.0)

    # Indoor lighting
    scene.set_ambient_light([0.4, 0.38, 0.36])
    scene.add_directional_light([0.3, 0.6, -1], [0.7, 0.68, 0.65], shadow=True)
    scene.add_point_light([0.3, 0.0, 2.5], [0.5, 0.48, 0.45], shadow=True)
    scene.add_point_light([-1.0, -0.5, 1.5], [0.25, 0.25, 0.28], shadow=False)

    # Floor (doubles as ground plane for collision)
    floor_builder = scene.create_actor_builder()
    floor_mat = sapien.render.RenderMaterial(
        base_color=[0.75, 0.70, 0.63, 1.0], roughness=0.9, metallic=0.0)
    floor_builder.add_box_visual(half_size=(5, 5, 0.01), material=floor_mat)
    floor_builder.add_box_collision(half_size=(5, 5, 0.01))
    floor = floor_builder.build_static(name="floor")
    floor.set_pose(sapien.Pose(p=[0, 0, -0.01]))

    # Back wall
    wall_builder = scene.create_actor_builder()
    wall_mat = sapien.render.RenderMaterial(
        base_color=[0.90, 0.88, 0.85, 1.0], roughness=0.95, metallic=0.0)
    wall_builder.add_box_visual(half_size=(5, 0.01, 3), material=wall_mat)
    wall = wall_builder.build_static(name="back_wall")
    wall.set_pose(sapien.Pose(p=[0, 2.5, 3.0]))

    # Side wall (right)
    side_wall_builder = scene.create_actor_builder()
    side_wall_mat = sapien.render.RenderMaterial(
        base_color=[0.88, 0.86, 0.83, 1.0], roughness=0.95, metallic=0.0)
    side_wall_builder.add_box_visual(half_size=(0.01, 5, 3), material=side_wall_mat)
    side_wall = side_wall_builder.build_static(name="side_wall")
    side_wall.set_pose(sapien.Pose(p=[3.0, 0, 3.0]))

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = 0.6
    cabinet = loader.load(os.path.join(base_dir, "40147", "mobility_vhacd.urdf"))
    cabinet.set_pose(sapien.Pose(p=[0.8, 0.0, 0.4]))

    loader_robot = scene.create_urdf_loader()
    loader_robot.fix_root_link = True
    panda = loader_robot.load(os.path.join(base_dir, "panda", "panda_v2.urdf"))
    panda.set_pose(sapien.Pose(p=[0.0, 0.222, 0.0]))

    panda_links = panda.get_links()
    hand_link = next(l for l in panda_links if l.name == 'panda_hand')
    lf_link = next(l for l in panda_links if l.name == 'panda_leftfinger')
    rf_link = next(l for l in panda_links if l.name == 'panda_rightfinger')

    active_joints = panda.get_active_joints()
    arm_joints = [j for j in active_joints if j.name.startswith('panda_joint')]
    gripper_joints = [j for j in active_joints if 'finger' in j.name]
    arm_indices = [active_joints.index(j) for j in arm_joints]
    gripper_indices = [active_joints.index(j) for j in gripper_joints]

    for j in arm_joints:
        j.set_drive_property(stiffness=2000, damping=100)
    for j in gripper_joints:
        j.set_drive_property(stiffness=50000, damping=5000, force_limit=2000000.0)
        j.set_armature(np.ones(j.dof, dtype=np.float32) * 5.0)

    init_qpos = np.array(
        [0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.785, 0.04, 0.04])
    panda.set_qpos(init_qpos)
    for i, ji in enumerate(arm_indices):
        arm_joints[i].set_drive_target(float(init_qpos[ji]))
    for i, ji in enumerate(gripper_indices):
        gripper_joints[i].set_drive_target(float(init_qpos[ji]))

    for _ in range(200):
        scene.step()

    for joint in cabinet.get_active_joints():
        joint.set_friction(0.02)
        joint.set_drive_property(stiffness=0, damping=2)

    # high-friction materials so the grip actually pulls the drawer
    grip_mat = scene.create_physical_material(
        static_friction=5.0, dynamic_friction=5.0, restitution=0.0)
    for link in cabinet.get_links():
        for comp in link.entity.get_components():
            if isinstance(comp, sapien.physx.PhysxCollisionShape):
                comp.set_material(grip_mat)
    for link in [lf_link, rf_link]:
        for comp in link.entity.get_components():
            if isinstance(comp, sapien.physx.PhysxCollisionShape):
                comp.set_material(grip_mat)

    # Primary camera: 3/4 view from the drawer-pull side so the drawer
    # extrusion is clearly visible (the straight-on view used by the client
    # looks too flat because the pull axis goes away from the camera).
    cam = scene.add_camera('rgb', 1280, 960, np.deg2rad(55), 0.01, 10.0)
    cam.entity.set_pose(sapien_look_at(
        [-0.2, 1.4, 1.0], [0.5, 0.1, 0.5]))

    # --- Waypoints ---
    # Starting hand pose (from init_qpos): query after stepping.
    scene.update_render()
    start_pose = hand_link.entity.get_pose()
    start_pos = np.array(start_pose.p)
    start_quat = np.array(start_pose.q)  # wxyz
    print(f"[info] start hand pos = {np.round(start_pos, 3)}")

    # Target grasp orientation: hand's +Z (forward through fingertips) along
    # world +X so fingers approach the drawer head-on, and finger opening axis
    # along world +Z so the two fingers straddle the horizontal handle bar
    # from above / below.
    #   world_R_hand: col0 (hand_x) = world +Y,
    #                 col1 (hand_y) = world +Z,
    #                 col2 (hand_z) = world +X.
    grasp_mat = np.array([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]], dtype=float)
    q_xyzw = R.from_matrix(grasp_mat).as_quat()
    grasp_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # wxyz
    print(f"[info] grasp quat (wxyz) = {np.round(grasp_quat, 3)}")

    handle_center = np.array([0.552, 0.222, 0.672])
    FINGER_OFFSET = 0.11
    handle_pos    = handle_center + np.array([-FINGER_OFFSET, 0.0, 0.0])
    pre_grasp_pos = handle_pos   + np.array([-0.15, 0.0, 0.0])
    pull_pos      = handle_pos   + np.array([-0.38, 0.0, 0.0])  # full drawer travel (0.36m) + margin

    # Visual markers for target trajectory
    def add_marker(p, rgba, radius=0.012, name="marker"):
        b = scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(
            base_color=rgba, roughness=0.3, metallic=0.0)
        b.add_sphere_visual(radius=radius, material=mat)
        a = b.build_kinematic(name=name)
        a.set_pose(sapien.Pose(p=p.tolist()))
        return a

    # Key waypoint markers
    add_marker(handle_center, [1, 0, 0, 1],    radius=0.015, name="mk_handle")      # red = handle
    add_marker(handle_pos,    [1, 1, 0, 1],    radius=0.015, name="mk_grasp")       # yellow = grasp
    add_marker(pre_grasp_pos, [0, 0.4, 1, 1],  radius=0.015, name="mk_pregrasp")    # blue = pre-grasp
    add_marker(pull_pos,      [0, 1, 0, 1],    radius=0.015, name="mk_pull")        # green = pull end

    # Interpolated trajectory path (small dots between waypoints)
    traj_points = [
        (start_pos, pre_grasp_pos),
        (pre_grasp_pos, handle_pos),
        (handle_pos, pull_pos),
    ]
    traj_colors = [
        [0.5, 0.5, 1, 0.8],   # light blue: start -> pre-grasp
        [1, 0.8, 0, 0.8],     # orange: pre-grasp -> handle
        [0.2, 0.9, 0.2, 0.8], # light green: handle -> pull
    ]
    for seg_i, ((p0, p1), color) in enumerate(zip(traj_points, traj_colors)):
        n_dots = max(2, int(np.linalg.norm(p1 - p0) / 0.03))
        for di in range(1, n_dots):
            t = di / n_dots
            pt = p0 * (1 - t) + p1 * t
            add_marker(pt, color, radius=0.006, name=f"traj_{seg_i}_{di}")

    # Waypoint schedule: (label, target_pos, target_quat_wxyz, grip, hold_steps)
    # No drive_drawer — rely on physical contact between gripper and handle.
    waypoints = [
        ("00_start",       start_pos.copy(),  start_quat, 0.04, 400),
        ("01_reorient",    pre_grasp_pos,     grasp_quat, 0.04, 3000),
        ("02_at_handle",   handle_pos,        grasp_quat, 0.04, 1500),
        ("03_close_grip",  handle_pos,        grasp_quat, 0.00, 1500),
        ("04_pull",        pull_pos,          grasp_quat, 0.00, 5000),
        ("05_hold",        pull_pos,          grasp_quat, 0.00, 500),
        ("06_push_back",   handle_pos,        grasp_quat, 0.00, 3000),
        ("07_release",     handle_pos,        grasp_quat, 0.04, 500),
    ]

    frame_idx = 0
    prev_ctrl_pos = start_pos.copy()
    ctrl_quat = start_quat.copy()
    current_grip = 0.04
    target_qpos = init_qpos.copy()

    DT = 1/1000.0
    MAX_POS_SPEED = 0.5       # m/s end-effector linear speed
    MAX_GRIP_SPEED = 0.3      # m/s finger close speed
    VELOCITY_FACTOR = 1.0

    def render_frame(label):
        nonlocal frame_idx
        scene.update_render()
        fn = os.path.join(out_dir, f"frame_{frame_idx:04d}_{label}.png")
        save_rgb(cam, fn)
        frame_idx += 1

    render_frame("init")

    # Pre-compute IK targets for the pull trajectory to avoid teleporting
    # the robot during the grip/pull phases (which breaks contact).
    pull_ik_targets = {}
    n_pull_samples = 20
    for i in range(n_pull_samples + 1):
        t = i / n_pull_samples
        pos = handle_pos * (1 - t) + pull_pos * t
        q = compute_ik(panda, hand_link, arm_indices, pos, grasp_quat,
                       max_iter=50, tol=1e-4, lr=0.3)
        pull_ik_targets[i] = q.copy()

    # === Phase 2 prep: Door geometry & pre-computed arc IK ===
    hinge_xy = np.array([0.567, 0.042])
    door_handle_center_world = np.array([0.544, 0.373, 0.481])
    door_radius = np.linalg.norm(door_handle_center_world[:2] - hinge_xy)
    door_angle_0 = np.arctan2(door_handle_center_world[1] - hinge_xy[1],
                               door_handle_center_world[0] - hinge_xy[0])
    DOOR_OPEN_ANGLE = np.deg2rad(60)

    def door_handle_at_theta(theta):
        angle = door_angle_0 + theta
        return np.array([hinge_xy[0] + door_radius * np.cos(angle),
                         hinge_xy[1] + door_radius * np.sin(angle),
                         door_handle_center_world[2]])

    def door_hand_pos_at_theta(theta):
        hc = door_handle_at_theta(theta)
        return hc - FINGER_OFFSET * np.array([np.cos(theta), np.sin(theta), 0])

    def door_quat_at_theta(theta):
        # Fingers open in Y (tangential to arc) — handle is only 21mm in Y, grippable
        # hand_x = -Z, hand_y = [-st, ct, 0] (tangential), hand_z = [ct, st, 0] (radial)
        ct, st = np.cos(theta), np.sin(theta)
        mat = np.array([[0, -st, ct],
                        [0,  ct, st],
                        [-1,  0,  0]])
        q_xyzw = R.from_matrix(mat).as_quat()
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    door_quat_0 = door_quat_at_theta(0)
    door_handle_pos_0 = door_hand_pos_at_theta(0)
    door_pre_grasp = door_handle_pos_0 + np.array([-0.15, 0.0, 0.0])
    retract_pos = np.array([0.2, 0.28, 0.7])

    # === Phase 1: Pull top drawer ===
    for label, tgt_pos, tgt_quat, tgt_grip, hold_steps in waypoints:
        print(f"[wp] {label} -> pos={np.round(tgt_pos,3)}  grip={tgt_grip:.2f}")
        goal_pos = tgt_pos.copy()
        goal_quat = tgt_quat.copy()
        goal_grip = float(tgt_grip)
        gripping = tgt_grip < 0.01  # gripper is closed

        for step in range(hold_steps):
            # slew ctrl toward goal
            diff = goal_pos - prev_ctrl_pos
            dist = np.linalg.norm(diff)
            step_pos = VELOCITY_FACTOR * MAX_POS_SPEED * DT
            if dist > step_pos:
                prev_ctrl_pos = prev_ctrl_pos + diff / dist * step_pos
            else:
                prev_ctrl_pos = goal_pos.copy()

            diff_g = goal_grip - current_grip
            step_g = VELOCITY_FACTOR * MAX_GRIP_SPEED * DT
            if abs(diff_g) > step_g:
                current_grip += np.sign(diff_g) * step_g
            else:
                current_grip = goal_grip

            if step % 10 == 0:
                if gripping and label.startswith("04"):
                    # Use pre-computed IK for pull (no teleporting)
                    pull_range = np.linalg.norm(pull_pos - handle_pos)
                    traveled = np.linalg.norm(prev_ctrl_pos - handle_pos)
                    t = np.clip(traveled / pull_range, 0, 1)
                    idx = int(t * n_pull_samples)
                    target_qpos = pull_ik_targets[min(idx, n_pull_samples)].copy()
                elif gripping and label.startswith("06"):
                    # Push back: reverse the pull IK targets
                    push_range = np.linalg.norm(handle_pos - pull_pos)
                    traveled = np.linalg.norm(prev_ctrl_pos - pull_pos)
                    t = np.clip(traveled / push_range, 0, 1)
                    idx = n_pull_samples - int(t * n_pull_samples)
                    target_qpos = pull_ik_targets[max(idx, 0)].copy()
                elif not gripping:
                    # Safe to call IK when gripper is open
                    target_qpos = compute_ik(
                        panda, hand_link, arm_indices,
                        prev_ctrl_pos, goal_quat,
                        max_iter=20, tol=1e-3, lr=0.5)
                # When gripping but not pulling (close_grip, hold): keep last target

            for i, ji in enumerate(arm_indices):
                arm_joints[i].set_drive_target(float(target_qpos[ji]))
            for i, ji in enumerate(gripper_indices):
                gripper_joints[i].set_drive_target(float(current_grip))

            scene.step()

            # Sample ~30fps from 1kHz sim steps.
            if step % 33 == 0:
                render_frame(label)

        # one render at end of waypoint
        render_frame(label + "_end")
        q_drawer = cabinet.get_qpos()
        print(f"       drawer qpos = {np.round(q_drawer, 3)}")

    # === Phase 2: Open bottom door (multi-pull) ===
    print("\n=== Phase 2: Open bottom door ===")

    # Door damping=2 retains position after grip breaks; low friction
    door_joint = cabinet.get_active_joints()[0]  # joint_0 = door
    door_joint.set_drive_property(stiffness=0, damping=2)
    door_joint.set_friction(0.001)

    # Helper: run a sequence of waypoints with velocity-limited control
    def run_waypoints(wps):
        nonlocal prev_ctrl_pos, current_grip, target_qpos, frame_idx
        for label, tgt_pos, tgt_quat, tgt_grip, hold_steps in wps:
            print(f"[wp] {label} -> pos={np.round(tgt_pos,3)}  grip={tgt_grip:.2f}")
            goal_pos = tgt_pos.copy()
            goal_quat = tgt_quat.copy()
            goal_grip = float(tgt_grip)
            smooth_qpos = target_qpos.copy()  # for joint-level smoothing
            for step in range(hold_steps):
                diff = goal_pos - prev_ctrl_pos
                dist = np.linalg.norm(diff)
                step_pos = VELOCITY_FACTOR * MAX_POS_SPEED * DT
                if dist > step_pos:
                    prev_ctrl_pos = prev_ctrl_pos + diff / dist * step_pos
                else:
                    prev_ctrl_pos = goal_pos.copy()
                diff_g = goal_grip - current_grip
                step_g = VELOCITY_FACTOR * MAX_GRIP_SPEED * DT
                if abs(diff_g) > step_g:
                    current_grip += np.sign(diff_g) * step_g
                else:
                    current_grip = goal_grip
                if step % 10 == 0:
                    raw_ik = compute_ik(
                        panda, hand_link, arm_indices,
                        prev_ctrl_pos, goal_quat,
                        max_iter=30, tol=5e-4, lr=0.4)
                    # Blend IK result to avoid sudden jumps between solutions
                    smooth_qpos = smooth_qpos + 0.3 * (raw_ik - smooth_qpos)
                    target_qpos = smooth_qpos.copy()
                for i, ji in enumerate(arm_indices):
                    arm_joints[i].set_drive_target(float(target_qpos[ji]))
                for i, ji in enumerate(gripper_indices):
                    gripper_joints[i].set_drive_target(float(current_grip))
                scene.step()
                if step % 33 == 0:
                    render_frame(label)
            render_frame(label + "_end")
            print(f"       cabinet qpos = {np.round(cabinet.get_qpos(), 3)}")

    # Door trajectory markers: arc path from closed to open
    add_marker(door_handle_pos_0, [1, 0.2, 0.2, 1], radius=0.015, name="mk_door_handle")
    add_marker(door_pre_grasp,    [0.3, 0.3, 1, 1],  radius=0.015, name="mk_door_pre")
    n_arc_markers = 12
    for ai in range(n_arc_markers + 1):
        theta = DOOR_OPEN_ANGLE * ai / n_arc_markers
        pt = door_hand_pos_at_theta(theta)
        add_marker(pt, [1, 0.5, 0, 0.8], radius=0.006, name=f"arc_{ai}")

    # Initial approach to door handle
    run_waypoints([
        ("08_retract",     retract_pos,        door_quat_0,  0.04, 3000),
        ("09_door_pre",    door_pre_grasp,      door_quat_0,  0.04, 3000),
        ("10_at_door",     door_handle_pos_0,   door_quat_0,  0.04, 1500),
    ])

    # Multi-pull: grip, pull ~20°, release, re-approach with matching orientation
    PULL_ARC_DEG = 20.0
    N_PULL_ROUNDS = 5
    n_arc_samples = 40
    PULL_STEPS = 6000

    for rnd in range(N_PULL_ROUNDS):
        # Refresh door angle right before gripping
        door_angle_start = max(0.0, float(cabinet.get_qpos()[0]))
        arc_target = min(door_angle_start + np.deg2rad(PULL_ARC_DEG), DOOR_OPEN_ANGLE)
        grip_quat = door_quat_at_theta(door_angle_start)
        print(f"\n--- Pull round {rnd+1}: {np.degrees(door_angle_start):.1f}° -> {np.degrees(arc_target):.1f}° ---")

        # Grip position: use rotating offset to match hand orientation
        grip_pos = door_hand_pos_at_theta(door_angle_start)

        # Grip the handle with orientation matching current door angle
        run_waypoints([
            (f"R{rnd}_grip", grip_pos, grip_quat, 0.00, 1500),
        ])

        # Re-read actual angle after grip (door may have shifted)
        door_angle_start = max(0.0, float(cabinet.get_qpos()[0]))
        arc_target = min(door_angle_start + np.deg2rad(PULL_ARC_DEG), DOOR_OPEN_ANGLE)

        # Pre-compute IK for this pull arc from actual config (rotating orientation)
        actual_qpos = panda.get_qpos().copy()
        arc_ik = {}
        seed = actual_qpos.copy()
        for i in range(n_arc_samples + 1):
            t = i / n_arc_samples
            theta = door_angle_start + (arc_target - door_angle_start) * t
            pos = door_hand_pos_at_theta(theta)
            quat = door_quat_at_theta(theta)
            panda.set_qpos(seed)
            q = compute_ik(panda, hand_link, arm_indices, pos, quat,
                           max_iter=50, tol=1e-4, lr=0.3)
            arc_ik[i] = q.copy()
            seed = q.copy()
        panda.set_qpos(actual_qpos)

        # Pull along arc with time-based + angle-clamped indexing
        pull_label = f"R{rnd}_pull"
        print(f"[wp] {pull_label} -> arc pull {PULL_STEPS} steps")
        smooth_theta = door_angle_start  # EMA-smoothed target angle
        SMOOTH_ALPHA = 0.003             # smoothing factor (lower = smoother)
        prev_target_qpos = panda.get_qpos().copy()  # for joint-level smoothing
        QPOS_SMOOTH = 0.02               # joint target EMA
        for step in range(PULL_STEPS):
            t_time = step / PULL_STEPS
            theta_time = door_angle_start + (arc_target - door_angle_start) * t_time
            actual_angle = max(0.0, float(cabinet.get_qpos()[0]))
            max_lead = np.deg2rad(3)
            # Clamp: don't exceed actual+3°, don't go behind actual angle
            theta_raw = min(theta_time, actual_angle + max_lead, arc_target)
            theta_raw = max(theta_raw, actual_angle)
            # EMA + monotonic: never let target angle decrease (prevents
            # oscillation when grip slips and door bounces back)
            smooth_theta = smooth_theta + SMOOTH_ALPHA * (theta_raw - smooth_theta)
            smooth_theta = max(smooth_theta, door_angle_start)  # floor
            theta = np.clip(smooth_theta, door_angle_start, arc_target)
            t_arc = (theta - door_angle_start) / max(arc_target - door_angle_start, 1e-6)
            t_arc = np.clip(t_arc, 0, 1)
            # Linearly interpolate between adjacent IK samples instead of snapping
            fidx = t_arc * n_arc_samples
            idx_lo = int(fidx)
            idx_hi = min(idx_lo + 1, n_arc_samples)
            frac = fidx - idx_lo
            raw_qpos = (1 - frac) * arc_ik[idx_lo] + frac * arc_ik[idx_hi]
            # Joint-level EMA to smooth remaining jitter
            target_qpos = prev_target_qpos + QPOS_SMOOTH * (raw_qpos - prev_target_qpos)
            prev_target_qpos = target_qpos.copy()
            prev_ctrl_pos = door_hand_pos_at_theta(theta)

            for i, ji in enumerate(arm_indices):
                arm_joints[i].set_drive_target(float(target_qpos[ji]))
            for i, ji in enumerate(gripper_indices):
                gripper_joints[i].set_drive_target(0.0)
            scene.step()
            if step % 33 == 0:
                render_frame(pull_label)
            if step % 1000 == 0:
                print(f"  step {step}: door={np.degrees(actual_angle):.1f}° tgt={np.degrees(theta):.1f}°")

        render_frame(pull_label + "_end")
        door_now = float(cabinet.get_qpos()[0])
        print(f"  Pull round {rnd+1} done: door={np.degrees(door_now):.1f}°")

        # Release + retract
        release_pos = door_hand_pos_at_theta(max(0, door_now))
        release_quat = door_quat_at_theta(max(0, door_now))
        retract_mid = release_pos + np.array([-0.12, 0.0, 0.05])
        run_waypoints([
            (f"R{rnd}_release", release_pos,  release_quat, 0.04, 800),
            (f"R{rnd}_retract", retract_mid,  release_quat, 0.04, 1500),
        ])

        if door_now >= DOOR_OPEN_ANGLE - np.deg2rad(5):
            print(f"  Door sufficiently open ({np.degrees(door_now):.1f}°), stopping.")
            break

        # Pre-grip for next round: approach at current door angle with matching orientation
        next_angle = max(0.0, float(cabinet.get_qpos()[0]))
        next_quat = door_quat_at_theta(next_angle)
        next_grip = door_hand_pos_at_theta(next_angle)
        next_pre = next_grip + np.array([-0.12, 0.0, 0.0])
        run_waypoints([
            (f"R{rnd}_pre_next", next_pre,   next_quat, 0.04, 2000),
            (f"R{rnd}_at_next",  next_grip,  next_quat, 0.04, 1500),
        ])

    # Diagnostic: actual world pose of link_1 (drawer)
    link_1 = next(l for l in cabinet.get_links() if l.name == "link_1")
    print(f"[diag] link_1.pose.p at end = {np.round(link_1.entity.get_pose().p, 3)}")
    link_0 = next(l for l in cabinet.get_links() if l.name == "link_0")
    print(f"[diag] link_0.pose.p at end = {np.round(link_0.entity.get_pose().p, 3)}")
    print(f"[diag] final cabinet qpos = {np.round(cabinet.get_qpos(), 3)}")

    # Second camera angle for a clearer view of the drawer state
    side_cam = scene.add_camera('side', 1280, 960, np.deg2rad(55), 0.01, 10.0)
    side_cam.entity.set_pose(sapien_look_at(
        [0.1, 1.4, 1.0], [0.5, 0.1, 0.5]))
    scene.update_render()
    save_rgb(side_cam, os.path.join(out_dir, "final_side.png"))

    # Summary render
    render_frame("final")
    print(f"\n[OK] wrote {frame_idx} frames to {out_dir}")

    # stitch into mp4 with ffmpeg if available
    mp4 = os.path.join(base_dir, "out", "interact.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "30",
            "-pattern_type", "glob", "-i", os.path.join(out_dir, "frame_*.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=960:720",
            mp4
        ], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        print(f"[OK] video -> {mp4}")
    except Exception as e:
        print(f"[warn] ffmpeg not available or failed: {e}")


if __name__ == "__main__":
    main()
