import os
import numpy as np
import sapien
import sapien.physx as physx
import time

def setup_sapien_cabinet_with_panda_v3():
    # 1. 路径准备
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cabinet_model_id = "46230"
    cabinet_urdf = os.path.join(base_dir, cabinet_model_id, "mobility.urdf")
    panda_urdf = "/home/wby/active_vision/ManiSkill/mani_skill/assets/robots/panda/panda_v2.urdf"

    if not os.path.exists(cabinet_urdf) or not os.path.exists(panda_urdf):
        print("Error: URDF paths not found.")
        return

    # 2. 初始化 SAPIEN 场景
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    
    scene.add_ground(altitude=0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [1, 1, 1], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)

    # 3. 加载柜子和机械臂
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    
    print(f"Loading Cabinet {cabinet_model_id}...")
    cabinet = loader.load(cabinet_urdf)
    # 抬高柜子，防止埋入地下
    cabinet.set_pose(sapien.Pose(p=[0.4, 0, 0.82])) 
    
    print(f"Loading Franka Panda...")
    panda = loader.load(panda_urdf)
    # 调整位置，使其面对柜子
    panda.set_pose(sapien.Pose(p=[-0.8, 0, 0])) 
    
    # 4. 获取所有活动关节
    cabinet_joints = cabinet.get_active_joints()
    panda_joints = panda.get_active_joints()
    all_joints = cabinet_joints + panda_joints
    
    print(f"\nJoints found: {len(all_joints)} (Cabinet: {len(cabinet_joints)}, Panda: {len(panda_joints)})")

    # 5. 可视化窗口
    from sapien.utils import Viewer
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=1.8, y=1.2, z=1.5)
    viewer.set_camera_rpy(r=0, p=-0.4, y=3.8)
    
    print("\nAnimation Active: Every joint is moving through its full range.")

    # 6. 仿真主循环
    while not viewer.closed:
        t = time.time()
        for j in all_joints:
            q_limit = j.limit
            if q_limit is not None and len(q_limit) > 0:
                low, high = q_limit[0]
                if low > -1e3 and high < 1e3:
                    center = (low + high) / 2.0
                    extent = (high - low) / 2.0
                    
                    # 使用全行程 (0.95倍) 摆动，确保动作明显
                    speed = 2.0 if "panda" in j.name else 1.2
                    target = center + extent * 0.95 * np.sin(t * speed)
                    
                    j.set_drive_target(target)
                    j.set_drive_property(stiffness=1500, damping=150)

        scene.step()
        scene.update_render()
        viewer.render()

if __name__ == "__main__":
    setup_sapien_cabinet_with_panda_v3()
