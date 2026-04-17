import os
import numpy as np
import sapien
import sapien.physx as physx
import time

def setup_sapien_cabinet():
    # 1. 路径准备
    base_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(base_dir, "40147", "mobility.urdf")
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF not found at {urdf_path}")
        return

    # 2. 初始化 SAPIEN 场景
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    
    # 添加地面和光照
    scene.add_ground(altitude=0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [1, 1, 1], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)

    # 3. 加载柜子 URDF
    # SAPIEN 的 URDFLoader 非常强大，会自动处理相对路径和贴图
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    
    print(f"Loading URDF from: {urdf_path}")
    cabinet = loader.load(urdf_path)
    
    if cabinet is None:
        print("Failed to load cabinet.")
        return

    cabinet.set_pose(sapien.Pose(p=[0, 0, 0]))
    
    # 4. 获取关节信息
    active_joints = cabinet.get_active_joints()
    print(f"\nCabinet Loaded Success!")
    print(f"Active Joints: {len(active_joints)}")
    for i, j in enumerate(active_joints):
        # SAPIEN 3.x 关节属性访问略有不同
        print(f"  [{i}] Name: {j.name}, Type: {j.type}, Limits: {j.limit}")

    # 5. 打开可视化窗口 (SAPIEN 3.0 标准用法)
    from sapien.utils import Viewer
    viewer = Viewer()
    viewer.set_scene(scene)
    
    # 设置相机位置
    viewer.set_camera_xyz(x=1.2, y=1.2, z=1.2)
    viewer.set_camera_rpy(r=0, p=-0.4, y=3.9)
    
    print("\nSimulation Running...")
    print("Close the window to stop.")

    # 6. 仿真主循环
    while not viewer.closed:
        # 让关节动起来，测试活动性
        t = time.time()
        for j in active_joints:
            q_limit = j.limit
            center = (q_limit[0][0] + q_limit[0][1]) / 2.0
            extent = (q_limit[0][1] - q_limit[0][0]) / 2.0
            target = center + extent * np.sin(t * 1.5)
            # SAPIEN 3.x Drive 设置
            j.set_drive_target(target)
            j.set_drive_property(stiffness=1000, damping=100)

        # 步进物理和渲染
        scene.step()
        scene.update_render()
        viewer.render()
        
    print("Simulator closed.")

if __name__ == "__main__":
    setup_sapien_cabinet()
