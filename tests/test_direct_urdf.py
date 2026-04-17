import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import re

def run_pure_direct_urdf():
    # 路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 40147 文件夹的根路径
    cabinet_root = os.path.join(base_dir, "40147")
    urdf_path = os.path.join(cabinet_root, "mobility_modified_vhacd_Mesh_X_1_5.urdf")

    if not os.path.exists(urdf_path):
        print(f"Error: URDF not found at {urdf_path}")
        return

    # 1. 构建全路径资产字典 (不修改文件内容，完全模拟文件系统的相对关系)
    assets = {}
    
    # 我们需要模拟 URDF 中出现的 "data/dataset/40147/" 这一层级
    # 这样 MuJoCo 的路径解析器就能根据相对关系找到所有东西
    prefix = "data/dataset/40147/"

    def add_to_assets(relative_dir):
        full_path = os.path.join(cabinet_root, relative_dir)
        if os.path.exists(full_path):
            for f in os.listdir(full_path):
                f_relative = os.path.join(relative_dir, f).replace("\\", "/")
                # 存入两个版本的 Key：一个是 URDF 带前缀的，一个是纯相对的
                with open(os.path.join(full_path, f), 'rb') as file:
                    content = file.read()
                    assets[f_relative] = content
                    assets[prefix + f_relative] = content
                    # 为了兼容 MTL 内部的 "../images/" 这种跨目录引用
                    if relative_dir == "images":
                        assets[prefix + "textured_objs/../images/" + f] = content
                        assets["textured_objs/../images/" + f] = content

    print("Mapping assets with high-fidelity paths...")
    add_to_assets("textured_objs")
    add_to_assets("Translated_Mesh_X_1_5")
    add_to_assets("images")

    # 2. 读取原始 URDF (不做任何字符串修改，保持原汁原味)
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    # 3. 直读 URDF
    print("Directly loading URDF with deep path resolution...")
    try:
        # 传入 assets 字典，MuJoCo 会根据 URDF 中的 filename="..." 自行查找
        model = mujoco.MjModel.from_xml_string(urdf_content, assets=assets)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Pure direct loading failed: {e}")
        # 如果报错 qhull，说明必须得处理 collision 转换，这里我们沿用之前的 smart fix
        print("Attempting with VHACD-Collision fix...")
        fixed_content = re.sub(r'<collision>.*?</collision>', 
                               lambda m: m.group(0).replace(".obj", "_vhacd.obj"), 
                               urdf_content, flags=re.DOTALL)
        model = mujoco.MjModel.from_xml_string(fixed_content, assets=assets)
        data = mujoco.MjData(model)

    # 4. 可视化与环境补全 (灯光对于看清纹理至关重要)
    # 我们通过修改编译后的 model 来动态添加灯光，而不是修改 XML
    print("Viewer opening. Texture visualization enabled...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 强制开启纹理渲染
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
        
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            elapsed = time.time() - start_time
            
            # 关节演练
            for i in range(model.njnt):
                if model.jnt_range[i][1] > model.jnt_range[i][0]:
                    center = (model.jnt_range[i][0] + model.jnt_range[i][1]) / 2.0
                    ampl = (model.jnt_range[i][1] - model.jnt_range[i][0]) / 2.0
                    data.qpos[model.jnt_qposadr[i]] = center + ampl * np.sin(elapsed * 2.0)
            
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, 1.0/60.0 - (time.time() - step_start)))

if __name__ == "__main__":
    run_pure_direct_urdf()
