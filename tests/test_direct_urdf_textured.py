import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET
import re

def run_textured_direct_urdf():
    # 路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cabinet_root = os.path.join(base_dir, "40147")
    urdf_path = os.path.join(cabinet_root, "mobility_modified_vhacd_Mesh_X_1_5.urdf")

    if not os.path.exists(urdf_path):
        print(f"Error: URDF not found at {urdf_path}")
        return

    print("Building flattened asset dictionary with material redirection...")
    assets = {}
    
    # 1. 加载所有贴图 ( images/ )
    img_dir = os.path.join(cabinet_root, "images")
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                with open(os.path.join(img_dir, f), 'rb') as file:
                    assets[f] = file.read()

    # 2. 加载网格和材质描述
    for sub in ["textured_objs", "Translated_Mesh_X_1_5"]:
        d_path = os.path.join(cabinet_root, sub)
        if os.path.exists(d_path):
            for f in os.listdir(d_path):
                f_path = os.path.join(d_path, f)
                if f.lower().endswith(".obj"):
                    with open(f_path, 'r') as file:
                        obj_data = file.read()
                    # 修正 OBJ 内部对 MTL 的引用：剥离路径
                    fixed_obj = re.sub(r'mtllib\s+([^/]+/)*', 'mtllib ', obj_data)
                    assets[f] = fixed_obj.encode()
                elif f.lower().endswith(".mtl"):
                    with open(f_path, 'r') as file:
                        mtl_data = file.read()
                    # 修正 MTL 内部对贴图的引用：剥离 ../images/ 等路径
                    fixed_mtl = re.sub(r'map_(\w+)\s+([^/]+/)*', r'map_\1 ', mtl_data)
                    assets[f] = fixed_mtl.encode()

    # 3. 读取并深度重构 URDF 字符串
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall(".//link"):
        # 视觉：用原始 OBJ + 纹理
        for visual in link.findall("visual"):
            mesh_tag = visual.find(".//mesh")
            if mesh_tag is not None:
                fname = os.path.basename(mesh_tag.get("filename"))
                mesh_tag.set("filename", fname)
        
        # 碰撞：用 VHACD 保证稳定
        for collision in link.findall("collision"):
            mesh_tag = collision.find(".//mesh")
            if mesh_tag is not None:
                fname = os.path.basename(mesh_tag.get("filename"))
                vhacd_f = fname.replace(".obj", "_vhacd.obj")
                # 只有资产库里确实有 vhacd 才换，否则用原样（如修改过的把手）
                mesh_tag.set("filename", vhacd_f if vhacd_f in assets else fname)

    urdf_string = ET.tostring(root, encoding='unicode')

    # 4. 直读入 MuJoCo
    print("Directly loading URDF into MuJoCo memory...")
    try:
        model = mujoco.MjModel.from_xml_string(urdf_string, assets=assets)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Loading failed: {e}")
        return

    # 5. 启动 Viewer 与演示
    print("Viewer opening. Wood textures should now be loaded from original OBJs.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
        
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            elapsed = time.time() - start_time
            # 关节运动
            for i in range(model.njnt):
                if model.jnt_range[i][1] > model.jnt_range[i][0]:
                    center = (model.jnt_range[i][0] + model.jnt_range[i][1]) / 2.0
                    ampl = (model.jnt_range[i][1] - model.jnt_range[i][0]) / 2.0
                    data.qpos[model.jnt_qposadr[i]] = center + ampl * np.sin(elapsed * 2.5)
            
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, 1.0/60.0 - (time.time() - step_start)))

if __name__ == "__main__":
    run_textured_direct_urdf()
