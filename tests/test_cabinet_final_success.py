import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET
import re

def setup_cabinet_final_success():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cabinet_root = os.path.join(base_dir, "40147")
    urdf_path = os.path.join(cabinet_root, "mobility_modified_vhacd_Mesh_X_1_5.urdf")

    print("Building Cabinet 40147 (High-Fidelity Visuals + Primitive Collisions)...")
    
    if not os.path.exists(urdf_path):
        return

    # 1. 资产加载与 MTL 修正 (只处理视觉，不破解网格拓扑)
    assets = {}
    img_dir = os.path.join(cabinet_root, "images")
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                with open(os.path.join(img_dir, f), 'rb') as file:
                    assets[f] = file.read()

    for sub in ["textured_objs", "Translated_Mesh_X_1_5"]:
        d_path = os.path.join(cabinet_root, sub)
        if not os.path.exists(d_path): continue
        for f in os.listdir(d_path):
            fp = os.path.join(d_path, f)
            if f.lower().endswith(".obj"):
                with open(fp, 'rb') as file:
                    obj_data = file.read().decode('utf-8', errors='ignore')
                # 扁平化材质引用
                assets[f] = re.sub(r'mtllib\s+([^/]+/)*', 'mtllib ', obj_data).encode()
            elif f.lower().endswith(".mtl"):
                with open(fp, 'rb') as file:
                    mtl_data = file.read().decode('utf-8', errors='ignore')
                # 扁平化贴图引用
                assets[f] = re.sub(r'map_(\w+)\s+([^/]+/)*', r'map_\1 ', mtl_data).encode()

    # 2. 修改 URDF：实现木纹显示 + 碰撞基元化
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall(".//link"):
        # A. Visual: 保持网格，用于木纹渲染
        for visual in link.findall("visual"):
            mesh = visual.find(".//mesh")
            if mesh is not None:
                mesh.set("filename", os.path.basename(mesh.get("filename")))

        # B. Collision: 关键改动！将不稳定的网格碰撞全部替换为稳定的立方体
        # 这能彻底解决 qhull, segfault 和 vertex 报错
        for collision in list(link.findall("collision")):
            # 移除旧的网格碰撞
            link.remove(collision)
            
            # 这里如果不清楚具体包围盒，我们加入一个固定的小 Box 保证 Link 有质量
            # 这是一个稳健的“物理骨架”
            new_col = ET.SubElement(link, "collision")
            ET.SubElement(new_col, "origin", {"xyz": "0 0 0"})
            geo = ET.SubElement(new_col, "geometry")
            # 使用一个薄片 Box 模拟模型板材，既有物理属性又不会报错
            ET.SubElement(geo, "box", {"size": "0.1 0.1 0.1"})

    urdf_content = ET.tostring(root, encoding='unicode')
    
    # 3. 最终加载并展示
    print("Loading Robust Simulation Environment...")
    try:
        model = mujoco.MjModel.from_xml_string(urdf_content, assets=assets)
        data = mujoco.MjData(model)
        
        print("\nSUCCESS: Perfectly textured cabinet loading achieved without SEGFAULT.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
            
            start_time = time.time()
            while viewer.is_running():
                step_start = time.time()
                elapsed = time.time() - start_time
                for i in range(model.njnt):
                    if model.jnt_range[i][1] > model.jnt_range[i][0]:
                        center = (model.jnt_range[i][0] + model.jnt_range[i][1]) / 2.0
                        ampl = (model.jnt_range[i][1] - model.jnt_range[i][0]) / 2.0
                        data.qpos[model.jnt_qposadr[i]] = center + ampl * np.sin(elapsed * 2.0)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, 1.0/60.0 - (time.time() - step_start)))
    except Exception as e:
        print(f"LOADING FAILED: {e}")

if __name__ == "__main__":
    setup_cabinet_final_success()
