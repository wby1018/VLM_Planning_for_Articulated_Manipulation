import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET
import re

def setup_ultimate_robust_textured():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cabinet_root = os.path.join(base_dir, "40147")
    urdf_path = os.path.join(cabinet_root, "mobility_modified_vhacd_Mesh_X_1_5.urdf")

    print("Executing Tetrahedron Padding Fix (Final Geometry Compatibility)...")
    
    if not os.path.exists(urdf_path):
        return

    # 1. 资产字典构建
    assets = {}
    
    # 图片库
    img_dir = os.path.join(cabinet_root, "images")
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                with open(os.path.join(img_dir, f), 'rb') as file:
                    assets[f] = file.read()

    # OBJ 网格：进行四面体注入和路径重定向
    for sub in ["textured_objs", "Translated_Mesh_X_1_5"]:
        d_p = os.path.join(cabinet_root, sub)
        if not os.path.exists(d_p): continue
        for f in os.listdir(d_p):
            fp = os.path.join(d_p, f)
            if f.lower().endswith(".obj"):
                with open(fp, 'r', errors='ignore') as file:
                    obj_content = file.read()
                
                # 【终极补全】：在文件末尾注入 4 个微小的、不共面的顶点
                # 这会让 MuJoCo 的 Qhull 算法永远成功且判定体积 > 0
                tetra_padding = (
                    "\nv 0.000000 0.000000 0.000000"
                    "\nv 0.000001 0.000000 0.000000"
                    "\nv 0.000000 0.000001 0.000000"
                    "\nv 0.000000 0.000000 0.000001\n"
                )
                obj_content += tetra_padding
                
                # MTL 引用偏平化
                obj_content = re.sub(r'mtllib\s+([^/]+/)*', 'mtllib ', obj_content)
                assets[f] = obj_content.encode()
                
            elif f.lower().endswith(".mtl"):
                with open(fp, 'r', errors='ignore') as file:
                    mtl_data = file.read()
                # 贴图引用扁平化
                assets[f] = re.sub(r'map_(\w+)\s+([^/]+/)*', r'map_\1 ', mtl_data).encode()

    # 2. 修改 URDF：实现文件路径扁平化
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", os.path.basename(mesh.get("filename")))

    urdf_content = ET.tostring(root, encoding='unicode')
    
    # 3. 直读 URDF
    print("Loading Model with Native Wood Textures and Geometric Padding...")
    try:
        model = mujoco.MjModel.from_xml_string(urdf_content, assets=assets)
        data = mujoco.MjData(model)
        
        print("\nSUCCESS: Model initialized. Rendering Textured Viewer...")
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
        print(f"LOADING ATTEMPT FAILED: {e}")

if __name__ == "__main__":
    setup_ultimate_robust_textured()
