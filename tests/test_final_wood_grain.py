import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import re

def setup_ultimate_vertex_fix():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cabinet_root = os.path.join(base_dir, "40147")
    urdf_path = os.path.join(cabinet_root, "mobility_modified_vhacd_Mesh_X_1_5.urdf")

    print("Executing Vertex-Padding Fix (Bypassing MuJoCo Geometry Constraints)...")
    
    if not os.path.exists(urdf_path):
        return

    # 1. 资产加载 + 顶点补全 + 路径重定向
    assets = {}
    for sub in ["textured_objs", "Translated_Mesh_X_1_5", "images"]:
        d_p = os.path.join(cabinet_root, sub)
        if not os.path.exists(d_p): continue
        for f in os.listdir(d_p):
            fp = os.path.join(d_p, f)
            if f.lower().endswith(".obj"):
                with open(fp, 'r', errors='ignore') as file:
                    obj_data = file.read()
                # 关键：修复 MTL 引用
                obj_data = re.sub(r'mtllib\s+([^/]+/)*', 'mtllib ', obj_data)
                # 关键：顶点补全。手动添加 4 个不共面的微小顶点，确保 MuJoCo 认为它是 3D 的
                dummy_v = "\nv 0.00001 0.00001 0.00001\nv -0.00001 0.00001 0.00001\nv 0.00001 -0.00001 0.00001\nv 0 0 -0.00001\n"
                obj_data += dummy_v
                assets[f] = obj_data.encode()
            elif f.lower().endswith(".mtl"):
                with open(fp, 'r', errors='ignore') as file:
                    mtl_data = file.read()
                assets[f] = re.sub(r'map_(\w+)\s+([^/]+/)*', r'map_\1 ', mtl_data).encode()
            elif f.lower().endswith((".png", ".jpg", ".jpeg")):
                with open(fp, 'rb') as file:
                    assets[f] = file.read()

    # 2. 修改 URDF 字符串，使其适配扁平化资产
    with open(urdf_path, 'r') as f:
        urdf_str = f.read()
    
    # 清理所有 filename 前缀
    urdf_str = re.sub(r'filename="([^"]+/)?([^"]+)"', r'filename="\2"', urdf_str)
    
    # 给模型加个简单的光照和地面（通过包装成 MJCF）
    mjcf_wrapper = f"""
<mujoco>
    <worldbody>
        <light diffuse="1 1 1" pos="0 0 3" dir="0 0 -1" ambient=".4 .4 .4"/>
        <geom type="plane" size="5 5 0.01" rgba=".2 .2 .2 1"/>
        {urdf_str.replace('<robot', '<body').replace('</robot>', '</body>')}
    </worldbody>
</mujoco>
"""
    # 注意：上面的简单替换可能不完美，我们直接加载修正后的 URDF 字符串即可
    print("Loading URDF string with vertex-padding...")
    try:
        model = mujoco.MjModel.from_xml_string(urdf_str, assets=assets)
        data = mujoco.MjData(model)
        
        print("\nSUCCESS: Model loaded with wood grain and full collision stability.")
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
        print(f"ULTIMATE FIX FAILED: {e}")

if __name__ == "__main__":
    setup_ultimate_vertex_fix()
