import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET
import re

def setup_cabinet_simulation():
    # 1. 路径准备
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cabinet_dir = os.path.join(base_dir, "40147")
    urdf_path = os.path.join(cabinet_dir, "mobility_modified_vhacd_Mesh_X_1_5.urdf")
    output_xml = os.path.join(cabinet_dir, "cabinet_fixed.xml")

    print(f"Loading Cabinet 40147 with Forced Texture Redirection...")
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF not found at {urdf_path}")
        return

    # 2. 深度加载资产并修复材质引用
    assets = {}
    
    # 加载贴图 (扁平化存储，不带任何路径前缀)
    images_dir = os.path.join(cabinet_dir, "images")
    if os.path.exists(images_dir):
        for f in os.listdir(images_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                with open(os.path.join(images_dir, f), 'rb') as file:
                    assets[f] = file.read()

    # 加载网格和材质文件，并修复 MTL 内部的贴图引用
    search_dirs = ["textured_objs", "Translated_Mesh_X_1_5"]
    for d in search_dirs:
        full_d = os.path.join(cabinet_dir, d)
        if os.path.exists(full_d):
            for f in os.listdir(full_d):
                if f.lower().endswith(".obj"):
                    with open(os.path.join(full_d, f), 'rb') as file:
                        assets[f] = file.read()
                elif f.lower().endswith(".mtl"):
                    with open(os.path.join(full_d, f), 'r') as file:
                        mtl_content = file.read()
                    # 关键修复：将 MTL 里的 ../images/texture_X.jpg 替换为 texture_X.jpg
                    # 这样它就能在 flattened 的 assets 字典里找到贴图
                    fixed_mtl = re.sub(r'map_(\w+)\s+(?:\.\./)?(?:images/)?([\w\.]+)', r'map_\1 \2', mtl_content)
                    assets[f] = fixed_mtl.encode()

    # 3. 解析并修改 URDF (分离 Visual 和 Collision)
    tree_urdf = ET.parse(urdf_path)
    root_urdf = tree_urdf.getroot()

    for link in root_urdf.findall(".//link"):
        # Visual 保持原始，启用纹理
        for visual in link.findall("visual"):
            mesh_tag = visual.find(".//mesh")
            if mesh_tag is not None:
                mesh_tag.set("filename", os.path.basename(mesh_tag.get("filename")))

        # Collision 切换为 VHACD，保证物理稳定
        for collision in link.findall("collision"):
            mesh_tag = collision.find(".//mesh")
            if mesh_tag is not None:
                fname = os.path.basename(mesh_tag.get("filename"))
                vhacd_f = fname.replace(".obj", "_vhacd.obj")
                if vhacd_f in assets:
                    mesh_tag.set("filename", vhacd_f)
                else:
                    mesh_tag.set("filename", fname)

    # 4. 转换到 MJCF
    fixed_urdf_str = ET.tostring(root_urdf, encoding='unicode')
    try:
        model = mujoco.MjModel.from_xml_string(fixed_urdf_str, assets=assets)
        temp_xml = "temp_raw.xml"
        mujoco.mj_saveLastXML(temp_xml, model)
        
        tree = ET.parse(temp_xml)
        root = tree.getroot()
        
        # 增加环境渲染细节
        worldbody = root.find('worldbody')
        if worldbody is not None:
            # 采用纹理地面
            asset_tag = root.find('asset')
            if asset_tag is None: asset_tag = ET.SubElement(root, 'asset')
            ET.SubElement(asset_tag, 'texture', {'name': 'grid', 'type': '2d', 'builtin': 'checker', 'rgb1': '.1 .1 .1', 'rgb2': '.2 .2 .2', 'width': '512', 'height': '512'})
            ET.SubElement(asset_tag, 'material', {'name': 'grid', 'texture': 'grid', 'texrepeat': '5 5'})
            
            ET.SubElement(worldbody, 'geom', {'name': 'floor', 'type': 'plane', 'pos': '0 0 0', 'size': '5 5 0.05', 'material': 'grid'})
            ET.SubElement(worldbody, 'light', {'diffuse': '.8 .8 .8', 'pos': '0 0 5', 'dir': '0 0 -1', 'castshadow': 'true', 'ambient': '.3 .3 .3'})
        
        # 渲染器参数微调：增强阴影和纹理清晰度
        visual = root.find('visual')
        if visual is None: visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'quality', {'shadowsize': '4096'})
        
        tree.write(output_xml)
        if os.path.exists(temp_xml): os.remove(temp_xml)
        print(f"Successfully generated Wood-Textured MJCF: {output_xml}")

    except Exception as e:
        print(f"Error during conversion: {e}")
        return

    # 5. 最终加载运行 (强制显示纹理)
    model = mujoco.MjModel.from_xml_path(output_xml, assets=assets)
    data = mujoco.MjData(model)

    print("\nStarting Viewer (Visual: Original Textures | Collision: VHACD)...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
        
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            elapsed = time.time() - start_time
            # 演示运动
            for j_id in range(model.njnt):
                jnt_range = model.jnt_range[j_id]
                if jnt_range[1] > jnt_range[0]:
                    center = (jnt_range[0] + jnt_range[1]) / 2.0
                    ampl = (jnt_range[1] - jnt_range[0]) / 2.0
                    data.qpos[model.jnt_qposadr[j_id]] = center + ampl * np.sin(elapsed * 1.5)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, 1.0/60.0 - (time.time() - step_start)))

if __name__ == "__main__":
    setup_cabinet_simulation()
