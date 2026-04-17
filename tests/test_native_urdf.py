import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import xml.etree.ElementTree as ET

def run_native_urdf_final_v2():
    # 路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cabinet_root = os.path.join(base_dir, "40147")
    orig_urdf_path = os.path.join(cabinet_root, "mobility_modified_vhacd_Mesh_X_1_5.urdf")
    native_urdf_path = os.path.join(cabinet_root, "mujoco_direct_load.urdf")

    print(f"Generating clean URDF for native MuJoCo loading...")
    
    if not os.path.exists(orig_urdf_path):
        print(f"Error: URDF not found at {orig_urdf_path}")
        return

    # 使用 ElementTree 解析，比正则更安全
    tree = ET.parse(orig_urdf_path)
    root = tree.getroot()

    # 扫描磁盘确定文件位置
    def get_correct_rel_path(fname, is_collision=False):
        # 尝试寻找 VHACD 版本
        if is_collision:
            vhacd_name = fname.replace(".obj", "_vhacd.obj")
            for sub in ["textured_objs", "Translated_Mesh_X_1_5"]:
                if os.path.exists(os.path.join(cabinet_root, sub, vhacd_name)):
                    return f"{sub}/{vhacd_name}"
        
        # 寻找原始版本
        for sub in ["textured_objs", "Translated_Mesh_X_1_5"]:
            if os.path.exists(os.path.join(cabinet_root, sub, fname)):
                return f"{sub}/{fname}"
        return fname

    # 遍历所有 link 进行路径修复
    for link in root.findall(".//link"):
        # 修复 Visual
        for visual in link.findall("visual"):
            mesh = visual.find(".//mesh")
            if mesh is not None:
                orig_file = os.path.basename(mesh.get("filename"))
                mesh.set("filename", get_correct_rel_path(orig_file, is_collision=False))
        
        # 修复 Collision
        for collision in link.findall("collision"):
            mesh = collision.find(".//mesh")
            if mesh is not None:
                orig_file = os.path.basename(mesh.get("filename"))
                mesh.set("filename", get_correct_rel_path(orig_file, is_collision=True))

    # 保存新的 URDF
    tree.write(native_urdf_path)

    # 3. 直读 URDF 文件
    print(f"Loading URDF natively from: {native_urdf_path}")
    try:
        # 没有任何 assets 字典，纯靠文件系统解析
        model = mujoco.MjModel.from_xml_path(native_urdf_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Native loading failed: {e}")
        return

    # 4. 启动 Viewer 与演示
    print("Viewer opening. High-fidelity textures should now be visible.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 开启高级渲染选项
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SHADOW] = True
        
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
    run_native_urdf_final_v2()
