import mujoco
import os
import re

# 定义路径
urdf_path = "mobility.urdf"
xml_path = "cabinet.xml"
mesh_dir = "textured_objs"

def make_obj_3d(obj_bytes):
    """
    核心修复逻辑：检测0厚度的扁平面片，并赋予1毫米的微小厚度
    """
    text = obj_bytes.decode('utf-8')
    lines = text.split('\n')
    v_lines = []
    
    # 提取所有顶点
    for i, line in enumerate(lines):
        if line.startswith('v '):
            parts = line.strip().split()
            if len(parts) >= 4:
                v_lines.append((i, [float(parts[1]), float(parts[2]), float(parts[3])]))
                
    if len(v_lines) < 4:
        return obj_bytes
        
    # 计算 X, Y, Z 三个维度的极差（厚度）
    x = [v[1][0] for v in v_lines]
    y = [v[1][1] for v in v_lines]
    z = [v[1][2] for v in v_lines]
    ranges = [max(x)-min(x), max(y)-min(y), max(z)-min(z)]
    
    modified = False
    new_verts = [v[1][:] for v in v_lines]
    
    # 如果某个维度的厚度小于 1mm，强制将其拉伸 2mm
    for dim in range(3):
        if ranges[dim] < 1e-3:
            modified = True
            mid = len(v_lines) // 2
            # 一半顶点向外推 1mm，一半向内推 1mm，形成立体厚度
            for j in range(len(v_lines)):
                if j < mid:
                    new_verts[j][dim] -= 0.001
                else:
                    new_verts[j][dim] += 0.001
                    
    # 如果被修改过，重新写回 obj 格式
    if modified:
        for idx, (line_idx, _) in enumerate(v_lines):
            lines[line_idx] = f"v {new_verts[idx][0]:.6f} {new_verts[idx][1]:.6f} {new_verts[idx][2]:.6f}"
        return '\n'.join(lines).encode('utf-8')
        
    return obj_bytes

# 1. 构建 Assets 字典，并在加载时“清洗”所有具有致命缺陷的 3D 模型
assets = {}
if os.path.exists(mesh_dir):
    print("正在扫描并修复 0厚度 模型...")
    for file in os.listdir(mesh_dir):
        file_path = os.path.join(mesh_dir, file)
        if file.endswith('.obj'):
            with open(file_path, 'rb') as f:
                # 调用修复函数拦截处理
                assets[file] = make_obj_3d(f.read())
        elif file.endswith(('.mtl', '.png', '.jpg')):
            with open(file_path, 'rb') as f:
                assets[file] = f.read()

try:
    print(f"正在读取 {urdf_path} 并编译...")
    
    with open(urdf_path, 'r', encoding='utf-8') as f:
        urdf_content = f.read()
    
    # 2. 注入 discardvisual 确保视觉模型保留
    if "<mujoco>" not in urdf_content:
        injection = """\n  <mujoco>\n    <compiler discardvisual="false"/>\n  </mujoco>"""
        urdf_content = re.sub(r'(<robot[^>]*>)', r'\1' + injection, urdf_content, count=1)

    # 3. 交给 MuJoCo 编译
    model = mujoco.MjModel.from_xml_string(urdf_content, assets=assets)
    mujoco.mj_saveLastXML(xml_path, model)
    
    # 4. 修正 meshdir 路径
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'meshdir="' not in content:
        content = content.replace('<compiler', f'<compiler meshdir="{mesh_dir}/"')
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    print(f"\n✅ 转换彻底成功！已生成 {xml_path}")

except Exception as e:
    print(f"\n❌ 转换失败，报错信息: {e}")