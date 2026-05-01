#!/usr/bin/env python3
import os
import torch
import cv2
import numpy as np
import kornia as K
from kornia.feature import LoFTR
from pathlib import Path

def main():
    # 路径配置
    workspace = Path("/home/wby/active_vision/vlm_based")
    data_dir = workspace / "record_pull_arc_with_joint_0/rgb"
    output_path = "/home/wby/active_vision/vlm_based/loftr_matches.png"
    
    # 选取第 0 帧和第 30 帧作为对比
    img0_path = str(data_dir / "rgb_000100.png")
    img1_path = str(data_dir / "rgb_000150.png")
    
    if not os.path.exists(img0_path) or not os.path.exists(img1_path):
        print(f"Error: 找不到图像文件 {img0_path} 或 {img1_path}")
        return

    # 1. 加载图像并预处理
    img0_raw = cv2.imread(img0_path)
    img1_raw = cv2.imread(img1_path)
    
    # LoFTR 期望灰度图，且尺寸通常需要是 8 的倍数
    # 为了演示，我们将图像缩放到 640x480 以提高速度并符合模型预期
    target_size = (640, 480)
    img0_resized = cv2.resize(img0_raw, target_size)
    img1_resized = cv2.resize(img1_raw, target_size)
    
    # 转为 tensor [B, C, H, W]
    t0 = K.image_to_tensor(img0_resized, keepdim=False).float() / 255.
    t1 = K.image_to_tensor(img1_resized, keepdim=False).float() / 255.
    
    # 转为灰度
    t0_gray = K.color.rgb_to_grayscale(t0)
    t1_gray = K.color.rgb_to_grayscale(t1)
    
    # 2. 初始化 LoFTR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 使用预训练模型 (outdoor 或 indoor)
    matcher = LoFTR(pretrained='outdoor').to(device).eval()
    
    t0_gray = t0_gray.to(device)
    t1_gray = t1_gray.to(device)
    
    # 3. 推理
    input_dict = {"image0": t0_gray, "image1": t1_gray}
    with torch.no_grad():
        correspondences = matcher(input_dict)
    
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    mconf = correspondences['confidence'].cpu().numpy()
    
    # 4. 过滤并可视化
    # 只取置信度较高的点
    conf_mask = mconf > 0.5
    mkpts0 = mkpts0[conf_mask]
    mkpts1 = mkpts1[conf_mask]
    
    print(f"找到 {len(mkpts0)} 个匹配点")
    
    # 手动画匹配线
    h0, w0 = img0_raw.shape[:2]
    h1, w1 = img1_raw.shape[:2]
    
    # 因为推理是在缩放后的图上做的，需要将坐标恢复到原图尺寸
    scale_x = w0 / target_size[0]
    scale_y = h0 / target_size[1]
    mkpts0[:, 0] *= scale_x; mkpts0[:, 1] *= scale_y
    mkpts1[:, 0] *= scale_x; mkpts1[:, 1] *= scale_y
    
    # 拼接图像展示
    canvas = np.hstack([img0_raw, img1_raw])
    
    # 随机选 50 个点画线，避免太乱
    indices = np.arange(len(mkpts0))
    if len(indices) > 500:
        indices = np.random.choice(indices, 900, replace=False)
        
    for i in indices:
        p1 = tuple(mkpts0[i].astype(int))
        p2 = tuple((mkpts1[i] + [w0, 0]).astype(int))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, p1, 4, color, -1)
        cv2.circle(canvas, p2, 4, color, -1)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        
    cv2.imwrite(output_path, canvas)
    print(f"结果已保存至 {output_path}")

if __name__ == "__main__":
    main()
