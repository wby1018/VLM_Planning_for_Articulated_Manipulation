#!/usr/bin/env python3
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    # 路径配置
    workspace = Path("/home/wby/active_vision/vlm_based")
    data_dir = workspace / "record_pull_arc_with_joint_0/rgb"
    output_video_path = "/home/wby/active_vision/vlm_based/cotracker_result.mp4"
    
    # 1. 加载图像序列
    img_paths = sorted(data_dir.glob("*.png"))
    if not img_paths:
        print(f"Error: 在 {data_dir} 没找到图像")
        return
    
    print(f"加载 {len(img_paths)} 帧图像...")
    frames = []
    for p in img_paths[:239]:  # 限制前 100 帧演示
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    
    # 转为 tensor [B, T, 3, H, W]
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float()
    
    # 2. 初始化 CoTracker3 Online
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试清理 torch hub 缓存后重试...")
        return

    video = video.to(device)
    B, T, C, H, W = video.shape
    
    # 3. 初始化跟踪点 (网格)
    grid_size = 20
    # cotracker3_online 可以自动生成网格，或者传入 queries
    # 我们先调用一次带 is_first_step=True 的初始化
    # video_chunk 建议长度为 cotracker.step * 2
    step = cotracker.step
    print(f"Model step size: {step}")
    
    # 4. 在线跟踪循环 + 实时可视化
    cv2.namedWindow("CoTracker3 Online", cv2.WINDOW_NORMAL)
    
    # 全局轨道引用 (用于变色判定)
    ref_points = None

    def draw_and_show(all_tracks, all_vis, start_idx, num_frames_to_show):
        nonlocal ref_points
        if ref_points is None:
            ref_points = all_tracks[0, 0].cpu().numpy()

        for t in range(start_idx, start_idx + num_frames_to_show):
            if t >= all_tracks.shape[1]: 
                break
            
            img = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
            curr_tracks = all_tracks[0, t].cpu().numpy()
            curr_vis    = all_vis[0, t].cpu().numpy()
            
            # 方案优化：先计算所有点的颜色，减少循环内逻辑
            dists = np.linalg.norm(curr_tracks - ref_points, axis=1)
            is_active = dists > 5.0
            
            for n in range(curr_tracks.shape[0]):
                if curr_vis[n] > 0.5:
                    active = is_active[n]
                    color = (0, 0, 255) if active else (0, 255, 0)
                    
                    p_curr = (int(curr_tracks[n, 0]), int(curr_tracks[n, 1]))
                    cv2.circle(img, p_curr, 2, color, -1)
                    
                    # 性能核心优化：只给红点（活跃点）画轨迹线
                    if active:
                        for prev_t in range(max(0, t-10), t):
                            p1 = all_tracks[0, prev_t, n].cpu().numpy()
                            p2 = all_tracks[0, prev_t+1, n].cpu().numpy()
                            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)

            cv2.imshow("CoTracker3 Online", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return True
        return False

    # 循环控制
    step = cotracker.step
    window_len = step * 2  
    
    for ind in tqdm(range(0, T - window_len, step), desc="Tracking"):
        video_chunk = video[:, ind : ind + window_len]
        is_first = (ind == 0)
        
        with torch.no_grad():
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video_chunk, 
                is_first_step=is_first, 
                grid_size=30 # 适当降低密度，换取流畅度
            )
            
            if pred_tracks is not None:
                show_start = ind if not is_first else 0
                show_num   = step if not is_first else window_len
                if draw_and_show(pred_tracks, pred_visibility, show_start, show_num):
                    break

    print("\n处理完毕。")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
