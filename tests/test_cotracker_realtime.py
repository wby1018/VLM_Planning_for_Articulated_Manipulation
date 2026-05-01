#!/usr/bin/env python3
import os
import torch
import cv2
import numpy as np
import threading
import queue
import time
from pathlib import Path

class RealTimeCoTracker:
    def __init__(self, data_dir, grid_size=10, device="cuda"):
        self.data_dir = Path(data_dir)
        self.grid_size = grid_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print("正在加载 CoTracker3 Online 模型...")
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(self.device).eval()
        self.step = self.model.step # 默认应该是 8
        
        self.frame_queue = queue.Queue(maxsize=200)
        self.results_lock = threading.Lock()
        
        self.latest_frame = None
        self.latest_tracks = None
        self.latest_vis = None
        self.ref_points = None
        self.is_running = True
        self.roi = None
        
    def select_roi(self, first_frame):
        window_name = "Select ROI - Press SPACE/ENTER to Confirm"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(window_name, first_frame, False)
        cv2.destroyWindow(window_name)
        return roi

    def generate_queries(self, roi):
        x, y, w, h = roi
        grid_y, grid_x = np.meshgrid(
            np.linspace(y, y + h, self.grid_size),
            np.linspace(x, x + w, self.grid_size)
        )
        points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
        queries = np.zeros((1, points.shape[0], 3))
        queries[0, :, 0] = 0
        queries[0, :, 1:] = points
        return torch.from_numpy(queries).float().to(self.device)

    def capture_loop(self, img_paths):
        # 模拟相机频率
        fps_delay = 1.0 / 10.0
        
        for i, p in enumerate(img_paths):
            if not self.is_running:
                break
                
            start_time = time.time()
            img = cv2.imread(str(p))
            if img is None: continue
            
            self.latest_frame = img.copy()
            
            if not self.frame_queue.full():
                # 修正：OpenCV 读取的是 BGR，CoTracker 需要的是 RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.frame_queue.put((i, img_rgb))
            
            elapsed = time.time() - start_time
            if elapsed < fps_delay:
                time.sleep(fps_delay - elapsed)
        
        # 等待数据处理完
        while not self.frame_queue.empty() and self.is_running:
            time.sleep(0.1)
        self.is_running = False

    def tracker_loop(self, queries):
        all_frames = []
        is_first_step = True
        
        while self.is_running:
            try:
                # 获取图像数据
                idx, frame = self.frame_queue.get(timeout=1)
                all_frames.append(frame)
                
                # 遵循官方 16 帧滑动窗口策略 (Overlap 8 frames)
                cur_len = len(all_frames)
                if cur_len >= self.step * 2 and (cur_len - self.step * 2) % self.step == 0:
                    chunk_frames = all_frames[-self.step * 2:]
                    chunk = torch.from_numpy(np.stack(chunk_frames)).permute(0, 3, 1, 2)[None].float().to(self.device)
                    
                    with torch.no_grad():
                        # 核心修改：一旦调用成功，无论是否返回 tracks，后续都切换为非初始化模式
                        tracks, vis = self.model(
                            video_chunk=chunk, 
                            is_first_step=is_first_step, 
                            queries=queries if is_first_step else None,
                            grid_size=None
                        )
                    
                    if is_first_step:
                        print(f"初始化指令已发送 (当前缓冲 {cur_len} 帧)")
                        is_first_step = False
                    
                    if tracks is not None:
                        with self.results_lock:
                            self.latest_tracks = tracks
                            self.latest_vis = vis
                            if self.ref_points is None:
                                self.ref_points = tracks[0, 0].cpu().numpy()
                                print("首批轨迹已就绪！")
                            
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"跟踪线程崩溃: {e}")
                import traceback
                traceback.print_exc()
                self.is_running = False

    def run(self):
        img_paths = sorted(Path(self.data_dir).glob("*.png"))[:240]
        if not img_paths:
            print("没有找到图像数据")
            return

        first_img = cv2.imread(str(img_paths[0]))
        self.roi = self.select_roi(first_img)
        if self.roi[2] == 0 or self.roi[3] == 0:
            print("未选择 ROI")
            return
            
        queries = self.generate_queries(self.roi)
        
        t_capture = threading.Thread(target=self.capture_loop, args=(img_paths,))
        t_tracker = threading.Thread(target=self.tracker_loop, args=(queries,))
        
        t_capture.start()
        t_tracker.start()
        
        cv2.namedWindow("Real-Time CoTracker3", cv2.WINDOW_NORMAL)
        
        while self.is_running:
            if self.latest_frame is None:
                time.sleep(0.01)
                continue
                
            img_display = self.latest_frame.copy()
            
            with self.results_lock:
                if self.latest_tracks is not None:
                    t_idx = self.latest_tracks.shape[1] - 1
                    tracks = self.latest_tracks[0, t_idx].cpu().numpy()
                    vis = self.latest_vis[0, t_idx].cpu().numpy()
                    
                    for n in range(tracks.shape[0]):
                        if vis[n] > 0.5:
                            dist = np.linalg.norm(tracks[n] - self.ref_points[n])
                            color = (0, 0, 255) if dist > 5.0 else (0, 255, 0)
                            
                            pt = (int(tracks[n, 0]), int(tracks[n, 1]))
                            cv2.circle(img_display, pt, 3, color, -1)
                            
                            # 轨迹拖尾
                            for i in range(max(0, t_idx-10), t_idx):
                                p1 = self.latest_tracks[0, i, n].cpu().numpy()
                                p2 = self.latest_tracks[0, i+1, n].cpu().numpy()
                                cv2.line(img_display, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)

            cv2.imshow("Real-Time CoTracker3", img_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break
        
        self.is_running = False
        t_capture.join()
        t_tracker.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealTimeCoTracker(data_dir="/home/wby/active_vision/vlm_based/record_pull_arc_with_joint_0/rgb")
    app.run()
