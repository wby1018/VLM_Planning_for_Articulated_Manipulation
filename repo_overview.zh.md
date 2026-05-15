# VLM_Planning_for_Articulated_Manipulation — 整合参考笔记

> 这是一份内部阅读笔记，用于理解 repo 完成的任务、模块边界，以及后续整合时的接口契约。
> 适合配合代码（特别是 `action_server.py`、`det_pipeline.py`、`client_sapien_*.py`、`loftr_fg.py`）一起读。

---

## 一、任务 What

> 给定一个**仿真柜子** + **自然语言指令**（如 `"open the second bottom drawer"`），让 Panda 机械臂**自主**完成：感知 → 规划 → 抓住把手 → 判断"是抽屉还是柜门" → 拉出/转开。

输入是 RGB-D 帧 + 文字；输出是逐步的 EE 目标位姿（位置 + rot6d + 夹爪开度），每仿真步发一次。

---

## 二、架构（3 进程 + 1 在线线程）

```
                          ┌───────────────┐
   det_pipeline.py        │  OpenAI GPT   │
   (FastAPI :8001)        │   (VLM API)   │
   OWLv2 + MobileSAM      └──────┬────────┘
        ▲                        │
        │ POST /detect           │ data-url + prompt
        │ (RGB+queries→masks)    │
        │                        ▼
        │              ┌──────────────────────┐
        │              │   action_server.py   │
        └──────────────┤   (ZMQ REQ-REP :5555)│
                       │  ActionPlanner 状态机│
                       │  ├─ TypeCheck (物理) │
                       │  └─ LoFTREstimator   │◄── 后台线程精化转轴
                       │              (loftr_fg.py)
                       └──────────┬───────────┘
                                  │ obs↔action (compressed pickle)
                                  ▼
                       ┌──────────────────────┐
                       │  client_sapien_*.py  │
                       │  SAPIEN 仿真 + 渲染  │
                       │  Jacobian IK 控制    │
                       │  PartNet URDF + Panda│
                       └──────────────────────┘
```

> 注意端口：在我们这台机器上 8000 被占用，已把 det_pipeline 改到 **8001**（`det_pipeline.py:394`、`action_server.py:64`）。

---

## 三、模块拆解（按功能 + 文件位置）

### ① 感知层 — `det_pipeline.py`

| | |
|---|---|
| 模型 | OWLv2 (`google/owlv2-base-patch16-ensemble`) + MobileSAM (vit_t, `weights/mobile_sam.pt`) |
| 接口 | `POST /detect`，multipart：`file`=jpg, `text_queries`=JSON 列表, `score_threshold` |
| 输出 | `{"detections":[{index, detection:[{label,score}], box:[x1,y1,x2,y2], mask:base64png}]}` |
| 调用方 | 只被 `action_server.call_detection()`（`action_server.py:670` 附近）调用 |
| 独立性 | ✅ **完全独立的微服务**，可单独替换成你想要的 detector + segmenter |

### ② 高层规划 — VLM (`action_server.call_vlm`, line 870)

| | |
|---|---|
| 输入 | annotated 图（带框/编号） + user_instruction + detections 列表 |
| Prompt | line 815–865 写死的 system prompt，约束输出 JSON schema |
| 输出 | `{target_handle_id, parent_object_id?, motion_type∈{Translation,Rotation}, gripper_orientation∈{V,H}, plan: ['MoveTo','Grasp','Pull_Linear|Pull_Arc','Release']}` |
| Fallback | `OPENAI_API_KEY` 没设或 API 出错时返回 hardcoded plan（line 921–925） |
| 独立性 | ✅ **可换** —— 只是个 dict 生产函数，里面是 LLM 不重要，重要的是输出 schema |

### ③ 几何估计（一次性，在 `_initialize` 里做）

`action_server.py` 里 `class ActionPlanner._initialize()`（line 1190 附近）：

| 步骤 | 函数 | 作用 |
|---|---|---|
| 4/7 | `backproject_pixel` | 用 box 中心 + depth → handle 3D 世界坐标 |
| 5/7 | `estimate_panel_normal` | 在 handle 周围采样有效深度点 → PCA → 柜门表面法向（不依赖 mask 也行，最新版） |
| 6/7 | `look_at_rotation` | 用 normal 算 approach quaternion |

输出固定字段：`handle_3d`, `panel_normal`, `approach_pos`, `grasp_target_pos`, `approach_quat`

### ④ 物理探测 + TypeCheck（核心创新点）

最关键的一个模块，建议你重点看。

**ProbePull**（`action_server.py:1410+` 在 `_target("ProbePull")`）：抓住后沿 panel_normal 微拉 5 cm，**保存 P0/P1 两帧的目标点云**（柜子部分的点云，已用 segmentation mask 过滤）。

**TypeCheck**（`action_server.py:404–508` 的 `decide_motion_type` / 在 ActionPlanner 里 line ~1620 调用）：

- 比较 **5 个假设**：
  - **H0 = Translation**：把 P0 沿 Δgripper 平移 → 和 P1 求 Chamfer error E_T
  - **H1–H4 = Rotation 4 个候选轴**（柜门 left/right/top/bottom 边缘各一个）：每个轴算 θ 然后旋转 P0 → 和 P1 比 → E_R
- 选 E 最低那个，得到 `motion_type` + `side`（如果是 Rotation 还附 `best_axis_point`/`best_axis_dir`）
- **Stage list 自动改写**（line 1648–1657）：原 plan 是 Pull_Arc 但实测判 Translation → 改 Pull_Linear，反之亦然

> **这是 repo 的核心思想**：VLM 可以错（看图判抽屉/门可能不准），但靠"抓住后小幅探测一下"的物理事实兜底。

### ⑤ 运动控制 — ActionPlanner 状态机

`action_server.py:1020+` 的 `class ActionPlanner`，stages 7 步：

| Stage | 目标位置 | 夹爪 | 完成条件 |
|---|---|---|---|
| MoveTo | `approach_pos`(handle 后 20 cm) | OPEN | `pos_err<2cm` |
| Approach | `grasp_target_pos`(handle - 8.5cm × normal) | OPEN | `pos_err<2cm` |
| Grasp | 同上 | CLOSED | `grip_err<4mm` 或 timeout |
| ProbePull | `grasp_pos + normal × 0.05` | CLOSED | 到位 |
| TypeCheck | hold | — | 立即（compute-only） |
| Pull_Linear | 沿 normal 渐进 38 cm（每步 5 mm） | CLOSED | 累计 ≥ 38 cm |
| Pull_Arc | 绕 axis 转 90°（每步 ARC_STEP） | CLOSED | swept ≥ 90° |
| Release | 停在原地 | OPEN | `grip_err<4mm` 或 timeout |

每帧调 `_step()` → `_target(stage)` 算目标 → `_is_stage_done()` 决定是否进下一个。

### ⑥ 在线轴精化 — LoFTREstimatorThread + `loftr_fg.py`

仅在 `motion_type=Rotation` 时启动一个后台线程：

- 每帧拿 RGB + depth + 相机外参 + EE 历史
- 用 LoFTR 在两帧间做特征匹配，找到柜门上的对应点对
- 用 **Factor Graph** + scipy `least_squares` 优化 screw axis ξ=(ω,v) 和 per-frame 角度 θ_i
- 把估计结果回灌给 ActionPlanner 的 `arc_axis_point/dir`，`Pull_Arc` 就用这个更准的轴
- （旧版 `loftr_pf.py` 用粒子滤波，5-07 commit 替换成 FG）

### ⑦ 客户端 — `client_sapien_<id>.py`

40147 / 44817 / 46230 三份，差别只是加载哪个 PartNet URDF + 柜子 z 高度。共同部分：

- `take_picture_once`：从 SAPIEN 相机一次性取 Color + Position + Segmentation 缓冲
- `get_point_cloud_from_buffers`：seg mask 过滤出柜子点云（只保留 cabinet 的 link IDs），fpsample 下采样到 1280 点
- `compute_ik`：**数值雅可比 IK**（手写 LM 解算，6×n_arm 雅可比，50 步迭代）—— 这是个独立可替换的 IK 求解器
- `ZMQCommunicationThread`：异步发观测、收 action（队列长 1，不阻塞渲染）

---

## 四、数据契约（适合整合的接口边界）

这是整合时最关心的。**所有跨模块通信都走这几个明确 schema**：

### A. det_pipeline 的 HTTP 接口

```
POST /detect  multipart: file=image, text_queries=json, score_threshold=float
→ {"detections": [{"index","detection":[{"label","score"}],"box","mask"}]}
```

### B. action_server 的 ZMQ 接口

```
client → server: compressed pickle of obs_dict
{rgb (H,W,3), depth (H,W), cam_pos (3,), cam_mat (3,3),
 fovy (1,), agent_pos (1,T,10), point_cloud (1,T,1280,3)}

server → client: compressed pickle of action_dict
{shape:(10,), data: target_pos(3) + target_rot6d(6) + target_gripper(1)}
```

### C. VLM 的内部契约（`call_vlm` 输出）

```python
{
  'target_handle_id': int,                    # detection 列表里的 index
  'parent_object_id': int | None,
  'motion_type': 'Translation' | 'Rotation',
  'gripper_orientation': 'Vertical' | 'Horizontal',
  'plan': ['MoveTo', 'Grasp', ('Pull_Linear'|'Pull_Arc'), 'Release']
}
```

### D. TypeCheck 的输出

```python
{
  'motion_type': 'Translation' | 'Rotation',
  'best_axis_side': 'left'|'right'|'top'|'bottom'|'N/A',
  'best_axis_point': np.ndarray (3,) | None,
  'best_axis_dir':   np.ndarray (3,) | None,
  'theta_axis': float,    # ProbePull 中已转过的角度
  'E_translation': float,
  'E_rotation':    float
}
```

---

## 五、可独立抽出做整合的子系统

按"耦合度从低到高"排：

1. **`det_pipeline.py`** — 任意可换成你自己的 OVS detection / open-vocab segmentation
2. **`call_vlm`** — 换 LLM/VLM（Claude / Gemini / 内部模型），只要遵守输出 schema
3. **`compute_ik`**（client 里的）— 换成 cuRobo / Pinocchio / curobo 都行，接口就是 `(panda, hand_link, target_pos, target_quat) → qpos`
4. **`estimate_panel_normal`** — 换成你自己的法向估计（如 RANSAC plane / 网络回归）
5. **`decide_motion_type`(TypeCheck)** — 换成更复杂的 articulation type classifier
6. **`LoFTREstimatorFG`** — 换成 ScrewNet / Ditto / NARF22 的 axis estimator，接口是 `push 帧 → get_pivot/axis`
7. **`ActionPlanner` 状态机** — 这是最"主干"的，整合通常以它为骨架，外面挂别的模块

> 注意 **TypeCheck 是核心创新** —— 它把"整套系统不依赖 VLM 准确性"的能力实现了。如果你整合到自己的项目里，这一块是最值得保留 / 抽象出来的。

---

## 六、你最可能遇到的整合痛点（根据这次复现的经验）

1. **依赖坏 pin** — `environment.yml` 是 pip freeze 出来的，里面 `clip==1.0`、`mobile-sam==1.0`、`scenic`、`bagpy` 等几个根本装不上。memory 里记了清单。
2. **VLM fallback brittle** — hardcoded `target_handle_id=1` 不一定能被 panda 抓到（44817 抽屉柜上面那个把手就够不到）。**实际上这个 repo 离了 `OPENAI_API_KEY` 跑不稳**。
3. **panda reach 边界** — PartNet 柜子位置是 `(1.367, 0.197)`，panda 在 `(0.2, 0.6)`，1.2+ 米对角距离，部分高位 handle 超出 0.855 m reach。原作者机器上能跑可能是因为他选的 handle 更低。
4. **5-hypothesis TypeCheck 假设柜门是矩形** — 4 个旋转候选是矩形 4 边各做一根轴，对非矩形面板可能不够。
5. **LoFTR-FG 只在 Rotation 启动**，Translation 不需要（直线不需要轴）。

---

## 七、跑起来的最小步骤（这台机器）

```bash
cd ~/.yifeng/VLM_Planning_for_Articulated_Manipulation

# OPENAI_API_KEY 已经在 ~/.openai_env 持久化，新开终端自动有
./run_all.sh 40147     # 柜子 + 抽屉混合（默认走柜门 / Pull_Arc）
./run_all.sh 44817     # 4 抽屉（开第二底抽屉，走 Pull_Linear）
./run_all.sh 46230     # 3 抽屉

# SAPIEN 窗口出来后 → 鼠标聚焦 → 按空格触发
# 日志: logs/det_pipeline.log, logs/action_server.log
```

退出：在 `run_all.sh` 那个终端按 Ctrl+C，trap 会自动 kill 后端。
偶尔 SAPIEN 关窗后 client 进程不退、`run_all.sh` 卡住，手动：

```bash
pkill -f run_all.sh
pkill -f "python.*(det_pipeline|action_server|client_sapien_)"
```
