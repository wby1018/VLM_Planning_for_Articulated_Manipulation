# det_pipeline.py
# OWLv2 改用 HuggingFace transformers 实现（Owlv2ForObjectDetection）
# 改为 FastAPI Server 形式
import os, io, json, base64, time, gc, datetime
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image

# ---- HuggingFace OWLv2 ----
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# ---- MobileSAM ----
from mobile_sam import SamPredictor, sam_model_registry

from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from contextlib import asynccontextmanager

# 提速
torch.backends.cudnn.benchmark = True

# ================== 配置 ==================
# OWLv2 HuggingFace checkpoint
OWL_CHECKPOINT = "google/owlv2-base-patch16-ensemble"

# MobileSAM 权重与类型
MOBILE_SAM_CKPT = "weights/mobile_sam.pt"
MOBILE_SAM_TYPE = "vit_t"  # 'vit_t' for MobileSAM-tiny

# SAM 批量与显存策略
SAM_MULTIMASK_OUTPUT    = False
SAM_INIT_BATCH_SIZE     = 64
USE_FP16_FOR_SAM        = False
USE_AUTOCast_INFER      = False
CPU_FALLBACK_ON_OOM     = True
EMPTY_CACHE_EVERY_CHUNK = True
# ==========================================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def mask_to_base64_png(mask_bool: np.ndarray) -> str:
    """将布尔 / 0-1 掩码(H×W)编码为PNG并Base64返回；输出与原图同尺寸。"""
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    img = Image.fromarray(mask_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------- MobileSAM 初始化/分割 ----------------
@torch.no_grad()
def init_mobilesam(device: torch.device):
    sam = sam_model_registry[MOBILE_SAM_TYPE](checkpoint=MOBILE_SAM_CKPT)
    if device.type == "cuda" and USE_FP16_FOR_SAM:
        sam = sam.to(device=device, dtype=torch.float16)
    else:
        sam = sam.to(device=device, dtype=torch.float32)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor


@torch.no_grad()
def sam_segment_boxes_adaptive(
    predictor: SamPredictor,
    image_rgb_u8: np.ndarray,
    boxes_xyxy: List[List[int]],
    *,
    multimask_output: bool = SAM_MULTIMASK_OUTPUT,
    init_batch_size: int = SAM_INIT_BATCH_SIZE,
    cpu_fallback: bool = CPU_FALLBACK_ON_OOM
) -> Tuple[List[np.ndarray], List[float]]:
    """
    自适应批量分割：每个批次如果 OOM 就减半重试；若最终 batch=1 仍 OOM 且允许，则切 CPU。
    返回：
      masks_bool_list: List[np.ndarray(H,W)]
      confs_list:      List[float]
    """
    image_rgb_u8 = np.ascontiguousarray(image_rgb_u8)
    H, W = image_rgb_u8.shape[:2]
    device = next(predictor.model.parameters()).device

    def _predict_on_device(dev: torch.device, batch_boxes: torch.Tensor):
        autocast_enabled = (dev.type == "cuda") and USE_AUTOCast_INFER
        with torch.autocast(device_type=dev.type, enabled=autocast_enabled):
            m, s, _ = predictor.predict_torch(
                point_coords=None, point_labels=None,
                boxes=batch_boxes, multimask_output=multimask_output
            )
        if m.shape[-2:] != (H, W):
            try:
                m = predictor.postprocess_masks(m, predictor.input_size, (H, W))
            except Exception:
                m = torch.nn.functional.interpolate(
                    m, size=(H, W), mode="bilinear", align_corners=False
                )
        if multimask_output:
            best = torch.argmax(s, dim=1)
            m = m[torch.arange(m.shape[0]), best]
            s = s.max(dim=1).values
        else:
            m = m[:, 0]
            s = s[:, 0] if s.ndim == 2 else s.squeeze()
        return m, s

    predictor.set_image(image_rgb_u8)
    masks_out: List[np.ndarray] = []
    confs_out: List[float] = []

    if hasattr(predictor, "predict_torch"):
        dev = device
        boxes_t = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=dev)
        boxes_trans = predictor.transform.apply_boxes_torch(boxes_t, (H, W))

        bs = max(1, int(init_batch_size))
        i = 0
        while i < boxes_trans.shape[0]:
            j = min(i + bs, boxes_trans.shape[0])
            batch = boxes_trans[i:j]
            try:
                m, s = _predict_on_device(dev, batch)
                masks_cpu = m.detach().to("cpu").float()
                masks_cpu = (masks_cpu > 0.5).to(torch.bool).numpy()
                confs_cpu = s.detach().to("cpu").float().numpy().tolist()
                for k in range(masks_cpu.shape[0]):
                    masks_out.append(masks_cpu[k].copy())
                    confs_out.append(float(confs_cpu[k]))
                i = j
                del m, s, masks_cpu
                if EMPTY_CACHE_EVERY_CHUNK and dev.type == "cuda":
                    torch.cuda.empty_cache(); gc.collect()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if bs > 1:
                        bs = max(1, bs // 2)
                        if dev.type == "cuda":
                            torch.cuda.empty_cache(); gc.collect()
                        continue
                    else:
                        if cpu_fallback:
                            print("[WARN] CUDA OOM at batch=1, falling back to CPU for remaining boxes.")
                            predictor.model.to("cpu")
                            predictor.set_image(image_rgb_u8)
                            dev_cpu = torch.device("cpu")
                            boxes_cpu = predictor.transform.apply_boxes_torch(
                                boxes_t.to(dev_cpu), (H, W)
                            )
                            for k in range(i, boxes_cpu.shape[0]):
                                bm = boxes_cpu[k:k+1]
                                m2, s2 = _predict_on_device(dev_cpu, bm)
                                m2 = (m2.to("cpu").float() > 0.5).numpy()[0]
                                s2 = float(s2.to("cpu").numpy().reshape(-1)[0])
                                masks_out.append(m2.copy())
                                confs_out.append(s2)
                            return masks_out, confs_out
                        else:
                            raise
                else:
                    raise
    else:
        for box in boxes_xyxy:
            m, s, _ = predictor.predict(
                box=np.array(box, dtype=np.float32)[None, :],
                multimask_output=multimask_output
            )
            if m.shape[-2:] != (H, W):
                from skimage.transform import resize as _resize
                m = _resize(m, (m.shape[0], 1, H, W), order=1, mode="reflect", anti_aliasing=True)
            if multimask_output:
                idx = int(np.argmax(s))
                masks_out.append((m[idx] > 0.5).astype(bool).copy())
                confs_out.append(float(s[idx]))
            else:
                masks_out.append((m[0] > 0.5).astype(bool).copy())
                confs_out.append(float(np.squeeze(s)))
    return masks_out, confs_out


# ---------------- HuggingFace OWLv2 初始化/检测 ----------------
def build_owl(device: torch.device):
    """加载 HuggingFace OWLv2 模型和处理器。"""
    print(f"[OWL] Loading checkpoint: {OWL_CHECKPOINT}")
    processor = Owlv2Processor.from_pretrained(OWL_CHECKPOINT)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_CHECKPOINT)
    model.to(device)
    model.eval()
    print("[OWL] Model loaded.")
    return processor, model


@torch.no_grad()
def owl_detect(
    image_rgb_u8: np.ndarray,
    processor: Owlv2Processor,
    model: Owlv2ForObjectDetection,
    device: torch.device,
    text_queries: List[str],
    score_threshold: float
) -> List[Dict[str, Any]]:
    """
    对单张图做 OWLv2 检测，返回每个候选的:
      {
        "box": [x1, y1, x2, y2],          # 原图像素坐标
        "detection": [{"label": str, "score": float}, ...]  # 按得分降序
      }
    """
    image_pil = Image.fromarray(image_rgb_u8)
    h, w = image_rgb_u8.shape[:2]

    inputs = processor(
        text=[text_queries],
        images=image_pil,
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)

    target_sizes = torch.tensor([[h, w]], dtype=torch.float32).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=score_threshold
    )

    boxes_px  = results[0]["boxes"]   # (N, 4) xyxy pixel coords
    scores    = results[0]["scores"]  # (N,)
    label_ids = results[0]["labels"]  # (N,)

    detections: List[Dict[str, Any]] = []
    for box, score, label_id in zip(boxes_px, scores, label_ids):
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(0, min(int(round(x2)), w - 1))
        y2 = max(0, min(int(round(y2)), h - 1))
        if x2 <= x1: x2 = min(w - 1, x1 + 1)
        if y2 <= y1: y2 = min(h - 1, y1 + 1)

        label_text = text_queries[label_id.item()]
        sc = float(score.item())

        # 与 pipeline 其余部分兼容：detection 列表（仅包含命中类别）
        detections.append({
            "detection": [{"label": label_text, "score": sc}],
            "box": [x1, y1, x2, y2]
        })

    return detections


def random_bgr_color(min_val=80, max_val=255):
    return np.random.randint(min_val, max_val, size=3).astype(np.float32)


def render_detection_png(image_rgb: np.ndarray, detections: list, save_path: str):
    """绘制 bbox + mask 叠加 + label:score，保存为 PNG。"""
    overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = overlay.shape[:2]

    for det in detections:
        b64 = det["mask"]
        mask_bytes = base64.b64decode(b64)
        mask_arr = np.frombuffer(mask_bytes, np.uint8)
        mask_gray = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            continue
        if mask_gray.shape != (h, w):
            mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = (mask_gray > 128)

        color_bgr = random_bgr_color()
        alpha = 0.4
        region = overlay[mask_bool].astype(np.float32)
        overlay[mask_bool] = (region * (1 - alpha) + color_bgr * alpha).astype(np.uint8)

        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

        top = det["detection"][0]
        idx = det.get("index", "?")
        txt = f"[{idx}] {top['label']}:{top['score']:.2f}"
        cv2.putText(
            overlay, txt,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2,
            cv2.LINE_AA
        )

    cv2.imwrite(save_path, overlay)


# ---------------- FastAPI 主流程 ----------------

model_context = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing models...")
    owl_processor, owl_model = build_owl(device)
    sam_predictor = init_mobilesam(device)
    model_context['device'] = device
    model_context['owl_processor'] = owl_processor
    model_context['owl_model'] = owl_model
    model_context['sam_predictor'] = sam_predictor
    print("Models initialized successfully.")
    yield
    model_context.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    text_queries: str = Form(...),
    score_threshold: float = Form(0.02)
):
    """
    接收一张图像，候选列表，以及阈值，返回检测结果。
    """
    start = time.time()
    
    # 解析候选列表
    try:
        queries = json.loads(text_queries)
    except:
        queries = [text_queries] # Fallback 如果解析失败就把整个字符串当作一个query
        
    # 读取图像
    image_bytes = await file.read()
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    device = model_context['device']
    owl_processor = model_context['owl_processor']
    owl_model = model_context['owl_model']
    sam_predictor = model_context['sam_predictor']

    # 1) OWLv2 检测
    candidates = owl_detect(
        image_rgb,
        owl_processor,
        owl_model,
        device,
        text_queries=queries,
        score_threshold=score_threshold
    )

    results = {"detections": []}
    if len(candidates) > 0:
        # 2) MobileSAM 批量分割
        boxes = [c["box"] for c in candidates]
        masks_list, confs = sam_segment_boxes_adaptive(
            sam_predictor, image_rgb, boxes,
            multimask_output=SAM_MULTIMASK_OUTPUT,
            init_batch_size=SAM_INIT_BATCH_SIZE,
            cpu_fallback=CPU_FALLBACK_ON_OOM
        )

        # 3) 组装 JSON
        for i, (cand, mask) in enumerate(zip(candidates, masks_list)):
            results["detections"].append({
                "index": i,
                "detection": cand["detection"],
                "box": cand["box"],
                "mask": mask_to_base64_png(mask)
            })

        del masks_list, confs

    # 可以选择在服务端也保存一张测试图
    # OUTPUT_DIR = "server_detections"
    # ensure_dir(OUTPUT_DIR)
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # save_path = os.path.join(OUTPUT_DIR, f"server_det_{timestamp}.png")
    # render_detection_png(image_rgb, results["detections"], save_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    dur = time.time() - start
    print(f"[Server] Processed {file.filename} -> {len(results['detections'])} masks  ({dur:.2f}s)")

    return results

if __name__ == "__main__":
    uvicorn.run("det_pipeline:app", host="0.0.0.0", port=8000, reload=False)
