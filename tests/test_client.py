import requests
import json
import argparse
import time
import cv2
import numpy as np
import base64

def random_bgr_color(min_val=80, max_val=255):
    return np.random.randint(min_val, max_val, size=3).astype(np.float32)

def render_detection_png(image_bgr: np.ndarray, detections: list, save_path: str):
    overlay = image_bgr.copy()
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to input image", default="test.png")
    parser.add_argument("--url", default="http://127.0.0.1:8000/detect", help="Server URL")
    parser.add_argument("--text_queries", nargs="+", default=["drawer", "door", "handle"], help="List of object names to detect")
    parser.add_argument("--threshold", type=float, default=0.15, help="Detection score threshold")
    parser.add_argument("--output", default="client_result.png", help="Output path for visualized detection")
    args = parser.parse_args()

    print(f"Sending request to {args.url} ...")
    print(f"Image: {args.image}")
    print(f"Queries: {args.text_queries}")
    
    start = time.time()
    with open(args.image, "rb") as f:
        files = {"file": (args.image, f, "image/jpeg")}
        data = {
            "text_queries": json.dumps(args.text_queries),
            "score_threshold": args.threshold
        }
        res = requests.post(args.url, files=files, data=data)
    
    dur = time.time() - start
    
    if res.status_code == 200:
        result = res.json()
        detections = result.get('detections', [])
        print(f"Success! Detected {len(detections)} objects in {dur:.2f}s.")
        
        # Read the original image for rendering
        image_bgr = cv2.imread(args.image)
        if image_bgr is not None:
            render_detection_png(image_bgr, detections, args.output)
            print(f"Visualized result saved to {args.output}")
        else:
            print("Failed to read image for visualization.")
            
        with open("client_result.json", "w") as f:
            # removing 'mask' key from printing for clearer log, keeping mask in json
            json.dump(result, f)
        print("Raw JSON response saved to client_result.json")
    else:
        print(f"Error: {res.status_code}")
        print(res.text)

if __name__ == "__main__":
    main()
