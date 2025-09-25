from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import urllib.request

YOLO_FACE_URLS = [
    "https://huggingface.co/keremberke/yolov8n-face/resolve/main/yolov8n-face.pt?download=true",
    "https://github.com/akanametov/yolov8-face/raw/main/weights/yolov8n-face.pt",
]

def _try_download(url: str, dst: Path, logger) -> bool:
    try:
        logger.info(f"Downloading YOLO face model from {url} ...")
        urllib.request.urlretrieve(url, str(dst))
        return dst.exists() and dst.stat().st_size > 1024 * 100  # >100KB sanity
    except Exception as e:
        logger.warning(f"Download failed from {url}: {e}")
        return False

def ensure_model(path: str, logger):
    p = Path(path)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    for url in YOLO_FACE_URLS:
        if _try_download(url, p, logger):
            logger.info("Model download complete.")
            return
    raise FileNotFoundError(
        f"Could not download YOLO face model to {p}. "
        f"Please download manually from one of:\n- {YOLO_FACE_URLS[0]}\n- {YOLO_FACE_URLS[1]}\n"
        f"and save as {p}"
    )

class FaceDetector:
    def __init__(self, model_path: str, conf: float, iou: float, logger):
        ensure_model(model_path, logger)
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.logger = logger

    def detect(self, frame_bgr: np.ndarray):
        res = self.model.predict(source=frame_bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        detections = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(b.conf[0].cpu().numpy())
            detections.append((x1, y1, x2, y2, conf))
        return detections