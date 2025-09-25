import cv2
import numpy as np
from typing import Tuple, Optional
import insightface
from insightface.model_zoo.arcface_onnx import ArcFaceONNX

def preprocess_for_arcface(crop_bgr: np.ndarray, out_size: Tuple[int,int]=(112,112)) -> np.ndarray:
    img = cv2.resize(crop_bgr, out_size, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # normalize to [-1,1]
    img = np.transpose(img, (2,0,1))  # CHW
    return img

class FaceRecognizer:
    def __init__(self, model_name: str = "arcface_r100_v1", logger=None):
        self.logger = logger
        self.model: ArcFaceONNX = insightface.model_zoo.get_model(model_name)
        self.model.prepare(ctx_id=0)  # CPU-> use 0; if GPU with CUDA, adjust
        if self.logger:
            self.logger.info(f"Loaded InsightFace model: {model_name}")

    def embed(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            inp = preprocess_for_arcface(crop_bgr)
            emb = self.model.get(inp)  # (512,) float32
            # L2 normalize
            norm = np.linalg.norm(emb) + 1e-9
            emb = emb / norm
            return emb.astype(np.float32)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Embedding failed: {e}")
            return None

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def match(self, emb: np.ndarray, gallery: list, threshold: float) -> Tuple[Optional[str], float]:
        # gallery: list of (face_id, embedding)
        best_id, best_sim = None, -1.0
        for fid, g in gallery:
            sim = self.cosine_sim(emb, g)
            if sim > best_sim:
                best_sim, best_id = sim, fid
        if best_sim >= threshold:
            return best_id, best_sim
        return None, best_sim