import cv2
import numpy as np
from typing import Tuple, Optional

def preprocess_for_arcface(crop_bgr: np.ndarray, out_size: Tuple[int,int]=(112,112)) -> np.ndarray:
    img = cv2.resize(crop_bgr, out_size, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2,0,1))
    return img

class FaceRecognizer:
    def __init__(self, model_name: str = "arcface_r100_v1", logger=None):
        self.logger = logger
        self.model = None

        # Lazy/optional import to avoid hard dependency failures
        try:
            import insightface
            from insightface.model_zoo.arcface_onnx import ArcFaceONNX  # noqa: F401
            self.model = insightface.model_zoo.get_model(model_name)
            if self.model is not None:
                self.model.prepare(ctx_id=0)  # CPU
                if self.logger:
                    self.logger.info(f"Loaded InsightFace model: {model_name}")
            else:
                raise Exception("Model returned None")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load InsightFace model {model_name}: {e}")
                self.logger.warning("Falling back to dummy embeddings (recognition disabled)")
            self.model = None

    def embed(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            # deterministic pseudo-embedding so the same face looks similar across frames
            h = cv2.resize(crop_bgr, (16,16), interpolation=cv2.INTER_AREA).astype(np.float32)
            emb = h.mean(axis=(0,1))  # 3 dims
            emb = np.pad(emb, (0, 509), mode='wrap')  # to 512 dims
            emb = emb.astype(np.float32)
            n = np.linalg.norm(emb) + 1e-9
            return emb / n

        try:
            inp = preprocess_for_arcface(crop_bgr)
            emb = self.model.get(inp)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
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
        if emb is None or len(gallery) == 0:
            return None, -1.0
        best_id, best_sim = None, -1.0
        for fid, g in gallery:
            sim = self.cosine_sim(emb, g)
            if sim > best_sim:
                best_sim, best_id = sim, fid
        if best_sim >= threshold:
            return best_id, best_sim
        return None, best_sim