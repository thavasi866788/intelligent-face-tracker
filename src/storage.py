from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

class Storage:
    def __init__(self, base_dir: str, images_subdir: str, entries_subdir: str, exits_subdir: str):
        self.base = Path(base_dir)
        self.images = self.base / images_subdir
        self.entries = self.base / entries_subdir
        self.exits = self.base / exits_subdir
        self.images.mkdir(parents=True, exist_ok=True)
        self.entries.mkdir(parents=True, exist_ok=True)
        self.exits.mkdir(parents=True, exist_ok=True)

    def save_face(self, kind: str, face_bgr: np.ndarray, face_id: str, track_id: int) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        if kind == "entry":
            dirp = self.entries / datetime.utcnow().strftime("%Y-%m-%d")
        elif kind == "exit":
            dirp = self.exits / datetime.utcnow().strftime("%Y-%m-%d")
        else:
            dirp = self.images / datetime.utcnow().strftime("%Y-%m-%d")
        dirp.mkdir(parents=True, exist_ok=True)
        filename = f"{ts}_fid-{face_id}_tid-{track_id}.jpg"
        path = dirp / filename
        cv2.imwrite(str(path), face_bgr)
        return str(path)