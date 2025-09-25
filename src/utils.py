from datetime import datetime
import cv2
import numpy as np

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def crop(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()

def draw_label(frame, x1, y1, text):
    cv2.putText(frame, text, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2, cv2.LINE_AA)

def draw_box(frame, x1, y1, x2, y2, color=(0,255,0)):
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)