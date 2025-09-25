from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class MultiObjectTracker:
    def __init__(self, max_age=30, n_init=2, max_cosine_distance=0.2, nn_budget=100):
        # Use built-in embedder on CPU
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nms_max_overlap=1.0,
            nn_budget=nn_budget,
            embedder='mobilenet',        # lightweight default
            half=False,                  # CPU-friendly
            embedder_gpu=False           # force CPU
        )

    def update(self, detections, frame=None):
        # detections: list of (x1,y1,x2,y2,conf)
        ds = []
        for (x1,y1,x2,y2,conf) in detections:
            w = x2 - x1
            h = y2 - y1
            ds.append(([x1, y1, w, h], conf, None))  # (tlwh, confidence, class)
        # Pass the current frame so the tracker can compute embeddings internally
        tracks = self.tracker.update_tracks(ds, frame=frame)
        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            ltrb = t.to_ltrb()
            out.append((tid, int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])))
        return out