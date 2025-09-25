import cv2
import numpy as np
from typing import Dict, Tuple
from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .tracker import MultiObjectTracker
from .database import Database
from .storage import Storage
from .events import TrackState
from .utils import now_iso, crop, draw_box, draw_label

class FacePipeline:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

        # Modules
        self.detector = FaceDetector(
            model_path=cfg.get("detection","yolo_face_model"),
            conf=cfg.get("detection","conf_threshold"),
            iou=cfg.get("detection","iou_threshold"),
            logger=logger
        )
        self.recognizer = FaceRecognizer(model_name=cfg.get("recognition","embedding_model"), logger=logger)
        self.tracker = MultiObjectTracker(
            max_age=cfg.get("tracking","max_age"),
            n_init=cfg.get("tracking","n_init"),
            max_cosine_distance=cfg.get("tracking","max_cosine_distance"),
            nn_budget=cfg.get("tracking","nn_budget"),
        )
        self.db = Database(cfg.get("storage","db_path"))
        self.storage = Storage(
            base_dir=cfg.get("storage","base_dir"),
            images_subdir=cfg.get("storage","images_subdir"),
            entries_subdir=cfg.get("storage","entries_subdir"),
            exits_subdir=cfg.get("storage","exits_subdir"),
        )

        self.frame_skip = int(cfg.get("detection","frame_skip"))
        self.sim_threshold = float(cfg.get("recognition","similarity_threshold"))
        self.min_face_size = int(cfg.get("recognition","min_face_size"))

        self.track_states: Dict[int, TrackState] = {}
        self.frame_idx = 0

    def close(self):
        self.db.close()

    def process_stream(self, cap: cv2.VideoCapture, show: bool = True):
        gallery = self.db.get_all_embeddings()
        self.logger.info(f"Loaded {len(gallery)} known embeddings from DB.")

        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.info("End of stream or camera read failed.")
                break

            self.frame_idx += 1
            run_det = (self.frame_idx % (self.frame_skip + 1) == 1)
            dets = self.detector.detect(frame) if run_det else []
            tracks = self.tracker.update(dets if run_det else [], frame)
            # Map detections to tracks for recognition
            # Build a dict: track_id -> bbox
            track_bboxes = {}
            for (tid, x1, y1, x2, y2) in tracks:
                track_bboxes[tid] = (x1, y1, x2, y2)

            for tid, bbox in track_bboxes.items():
                x1,y1,x2,y2 = bbox
                w, h = x2 - x1, y2 - y1
                if w < self.min_face_size or h < self.min_face_size:
                    continue

                face_crop = crop(frame, x1, y1, x2, y2)
                if face_crop is None:
                    continue

                state = self.track_states.get(tid)
                if state is None:
                    # New track -> entry
                    emb = self.recognizer.embed(face_crop)
                    face_id = None
                    if emb is not None:
                        match_id, sim = self.recognizer.match(emb, gallery, self.sim_threshold)
                        if match_id is None:
                            # Register new face
                            face_id = f"F{len(gallery)+1:06d}"
                            self.db.upsert_face(face_id, emb)
                            gallery = self.db.get_all_embeddings()
                            self.logger.info(f"Registered new face {face_id}")
                            self.db.insert_event(face_id, tid, "register", now_iso(), None)
                        else:
                            face_id = match_id
                            self.db.insert_event(face_id, tid, "recognize", now_iso(), None)
                    else:
                        face_id = f"UNK{tid}"

                    # Save entry image
                    path = self.storage.save_face("entry", face_crop, face_id, tid)
                    self.db.insert_event(face_id, tid, "entry", now_iso(), path)

                    state = TrackState(face_id=face_id, entered=True)
                    self.track_states[tid] = state

                else:
                    state.last_seen = state.last_seen  # keep updated if needed

                # Draw UI
                draw_box(frame, x1,y1,x2,y2)
                draw_label(frame, x1, y1, f"{self.track_states[tid].face_id} (T{tid})")

            # Detect exited tracks (those no longer present)
            active_ids = set(track_bboxes.keys())
            for tid in list(self.track_states.keys()):
                if tid not in active_ids:
                    # Exit event once
                    state = self.track_states[tid]
                    if not state.exited:
                        # We don't have the last bbox to crop; skip image or use previous frame if stored.
                        self.db.insert_event(state.face_id, tid, "exit", now_iso(), None)
                        state.exited = True
                    del self.track_states[tid]

            if show:
                cv2.imshow("Face Tracker", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()