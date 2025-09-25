import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
import numpy as np

class Database:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            face_id TEXT PRIMARY KEY,
            first_seen TIMESTAMP NOT NULL
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            face_id TEXT NOT NULL,
            vector BLOB NOT NULL,
            FOREIGN KEY(face_id) REFERENCES faces(face_id)
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id TEXT,
            track_id INTEGER,
            event_type TEXT CHECK(event_type IN ('entry','exit','recognize','register')),
            timestamp TIMESTAMP NOT NULL,
            image_path TEXT,
            FOREIGN KEY(face_id) REFERENCES faces(face_id)
        );""")
        self.conn.commit()

    def upsert_face(self, face_id: str, embedding: np.ndarray):
        ts = datetime.utcnow().isoformat()
        self.conn.execute("INSERT OR IGNORE INTO faces(face_id, first_seen) VALUES (?, ?)", (face_id, ts))
        self.conn.execute("DELETE FROM embeddings WHERE face_id = ?", (face_id,))
        self.conn.execute("INSERT INTO embeddings(face_id, vector) VALUES (?, ?)", (face_id, embedding.tobytes()))
        self.conn.commit()

    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        cur = self.conn.cursor()
        cur.execute("SELECT face_id, vector FROM embeddings")
        rows = cur.fetchall()
        out = []
        for fid, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            out.append((fid, vec))
        return out

    def insert_event(self, face_id: Optional[str], track_id: Optional[int], event_type: str, timestamp: str, image_path: Optional[str]):
        self.conn.execute(
            "INSERT INTO events(face_id, track_id, event_type, timestamp, image_path) VALUES (?,?,?,?,?)",
            (face_id, track_id, event_type, timestamp, image_path)
        )
        self.conn.commit()

    def unique_visitor_count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM faces")
        return cur.fetchone()[0]

    def close(self):
        self.conn.close()