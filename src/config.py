import json
from pathlib import Path

class Config:
    def __init__(self, path: str):
        self.path = Path(path)
        # tolerate BOM and clean it if present
        with open(self.path, "r", encoding="utf-8-sig") as f:
            raw = f.read()
        if raw.startswith("\ufeff"):
            raw = raw.lstrip("\ufeff")
        self.data = json.loads(raw)

    def get(self, *keys, default=None):
        cur = self.data
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k, None)
            if cur is None:
                return default
        return cur