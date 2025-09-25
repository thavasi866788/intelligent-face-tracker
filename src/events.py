from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class TrackState:
    face_id: Optional[str] = None
    last_seen: datetime = datetime.utcnow()
    entered: bool = False
    exited: bool = False