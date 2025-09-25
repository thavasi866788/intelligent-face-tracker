# intelligent-face-tracker
## Intelligent Face Tracker with Auto-Registration and Visitor Counting

This project detects, tracks, recognizes, and auto-registers faces from a video or RTSP stream. It logs entry and exit events with cropped face images, timestamps, and stores metadata in SQLite.

### Features
- YOLOv8 Face-based detection (real-time)
- DeepSORT tracking with persistent `track_id`s
- InsightFace ArcFace embeddings for recognition
- Auto-registration for new faces with unique IDs
- Structured logging to filesystem and SQLite
- Configurable detection frame skipping via `config.json`
- Unique visitor counting via DB

### Quickstart
1) Create folder and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Put the provided video as `sample.mp4` alongside `config.json`, or set your RTSP URL in `config.json`.

3) Run:
```bash
python main.py
```

4) Outputs
- Logs: `logs/events.log`
- Images:
  - Entries: `logs/entries/YYYY-MM-DD/*.jpg`
  - Exits: `logs/exits/YYYY-MM-DD/*.jpg`
- Database: `data/events.db`

### Configuration (config.json)
- `detection.frame_skip`: number of frames to skip between detection cycles
- `recognition.similarity_threshold`: cosine similarity threshold to match known faces
- `storage.db_path`, `logging.log_file`: customize paths

### Architecture
```mermaid
flowchart LR
    A[Video/RTSP] --> B[YOLOv8 Face Detector]
    B --> C[DeepSORT Tracker]
    C -->|bbox crop| D[ArcFace Embedding (InsightFace)]
    D -->|match/register| E[SQLite DB]
    C -->|entry/exit| F[Filesystem Images]
    E --> G[Unique Visitor Count]
    subgraph Logging
      E --> H[events.log]
      B --> H
      D --> H
      C --> H
    end
```

### Assumptions
- Internet access available to download YOLO face model once.
- ArcFace embedding is computed on resized detection crop (approximate alignment).
- Exit event image is omitted if bbox is unavailable on last frame.

### Notes
- During interview, set `input.source` to `"rtsp"` and configure `input.rtsp_url`.
- You can disable display by editing `process_stream(show=False)` in `main.py`.

### Sample DB Queries
- Unique visitors:
```sql
SELECT COUNT(*) FROM faces;
```
- All entry/exit events:
```sql
SELECT event_type, face_id, timestamp, image_path FROM events ORDER BY timestamp;
```

This project is a part of a hackathon run by https://katomaran.com
