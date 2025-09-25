import cv2
import sys
from pathlib import Path
from src.config import Config
from src.logger import setup_logger
from src.pipeline import FacePipeline

def main():
    cfg_path = Path("config.json")
    if not cfg_path.exists():
        print("config.json not found")
        sys.exit(1)

    cfg = Config(str(cfg_path))
    logger = setup_logger(cfg.get("logging","log_file"), cfg.get("logging","level"))

    pipe = FacePipeline(cfg, logger)
    try:
        source_type = cfg.get("input","source")
        if source_type == "video":
            vid = cfg.get("input","video_path")
            cap = cv2.VideoCapture(vid)
        elif source_type == "rtsp":
            url = cfg.get("input","rtsp_url")
            cap = cv2.VideoCapture(url)
        elif source_type == "camera":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error("Failed to open video source.")
            sys.exit(2)

        logger.info("Starting processing...")
        pipe.process_stream(cap, show=True)
        logger.info("Processing finished.")

        count = pipe.db.unique_visitor_count()
        logger.info(f"Unique visitor count: {count}")
        print(f"Unique visitors: {count}")
    finally:
        pipe.close()

if __name__ == "__main__":
    main()