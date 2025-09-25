from loguru import logger
from pathlib import Path

def setup_logger(log_path: str, level: str = "INFO"):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(log_path, level=level, rotation="20 MB", retention="30 days", enqueue=True, backtrace=False, diagnose=False)
    logger.add(lambda msg: print(msg, end=""), level=level)
    return logger
