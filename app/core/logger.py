# /app/core/logger.py

import logging
from pathlib import Path

# Ensure log directory exists
Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler("logs/server.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)