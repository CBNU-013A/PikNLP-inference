from fastapi import Header, HTTPException, status
import os
from app.core.logger import logger


def verify_api_key(nlp_api_key: str | None = Header(None)):
    if not nlp_api_key:
        logger.warning("⚠️ Missing API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    if nlp_api_key != os.getenv("API_KEY"):
        logger.warning("⚠️ Invalid API key provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )