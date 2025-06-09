# /app/routes/common.py

from fastapi import APIRouter, Depends
from ..core.logger import logger
from ..services.inference_runner import model_loader
import os
import torch

router = APIRouter()

@router.get("/")
def top():
    return "It's Pik!"

@router.get("/health")
def health():
    logger.info("GET /health - Health check requested.")
    return {
        "status": "ok",
        "s_model_loaded": model_loader.s_model is not None,
        "c_model_loaded": model_loader.c_model is not None,
        "device": str(model_loader.device),
        "cuda_available": torch.cuda.is_available(),
        "API-MODE": os.getenv("ENV")
        }

if os.getenv("ENV") == "test":
    logger.info("Test mode enabled. add /error endpoint.")
    
    @router.get("/error")
    def raise_error():
        logger.info("GET /error - Raising an error for testing.")
        raise Exception("intentional error for testing")
