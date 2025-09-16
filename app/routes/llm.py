# /app/routes/llm.py

from fastapi import APIRouter, HTTPException
from ..core.logger import logger
from ..schemas.llm_schema import ReviewSummaryRequest, OverviewSummaryRequest, LLMResponse
from ..services.llm_runner import llm_runner
from fastapi import status
from fastapi import Body


def _sanitize_text(text: str) -> str:
    # 기본적인 제어문자 제거 및 공백 정리
    cleaned = text.replace("\r", "\n").replace("\t", " ")
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    # 유니코드 특수 따옴표 통일
    quotes_map = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
    }
    for k, v in quotes_map.items():
        cleaned = cleaned.replace(k, v)
    return cleaned.strip()

router = APIRouter()

@router.post("/reviewSummary", response_model=LLMResponse)
async def review_summary(request: ReviewSummaryRequest):
    logger.info("POST /reviewSummary - payload received")
    try:
        reviews = [t for t in ( _sanitize_text(t) for t in request.Reviews ) if t]
        if not reviews:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Reviews must not be empty after sanitization.")
        result = await llm_runner.review_summary(reviews)
        return LLMResponse(summary=result)
    except HTTPException:
        # 그대로 전파
        raise
    except Exception as e:
        logger.exception("POST /reviewSummary - error during review summary: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/overviewSummary", response_model=LLMResponse)
async def overview_summary(request: OverviewSummaryRequest):
    logger.info("POST /overviewSummary - payload received")
    try:
        overview = _sanitize_text(request.Overview)
        if not overview:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Overview must not be empty after sanitization.")
        result = await llm_runner.overview_summary(overview)
        return LLMResponse(summary=result)
    except HTTPException:
        # 그대로 전파
        raise
    except Exception as e:
        logger.exception("POST /overviewSummary - error during overview summary: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")