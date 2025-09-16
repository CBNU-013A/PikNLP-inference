# /app/schemas/llm_schema.py

from pydantic import BaseModel, Field

class LLMResponse(BaseModel):
    summary: str = Field(..., min_length=1)

class ReviewSummaryRequest(BaseModel):
    Reviews: list[str] = Field(..., min_length=1)

class OverviewSummaryRequest(BaseModel):
    Overview: str = Field(..., min_length=1)