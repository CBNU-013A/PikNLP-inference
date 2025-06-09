# /app/schemas/inference_schema.py

from pydantic import BaseModel, Field
from typing import Dict

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

class PredictResponse(BaseModel):
    sentiments: Dict[str, str]  # 카테고리별 감성 분석 결과 (pos, neg, none)
    categories: Dict[str, str]  # 카테고리별 카테고리 분석 결과

class CategoriesResponse(BaseModel):
    categories: list[str]  # 카테고리 목록 
    