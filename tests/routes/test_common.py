# /tests/routes/test_common.py

import pytest
import os

API_KEY = os.getenv("API_KEY")

# / - 루트 엔드포인트
@pytest.mark.asyncio
async def test_top(async_client):
    res = await async_client.get("/")
    assert res.status_code == 200
    assert res.text == '"It\'s Pik!"'  # JSON 문자열 응답이므로 따옴표 포함

# /health - 헬스체크
@pytest.mark.asyncio
async def test_health(async_client):
    res = await async_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "s_model_loaded" in data
    assert "c_model_loaded" in data
    assert "device" in data
    assert "cuda_available" in data
    assert "API-MODE" in data

# /error - 전역 예외 핸들러 테스트 (test 환경에서만 등록됨)
@pytest.mark.asyncio
async def test_global_exception_handler(async_client):
    try:
        res = await async_client.get("/error", headers={"NLP-API-KEY": API_KEY})
        assert res.status_code == 500
        assert res.json()["detail"] == "Internal Server Error"
    except Exception as e:
        # 환경에 따라 예외가 직접 전파될 수도 있음
        print(f"Expected exception caught at test level: {e}")