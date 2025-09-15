# /test/conftest.py

import pytest
import os
from dotenv import load_dotenv
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
from fastapi.testclient import TestClient

load_dotenv()

# 테스트 기본 환경 구성
os.environ["ENV"] = "test"
os.environ.setdefault("API_KEY", "test-key")
API_KEY = os.getenv("API_KEY")

from app.main import app

@pytest.fixture
async def async_client():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac

@pytest.fixture(scope="module")
def client():
    return TestClient(app)