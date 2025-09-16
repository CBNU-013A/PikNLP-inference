# /tests/routes/test_llm_routes.py

import os
import pytest
from fastapi.testclient import TestClient

# Ensure env for API key exists during tests
os.environ.setdefault("API_KEY", "test-key")

from app.main import app

client = TestClient(app)


def _headers():
    return {"nlp-api-key": os.getenv("API_KEY")}


@pytest.fixture(autouse=True)
def mock_openai_responses(monkeypatch):
    # Patch LLM client to avoid real network
    from app.services.llm_runner import llm_runner

    class MockResp:
        def __init__(self, text: str):
            self._text = text
        @property
        def output_text(self):
            return self._text

    class MockResponses:
        async def create(self, **kwargs):
            return MockResp("요약 결과")

    class MockClient:
        def __init__(self):
            self.responses = MockResponses()

    def ensure_client():
        return MockClient()

    monkeypatch.setattr(llm_runner, "_ensure_client", ensure_client)
    yield


def test_overview_summary_success():
    payload = {"Overview": "이것은 소개문입니다. 여러 문장으로 구성됩니다."}
    res = client.post("/api/v1/llm/overviewSummary", json=payload, headers=_headers())
    assert res.status_code == 200
    assert res.json()["summary"] == "요약 결과"


def test_review_summary_success():
    payload = {"Reviews": ["맛있어요", "친절해요"]}
    res = client.post("/api/v1/llm/reviewSummary", json=payload, headers=_headers())
    assert res.status_code == 200
    assert res.json()["summary"] == "요약 결과"


def test_overview_summary_422_on_empty_after_sanitize():
    payload = {"Overview": "\n\n\t   "}
    res = client.post("/api/v1/llm/overviewSummary", json=payload, headers=_headers())
    assert res.status_code == 422


def test_review_summary_422_on_empty_list():
    payload = {"Reviews": []}
    res = client.post("/api/v1/llm/reviewSummary", json=payload, headers=_headers())
    assert res.status_code == 422


def test_unauthorized_when_api_key_missing():
    payload = {"Overview": "간단 소개"}
    res = client.post("/api/v1/llm/overviewSummary", json=payload)  # no header
    # dependencies.verify_api_key returns 401 on missing
    assert res.status_code == 401 or res.status_code == 422


def test_unauthorized_when_api_key_invalid():
    payload = {"Overview": "간단 소개"}
    res = client.post(
        "/api/v1/llm/overviewSummary",
        json=payload,
        headers={"nlp-api-key": "wrong"},
    )
    assert res.status_code == 401


def test_quotes_and_whitespace_sanitization():
    # fancy quotes and tabs/newlines should be normalized and not cause 422
    payload = {"Overview": "“스페셜” 메뉴가\t인기입니다.\n‘추천’합니다."}
    res = client.post("/api/v1/llm/overviewSummary", json=payload, headers=_headers())
    assert res.status_code == 200
    assert res.json()["summary"] == "요약 결과"


def test_reviews_filtered_and_processed():
    payload = {"Reviews": ["\n\t", "  ", "좋아요", "“최고”"]}
    res = client.post("/api/v1/llm/reviewSummary", json=payload, headers=_headers())
    assert res.status_code == 200
    assert res.json()["summary"] == "요약 결과"


def test_llm_runner_raises_returns_500(monkeypatch):
    from app.services.llm_runner import llm_runner

    class MockRespFail:
        async def create(self, **kwargs):
            raise RuntimeError("network error")

    class MockClientFail:
        def __init__(self):
            self.responses = MockRespFail()

    def ensure_client_fail():
        return MockClientFail()

    monkeypatch.setattr(llm_runner, "_ensure_client", ensure_client_fail)

    res = client.post(
        "/api/v1/llm/overviewSummary",
        json={"Overview": "정상 입력"},
        headers=_headers(),
    )
    assert res.status_code == 500


# --- Additional sanitization robustness tests ---

def test_overview_with_emojis_and_zero_width():
    # zero-width space \u200b and emojis should not break sanitization
    txt = "맛있어요\u200b 😋 최고!"
    res = client.post(
        "/api/v1/llm/overviewSummary",
        json={"Overview": txt},
        headers=_headers(),
    )
    assert res.status_code == 200


def test_overview_html_tags_are_treated_as_text():
    txt = "<b>강조</b>와 <script>alert('xss')</script> 같은 태그 포함"
    res = client.post(
        "/api/v1/llm/overviewSummary",
        json={"Overview": txt},
        headers=_headers(),
    )
    assert res.status_code == 200


def test_overview_multiline_trim_behavior():
    txt = " 첫 줄  \n\t둘째 줄\n   셋째 줄  "
    res = client.post(
        "/api/v1/llm/overviewSummary",
        json={"Overview": txt},
        headers=_headers(),
    )
    assert res.status_code == 200


def test_overview_very_long_text():
    txt = ("가" * 5000) + "\n" + ("나" * 5000)
    res = client.post(
        "/api/v1/llm/overviewSummary",
        json={"Overview": txt},
        headers=_headers(),
    )
    assert res.status_code == 200


def test_reviews_mixed_collapse_to_single_valid():
    payload = {"Reviews": ["", "\n", "  ", "유효한 리뷰"]}
    res = client.post("/api/v1/llm/reviewSummary", json=payload, headers=_headers())
    assert res.status_code == 200


def test_reviews_all_invalid_after_sanitize_422():
    payload = {"Reviews": ["\n\t", "  ", "\r\n", "\t\t"]}
    res = client.post("/api/v1/llm/reviewSummary", json=payload, headers=_headers())
    assert res.status_code == 422
