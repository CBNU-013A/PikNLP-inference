# tests/services/test_inference_runner.py

import pytest
import torch

import os
import types
import pytest
import torch

from app.core.config import Config
import app.services.inference_runner as ir


@pytest.fixture(autouse=True)
def mock_external_dependencies(monkeypatch):
    # 항상 CPU로 강제
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # 가벼운 토크나이저/모델/설정 더미 구현
    class FakeTokenizer:
        def __init__(self, *_, **__):
            pass

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, text, text_pair, truncation, max_length, padding, return_tensors):
            seq_len = max_length
            return {
                "input_ids": torch.zeros((1, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
                "token_type_ids": torch.zeros((1, seq_len), dtype=torch.long),
            }

    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=16)

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
            batch, seq = input_ids.shape
            hidden = self.config.hidden_size
            last_hidden_state = torch.zeros((batch, seq, hidden))
            return types.SimpleNamespace(last_hidden_state=last_hidden_state)

    class FakeSentimentModel(torch.nn.Module):
        def __init__(self, num_labels=3):
            super().__init__()
            self.linear = torch.nn.Linear(16, num_labels)

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
            logits = torch.randn((1, 3))
            return types.SimpleNamespace(logits=logits)

    class FakeAutoConfig:
        def __init__(self, kind: str):
            if kind == "sentiment":
                self.id2label = {0: "pos", 1: "neg", 2: "none"}
                self.label2id = {v: k for k, v in self.id2label.items()}
            else:
                # category config 요구 필드
                self.category2label = {"분위기": ["아늑함", "시끌벅적"], "서비스": ["친절", "불친절"]}
                self.model_name = "monologg/koelectra-small-v3-discriminator"
                self.max_seq_length = 32
                self.dropout_rate = 0.1

        @classmethod
        def from_pretrained(cls, model_name: str, *_, **__):
            # 간단하게 이름으로 분기
            if "sentiment" in model_name:
                return cls("sentiment")
            return cls("category")

    # 외부 의존 모킹 적용
    monkeypatch.setattr(ir, "ElectraTokenizer", FakeTokenizer)
    monkeypatch.setattr(ir, "ElectraModel", FakeEncoder)
    monkeypatch.setattr(ir, "ElectraForSequenceClassification", FakeSentimentModel)
    monkeypatch.setattr(ir, "AutoConfig", FakeAutoConfig)
    monkeypatch.setattr(ir, "hf_hub_download", lambda **kwargs: "/tmp/dummy")
    monkeypatch.setattr(ir, "load_file", lambda path: {})
    monkeypatch.setattr(ir.MultiHeadClassifier, "load_state_dict", lambda self, sd: None)


@pytest.fixture(scope="module")
def model_loader():
    # 테스트 전용 설정: 장비는 cpu로, 나머지는 기본 yaml
    cfg = Config()
    cfg.device = "cpu"
    return ir.ModelLoader(cfg)


# TC1: convert_to_feature 기본 동작
def test_convert_to_feature_normal(model_loader):
    features = model_loader.convert_to_feature("좋았어요", "분위기")
    assert set(features.keys()) >= {"input_ids", "attention_mask", "token_type_ids"}
    for v in features.values():
        assert isinstance(v, torch.Tensor)


# TC2: predict 정상 동작
@pytest.mark.asyncio
async def test_predict_basic(model_loader):
    result = await model_loader.predict("정말 좋아요!")
    assert set(result.keys()) == {"sentiment", "category"}
    assert isinstance(result["sentiment"], dict)
    assert isinstance(result["category"], dict)


# TC3: get_categories 정상 동작
@pytest.mark.asyncio
async def test_get_categories_basic(model_loader):
    categories = await model_loader.get_categories()
    assert "sentiment_model" in categories and isinstance(categories["sentiment_model"], list)
    assert "category_map" in categories and isinstance(categories["category_map"], dict)

