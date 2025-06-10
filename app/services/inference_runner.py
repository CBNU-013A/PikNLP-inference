# /app/services/inference_runner.py

from transformers import ElectraTokenizer, ElectraForSequenceClassification, ElectraModel, AutoConfig
import torch
import torch.nn as nn
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from ..core.config import Config, CategoryConfig
from ..core.logger import logger

app_config = Config()

class MultiHeadClassifier(nn.Module):
    def __init__(self, config: CategoryConfig):
        super().__init__()
        self.encoder = ElectraModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.config = config

        # 카테고리마다 head를 따로 두는 구조
        self.heads = nn.ModuleDict({
            cat: nn.Linear(self.encoder.config.hidden_size, len(sub_labels))
            for cat, sub_labels in config.category2label.items()
        })

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = self.dropout(outputs.last_hidden_state[:, 0])  # [CLS] token

        logits = {
            cat: head(cls_output)
            for cat, head in self.heads.items()
        }
        return logits  # {"장소": logits1, "활동": logits2, ...}
    @classmethod
    def from_pretrained(cls, pretrained_dir: str | Path, config: CategoryConfig):
        # 1) 모델 빈 껍데기 생성
        model = cls(config)
        model.config = config
        # 2) safetensors 로드
        from safetensors.torch import load_file
        import os
        path = os.path.join(pretrained_dir, "model.safetensors")
        state_dict = load_file(path)
        # 3) 가중치 주입
        model.load_state_dict(state_dict)
        return model

class ModelLoader:
    def __init__(self, config: Config):
        # 1. Config
        self.config = config

        # 2. Device 설정
        self.device = torch.device(self.config.device if self.config.device == 'cuda' and torch.cuda.is_available() else 'cpu')
        logger.info("Using device: %s", self.device)

        # 3. Tokenizer: 공통
        self.tokenizer = ElectraTokenizer.from_pretrained(self.config.tokenizer_name)
        
        # 4. Sentiment Model
        sent_cfg = AutoConfig.from_pretrained(self.config.sentiment_model)
        self.s_id2label = {int(k): v for k, v in sent_cfg.id2label.items()}
        self.s_label2id = {v: int(k) for v, k in sent_cfg.label2id.items()}
        self.s_model = ElectraForSequenceClassification.from_pretrained(
            self.config.sentiment_model).to(self.device)
        logger.info("Loaded sentiment model: %s", self.config.sentiment_model)
        self.sentiment_map = self.config.sentiment_label_map
        
        # 5. Category Model
        # 5.1. Huggingface 모델 로드
        model_file = hf_hub_download(
            repo_id=self.config.category_model,
            filename="model.safetensors",
            repo_type="model"
        )
        config_file = hf_hub_download(
            repo_id=self.config.category_model,
            filename="config.json",
            repo_type="model"
        )
        # 5.2. 모델 로드
        cat_cfg = AutoConfig.from_pretrained(self.config.category_model)
        #5.4 MultiHeadClassifier 초기화
        self.c_model = MultiHeadClassifier(cat_cfg)
        state_dict = load_file(model_file)
        self.c_model.load_state_dict(state_dict)
        self.c_model.to(self.device).eval()
        logger.info("Loaded category model: %s", self.config.category_model)
        
        # 스레드 풀 생성 (CPU 작업용)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.inference.get("num_workers", 4))
    
        logger.info("✅ ModelLoader initialized successfully.")

    # TODO: Sentiment 검증
    def convert_to_feature(self, text: str, category: str):
        # 입력값 검증
        if not isinstance(text, str) or not isinstance(category, str):
            raise ValueError("text and category must be strings")

        max_length = self.c_model.config.max_seq_length
        encoded = self.tokenizer(
            text=text,
            text_pair=category,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded.get("token_type_ids", torch.zeros((1, max_length), dtype=torch.long))
        }
    
    async def _predict_category(self, text: str, category: str) -> tuple[str, str]:
        logger.debug("Starting prediction for category: %s", category)
        # 토크나이징은 CPU에서 수행
        inputs = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.convert_to_feature,
            text,
            category
        )
        
        # GPU 연산은 별도의 스레드에서 실행
        def run_cat():
            inp = {k:v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                # MultiHeadClassifier forward -> dict[cat: logits]
                logits_dict = self.c_model(**inp)
                logits = logits_dict[category]
                idx = torch.argmax(logits, dim=-1).item()
            return category, self.c_model.config.category2label[category][idx]
        cat, pred = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            run_cat
            )
        return cat, pred
    
    async def _predict_sentiment(self, text: str, sentiment_category: str) -> tuple[str, str]:
        logger.debug("Starting prediction for Sentiment: %s", sentiment_category)
        # 토크나이징은 CPU에서 수행
        inputs = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.convert_to_feature,
            text,
            sentiment_category
        )
        
        # GPU 연산은 별도의 스레드에서 실행
        def run_inference():
            inputs_gpu = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.s_model(**inputs_gpu)
                predictions = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
            return self.sentiment_map[predicted_class]
            
        sentiment = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            run_inference
        )
        logger.debug("Predicted sentiment for category %s: %s", sentiment_category, sentiment)
        
        return sentiment_category, sentiment

    # TODO: 엔드포인트 예측
    async def predict(self, text: str) -> dict:
        # 입력값 검증
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        logger.info("Starting prediction for text: %s", text)
        # 모든 카테고리에 대해 병렬로 추론 수행
        
        category_list = list(self.c_model.heads.keys())
        sentiment_tasks = [
            self._predict_sentiment(text, cat)
            for cat in self.s_id2label.values()
        ]
        category_tasks = [
            self._predict_category(text, cat)
            for cat in category_list
        ]
        # 병렬 실행
        sentiment_results = await asyncio.gather(*sentiment_tasks)
        category_results = await asyncio.gather(*category_tasks)

        # 결과 정리
        sentiment_map = {cat: lbl for cat, lbl in sentiment_results}
        category_map = {cat: lbl for cat, lbl in category_results}
        
        logger.info("✅ Completed prediction")
        return {
            "sentiment": sentiment_map,
            "category": category_map
        }

    
    # TODO: Sentiment, Category 모델 반영
    async def get_categories(self) -> dict:
        logger.info("Fetching categories")
        # 카테고리 목록 반환
        sentimnet_model = list(self.s_id2label.values())
        category_map = self.c_model.config.category2label

        logger.info("✅ Fetched categories: %s, %s", sentimnet_model, category_map)
        return {"sentiment_model": sentimnet_model, "category_map": category_map}

# 싱글톤 인스턴스 생성
model_loader = ModelLoader(app_config)
