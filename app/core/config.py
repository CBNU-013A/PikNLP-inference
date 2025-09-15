# /app/core/config.py

import os
import yaml
from ..core.logger import logger
from transformers import PretrainedConfig

def load_env():
    if "ENV" not in os.environ:
        from dotenv import load_dotenv
        load_dotenv()

class Config:
    def __init__(self, config_path: str = "app/services/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logger.info("Loaded configuration file from %s", config_path)

        self.model = cfg.get("model", {})
        self.category_model = self.model.get("c_name", "")
        self.sentiment_model = self.model.get("s_name", "")
        self.device = self.model.get("device", "cpu")
        self.tokenizer_name = self.model.get("tokenizer_name", "")
        self.dropout_rate = self.model.get("dropout_rate", 0.1)
        self.inference = cfg.get("inference", {})
        self.sentiment_label_map = cfg.get("sentiment_label_map", {})


class CategoryConfig(PretrainedConfig):
    def __init__(self,
                 category2label: dict[str, list[str]],
                 max_seq_length: int = 256,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.category2label = category2label
        self.categories = list(category2label.keys())
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate