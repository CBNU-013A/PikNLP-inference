# /app/services/llm_runner.py

import os
from openai import AsyncOpenAI

from ..core.logger import logger
from ..core.config import load_env
from .llm_prompts import review_summary_prompt, overview_summary_prompt


class LLMRunner:
    def __init__(self):
        self.client: AsyncOpenAI | None = None
        self.model = "gpt-5"
        self.reasoning = {"effort": "minimal"}

    def _ensure_client(self) -> AsyncOpenAI:
        if self.client is None:
            load_env()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set. Set env or .env before using LLM.")
            self.client = AsyncOpenAI(api_key=api_key)
        return self.client

    async def review_summary(self, reviews: list[str]) -> str:
        prompt_user = "\n".join(reviews)
        messages = [
            {"role": "system", "content": review_summary_prompt.strip()},
            {"role": "user", "content": prompt_user},
        ]
        client = self._ensure_client()
        resp = await client.responses.create(
            model=self.model,
            input=messages,
            reasoning=self.reasoning,
        )
        return resp.output_text or ""

    async def overview_summary(self, overview_text: str) -> str:
        messages = [
            {"role": "system", "content": overview_summary_prompt.strip()},
            {"role": "user", "content": overview_text},
        ]
        client = self._ensure_client()
        resp = await client.responses.create(
            model=self.model,
            input=messages,
            reasoning=self.reasoning,
        )
        return resp.output_text or ""


# Singleton instance for injection-less use
llm_runner = LLMRunner()