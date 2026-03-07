"""
OpenAI Embedding provider.

MOCK_MODE=true  → 더미 랜덤 벡터 반환 (API 호출 없음)
MOCK_MODE=false → OpenAI API 호출 (OPENAI_API_KEY 필요)
"""

import random

import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from providers.base import EmbedProvider

_BATCH_SIZE = 16
_MOCK_DIM = 1536  # text-embedding-3-small dimension


class OpenAIEmbedProvider(EmbedProvider):
    """OpenAI 공식 API로 임베딩 생성."""

    def __init__(self) -> None:
        self._client: OpenAI | None = None
        self._model: str = settings.OPENAI_EMBED_MODEL

        if not settings.MOCK_MODE and settings.OPENAI_API_KEY:
            self._client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                http_client=httpx.Client(verify=False),
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if settings.MOCK_MODE or self._client is None:
            return self._mock_embed(texts)
        return self._batched_embed(texts)

    def _batched_embed(self, texts: list[str]) -> list[list[float]]:
        all_embs: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            all_embs.extend(self._call_api(batch))
        return all_embs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        assert self._client is not None
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def _mock_embed(texts: list[str]) -> list[list[float]]:
        return [
            [random.uniform(-1, 1) for _ in range(_MOCK_DIM)]
            for _ in texts
        ]
