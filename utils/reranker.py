"""
Cohere Reranker 유틸리티.

ChromaDB 검색 결과를 Cohere rerank-multilingual-v3.0 모델로 재정렬하여
진짜 관련 있는 문서만 상위로 올린다.
"""

from __future__ import annotations

import logging

import cohere
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config import settings  # settings.COHERE_API_KEY 필요

logger = logging.getLogger(__name__)

_client: cohere.Client | None = None


def _get_client() -> cohere.Client:
    global _client
    if _client is None:
        _client = cohere.Client(settings.COHERE_API_KEY)
    return _client


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_rerank(query: str, documents: list[str], top_n: int):
    """Cohere rerank API 호출 (tenacity 재시도 적용)."""
    return _get_client().rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=documents,
        top_n=top_n,
    )


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = 5,
    min_score: float = 0.3,
) -> list[dict]:
    """
    Cohere Reranker로 청크를 재정렬.

    Args:
        query: 사용자 검색 질의
        chunks: ChromaDB에서 가져온 청크 리스트 [{content, metadata, chroma_id, relevance_score}, ...]
        top_n: 리랭킹 후 반환할 최대 개수
        min_score: 이 점수 미만은 제외 (0~1)

    Returns:
        재정렬된 청크 리스트 (relevance_score가 리랭크 점수로 교체됨)
    """
    if not chunks:
        return []

    documents = [c["content"] for c in chunks]

    try:
        response = _call_rerank(query, documents, top_n)
    except Exception as e:
        # 리랭킹 실패 시 원본 그대로 반환 (fallback)
        logger.error("[reranker] Cohere rerank 실패, 원본 반환: %s", e)
        return chunks[:top_n]

    reranked: list[dict] = []
    for result in response.results:
        idx = result.index
        score = result.relevance_score

        if score < min_score:
            continue

        chunk = chunks[idx].copy()
        chunk["relevance_score"] = round(score, 4)
        chunk["rerank_score"] = round(score, 4)  # 원본 score와 구분용
        reranked.append(chunk)

    return reranked