"""
심의 사례 검색 Tool — LangChain @tool 래퍼.

cases 컬렉션에서 과거 심의 사례를 하이브리드 검색(BM25+Vector+RRF) → Cohere 리랭킹하여 반환한다.
"""

from __future__ import annotations

from langchain_core.tools import tool

from providers.embed_openai import OpenAIEmbedProvider
from utils.hybrid_search import get_hybrid_engine
from utils.reranker import rerank_chunks

_embedder: OpenAIEmbedProvider | None = None


def _get_query_embedding(query: str) -> list[float]:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbedProvider()
    return _embedder.embed([query])[0]


@tool
def search_cases(query: str) -> dict:
    """심의 사례 검색: 주어진 질의와 유사한 과거 심의 사례를 하이브리드 검색합니다.

    Args:
        query: 검색할 질의 문자열 (예: "오늘만 한정 가격")

    Returns:
        case_chunks — 유사 심의 사례 청크 목록 (최대 20건)
    """
    query_embedding = _get_query_embedding(query)

    # 1단계: 하이브리드 검색 (BM25 + Vector + RRF)
    engine = get_hybrid_engine()
    hybrid_results = engine.search(
        collection_key="cases",
        query_text=query,
        query_embedding=query_embedding,
        vector_top_n=20,
        bm25_top_n=20,
        final_top_n=20,
    )

    # 2단계: Cohere 리랭킹으로 정밀 재정렬
    reranked = rerank_chunks(query=query, chunks=hybrid_results, top_n=5)

    return {"case_chunks": reranked}