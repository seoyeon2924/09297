"""
규정·지침 검색 Tool — LangChain @tool 래퍼.

regulations(법률 + 규정) 컬렉션과 guidelines(지침) 컬렉션에서
관련 근거를 하이브리드 검색(BM25+Vector+RRF) → Cohere 리랭킹하여 반환한다.
"""

from __future__ import annotations

from langchain_core.tools import tool

from providers.embed_openai import OpenAIEmbedProvider
from storage.chroma_store import chroma_store
from utils.hybrid_search import get_hybrid_engine
from utils.reranker import rerank_chunks

_LAW_DOC_TYPES = {"법령", "law"}
_embedder: OpenAIEmbedProvider | None = None


def _get_query_embedding(query: str) -> list[float]:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbedProvider()
    return _embedder.embed([query])[0]


@tool
def search_policy(query: str) -> dict:
    """규정·지침 검색: 주어진 질의와 관련된 법률, 규정, 지침 근거를 하이브리드 검색합니다.

    Args:
        query: 검색할 질의 문자열 (예: "방송 한정판매 긴급성 표현")

    Returns:
        law_chunks — 법률 관련 청크
        regulation_chunks — 규정 관련 청크
        guideline_chunks — 지침 관련 청크
    """
    query_embedding = _get_query_embedding(query)
    engine = get_hybrid_engine()

    # 1) regulations 컬렉션 하이브리드 검색
    reg_results = engine.search(
        collection_key="regulations",
        query_text=query,
        query_embedding=query_embedding,
        vector_top_n=20,
        bm25_top_n=20,
        final_top_n=20,
    )

    # 2) 리랭킹 먼저, 그 후 법률/규정 분류
    reg_reranked = rerank_chunks(query=query, chunks=reg_results, top_n=10)

    law_chunks: list[dict] = []
    regulation_chunks: list[dict] = []
    for chunk in reg_reranked:
        dt = chunk["metadata"].get("doc_type", "")
        if dt in _LAW_DOC_TYPES:
            law_chunks.append(chunk)
        else:
            regulation_chunks.append(chunk)

    # 3) guidelines 컬렉션 하이브리드 검색
    guide_results = engine.search(
        collection_key="guidelines",
        query_text=query,
        query_embedding=query_embedding,
        vector_top_n=15,
        bm25_top_n=15,
        final_top_n=15,
    )

    # 4) 지침도 리랭킹
    guideline_reranked = rerank_chunks(query=query, chunks=guide_results, top_n=2)

    return {
        "law_chunks": law_chunks,
        "regulation_chunks": regulation_chunks,
        "guideline_chunks": guideline_reranked,
    }


@tool
def fetch_chunk_by_id(chroma_id: str, collection_key: str) -> dict:
    """Chroma ID로 특정 청크 원문을 조회합니다."""
    coll = chroma_store.get_collection(collection_key)
    result = coll.get(ids=[chroma_id])

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    if not ids:
        return {"content": "", "metadata": {}, "chroma_id": chroma_id}

    return {
        "content": documents[0] if documents else "",
        "metadata": metadatas[0] if metadatas else {},
        "chroma_id": ids[0],
    }