"""
ChromaDB collection wrapper.

문서 유형별 컬렉션 분리:
  regulations — 법률(law) + 규정(regulation)
  guidelines  — 지침(guideline)
  cases       — 과거 심의 사례(case)
  general     — fallback
"""

import chromadb

from config import settings

COLLECTION_REGULATIONS = "regulations"
COLLECTION_GUIDELINES = "guidelines"
COLLECTION_CASES = "cases"
COLLECTION_GENERAL = "general"

ALL_COLLECTION_NAMES = (
    COLLECTION_REGULATIONS,
    COLLECTION_GUIDELINES,
    COLLECTION_CASES,
    COLLECTION_GENERAL,
)

DOC_TYPE_TO_COLLECTION = {
    "law": COLLECTION_REGULATIONS,
    "regulation": COLLECTION_REGULATIONS,
    "규정": COLLECTION_REGULATIONS,
    "법령": COLLECTION_REGULATIONS,
    "guideline": COLLECTION_GUIDELINES,
    "지침": COLLECTION_GUIDELINES,
    "case": COLLECTION_CASES,
    "사례": COLLECTION_CASES,
}


def get_collection_name_for_doc_type(doc_type: str) -> str:
    return DOC_TYPE_TO_COLLECTION.get(doc_type, COLLECTION_GENERAL)


class ChromaStore:
    """문서 유형별 컬렉션 관리 + upsert/query."""

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collections: dict[str, chromadb.Collection] = {}

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    def get_collection(self, name: str) -> chromadb.Collection:
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        out = {}
        for k, v in meta.items():
            if v is None:
                out[k] = ""
            elif isinstance(v, (int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v).strip()[:500]
        return out

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]] | None = None,
        collection_key: str | None = None,
    ) -> None:
        name = collection_key or COLLECTION_GENERAL
        coll = self.get_collection(name)
        clean_metas = [self._sanitize_metadata(m) for m in metadatas]
        kwargs: dict = {
            "ids": ids,
            "documents": documents,
            "metadatas": clean_metas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        coll.upsert(**kwargs)

    def query(
        self,
        collection_key: str,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        name = collection_key or COLLECTION_GENERAL
        coll = self.get_collection(name)
        kwargs: dict = {"n_results": n_results}
        if query_texts is not None:
            kwargs["query_texts"] = query_texts
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        if where is not None:
            kwargs["where"] = where
        return coll.query(**kwargs)

    def count(self, collection_key: str | None = None) -> int:
        if collection_key:
            return self.get_collection(collection_key).count()
        total = 0
        for name in ALL_COLLECTION_NAMES:
            try:
                total += self.client.get_collection(name).count()
            except Exception:
                pass
        return total

    def get_stats(self, sample_limit: int = 5) -> list[dict]:
        """
        컬렉션별 벡터 개수와 샘플 ID/문서를 반환.
        반환: [{"name": str, "count": int, "sample_ids": list[str], "sample_docs": list[str]}, ...]
        """
        result: list[dict] = []
        for name in ALL_COLLECTION_NAMES:
            try:
                coll = self.client.get_collection(name)
                n = coll.count()
                sample_ids: list[str] = []
                sample_docs: list[str] = []
                if n > 0 and sample_limit > 0:
                    peek = coll.peek(limit=min(sample_limit, n))
                    raw_ids = peek.get("ids")
                    if raw_ids:
                        # Chroma may return list or list of lists
                        flat = raw_ids[0] if (raw_ids and isinstance(raw_ids[0], list)) else raw_ids
                        sample_ids = list(flat)[:sample_limit] if flat else []
                    raw_docs = peek.get("documents")
                    if raw_docs:
                        flat_docs = raw_docs[0] if (raw_docs and isinstance(raw_docs[0], list)) else raw_docs
                        sample_docs = [
                            (d[:120] + "…" if d and len(d) > 120 else (d or ""))
                            for d in (list(flat_docs)[:sample_limit] if flat_docs else [])
                        ]
                result.append({
                    "name": name,
                    "count": n,
                    "sample_ids": sample_ids,
                    "sample_docs": sample_docs,
                })
            except Exception:
                result.append({
                    "name": name,
                    "count": 0,
                    "sample_ids": [],
                    "sample_docs": [],
                })
        return result

    def delete(self, ids: list[str], collection_key: str | None = None) -> None:
        name = collection_key or COLLECTION_GENERAL
        self.get_collection(name).delete(ids=ids)

    def reset_collection(self, collection_key: str) -> None:
        """컬렉션을 삭제 후 재생성 (기존 벡터 전체 초기화).

        Args:
            collection_key: 초기화할 컬렉션 키 (예: "cases")
        """
        name = collection_key or COLLECTION_GENERAL
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        # 캐시 제거 후 재생성
        self._collections.pop(name, None)
        self.get_collection(name)


chroma_store = ChromaStore()
