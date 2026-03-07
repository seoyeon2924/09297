"""
Ingest Service — 파일 업로드, 파싱, 청킹, 임베딩, Chroma 인덱싱.

Pipeline:
    파일 저장 → 파서(PDF/Excel) → 청킹 → 임베딩 → Chroma upsert → SQLite 저장
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import settings
from storage.models import DocStatus
from storage.repository import DocumentRepository, AuditRepository
from storage.chroma_store import (
    chroma_store,
    get_collection_name_for_doc_type,
)
from ingest.parser_pdf import PDFParser
from ingest.parser_excel import ExcelParser
from ingest.chunker import Chunker
from providers.embed_openai import OpenAIEmbedProvider

logger = logging.getLogger(__name__)

# 대용량 사례(Excel) 시 메모리·연결 부담 완화: 한 번에 처리하는 단위
_PIPELINE_BATCH = 50   # 이 개수씩 임베딩 후 Chroma 저장 (전체를 메모리에 쌓지 않음)
_EMBED_BATCH = 8       # OpenAI 임베딩 API 한 번에 보내는 개수 (embedder 내부 배치)
_CHROMA_BATCH = 30     # Chroma upsert 한 번에 보내는 개수
_CHROMA_SLEEP = 0.03   # Chroma 배치 간 짧은 대기(초) — 서버 과부하 완화

DOC_TYPE_NORMALIZE = {
    "법령": "law",
    "규정": "regulation",
    "지침": "guideline",
    "사례": "case",
}


class IngestService:

    _embedder: OpenAIEmbedProvider | None = None

    @classmethod
    def _get_embedder(cls) -> OpenAIEmbedProvider:
        if cls._embedder is None:
            cls._embedder = OpenAIEmbedProvider()
        return cls._embedder

    @staticmethod
    def upload_and_index(
        file,
        doc_type: str,
        category: str,
        scope: str,
        uploaded_by: str,
    ) -> dict:
        """전체 인덱싱 파이프라인 (동기)."""
        start_time = time.time()

        # 1. 파일 저장
        file_path = settings.UPLOAD_DIR / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # 2. 문서 레코드
        doc_id = DocumentRepository.create(
            filename=file.name,
            doc_type=doc_type,
            category=category,
            scope=scope,
            file_path=str(file_path),
            uploaded_by=uploaded_by,
        )
        DocumentRepository.update_status(doc_id, DocStatus.INDEXING.value)

        try:
            # 3. 파싱 + 청킹 (문서 유형별 구조 인식)
            suffix = Path(file.name).suffix.lower()
            normalized = DOC_TYPE_NORMALIZE.get(doc_type, doc_type)
            chunker = Chunker()

            if suffix == ".pdf":
                full_text = PDFParser.get_full_text(str(file_path))
                if not full_text.strip():
                    raise ValueError("문서에서 추출된 텍스트가 없습니다.")
                if normalized in ("law", "regulation"):
                    chunks = (
                        chunker.chunk_law(full_text, file.name)
                        if normalized == "law"
                        else chunker.chunk_regulation(full_text, file.name)
                    )
                elif normalized == "guideline":
                    chunks = chunker.chunk_guideline(full_text, file.name)
                else:
                    chunks = chunker.chunk_fallback(full_text, file.name)
            elif suffix == ".xlsx":
                rows = ExcelParser.parse(str(file_path))
                if not rows:
                    raise ValueError("문서에서 추출된 행이 없습니다.")
                chunks = chunker.chunk_cases(rows)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

            if not chunks:
                raise ValueError("문서에서 추출된 텍스트가 없습니다.")

            logger.debug(
                "청크 생성 완료: %d건 / 최대 %d자 / 평균 %d자",
                len(chunks),
                max(len(c["content"]) for c in chunks),
                sum(len(c["content"]) for c in chunks) // len(chunks),
            )

            # 임베딩용 텍스트 = content (순수 시맨틱 내용)
            embed_texts = [(c["content"].strip() or " ") for c in chunks]
            # Chroma documents용 = display_text (헤더 포함, LLM 컨텍스트용)
            # display_text가 없으면 content를 그대로 사용 (법률/규정/지침은 분리 불필요)
            display_texts = [
                (c.get("display_text") or c["content"]).strip() or " "
                for c in chunks
            ]
            total = len(embed_texts)
            collection_key = get_collection_name_for_doc_type(doc_type)
            chroma_ids = [f"{doc_id}_chunk_{c['chunk_index']}" for c in chunks]
            metadatas = [
                {
                    "document_id": doc_id,
                    "doc_type": doc_type,
                    "category": category or "",
                    "scope": scope or "",
                    "page_or_row": c.get("page_or_row", ""),
                    "source_file": c.get("source_file", ""),
                    "doc_structure_type": c.get("doc_structure_type", ""),
                    "case_number": c.get("case_number", ""),
                    "case_date": c.get("case_date", ""),
                    "chapter": c.get("chapter", ""),
                    "section": c.get("section", ""),
                    "article_number": c.get("article_number", ""),
                    "article_title": c.get("article_title", ""),
                    "major_section": c.get("major_section", ""),
                    "sub_section": c.get("sub_section", ""),
                    "sub_detail": c.get("sub_detail", ""),
                    "violation_type": c.get("violation_type", ""),
                    "limit_expression": c.get("limit_expression", ""),
                    "product_summary": c.get("product_summary", ""),
                }
                for c in chunks
            ]

            # 4·5. 배치 단위로 임베딩 → Chroma 저장 (전체 임베딩을 메모리에 쌓지 않음, 대용량 사례 대응)
            embedder = IngestService._get_embedder()
            progress = st.progress(0, text="임베딩 및 Chroma 저장 중...")
            for i in range(0, total, _PIPELINE_BATCH):
                end = min(i + _PIPELINE_BATCH, total)
                batch_embed = embed_texts[i:end]
                batch_display = display_texts[i:end]
                batch_embeddings = embedder.embed(batch_embed)
                # Chroma는 작은 단위로 나눠 보냄 (연결/타임아웃 완화)
                for j in range(0, len(batch_embed), _CHROMA_BATCH):
                    j_end = min(j + _CHROMA_BATCH, len(batch_embed))
                    chroma_store.upsert(
                        ids=chroma_ids[i + j : i + j_end],
                        documents=batch_display[j:j_end],
                        embeddings=batch_embeddings[j:j_end],
                        metadatas=metadatas[i + j : i + j_end],
                        collection_key=collection_key,
                    )
                    if _CHROMA_SLEEP and _CHROMA_SLEEP > 0:
                        time.sleep(_CHROMA_SLEEP)
                progress.progress(end / total, text=f"임베딩·Chroma 저장 중... ({end}/{total})")
            progress.empty()

            # Chroma 인덱싱 후 BM25 인덱스 캐시 무효화 (하이브리드 검색 인덱스 갱신)
            from utils.hybrid_search import get_hybrid_engine
            get_hybrid_engine().invalidate(collection_key)

            # 6. SQLite 청크 레코드
            chunk_records = [
                {
                    "chunk_index": c["chunk_index"],
                    "content_preview": c["content"][:200],
                    "page_or_row": c["page_or_row"],
                    "source_file": c["source_file"],
                    "doc_type": doc_type,
                    "chroma_id": chroma_ids[idx],
                }
                for idx, c in enumerate(chunks)
            ]
            chunk_count = DocumentRepository.create_chunks(doc_id, chunk_records)

            # 7. 완료
            elapsed = round(time.time() - start_time, 1)
            DocumentRepository.update_status(
                doc_id,
                DocStatus.INDEXED.value,
                chunk_count=chunk_count,
                indexed_at=datetime.utcnow(),
            )
            AuditRepository.create_log(
                event_type="INGEST",
                entity_type="ReferenceDocument",
                entity_id=doc_id,
                actor=uploaded_by,
                detail={
                    "filename": file.name,
                    "chunk_count": chunk_count,
                    "elapsed_sec": elapsed,
                    "mock_mode": settings.MOCK_MODE,
                },
            )
            return {
                "doc_id": doc_id,
                "filename": file.name,
                "chunk_count": chunk_count,
                "status": DocStatus.INDEXED.value,
                "elapsed_sec": elapsed,
            }

        except Exception as e:
            DocumentRepository.update_status(
                doc_id,
                DocStatus.INDEX_FAILED.value,
                error_message=str(e),
            )
            raise

    # ──────────────────────────────────────────
    # 조회
    # ──────────────────────────────────────────

    @staticmethod
    def list_documents() -> list[dict]:
        return DocumentRepository.list_all()

    @staticmethod
    def get_document(doc_id: str) -> dict | None:
        return DocumentRepository.get(doc_id)

    @staticmethod
    def get_chunks(doc_id: str) -> list[dict]:
        return DocumentRepository.list_chunks(doc_id)

    @staticmethod
    def generate_advanced_metadata(document_id: str) -> dict:
        return {
            "document_id": document_id,
            "status": "NOT_IMPLEMENTED",
            "message": "고급 메타데이터 생성은 추후 구현됩니다.",
        }