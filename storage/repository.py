"""
Repository layer — CRUD operations for all entities.

Every public method opens its own session and returns plain dicts
so callers (services, UI) are decoupled from SQLAlchemy.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from storage.database import SessionLocal
from storage.models import (
    ReferenceDocument,
    Chunk,
    ReviewRequest,
    ReviewItem,
    AiRecommendation,
    HumanDecision,
    AuditLog,
    DocStatus,
    ReviewStatus,
)


# ───────────────────────────────────────────
# Session helper
# ───────────────────────────────────────────

@contextmanager
def get_db():
    """Provide a transactional database session scope."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ═══════════════════════════════════════════
# ReferenceDocument / Chunk
# ═══════════════════════════════════════════

class DocumentRepository:

    @staticmethod
    def create(
        filename: str,
        doc_type: str,
        category: str,
        scope: str,
        file_path: str,
        uploaded_by: str,
    ) -> str:
        """Create a new document record. Returns doc id."""
        with get_db() as db:
            doc = ReferenceDocument(
                filename=filename,
                doc_type=doc_type,
                category=category,
                scope=scope,
                file_path=file_path,
                uploaded_by=uploaded_by,
            )
            db.add(doc)
            db.flush()
            return doc.id

    @staticmethod
    def list_all() -> list[dict]:
        with get_db() as db:
            docs = (
                db.query(ReferenceDocument)
                .order_by(ReferenceDocument.created_at.desc())
                .all()
            )
            return [
                {
                    "id": d.id,
                    "filename": d.filename,
                    "doc_type": d.doc_type,
                    "category": d.category,
                    "scope": d.scope,
                    "status": d.status,
                    "advanced_meta_status": d.advanced_meta_status,
                    "chunk_count": d.chunk_count,
                    "uploaded_by": d.uploaded_by,
                    "created_at": d.created_at,
                    "indexed_at": d.indexed_at,
                    "error_message": d.error_message,
                }
                for d in docs
            ]

    @staticmethod
    def get(doc_id: str) -> Optional[dict]:
        with get_db() as db:
            d = db.query(ReferenceDocument).filter_by(id=doc_id).first()
            if not d:
                return None
            return {
                "id": d.id,
                "filename": d.filename,
                "doc_type": d.doc_type,
                "category": d.category,
                "scope": d.scope,
                "status": d.status,
                "advanced_meta_status": d.advanced_meta_status,
                "chunk_count": d.chunk_count,
                "file_path": d.file_path,
                "uploaded_by": d.uploaded_by,
                "created_at": d.created_at,
                "indexed_at": d.indexed_at,
                "error_message": d.error_message,
            }

    @staticmethod
    def update_status(doc_id: str, status: str, **kwargs) -> None:
        with get_db() as db:
            doc = db.query(ReferenceDocument).filter_by(id=doc_id).first()
            if doc:
                doc.status = status
                for key, val in kwargs.items():
                    if hasattr(doc, key):
                        setattr(doc, key, val)

    @staticmethod
    def create_chunks(document_id: str, chunks_data: list[dict]) -> int:
        """Bulk-create chunks. Returns count created."""
        with get_db() as db:
            for cd in chunks_data:
                chunk = Chunk(
                    document_id=document_id,
                    chunk_index=cd["chunk_index"],
                    content_preview=cd.get("content_preview", ""),
                    page_or_row=cd.get("page_or_row"),
                    source_file=cd.get("source_file"),
                    doc_type=cd.get("doc_type"),
                    chroma_id=cd.get("chroma_id"),
                )
                db.add(chunk)
            # Update parent chunk_count
            doc = db.query(ReferenceDocument).filter_by(id=document_id).first()
            if doc:
                doc.chunk_count = len(chunks_data)
            return len(chunks_data)

    @staticmethod
    def list_chunks(document_id: str) -> list[dict]:
        """Return all chunks for a document, ordered by chunk_index."""
        with get_db() as db:
            chunks = (
                db.query(Chunk)
                .filter_by(document_id=document_id)
                .order_by(Chunk.chunk_index)
                .all()
            )
            return [
                {
                    "id": c.id,
                    "chunk_index": c.chunk_index,
                    "content_preview": c.content_preview,
                    "page_or_row": c.page_or_row,
                    "source_file": c.source_file,
                    "doc_type": c.doc_type,
                    "chroma_id": c.chroma_id,
                }
                for c in chunks
            ]


# ═══════════════════════════════════════════
# ReviewRequest / ReviewItem
# ═══════════════════════════════════════════

class ReviewRepository:

    @staticmethod
    def create_request(
        product_name: str,
        category: str,
        broadcast_type: str,
        requested_by: str,
        items: list[dict],
    ) -> dict:
        with get_db() as db:
            req = ReviewRequest(
                product_name=product_name,
                category=category,
                broadcast_type=broadcast_type,
                requested_by=requested_by,
            )
            db.add(req)
            db.flush()

            for item_data in items:
                item = ReviewItem(
                    request_id=req.id,
                    item_index=item_data["item_index"],
                    item_type=item_data["item_type"],
                    label=item_data["label"],
                    text=item_data["text"],
                )
                db.add(item)

            db.flush()
            return {
                "id": req.id,
                "status": req.status,
                "created_at": req.created_at,
            }

    @staticmethod
    def list_requests(status_filter: Optional[str] = None) -> list[dict]:
        with get_db() as db:
            q = db.query(ReviewRequest).order_by(
                ReviewRequest.created_at.desc()
            )
            if status_filter:
                q = q.filter(ReviewRequest.status == status_filter)
            requests = q.all()

            results = []
            for r in requests:
                item_count = (
                    db.query(ReviewItem).filter_by(request_id=r.id).count()
                )
                results.append(
                    {
                        "id": r.id,
                        "product_name": r.product_name,
                        "category": r.category,
                        "broadcast_type": r.broadcast_type,
                        "status": r.status,
                        "requested_by": r.requested_by,
                        "item_count": item_count,
                        "created_at": r.created_at,
                        "decided_at": r.decided_at,
                    }
                )
            return results

    @staticmethod
    def get_detail(request_id: str) -> Optional[dict]:
        with get_db() as db:
            req = db.query(ReviewRequest).filter_by(id=request_id).first()
            if not req:
                return None

            items = (
                db.query(ReviewItem)
                .filter_by(request_id=request_id)
                .order_by(ReviewItem.item_index)
                .all()
            )
            items_data = []
            for item in items:
                rec = (
                    db.query(AiRecommendation)
                    .filter_by(review_item_id=item.id)
                    .first()
                )
                rec_data = None
                if rec:
                    rec_data = {
                        "id": rec.id,
                        "judgment": rec.judgment,
                        "reason": rec.reason,
                        "references": rec.references,
                        "model_name": rec.model_name,
                        "prompt_version": rec.prompt_version,
                        "latency_ms": rec.latency_ms,
                        "created_at": rec.created_at,
                    }
                items_data.append(
                    {
                        "id": item.id,
                        "item_index": item.item_index,
                        "item_type": item.item_type,
                        "label": item.label,
                        "text": item.text,
                        "ai_recommendation": rec_data,
                    }
                )

            human_dec = (
                db.query(HumanDecision)
                .filter_by(request_id=request_id)
                .first()
            )
            dec_data = None
            if human_dec:
                dec_data = {
                    "id": human_dec.id,
                    "decision": human_dec.decision,
                    "comment": human_dec.comment,
                    "decided_by": human_dec.decided_by,
                    "created_at": human_dec.created_at,
                }

            return {
                "request": {
                    "id": req.id,
                    "product_name": req.product_name,
                    "category": req.category,
                    "broadcast_type": req.broadcast_type,
                    "status": req.status,
                    "requested_by": req.requested_by,
                    "created_at": req.created_at,
                    "decided_at": req.decided_at,
                },
                "items": items_data,
                "human_decision": dec_data,
            }

    @staticmethod
    def update_request_status(request_id: str, status: str, **kwargs) -> None:
        with get_db() as db:
            req = (
                db.query(ReviewRequest).filter_by(id=request_id).first()
            )
            if req:
                req.status = status
                for key, val in kwargs.items():
                    if hasattr(req, key):
                        setattr(req, key, val)

    @staticmethod
    def create_ai_recommendation(
        review_item_id: str,
        judgment: str,
        reason: str,
        references: list[dict],
        model_name: str,
        prompt_version: str,
        latency_ms: int,
    ) -> str:
        with get_db() as db:
            # Delete existing (supports re-run)
            db.query(AiRecommendation).filter_by(
                review_item_id=review_item_id
            ).delete()
            rec = AiRecommendation(
                review_item_id=review_item_id,
                judgment=judgment,
                reason=reason,
                references=references,
                model_name=model_name,
                prompt_version=prompt_version,
                latency_ms=latency_ms,
            )
            db.add(rec)
            db.flush()
            return rec.id

    @staticmethod
    def create_human_decision(
        request_id: str,
        decision: str,
        comment: str,
        decided_by: str,
    ) -> dict:
        with get_db() as db:
            # Delete existing (supports re-decision)
            db.query(HumanDecision).filter_by(request_id=request_id).delete()

            dec = HumanDecision(
                request_id=request_id,
                decision=decision,
                comment=comment,
                decided_by=decided_by,
            )
            db.add(dec)

            # Transition request status
            req = (
                db.query(ReviewRequest).filter_by(id=request_id).first()
            )
            if req:
                req.status = decision  # "DONE" or "REJECTED"
                req.decided_at = datetime.utcnow()

            db.flush()
            return {
                "id": dec.id,
                "decision": dec.decision,
                "comment": dec.comment,
                "decided_by": dec.decided_by,
                "created_at": dec.created_at,
            }


# ═══════════════════════════════════════════
# AuditLog
# ═══════════════════════════════════════════

class AuditRepository:

    @staticmethod
    def create_log(
        event_type: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        actor: Optional[str] = None,
        detail: Optional[dict] = None,
    ) -> str:
        with get_db() as db:
            log = AuditLog(
                event_type=event_type,
                entity_type=entity_type,
                entity_id=entity_id,
                actor=actor,
                detail=detail,
            )
            db.add(log)
            db.flush()
            return log.id

    @staticmethod
    def list_logs(
        entity_id: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        with get_db() as db:
            q = db.query(AuditLog).order_by(AuditLog.created_at.desc())
            if entity_id:
                q = q.filter(AuditLog.entity_id == entity_id)
            logs = q.limit(limit).all()
            return [
                {
                    "id": lg.id,
                    "event_type": lg.event_type,
                    "entity_type": lg.entity_type,
                    "entity_id": lg.entity_id,
                    "actor": lg.actor,
                    "detail": lg.detail,
                    "created_at": lg.created_at,
                }
                for lg in logs
            ]
