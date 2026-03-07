"""
SQLAlchemy ORM models — design v2.

Entities:
    ReferenceDocument, Chunk,
    ReviewRequest, ReviewItem,
    AiRecommendation, HumanDecision,
    AuditLog
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, Text, DateTime, ForeignKey, JSON,
)
from sqlalchemy.orm import relationship, DeclarativeBase


# ───────────────────────────────────────────
# Base
# ───────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


# ───────────────────────────────────────────
# Enums
# ───────────────────────────────────────────

class DocType(str, PyEnum):
    LAW = "법령"
    REGULATION = "규정"
    GUIDELINE = "지침"
    CASE = "사례"


class DocStatus(str, PyEnum):
    UPLOADED = "UPLOADED"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    INDEX_FAILED = "INDEX_FAILED"


class AdvancedMetaStatus(str, PyEnum):
    NONE = "NONE"
    RUNNING = "RUNNING"
    DONE = "DONE"
    PARTIAL_FAIL = "PARTIAL_FAIL"


class ReviewStatus(str, PyEnum):
    REQUESTED = "REQUESTED"
    AI_RUNNING = "AI_RUNNING"
    REVIEWING = "REVIEWING"
    DONE = "DONE"
    REJECTED = "REJECTED"


class ItemType(str, PyEnum):
    REQUEST_TEXT = "REQUEST_TEXT"
    EMPHASIS_BAR = "EMPHASIS_BAR"


class Judgment(str, PyEnum):
    VIOLATION = "위반소지"
    CAUTION = "주의"
    OK = "OK"


class DecisionType(str, PyEnum):
    DONE = "DONE"
    REJECTED = "REJECTED"


# ───────────────────────────────────────────
# Models
# ───────────────────────────────────────────

class ReferenceDocument(Base):
    __tablename__ = "reference_documents"

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False)
    doc_type = Column(String, nullable=False)
    category = Column(String, nullable=True)
    scope = Column(String, nullable=True)
    status = Column(String, nullable=False, default=DocStatus.UPLOADED.value)
    advanced_meta_status = Column(
        String, nullable=False, default=AdvancedMetaStatus.NONE.value
    )
    chunk_count = Column(Integer, default=0)
    file_path = Column(String, nullable=True)
    uploaded_by = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True)

    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(
        String, ForeignKey("reference_documents.id"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    content_preview = Column(String(200), nullable=True)

    # ── Basic meta (rule-based) ──
    page_or_row = Column(String, nullable=True)
    source_file = Column(String, nullable=True)
    doc_type = Column(String, nullable=True)

    # ── Advanced meta (LLM-based, nullable) ──
    section_title = Column(String, nullable=True)
    keywords = Column(JSON, nullable=True)
    advanced_meta_status = Column(
        String, default=AdvancedMetaStatus.NONE.value
    )

    chroma_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("ReferenceDocument", back_populates="chunks")


class ReviewRequest(Base):
    __tablename__ = "review_requests"

    id = Column(String, primary_key=True, default=_uuid)
    product_name = Column(String, nullable=False)
    category = Column(String, nullable=True)
    broadcast_type = Column(String, nullable=True)
    status = Column(
        String, nullable=False, default=ReviewStatus.REQUESTED.value
    )
    requested_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    decided_at = Column(DateTime, nullable=True)

    items = relationship(
        "ReviewItem", back_populates="request", cascade="all, delete-orphan"
    )
    human_decision = relationship(
        "HumanDecision",
        back_populates="request",
        uselist=False,
        cascade="all, delete-orphan",
    )


class ReviewItem(Base):
    """A single phrase or emphasis-bar submitted for review."""

    __tablename__ = "review_items"

    id = Column(String, primary_key=True, default=_uuid)
    request_id = Column(
        String, ForeignKey("review_requests.id"), nullable=False
    )
    item_index = Column(Integer, nullable=False)
    item_type = Column(String, nullable=False)   # REQUEST_TEXT | EMPHASIS_BAR
    label = Column(String, nullable=False)        # 요청문구1, 강조바2, …
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    request = relationship("ReviewRequest", back_populates="items")
    ai_recommendation = relationship(
        "AiRecommendation",
        back_populates="review_item",
        uselist=False,
        cascade="all, delete-orphan",
    )


class AiRecommendation(Base):
    __tablename__ = "ai_recommendations"

    id = Column(String, primary_key=True, default=_uuid)
    review_item_id = Column(
        String, ForeignKey("review_items.id"), nullable=False
    )
    judgment = Column(String, nullable=False)   # 위반소지 | 주의 | OK
    reason = Column(Text, nullable=True)
    references = Column(JSON, nullable=True)    # list[{doc_filename, …}]
    model_name = Column(String, nullable=True)
    prompt_version = Column(String, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    review_item = relationship(
        "ReviewItem", back_populates="ai_recommendation"
    )


class HumanDecision(Base):
    __tablename__ = "human_decisions"

    id = Column(String, primary_key=True, default=_uuid)
    request_id = Column(
        String, ForeignKey("review_requests.id"), nullable=False
    )
    decision = Column(String, nullable=False)   # DONE | REJECTED
    comment = Column(Text, nullable=True)
    decided_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    request = relationship("ReviewRequest", back_populates="human_decision")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=_uuid)
    event_type = Column(String, nullable=False)
    entity_type = Column(String, nullable=True)
    entity_id = Column(String, nullable=True)
    actor = Column(String, nullable=True)
    detail = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
