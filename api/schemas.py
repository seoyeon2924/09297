"""
FastAPI Request / Response 스키마.
"""

from __future__ import annotations

from pydantic import BaseModel


class ReviewItemIn(BaseModel):
    item_type: str          # "REQUEST_TEXT" | "EMPHASIS_BAR"
    label: str              # "요청문구1" 등
    text: str
    item_index: int


class CreateReviewIn(BaseModel):
    product_name: str
    category: str
    broadcast_type: str
    requested_by: str = ""
    items: list[ReviewItemIn]


class DecisionIn(BaseModel):
    decision: str           # "DONE" | "REJECTED"
    comment: str = ""
    decided_by: str = ""
