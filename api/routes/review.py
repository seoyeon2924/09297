"""
Review API 라우터 — CRUD + SSE 스트리밍.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from api.schemas import CreateReviewIn, DecisionIn
from services.rag_service import RAGService
from services.review_service import ReviewService

logger = logging.getLogger(__name__)
router = APIRouter()


# ── CRUD ─────────────────────────────────────────────────────────

@router.post("/reviews", status_code=201)
def create_review(body: CreateReviewIn):
    """심의 요청 등록."""
    try:
        result = ReviewService.create_request(
            product_name=body.product_name,
            category=body.category,
            broadcast_type=body.broadcast_type,
            requested_by=body.requested_by,
            items=[item.model_dump() for item in body.items],
        )
        return jsonable_encoder(result)
    except Exception as e:
        logger.error("create_review 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reviews")
def list_reviews(status: str | None = None):
    """심의 요청 목록 조회."""
    try:
        return jsonable_encoder(ReviewService.list_requests(status_filter=status))
    except Exception as e:
        logger.error("list_reviews 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reviews/{request_id}")
def get_review(request_id: str):
    """심의 요청 상세 조회."""
    detail = ReviewService.get_detail(request_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Review request not found")
    return jsonable_encoder(detail)


@router.post("/reviews/{request_id}/decision")
def submit_decision(request_id: str, body: DecisionIn):
    """최종 심의 판단 저장."""
    try:
        result = ReviewService.submit_decision(
            request_id=request_id,
            decision=body.decision,
            comment=body.comment,
            decided_by=body.decided_by,
        )
        return jsonable_encoder(result)
    except Exception as e:
        logger.error("submit_decision 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── SSE 스트리밍 ──────────────────────────────────────────────────

@router.get("/reviews/{request_id}/stream")
async def stream_review(request_id: str, request: Request):
    """LangGraph 노드별 실시간 스트리밍 (Server-Sent Events).

    Event format:
        data: {"node": "orchestrator", "status": "done", "summary": "...", "elapsed": 1.2, "item_label": "요청문구1"}

    종료 이벤트:
        data: {"done": true}

    에러 이벤트:
        data: {"error": "에러 메시지"}
    """
    async def event_generator():
        try:
            async for event in RAGService.async_stream_recommendation(request_id):
                if await request.is_disconnected():
                    logger.info("Client disconnected during stream: %s", request_id)
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            logger.error("SSE stream error [%s]: %s", request_id, e)
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            yield 'data: {"done": true}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
