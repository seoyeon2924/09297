"""
FastAPI 클라이언트 — Streamlit 페이지 → FastAPI 백엔드 httpx 래퍼.

사용 예:
    from ui.api_client import list_reviews, stream_review_sse
"""

from __future__ import annotations

import json
import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)


def _base() -> str:
    return settings.FASTAPI_BASE_URL


# ── Review requests ───────────────────────────────────────────────

def create_review(payload: dict) -> dict:
    """POST /api/reviews — 심의 요청 생성."""
    r = httpx.post(f"{_base()}/api/reviews", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def list_reviews(status_filter: str | None = None) -> list:
    """GET /api/reviews — 심의 요청 목록."""
    params = {}
    if status_filter:
        params["status"] = status_filter
    r = httpx.get(f"{_base()}/api/reviews", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def get_review_detail(request_id: str) -> dict | None:
    """GET /api/reviews/{id} — 심의 요청 상세."""
    r = httpx.get(f"{_base()}/api/reviews/{request_id}", timeout=10)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def submit_review_decision(request_id: str, payload: dict) -> dict:
    """POST /api/reviews/{id}/decision — 최종 심의 판단 저장."""
    r = httpx.post(
        f"{_base()}/api/reviews/{request_id}/decision",
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def stream_review_sse(request_id: str):
    """GET /api/reviews/{id}/stream — SSE 스트리밍 sync 제너레이터.

    Streamlit에서 직접 소비:
        for event in stream_review_sse(request_id):
            # event: {"node": ..., "status": ..., "summary": ..., "elapsed": ...}
    """
    url = f"{_base()}/api/reviews/{request_id}/stream"
    try:
        with httpx.stream("GET", url, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        if event.get("done"):
                            break
                        if event.get("error"):
                            logger.error("SSE stream error: %s", event["error"])
                            raise RuntimeError(event["error"])
                        yield event
                    except json.JSONDecodeError:
                        continue
    except httpx.ConnectError:
        raise ConnectionError(
            f"FastAPI 서버에 연결할 수 없습니다. "
            f"{_base()} 에서 서버가 실행 중인지 확인하세요.\n"
            f"  uvicorn api.main:app --port 8001 --reload"
        )


# ── 날짜 유틸 ─────────────────────────────────────────────────────

def fmt_date(val) -> str:
    """datetime 또는 ISO string → 'YYYY-MM-DD HH:MM' 문자열."""
    if not val:
        return "-"
    if isinstance(val, str):
        return val[:16].replace("T", " ")
    try:
        return val.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(val)
