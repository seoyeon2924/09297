"""
FastAPI 클라이언트 — Streamlit 페이지 → FastAPI 백엔드 httpx 래퍼.

httpx.Client를 재사용하여 HTTP keep-alive로 TCP 연결 오버헤드를 제거한다.

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


# keep-alive 연결을 재사용하는 싱글턴 클라이언트
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(
            base_url=_base(),
            timeout=httpx.Timeout(10.0, connect=3.0),
        )
    return _client


# ── Review requests ───────────────────────────────────────────────

def create_review(payload: dict) -> dict:
    """POST /api/reviews — 심의 요청 생성."""
    try:
        r = _get_client().post("/api/reviews", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise ConnectionError(_conn_err_msg())


def list_reviews(status_filter: str | None = None) -> list:
    """GET /api/reviews — 심의 요청 목록."""
    params = {}
    if status_filter:
        params["status"] = status_filter
    try:
        r = _get_client().get("/api/reviews", params=params)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise ConnectionError(_conn_err_msg())


def get_review_detail(request_id: str) -> dict | None:
    """GET /api/reviews/{id} — 심의 요청 상세."""
    try:
        r = _get_client().get(f"/api/reviews/{request_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise ConnectionError(_conn_err_msg())


def submit_review_decision(request_id: str, payload: dict) -> dict:
    """POST /api/reviews/{id}/decision — 최종 심의 판단 저장."""
    try:
        r = _get_client().post(
            f"/api/reviews/{request_id}/decision",
            json=payload,
        )
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise ConnectionError(_conn_err_msg())


def stream_review_sse(request_id: str):
    """GET /api/reviews/{id}/stream — SSE 스트리밍 sync 제너레이터.

    SSE는 장시간 연결이므로 별도 httpx.stream 사용 (keep-alive 클라이언트와 분리).
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
        raise ConnectionError(_conn_err_msg())


def _conn_err_msg() -> str:
    return (
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
