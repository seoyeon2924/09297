"""
Reusable status badge renderer for Streamlit.
"""

import streamlit as st

STATUS_CONFIG: dict[str, tuple[str, str]] = {
    # ReviewRequest statuses
    "REQUESTED": ("\U0001F7E1", "요청"),
    "AI_RUNNING": ("\U0001F535", "AI 실행중"),
    "REVIEWING": ("\U0001F7E0", "검토중"),
    "DONE": ("\U0001F7E2", "완료"),
    "REJECTED": ("\U0001F534", "반려"),
    # ReferenceDocument statuses
    "UPLOADED": ("\u2B1C", "업로드됨"),
    "INDEXING": ("\U0001F535", "인덱싱중"),
    "INDEXED": ("\U0001F7E2", "인덱싱완료"),
    "INDEX_FAILED": ("\U0001F534", "인덱싱실패"),
    # AdvancedMetaStatus
    "NONE": ("\u2B1C", "없음"),
    "RUNNING": ("\U0001F535", "생성중"),
    "PARTIAL_FAIL": ("\U0001F7E1", "일부실패"),
}


def render_status_badge(status: str) -> None:
    """Render a coloured icon + label for the given status string."""
    icon, label = STATUS_CONFIG.get(status, ("\u2B1C", status))
    st.markdown(f"{icon} **{label}**")
