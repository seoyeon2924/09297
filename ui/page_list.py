"""
Page 3 — 심의요청 목록 (심의자)
View all review requests with status filter; navigate to detail.
"""

import streamlit as st

from ui.api_client import list_reviews, fmt_date
from ui.components.status_badge import render_status_badge


def render() -> None:
    st.header("심의요청 목록")

    # ── Filters ──
    col_filter, col_refresh = st.columns([3, 1])
    with col_filter:
        status_filter = st.selectbox(
            "상태 필터",
            ["전체", "REQUESTED", "AI_RUNNING", "REVIEWING", "DONE", "REJECTED"],
        )
    with col_refresh:
        st.write("")  # vertical spacer
        st.write("")
        if st.button("새로고침"):
            st.rerun()

    filter_val = None if status_filter == "전체" else status_filter

    try:
        requests = list_reviews(status_filter=filter_val)
    except ConnectionError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"목록 조회 실패: {e}")
        return

    if not requests:
        st.info("등록된 심의 요청이 없습니다.")
        return

    # ── Table header ──
    header_cols = st.columns([1.2, 2, 1.2, 1, 1.5, 0.8])
    header_cols[0].markdown("**요청 ID**")
    header_cols[1].markdown("**상품명**")
    header_cols[2].markdown("**요청자**")
    header_cols[3].markdown("**상태**")
    header_cols[4].markdown("**요청일**")
    header_cols[5].markdown("**보기**")

    # ── Table rows ──
    for req in requests:
        cols = st.columns([1.2, 2, 1.2, 1, 1.5, 0.8])
        cols[0].text(req["id"][:8] + "...")
        cols[1].text(req["product_name"])
        cols[2].text(req["requested_by"] or "-")
        with cols[3]:
            render_status_badge(req["status"])
        cols[4].text(fmt_date(req.get("created_at")))
        with cols[5]:
            if st.button("보기", key=f"view_{req['id']}"):
                st.session_state.selected_request_id = req["id"]
                st.session_state.current_page = "detail"
                st.rerun()
