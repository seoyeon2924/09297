"""
Page 2 — 심의요청 등록 (PD/MD)
Input request phrases, emphasis bars, and product metadata.
"""

import streamlit as st

from ui.api_client import create_review

CATEGORIES = [
    "건강기능식품",
    "식품/음료",
    "화장품/뷰티",
    "의류/패션",
    "가전/디지털",
    "주방/생활용품",
    "스포츠/레저",
    "유아/완구",
    "가구/인테리어",
    "여행/서비스",
    "보험/금융",
    "기타",
]

BROADCAST_TYPES = [
    "생방송",
    "녹화방송",
    "T커머스",
    "인포머셜",
    "데이터홈쇼핑",
]


@st.dialog("심의 요청 등록 완료")
def _show_success_dialog(request_id: str):
    st.write(f"심의 요청이 등록되었습니다.\n\n요청 ID: {request_id[:8]}...")
    if st.button("확인", type="primary", use_container_width=True):
        st.rerun()


def render() -> None:
    st.header("심의요청 등록")
    st.caption("심의할 문구를 입력하고 요청을 등록합니다.")

    with st.form("request_form", clear_on_submit=True):
        # ── Product metadata ──
        col1, col2, col3 = st.columns(3)
        with col1:
            product_name = st.text_input(
                "상품명 *", placeholder="예: 콜라겐 젤리"
            )
        with col2:
            category = st.selectbox(
                "카테고리 *", CATEGORIES
            )
        with col3:
            broadcast_type = st.selectbox(
                "방송유형 *", BROADCAST_TYPES
            )

        requested_by = st.text_input(
            "요청자", placeholder="예: 김PD"
        )

        st.divider()

        # ── Request texts ──
        st.subheader("요청문구 (최대 3개)")
        text1 = st.text_area(
            "요청문구 1 *", placeholder="심의할 문구를 입력하세요", height=80
        )
        text2 = st.text_area(
            "요청문구 2 (선택)", placeholder="", height=80
        )
        text3 = st.text_area(
            "요청문구 3 (선택)", placeholder="", height=80
        )

        st.divider()

        # ── Emphasis bars ──
        st.subheader("강조바 (최대 3개)")
        bar1 = st.text_area(
            "강조바 1 (선택)", placeholder="강조바 문구를 입력하세요", height=80
        )
        bar2 = st.text_area(
            "강조바 2 (선택)", placeholder="", height=80
        )
        bar3 = st.text_area(
            "강조바 3 (선택)", placeholder="", height=80
        )

        submitted = st.form_submit_button(
            "심의 요청 등록", type="primary"
        )

    # ── Handle submission (outside form context) ──
    if submitted:
        if not product_name or not text1:
            st.error("상품명과 요청문구 1은 필수입니다.")
            return

        items: list[dict] = []

        for text, label in [
            (text1, "요청문구1"),
            (text2, "요청문구2"),
            (text3, "요청문구3"),
        ]:
            if text and text.strip():
                items.append(
                    {
                        "item_type": "REQUEST_TEXT",
                        "label": label,
                        "text": text.strip(),
                        "item_index": len(items) + 1,
                    }
                )

        for text, label in [
            (bar1, "강조바1"),
            (bar2, "강조바2"),
            (bar3, "강조바3"),
        ]:
            if text and text.strip():
                items.append(
                    {
                        "item_type": "EMPHASIS_BAR",
                        "label": label,
                        "text": text.strip(),
                        "item_index": len(items) + 1,
                    }
                )

        if not items:
            st.error("최소 하나의 문구를 입력해주세요.")
            return

        try:
            result = create_review({
                "product_name": product_name,
                "category": category,
                "broadcast_type": broadcast_type,
                "requested_by": requested_by,
                "items": items,
            })
            _show_success_dialog(result["id"])
        except ConnectionError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"등록 실패: {e}")
