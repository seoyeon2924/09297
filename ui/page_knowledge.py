"""
Page 1 — 기준지식 관리 (관리자)
Upload documents, run indexing, view document list.
"""

import streamlit as st

from services.ingest_service import IngestService
from storage.chroma_store import chroma_store
from ui.components.status_badge import render_status_badge

CATEGORIES = [
    "전체",
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

SCOPES = [
    "전체",
    "TV홈쇼핑",
    "T커머스",
    "데이터홈쇼핑",
    "인포머셜",
    "라이브커머스",
]


def render() -> None:
    st.header("기준지식 관리")
    st.caption("법령 / 규정 / 지침 / 사례 문서를 업로드하고 인덱싱합니다.")

    # ──────────────────────────────────
    # 벡터DB(Chroma) 상태
    # ──────────────────────────────────
    with st.expander("벡터DB(Chroma) 상태", expanded=False):
        try:
            stats = chroma_store.get_stats(sample_limit=5)
            total = sum(s["count"] for s in stats)
            st.metric("총 벡터 수", f"{total:,}개")
            for s in stats:
                if s["count"] == 0:
                    continue
                st.markdown(f"**{s['name']}** — {s['count']:,}개")
                if s["sample_ids"]:
                    st.caption("샘플 ID:")
                    for sid in s["sample_ids"][:5]:
                        st.code(sid, language=None)
                if s["sample_docs"]:
                    st.caption("샘플 문서(앞 120자):")
                    for i, doc in enumerate(s["sample_docs"][:3], 1):
                        st.text_area(f"샘플 {i}", doc, height=60, disabled=True, label_visibility="collapsed")
        except Exception as e:
            st.error(f"Chroma 연결 실패: {e}")

        st.divider()
        st.caption("⚠️ 심의 사례 컬렉션을 초기화하면 기존 벡터가 모두 삭제됩니다. 새 파일 인덱싱 전에 실행하세요.")
        if st.button("심의사례(cases) 컬렉션 초기화", type="secondary", key="reset_cases"):
            try:
                chroma_store.reset_collection("cases")
                st.success("cases 컬렉션이 초기화되었습니다. 새 심의사례 파일을 업로드하여 인덱싱하세요.")
                st.rerun()
            except Exception as e:
                st.error(f"초기화 실패: {e}")

    # ──────────────────────────────────
    # Upload & Index
    # ──────────────────────────────────
    with st.expander("문서 업로드 & 인덱싱", expanded=True):
        uploaded_file = st.file_uploader(
            "파일 선택 (PDF / Excel)",
            type=["pdf", "xlsx", "xls"],
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            doc_type = st.selectbox(
                "문서 유형", ["법령", "규정", "지침", "사례"]
            )
        with col2:
            category = st.selectbox(
                "카테고리", CATEGORIES
            )
        with col3:
            scope = st.selectbox(
                "적용범위", SCOPES
            )

        if st.button(
            "업로드 & 인덱싱 실행",
            type="primary",
            disabled=(uploaded_file is None),
        ):
            if uploaded_file:
                with st.spinner("인덱싱 중..."):
                    try:
                        result = IngestService.upload_and_index(
                            file=uploaded_file,
                            doc_type=doc_type,
                            category=category,
                            scope=scope,
                            uploaded_by="관리자",
                        )
                        st.success(
                            f"인덱싱 완료: {result['filename']} "
                            f"({result['chunk_count']}개 청크, "
                            f"{result['elapsed_sec']}초)"
                        )
                    except Exception as e:
                        st.error(f"인덱싱 실패: {e}")

    st.divider()

    # ──────────────────────────────────
    # Document List
    # ──────────────────────────────────
    st.subheader("문서 목록")
    documents = IngestService.list_documents()

    if not documents:
        st.info("등록된 문서가 없습니다.")
        return

    cols = st.columns([3, 1.2, 0.8, 1.4, 1.4, 1.0, 1.0])
    cols[0].markdown("**파일명**")
    cols[1].markdown("**유형**")
    cols[2].markdown("**청크수**")
    cols[3].markdown("**인덱싱 상태**")
    cols[4].markdown("**고급 메타**")
    cols[5].markdown("**청크**")
    cols[6].markdown("**작업**")

    for doc in documents:
        cols = st.columns([3, 1.2, 0.8, 1.4, 1.4, 1.0, 1.0])
        cols[0].text(doc["filename"])
        cols[1].text(doc["doc_type"])
        cols[2].text(str(doc["chunk_count"]))
        with cols[3]:
            render_status_badge(doc["status"])
        with cols[4]:
            render_status_badge(doc["advanced_meta_status"])
        with cols[5]:
            is_indexed = doc["status"] == "INDEXED"
            if st.button(
                "청크 보기",
                key=f"chunks_{doc['id']}",
                disabled=not is_indexed,
            ):
                st.session_state["view_chunks_doc_id"] = doc["id"]
        with cols[6]:
            adv_eligible = doc["advanced_meta_status"] in (
                "NONE",
                "PARTIAL_FAIL",
            )
            if st.button(
                "고급 메타",
                key=f"adv_{doc['id']}",
                disabled=not (is_indexed and adv_eligible),
            ):
                result = IngestService.generate_advanced_metadata(doc["id"])
                st.info(result.get("message", "완료"))

    # ──────────────────────────────────
    # 청크 미리보기
    # ──────────────────────────────────
    view_doc_id = st.session_state.get("view_chunks_doc_id")
    if view_doc_id:
        _render_chunk_preview(view_doc_id)


def _render_chunk_preview(doc_id: str) -> None:
    """선택한 문서의 청크 목록을 미리보기로 표시."""
    doc = IngestService.get_document(doc_id)
    if not doc:
        return

    st.divider()
    st.subheader(f"청크 미리보기: {doc['filename']}")

    chunks = IngestService.get_chunks(doc_id)
    if not chunks:
        st.info("저장된 청크가 없습니다.")
        return

    st.caption(f"총 {len(chunks)}개 청크")
    for c in chunks:
        with st.expander(
            f"#{c['chunk_index']}  |  {c.get('page_or_row', '')}  |  "
            f"{(c.get('content_preview') or '')[:60]}…"
        ):
            st.text(c.get("content_preview", ""))
            st.caption(
                f"chroma_id: {c.get('chroma_id', '')}  |  "
                f"source: {c.get('source_file', '')}"
            )

    if st.button("닫기", key="close_chunks"):
        del st.session_state["view_chunks_doc_id"]
        st.rerun()
