"""
Page 4 — 심의 상세 (심의자)
AI recommendation + human final decision in a single view.
"""

import streamlit as st

from services.rag_service import NODE_DISPLAY
from ui.api_client import (
    fmt_date,
    get_review_detail,
    stream_review_sse,
    submit_review_decision,
)
from ui.components.status_badge import render_status_badge
from ui.components.pipeline_viz import (
    render_pipeline_result,
    render_execution_summary,
)


# ── 실시간 스트리밍 AI 실행 ────────────────────────────────────────

def _run_ai_with_streaming(request_id: str) -> None:
    """st.status()로 노드별 실시간 진행 표시하면서 AI 심의 실행 (FastAPI SSE 경유)."""

    with st.status("🚀 AI 심의 파이프라인 실행 중...", expanded=True) as status:
        try:
            current_item = ""
            for event in stream_review_sse(request_id):
                node = event.get("node", "")
                summary = event.get("summary", "")
                elapsed = event.get("elapsed", 0)
                item_label = event.get("item_label", "")
                display = NODE_DISPLAY.get(node, {"icon": "⚙️", "label": node})

                if item_label and item_label != current_item:
                    current_item = item_label
                    st.markdown(f"---\n**📝 {item_label}** ⏳")

                st.write(
                    f"{display['icon']} **{display['label']}** — {summary}  `{elapsed:.1f}s`"
                )

            status.update(label="✅ AI 심의 완료!", state="complete", expanded=True)

        except ConnectionError as e:
            status.update(label="⛔ 서버 연결 실패", state="error")
            st.error(str(e))
        except Exception as e:
            status.update(label=f"⛔ 실패: {e}", state="error")
            st.error(f"AI 추천 실패: {e}")


def render() -> None:
    st.header("심의 상세")

    request_id = st.session_state.get("selected_request_id")
    if not request_id:
        st.warning(
            "심의 요청을 먼저 선택해주세요. "
            "[심의요청 목록]에서 '보기'를 클릭하세요."
        )
        return

    try:
        detail = get_review_detail(request_id)
    except ConnectionError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"상세 조회 실패: {e}")
        return

    if not detail:
        st.error("요청을 찾을 수 없습니다.")
        return

    req = detail["request"]
    items = detail["items"]
    human_dec = detail["human_decision"]

    # ──────────────────────────────────
    # 1. Request Summary
    # ──────────────────────────────────
    st.subheader("요청 요약")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("상품명", req["product_name"])
    mc2.metric("카테고리", req["category"] or "-")
    mc3.metric("방송유형", req["broadcast_type"] or "-")
    with mc4:
        st.markdown("**상태**")
        render_status_badge(req["status"])

    st.caption(
        f"요청자: {req['requested_by'] or '-'} · "
        f"요청일: {fmt_date(req.get('created_at'))} · "
        f"문구 수: {len(items)}"
    )

    st.divider()

    # ──────────────────────────────────
    # 2. AI Recommendation
    # ──────────────────────────────────
    st.subheader("AI 심의 추천")

    can_run_ai = req["status"] in ("REQUESTED",)
    if st.button(
        "AI 심의 추천 실행",
        type="primary",
        disabled=not can_run_ai,
    ):
        _run_ai_with_streaming(request_id)
        st.rerun()

    if req["status"] == "AI_RUNNING":
        st.info("AI가 추천 결과를 생성 중입니다...")

    # ── Item tabs ──
    if items:
        tab_labels = [item["label"] for item in items]
        tabs = st.tabs(tab_labels)

        for tab, item in zip(tabs, items):
            with tab:
                type_label = (
                    "요청문구" if item["item_type"] == "REQUEST_TEXT"
                    else "강조바"
                )
                st.markdown(
                    f"**{item['label']}** · `{type_label}`"
                )
                st.info(item["text"])

                rec = item.get("ai_recommendation")
                if rec:
                    _render_recommendation(rec, item_id=item.get("id", ""))
                else:
                    st.caption(
                        "아직 AI 추천이 실행되지 않았습니다."
                    )

    st.divider()

    # ──────────────────────────────────
    # 3. Human Decision
    # ──────────────────────────────────
    st.subheader("최종 심의 판단")

    if human_dec:
        icon = "완료" if human_dec["decision"] == "DONE" else "반려"
        st.success(f"최종 결정: {icon}")
        st.markdown(f"**코멘트:** {human_dec['comment'] or '-'}")
        st.caption(
            f"심의자: {human_dec['decided_by'] or '-'} · "
            f"결정일: {fmt_date(human_dec.get('created_at'))}"
        )
    else:
        can_decide = req["status"] == "REVIEWING"

        if not can_decide:
            st.info("AI 추천 실행 후 최종 판단을 내릴 수 있습니다.")

        with st.form("decision_form"):
            decision = st.radio(
                "최종 결과",
                ["DONE", "REJECTED"],
                format_func=lambda x: (
                    "완료 (DONE)" if x == "DONE" else "반려 (REJECTED)"
                ),
                horizontal=True,
            )
            draft = _generate_comment_draft(items)
            comment = st.text_area(
                "심의 코멘트",
                value=draft,
                placeholder="심의 의견을 작성하세요",
                height=200,
            )
            decided_by = st.text_input(
                "심의자", placeholder="예: 박심의위원"
            )

            submitted = st.form_submit_button(
                "최종 판단 저장",
                type="primary",
                disabled=not can_decide,
            )

        if submitted and can_decide:
            try:
                submit_review_decision(
                    request_id=request_id,
                    payload={
                        "decision": decision,
                        "comment": comment,
                        "decided_by": decided_by,
                    },
                )
                label = "완료" if decision == "DONE" else "반려"
                st.success(f"최종 판단이 저장되었습니다: {label}")
                st.rerun()
            except ConnectionError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"저장 실패: {e}")


# ──────────────────────────────────
# Helper
# ──────────────────────────────────

_JUDGMENT_ICON = {
    "위반소지": "\U0001F534",
    "주의": "\U0001F7E1",
    "OK": "\U0001F7E2",
}


def _generate_comment_draft(items: list[dict]) -> str:
    """AI 추천 결과를 기반으로 심의 코멘트 초안을 생성한다. LLM 호출 없이 포맷팅만."""
    lines = []
    judgment_counts = {"위반소지": 0, "주의": 0, "OK": 0}

    for item in items:
        rec = item.get("ai_recommendation")
        if not rec:
            continue
        judgment = rec.get("judgment", "")
        reason = rec.get("reason", "")
        icon = _JUDGMENT_ICON.get(judgment, "⚪")
        judgment_counts[judgment] = judgment_counts.get(judgment, 0) + 1

        lines.append(f"[{item['label']}] {icon} {judgment}")
        if reason:
            lines.append(f"  - {reason}")
        lines.append("")

    summary_parts = []
    for j, count in judgment_counts.items():
        if count > 0:
            summary_parts.append(f"{j} {count}건")
    if summary_parts:
        lines.append(f"종합: {', '.join(summary_parts)}")

    return "\n".join(lines)


def _render_recommendation(rec: dict, item_id: str = "") -> None:
    """Render a single AI recommendation block."""
    icon = _JUDGMENT_ICON.get(rec["judgment"], "\u26AA")
    st.markdown(f"**판단:** {icon} {rec['judgment']}")
    st.markdown(f"**사유:** {rec['reason']}")

    # ── 파이프라인 실행 로그 시각화 (session_state에 로그가 있을 때만) ──
    pipeline_data = st.session_state.get("pipeline_logs", {}).get(item_id)
    if pipeline_data:
        with st.expander("🔬 AI 파이프라인 실행 상세", expanded=False):
            render_execution_summary(pipeline_data.get("tool_logs", []))
            render_pipeline_result(
                pipeline_data.get("tool_logs", []),
                judgment=pipeline_data.get("judgment", ""),
            )

    all_refs = rec.get("references") or []
    refs = [r for r in all_refs if (r.get("content") or "").strip()]

    if refs:
        st.markdown(f"**근거:** ({len(refs)}건)")
        for i, ref in enumerate(refs, 1):
            doc_type = ref.get("doc_type", "-")
            case_number = ref.get("case_number", "")
            case_date = ref.get("case_date", "")
            article_number = ref.get("article_number", "")
            doc_filename = ref.get("doc_filename", "-")
            section_title = ref.get("section_title", "")
            score = ref.get("relevance_score", "-")

            if doc_type == "사례" and case_number:
                date_str = f" ({case_date})" if case_date else ""
                label = f"[{doc_type}] 처리번호 {case_number}{date_str} · score: {score}"
            elif article_number:
                section_str = f" ({section_title})" if section_title else ""
                label = f"[{doc_type}] `{doc_filename}` {article_number}{section_str} · score: {score}"
            else:
                section_str = f" · {section_title}" if section_title else ""
                label = f"[{doc_type}] `{doc_filename}`{section_str} · score: {score}"

            st.caption(f"{i}. {label}")
    elif all_refs:
        st.caption("⚠️ 근거 문서를 검색했으나 유효한 내용을 가져오지 못했습니다.")

    st.caption(
        f"모델: {rec.get('model_name', '-')} · "
        f"프롬프트: {rec.get('prompt_version', '-')} · "
        f"지연: {rec.get('latency_ms', '-')}ms"
    )
