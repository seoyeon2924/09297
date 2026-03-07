"""
LangGraph 파이프라인 시각화 컴포넌트.

심의 실행 과정을 Streamlit에서 시각적으로 표시:
  1. 실시간 진행 표시 (spinner + step indicator)
  2. 완료 후 파이프라인 다이어그램 + tool_logs 상세 표시
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


# ── 노드 설정 ──────────────────────────────────────────────────────

_NODES = [
    {
        "id": "orchestrator",
        "label": "🎯 Orchestrator",
        "desc": "위험유형 분류 + 검색 쿼리 생성",
        "steps": ["orchestrator"],
    },
    {
        "id": "case_agent",
        "label": "📋 CaseAgent",
        "desc": "사례 검색 → 평가 → 쿼리 개선 루프",
        "steps": ["case_retrieve", "case_grade", "case_rewrite"],
    },
    {
        "id": "policy_agent",
        "label": "📜 PolicyAgent",
        "desc": "법령·규정·지침 검색 → 평가 → 쿼리 개선 루프",
        "steps": ["policy_retrieve", "policy_grade", "policy_rewrite"],
    },
    {
        "id": "synthesizer",
        "label": "⚖️ Synthesizer",
        "desc": "근거 통합 → 최종 판정 생성",
        "steps": ["synthesizer"],
    },
    {
        "id": "grade_answer",
        "label": "✅ GradeAnswer",
        "desc": "판정 품질 검증 (fail 시 재시도)",
        "steps": ["grade_answer"],
    },
]

_STEP_LABELS = {
    "orchestrator": "위험유형 분류",
    "case_retrieve": "사례 검색",
    "case_grade": "사례 관련성 평가",
    "case_rewrite": "사례 쿼리 재작성",
    "policy_retrieve": "법규 검색",
    "policy_grade": "법규 관련성 평가",
    "policy_rewrite": "법규 쿼리 재작성",
    "synthesizer": "판정 생성",
    "grade_answer": "품질 검증",
    "error": "오류",
}

_JUDGMENT_STYLE = {
    "위반소지": ("🔴", "#FF4B4B", "위반소지"),
    "주의": ("🟡", "#FFA500", "주의"),
    "OK": ("🟢", "#00CC66", "OK"),
}


# ── 파이프라인 다이어그램 (Mermaid) ────────────────────────────────

def render_pipeline_diagram() -> None:
    """LangGraph 워크플로우를 HTML/CSS 다이어그램으로 표시 (외부 JS 의존성 없음)."""
    n = "display:inline-block;padding:8px 18px;border-radius:6px;font-weight:bold;font-size:13px;color:#fff;text-align:center;line-height:1.5;"
    p = "display:inline-block;padding:6px 18px;border-radius:20px;font-weight:bold;font-size:13px;color:#fff;"
    a = "color:#9CA3AF;font-size:18px;"
    sub = "font-size:10px;font-weight:normal;"
    html = f"""<!DOCTYPE html><html>
<body style="background:transparent;margin:0;padding:12px;font-family:sans-serif;">
<div style="display:flex;flex-direction:column;align-items:center;gap:5px;">
  <span style="{p}background:#10B981;">&#9654; START</span>
  <span style="{a}">&#8595;</span>
  <span style="{n}background:#3B82F6;">&#127919; Orchestrator<br><span style="{sub}">위험유형 분류 + 검색 쿼리 생성</span></span>
  <span style="{a}">&#8595;</span>
  <div style="display:flex;gap:16px;align-items:center;">
    <span style="{n}background:#F59E0B;">&#128203; CaseAgent<br><span style="{sub}">사례 검색·평가·재작성</span></span>
    <span style="color:#D1D5DB;font-size:20px;">&#8214;</span>
    <span style="{n}background:#8B5CF6;">&#128220; PolicyAgent<br><span style="{sub}">법령·규정·지침 검색·평가·재작성</span></span>
  </div>
  <span style="{a}">&#8595;</span>
  <span style="{n}background:#EC4899;">&#9878;&#65039; Synthesizer<br><span style="{sub}">근거 통합 → 최종 판정</span></span>
  <span style="{a}">&#8595;</span>
  <span style="{n}background:#14B8A6;">&#9989; GradeAnswer<br><span style="{sub}">판정 품질 검증</span></span>
  <div style="display:flex;gap:40px;margin-top:4px;">
    <div style="display:flex;flex-direction:column;align-items:center;gap:3px;">
      <span style="font-size:11px;color:#9CA3AF;">pass</span>
      <span style="{a}">&#8595;</span>
      <span style="{p}background:#6366F1;">&#9873; END</span>
    </div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:3px;">
      <span style="font-size:11px;color:#9CA3AF;">fail</span>
      <span style="{a}">&#8593;</span>
      <span style="font-size:11px;color:#9CA3AF;">Synthesizer 재시도</span>
    </div>
  </div>
</div>
</body></html>"""
    components.html(html, height=360, scrolling=False)


# ── tool_logs 파싱 ─────────────────────────────────────────────────

def _group_logs_by_node(tool_logs: list[dict]) -> dict[str, list[dict]]:
    """tool_logs를 노드별로 그룹핑."""
    groups: dict[str, list[dict]] = {}
    for log in tool_logs:
        step = log.get("step", "unknown")
        # step → node_id 매핑
        node_id = "unknown"
        for node in _NODES:
            if step in node["steps"]:
                node_id = node["id"]
                break
        groups.setdefault(node_id, []).append(log)
    return groups


def _total_elapsed(tool_logs: list[dict]) -> float:
    """전체 소요 시간 추정 (CaseAgent∥PolicyAgent 병렬 실행은 최대값으로 계산)."""
    grouped = _group_logs_by_node(tool_logs)
    case_time = sum(lg.get("elapsed", 0) for lg in grouped.get("case_agent", []))
    policy_time = sum(lg.get("elapsed", 0) for lg in grouped.get("policy_agent", []))
    sequential_time = sum(
        sum(lg.get("elapsed", 0) for lg in grouped.get(nid, []))
        for nid in ("orchestrator", "synthesizer", "grade_answer")
    )
    return max(case_time, policy_time) + sequential_time


# ── 메인 시각화 함수 ───────────────────────────────────────────────

def render_pipeline_result(tool_logs: list[dict], judgment: str = "") -> None:
    """심의 완료 후 파이프라인 실행 결과를 시각화.

    Args:
        tool_logs: ReviewChain.run()에서 반환된 tool_logs
        judgment: 최종 판정 (위반소지/주의/OK)
    """
    if not tool_logs:
        st.caption("파이프라인 실행 로그가 없습니다.")
        return

    total_time = _total_elapsed(tool_logs)
    grouped = _group_logs_by_node(tool_logs)

    # ── 헤더: 판정 + 총 시간 ──
    if judgment:
        icon, color, label = _JUDGMENT_STYLE.get(judgment, ("⚪", "#888", judgment))
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">'
            f'<span style="font-size:2rem;">{icon}</span>'
            f'<span style="font-size:1.4rem;font-weight:700;color:{color};">{label}</span>'
            f'<span style="font-size:0.9rem;color:#888;">총 {total_time:.1f}초</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 노드별 실행 결과 ──
    for node in _NODES:
        node_logs = grouped.get(node["id"], [])
        if not node_logs:
            continue

        node_time = sum(lg.get("elapsed", 0) for lg in node_logs)
        step_count = len(node_logs)

        with st.expander(
            f"{node['label']}  —  {step_count}단계 · {node_time:.1f}초",
            expanded=False,
        ):
            st.caption(node["desc"])

            for idx, log in enumerate(node_logs):
                step = log.get("step", "unknown")
                step_label = _STEP_LABELS.get(step, step)
                elapsed = log.get("elapsed", 0)

                _render_step_card(step, step_label, elapsed, log)
                if idx < len(node_logs) - 1:
                    st.markdown("---")


def _render_step_card(step: str, label: str, elapsed: float, log: dict) -> None:
    """개별 step 카드 렌더링."""

    if step == "orchestrator":
        risk_types = log.get("risk_types", [])
        st.markdown(
            f"**{label}** `{elapsed:.1f}s`  \n"
            f"위험유형: {', '.join(risk_types) if risk_types else '(미분류)'}"
        )

    elif step in ("case_retrieve", "policy_retrieve"):
        query = log.get("query", "")
        total = log.get("total", 0)
        retry = log.get("retry", 0)
        status = "🔄 재시도" if retry > 0 else "🔍 검색"
        st.markdown(
            f"{status} **{label}** `{elapsed:.1f}s`  \n"
            f"쿼리: _{query[:60]}_  \n"
            f"결과: **{total}건** {'(retry #' + str(retry) + ')' if retry > 0 else ''}"
        )

    elif step in ("case_grade", "policy_grade"):
        total = log.get("total", 0)
        relevant = log.get("relevant", 0)
        rate = f"{relevant}/{total}" if total else "0/0"
        bar_pct = (relevant / total * 100) if total else 0
        st.markdown(
            f"📊 **{label}** `{elapsed:.1f}s`  \n"
            f"관련: **{rate}** ({bar_pct:.0f}%)"
        )

    elif step in ("case_rewrite", "policy_rewrite"):
        old_q = log.get("old_query", "")[:40]
        new_q = log.get("new_query", "")[:40]
        st.markdown(
            f"✏️ **{label}** `{elapsed:.1f}s`  \n"
            f"~~{old_q}~~ → _{new_q}_"
        )

    elif step == "synthesizer":
        j = log.get("judgment", "")
        icon = _JUDGMENT_STYLE.get(j, ("⚪", "#888", j))[0]
        st.markdown(f"{icon} **{label}** `{elapsed:.1f}s` → {j}")

    elif step == "grade_answer":
        grade = log.get("grade", "")
        icon = "✅" if grade == "pass" else "🔄"
        st.markdown(f"{icon} **{label}** `{elapsed:.1f}s` → {grade}")

    elif step == "error":
        msg = log.get("message", "알 수 없는 오류")
        st.error(f"⛔ {msg}")

    else:
        st.markdown(f"**{label}** `{elapsed:.1f}s`")


# ── 실시간 진행 표시 (AI 실행 중 사용) ──────────────────────────────

def render_progress_header() -> None:
    """AI 심의 실행 전 파이프라인 안내 표시."""
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1e3a5f,#2d5a87);'
        'border-radius:12px;padding:20px;margin-bottom:16px;">'
        '<p style="color:#93c5fd;font-size:0.85rem;margin:0 0 4px 0;">Multi-Agent Self-Corrective RAG</p>'
        '<p style="color:#fff;font-size:1.1rem;font-weight:600;margin:0;">'
        '🎯 Orchestrator → 📋 CaseAgent ∥ 📜 PolicyAgent → ⚖️ Synthesizer → ✅ GradeAnswer'
        '</p></div>',
        unsafe_allow_html=True,
    )


def render_execution_summary(tool_logs: list[dict]) -> None:
    """tool_logs를 요약 metrics로 표시 (상단 KPI 카드용)."""
    if not tool_logs:
        return

    grouped = _group_logs_by_node(tool_logs)
    total_time = _total_elapsed(tool_logs)

    # 사례 건수 — 마지막 grade 결과를 사용 (retry 시 중복 합산 방지)
    case_logs = grouped.get("case_agent", [])
    case_grade_logs = [lg for lg in case_logs if lg.get("step") == "case_grade"]
    case_total = case_grade_logs[-1].get("relevant", 0) if case_grade_logs else 0

    # 법규 건수 — 마지막 grade 결과를 사용
    policy_logs = grouped.get("policy_agent", [])
    policy_grade_logs = [lg for lg in policy_logs if lg.get("step") == "policy_grade"]
    policy_total = policy_grade_logs[-1].get("relevant", 0) if policy_grade_logs else 0

    # grade 결과
    grade_logs = grouped.get("grade_answer", [])
    grade_result = grade_logs[-1].get("grade", "N/A") if grade_logs else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⏱️ 총 소요", f"{total_time:.1f}s")
    c2.metric("📋 사례", f"{case_total}건")
    c3.metric("📜 법규", f"{policy_total}건")
    c4.metric("✅ 품질검증", grade_result)