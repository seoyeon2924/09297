"""
ReviewChain — 멀티 에이전트 Self-Corrective RAG (LangGraph).

흐름:
    orchestrator
        ├─► case_agent_node    (CaseAgent: 사례 전용 retrieve→grade→rewrite 루프)
        └─► policy_agent_node  (PolicyAgent: 법령·규정·지침 전용 retrieve→grade→rewrite 루프)
    ↓ (두 에이전트 병렬 완료 후)
    synthesizer  →  grade_answer  →  end
                         └─(fail)─► synthesizer (최대 2회 재시도)
"""

from __future__ import annotations

import json
import logging
import operator
import re
import time
from typing import Annotated, TypedDict

import httpx
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from chains.case_agent import get_case_agent
from config import settings
from prompts.generator import GENERATOR_HUMAN, GENERATOR_SYSTEM
from prompts.grader import GRADE_ANSWER_HUMAN, GRADE_ANSWER_SYSTEM
from prompts.planner import PLANNER_HUMAN, PLANNER_SYSTEM
from tools.policy_tools import search_policy

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────

class ReviewState(TypedDict):
    """LangGraph 공유 상태.
    case_agent_node와 policy_agent_node가 각각 다른 키에 쓰므로 충돌 없음.
    tool_logs만 Annotated[list, operator.add]로 병렬 병합.
    """

    # 입력 (불변)
    item_text: str
    category: str
    broadcast_type: str

    # orchestrator 출력
    plan: dict                          # risk_types, search_queries(policy/cases)

    # CaseAgent 출력  ← case_agent_node가 씀
    case_context: list[dict]

    # PolicyAgent 출력  ← policy_agent_node가 씀
    law_chunks: list[dict]
    regulation_chunks: list[dict]
    guideline_chunks: list[dict]

    # synthesizer / grade_answer 출력
    result: dict
    answer_grade: str
    retry_count: int
    max_retries: int

    # 각 노드 실행 로그 (병렬 노드가 동시에 append → operator.add로 안전 병합)
    tool_logs: Annotated[list, operator.add]


# ── LLM / Parser ─────────────────────────────────────────────────

_http_client = httpx.Client(verify=False)
_http_async_client = httpx.AsyncClient(verify=False)


def _get_llm(model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=settings.OPENAI_API_KEY or "dummy",
        http_client=_http_client,
        http_async_client=_http_async_client,
        request_timeout=90,
    )


_llm = _get_llm()
_json_parser = JsonOutputParser()


# ── PolicyAgent 전용 인라인 프롬프트 ─────────────────────────────

_POLICY_GRADE_SYSTEM = """당신은 방송심의 법령·규정·지침 관련성 평가 전문가입니다.
주어진 심의 대상 문구와 위험 유형에 대해, 각 문서가 심의 판단에 관련이 있는지 평가하세요.

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "grades": [
        {{"doc_index": 1, "relevance": "relevant", "reason": "이유"}},
        {{"doc_index": 2, "relevance": "irrelevant", "reason": "이유"}}
    ]
}}
relevance는 반드시 "relevant" 또는 "irrelevant" 중 하나입니다."""

_POLICY_GRADE_HUMAN = """## 심의 대상 문구
{item_text}

## 위험 유형
{risk_type}

## 평가 대상 문서들
{docs_text}

위 문서들 각각에 대해 심의 대상 문구와의 관련성을 평가해주세요."""

_POLICY_REWRITE_SYSTEM = """당신은 방송심의 법령·규정·지침 검색 전문가입니다.
이전 검색에서 관련 문서를 충분히 찾지 못했습니다.
법조문 키워드, 규정 용어, 조항 관련 표현에 집중하여 더 효과적인 검색 쿼리를 생성하세요.

반드시 아래 JSON 형식으로만 응답하세요:
{{"policy_query": "개선된 검색 쿼리"}}"""

_POLICY_REWRITE_HUMAN = """## 심의 대상 문구
{item_text}

## 위험 유형
{risk_type}

## 이전 검색 쿼리
{old_query}

더 관련성 높은 법령·규정·지침을 찾기 위한 새로운 검색 쿼리를 생성해주세요."""


# ── 헬퍼 ─────────────────────────────────────────────────────────

def _strip_display_header(text: str) -> str:
    """display_text 앞의 [처리번호: ...][처리일자: ...][심의지적코드: ...] 헤더 라인 제거."""
    # 첫 줄이 [처리번호: ...] 등의 연속 태그로만 이루어진 경우 제거
    stripped = re.sub(
        r"^(\[(?:처리번호|처리일자|심의지적코드)\s*:[^\]]*\]\s*)+\n?",
        "",
        text,
        count=1,
    )
    return stripped.strip()


def _format_chunks(chunks: list[dict], label: str) -> str:
    """검색된 chunk 리스트를 LLM 컨텍스트용 문자열로 포맷팅."""
    if not chunks:
        return f"(검색된 {label} 근거 없음)"

    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        score = chunk.get("relevance_score", 0)
        cid = chunk.get("chroma_id", "N/A")
        raw_content = (chunk.get("content") or "")[:500]
        source = meta.get("source_file") or meta.get("doc_filename") or "N/A"
        doc_type = meta.get("doc_type", "N/A")
        section = (
            meta.get("article_title")
            or meta.get("section")
            or meta.get("section_title")
            or "N/A"
        )

        case_number = meta.get("case_number", "")
        case_date = meta.get("case_date", "")
        violation_type = meta.get("violation_type", "")
        if case_number or case_date:
            # 사례 청크: metadata로 처리건 헤더를 구성하므로,
            # content(display_text)에서 [처리번호:...] 헤더 라인을 제거하여 이중 노출 방지
            content = _strip_display_header(raw_content)
            case_ref = ""
            if case_date:
                case_ref += f"{case_date}에 나온 "
            if case_number:
                case_ref += f"처리번호 {case_number} 건"
            if violation_type:
                case_ref += f"에서 [{violation_type}] 지적"
            lines.append(
                f"[{label} {i}] (유사도: {score})\n"
                f"  처리건: {case_ref}\n"
                f"  내용: {content}"
            )
        else:
            lines.append(
                f"[{label} {i}] (유사도: {score}, ID: {cid})\n"
                f"  출처: {source} | 유형: {doc_type} | 섹션: {section}\n"
                f"  내용: {raw_content}"
            )
    return "\n\n".join(lines)


def _context_to_refs(
    case_context: list[dict],
    law_chunks: list[dict],
    regulation_chunks: list[dict],
    guideline_chunks: list[dict],
) -> list[dict]:
    """에이전트별 컨텍스트를 reference 포맷으로 변환."""
    label_map = [
        ("사례", case_context),
        ("법령", law_chunks),
        ("규정", regulation_chunks),
        ("지침", guideline_chunks),
    ]
    refs = []
    for default_label, chunks in label_map:
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            refs.append({
                "chroma_id": chunk.get("chroma_id", ""),
                "doc_filename": meta.get("source_file") or meta.get("doc_filename") or "",
                "doc_type": meta.get("doc_type", default_label),
                "case_number": meta.get("case_number", ""),
                "case_date": meta.get("case_date", ""),
                "article_number": (
                    meta.get("article_title") or meta.get("article_number") or ""
                ),
                "section_title": meta.get("section") or meta.get("section_title") or "",
                "relevance_score": chunk.get("relevance_score", 0),
                "content": (chunk.get("content") or "")[:600],
            })
    return refs


def _query_to_str(q: str | list[str] | None, fallback: str) -> str:
    if q is None:
        return fallback
    if isinstance(q, str):
        return q.strip() or fallback
    if isinstance(q, list) and q:
        return (q[0].strip() if isinstance(q[0], str) else fallback) or fallback
    return fallback


# ── 노드 1: orchestrator ──────────────────────────────────────────

def orchestrator_node(state: ReviewState) -> dict:
    """위험 유형 분류 + CaseAgent·PolicyAgent 각각의 검색 쿼리 생성."""
    start = time.time()

    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM),
        ("human", PLANNER_HUMAN),
    ])
    chain = prompt | _llm | _json_parser

    try:
        plan = chain.invoke({
            "item_text": state["item_text"],
            "category": state.get("category", "미지정"),
            "broadcast_type": state.get("broadcast_type", "미지정"),
        })
        if not isinstance(plan, dict):
            plan = {}
    except Exception as e:
        logger.error("Orchestrator 실패: %s", e)
        plan = {}

    if not plan:
        plan = {
            "risk_types": ["방송심의일반"],
            "tools_to_use": ["policy_search", "case_search"],
            "search_queries": {
                "policy": state["item_text"],
                "cases": state["item_text"],
            },
        }

    elapsed = round(time.time() - start, 2)
    return {
        "plan": plan,
        "case_context": [],
        "law_chunks": [],
        "regulation_chunks": [],
        "guideline_chunks": [],
        "retry_count": 0,
        "max_retries": 2,
        "tool_logs": [{"step": "orchestrator", "risk_types": plan.get("risk_types", []), "elapsed": elapsed}],
    }


# ── 노드 2: case_agent_node ───────────────────────────────────────

def case_agent_node(state: ReviewState) -> dict:
    """CaseAgent 실행 — 사례 전용 retrieve→grade→rewrite 루프."""
    plan = state.get("plan", {})
    queries = plan.get("search_queries", {})
    item_text = state["item_text"]
    risk_types = plan.get("risk_types", [])
    risk_type = ", ".join(risk_types)

    case_query = _query_to_str(queries.get("cases"), item_text)

    agent = get_case_agent()
    result = agent.run(
        query=case_query,
        item_text=item_text,
        risk_type=risk_type,
    )

    logger.info("CaseAgent 완료: %d건 사례 확보", len(result.get("case_chunks", [])))
    return {
        "case_context": result.get("case_chunks", []),
        "tool_logs": result.get("tool_logs", []),
    }


# ── 노드 3: policy_agent_node ─────────────────────────────────────

def policy_agent_node(state: ReviewState) -> dict:
    """PolicyAgent — 법령·규정·지침 전용 retrieve→grade→rewrite 루프 (인라인)."""
    plan = state.get("plan", {})
    queries = plan.get("search_queries", {})
    item_text = state["item_text"]
    risk_types = plan.get("risk_types", [])
    risk_type = ", ".join(risk_types)
    max_retries = state.get("max_retries", 2)

    policy_query = _query_to_str(queries.get("policy"), item_text)

    tool_logs: list[dict] = []
    retry = 0
    law_chunks: list[dict] = []
    regulation_chunks: list[dict] = []
    guideline_chunks: list[dict] = []

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", _POLICY_GRADE_SYSTEM),
        ("human", _POLICY_GRADE_HUMAN),
    ])
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", _POLICY_REWRITE_SYSTEM),
        ("human", _POLICY_REWRITE_HUMAN),
    ])
    grade_chain = grade_prompt | _llm | _json_parser
    rewrite_chain = rewrite_prompt | _llm | _json_parser

    while True:
        # 1. Retrieve
        start = time.time()
        try:
            raw = search_policy.invoke({"query": policy_query})
            raw_law = raw.get("law_chunks", [])
            raw_reg = raw.get("regulation_chunks", [])
            raw_guide = raw.get("guideline_chunks", [])
            all_policy = raw_law + raw_reg + raw_guide
        except Exception as e:
            logger.error("PolicyAgent retrieve 실패: %s", e)
            all_policy = []

        tool_logs.append({
            "step": "policy_retrieve",
            "query": policy_query,
            "total": len(all_policy),
            "retry": retry,
            "elapsed": round(time.time() - start, 2),
        })

        if not all_policy:
            if retry >= max_retries:
                break
            policy_query = _rewrite_policy(rewrite_chain, policy_query, item_text, risk_type, tool_logs)
            retry += 1
            continue

        # 2. Grade (배치 평가, 상위 15개)
        start = time.time()
        docs_text_parts: list[str] = []
        for i, chunk in enumerate(all_policy[:15], 1):
            meta = chunk.get("metadata", {})
            source = meta.get("source_file", "N/A")
            doc_type = meta.get("doc_type", "N/A")
            content = (chunk.get("content") or "")[:400]
            docs_text_parts.append(
                f"[문서 {i}] 유형: {doc_type} | 출처: {source}\n내용: {content}"
            )

        try:
            grade_result = grade_chain.invoke({
                "item_text": item_text,
                "risk_type": risk_type,
                "docs_text": "\n\n---\n\n".join(docs_text_parts),
            })
            relevant_indices = {
                g.get("doc_index", 0) - 1
                for g in grade_result.get("grades", [])
                if g.get("relevance") == "relevant"
            }
            relevant_policy = [c for i, c in enumerate(all_policy[:15]) if i in relevant_indices]
            relevant_policy += all_policy[15:]   # 15개 초과분은 평가 없이 포함
        except Exception as e:
            logger.error("PolicyAgent grade 실패, 전체 포함: %s", e)
            relevant_policy = all_policy

        tool_logs.append({
            "step": "policy_grade",
            "total": len(all_policy),
            "relevant": len(relevant_policy),
            "elapsed": round(time.time() - start, 2),
        })

        if relevant_policy or retry >= max_retries:
            # doc_type 기준으로 재분류
            for chunk in relevant_policy:
                dt = chunk.get("metadata", {}).get("doc_type", "")
                if dt in {"법령", "law"}:
                    law_chunks.append(chunk)
                elif dt in {"지침", "guideline"}:
                    guideline_chunks.append(chunk)
                else:
                    regulation_chunks.append(chunk)
            break

        # 3. Rewrite
        policy_query = _rewrite_policy(rewrite_chain, policy_query, item_text, risk_type, tool_logs)
        retry += 1

    logger.info(
        "PolicyAgent 완료: 법령 %d건 / 규정 %d건 / 지침 %d건",
        len(law_chunks), len(regulation_chunks), len(guideline_chunks),
    )
    return {
        "law_chunks": law_chunks,
        "regulation_chunks": regulation_chunks,
        "guideline_chunks": guideline_chunks,
        "tool_logs": tool_logs,
    }


def _rewrite_policy(
    chain,
    old_query: str,
    item_text: str,
    risk_type: str,
    tool_logs: list[dict],
) -> str:
    start = time.time()
    try:
        result = chain.invoke({
            "item_text": item_text,
            "risk_type": risk_type,
            "old_query": old_query,
        })
        new_query = result.get("policy_query") or old_query
    except Exception as e:
        logger.error("PolicyAgent rewrite 실패: %s", e)
        new_query = old_query

    tool_logs.append({
        "step": "policy_rewrite",
        "old_query": old_query,
        "new_query": new_query,
        "elapsed": round(time.time() - start, 2),
    })
    return new_query


async def _arewrite_policy(
    chain,
    old_query: str,
    item_text: str,
    risk_type: str,
    tool_logs: list[dict],
) -> str:
    """_rewrite_policy()의 async 버전."""
    start = time.time()
    try:
        result = await chain.ainvoke({
            "item_text": item_text,
            "risk_type": risk_type,
            "old_query": old_query,
        })
        new_query = result.get("policy_query") or old_query
    except Exception as e:
        logger.error("PolicyAgent async rewrite 실패: %s", e)
        new_query = old_query

    tool_logs.append({
        "step": "policy_rewrite",
        "old_query": old_query,
        "new_query": new_query,
        "elapsed": round(time.time() - start, 2),
    })
    return new_query


# ── 노드 4: synthesizer ───────────────────────────────────────────

def synthesizer_node(state: ReviewState) -> dict:
    """두 에이전트의 결과를 통합하여 최종 판정 생성."""
    start = time.time()
    plan = state.get("plan", {})
    risk_types = plan.get("risk_types", ["방송심의일반"])

    case_context = state.get("case_context", [])
    law_chunks = state.get("law_chunks", [])
    regulation_chunks = state.get("regulation_chunks", [])
    guideline_chunks = state.get("guideline_chunks", [])

    default_result: dict = {
        "judgment": "주의",
        "reason": "",
        "risk_type": risk_types[0] if risk_types else "",
        "related_articles": [],
        "suggested_fix": "",
        "references": [],
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATOR_SYSTEM),
        ("human", GENERATOR_HUMAN),
    ])
    chain = prompt | _llm | _json_parser

    try:
        result = chain.invoke({
            "item_text": state["item_text"],
            "category": state.get("category", "미지정"),
            "risk_type": ", ".join(risk_types),
            "law_context": _format_chunks(law_chunks, "법률"),
            "regulation_context": _format_chunks(regulation_chunks, "규정"),
            "guideline_context": _format_chunks(guideline_chunks, "지침"),
            "case_context": _format_chunks(case_context, "사례"),
        })
        if isinstance(result, dict):
            default_result.update(result)
            result = default_result
        else:
            result = default_result
    except Exception as e:
        logger.error("Synthesizer 실패: %s", e)
        default_result["reason"] = f"AI 생성 중 오류 발생: {str(e)}"
        result = default_result

    # LLM 선택 references에 검색된 전체 청크 보완
    existing_ids = {r.get("chroma_id") for r in result.get("references", []) if r.get("chroma_id")}
    for ref in _context_to_refs(case_context, law_chunks, regulation_chunks, guideline_chunks):
        if ref["chroma_id"] not in existing_ids:
            result.setdefault("references", []).append(ref)
            existing_ids.add(ref["chroma_id"])

    elapsed = round(time.time() - start, 2)
    return {
        "result": result,
        "tool_logs": [{"step": "synthesizer", "judgment": result.get("judgment", ""), "elapsed": elapsed}],
    }


# ── 노드 5: grade_answer ──────────────────────────────────────────

def grade_answer_node(state: ReviewState) -> dict:
    """생성된 심의 의견 품질 검증."""
    start = time.time()
    result = state.get("result", {})

    all_chunks = (
        state.get("case_context", [])
        + state.get("law_chunks", [])
        + state.get("regulation_chunks", [])
        + state.get("guideline_chunks", [])
    )
    context_parts = [(c.get("content") or "")[:200] for c in all_chunks]
    context_summary = ("\n---\n".join(context_parts))[:2000] if context_parts else "(검색된 근거 없음)"

    prompt = ChatPromptTemplate.from_messages([
        ("system", GRADE_ANSWER_SYSTEM),
        ("human", GRADE_ANSWER_HUMAN),
    ])
    chain = prompt | _llm | _json_parser

    try:
        grade = chain.invoke({
            "item_text": state["item_text"],
            "answer_json": json.dumps(result, ensure_ascii=False, indent=2),
            "context_summary": context_summary,
        })
        answer_grade = grade.get("grade", "pass") if isinstance(grade, dict) else "pass"
    except Exception as e:
        logger.warning("답변 평가 실패, pass 처리: %s", e)
        answer_grade = "pass"

    elapsed = round(time.time() - start, 2)
    retry_count = state.get("retry_count", 0) + 1

    return {
        "answer_grade": answer_grade,
        "retry_count": retry_count,
        "tool_logs": [{"step": "grade_answer", "grade": answer_grade, "elapsed": elapsed}],
    }


# ── 라우터 ────────────────────────────────────────────────────────

def route_after_grade_answer(state: ReviewState) -> str:
    """grade_answer 결과에 따라 종료 또는 synthesizer 재시도 결정."""
    answer_grade = state.get("answer_grade", "pass")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if answer_grade == "pass":
        return "end"
    if retry_count >= max_retries:
        logger.warning("grade_answer: 재시도 %d회 초과, 현재 결과로 종료", max_retries)
        return "end"
    return "synthesizer"


# ── Async 노드 (graph.astream() / FastAPI SSE용) ──────────────────

async def orchestrator_node_async(state: ReviewState) -> dict:
    """orchestrator_node의 async 버전."""
    start = time.time()

    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM),
        ("human", PLANNER_HUMAN),
    ])
    chain = prompt | _llm | _json_parser

    try:
        plan = await chain.ainvoke({
            "item_text": state["item_text"],
            "category": state.get("category", "미지정"),
            "broadcast_type": state.get("broadcast_type", "미지정"),
        })
        if not isinstance(plan, dict):
            plan = {}
    except Exception as e:
        logger.error("Orchestrator async 실패: %s", e)
        plan = {}

    if not plan:
        plan = {
            "risk_types": ["방송심의일반"],
            "tools_to_use": ["policy_search", "case_search"],
            "search_queries": {
                "policy": state["item_text"],
                "cases": state["item_text"],
            },
        }

    elapsed = round(time.time() - start, 2)
    return {
        "plan": plan,
        "case_context": [],
        "law_chunks": [],
        "regulation_chunks": [],
        "guideline_chunks": [],
        "retry_count": 0,
        "max_retries": 2,
        "tool_logs": [{"step": "orchestrator", "risk_types": plan.get("risk_types", []), "elapsed": elapsed}],
    }


async def case_agent_node_async(state: ReviewState) -> dict:
    """case_agent_node의 async 버전 — CaseAgent.arun() 호출."""
    from chains.case_agent import get_case_agent
    plan = state.get("plan", {})
    queries = plan.get("search_queries", {})
    item_text = state["item_text"]
    risk_types = plan.get("risk_types", [])
    risk_type = ", ".join(risk_types)

    case_query = _query_to_str(queries.get("cases"), item_text)

    agent = get_case_agent()
    result = await agent.arun(
        query=case_query,
        item_text=item_text,
        risk_type=risk_type,
    )

    logger.info("CaseAgent async 완료: %d건 사례 확보", len(result.get("case_chunks", [])))
    return {
        "case_context": result.get("case_chunks", []),
        "tool_logs": result.get("tool_logs", []),
    }


async def policy_agent_node_async(state: ReviewState) -> dict:
    """policy_agent_node의 async 버전."""
    plan = state.get("plan", {})
    queries = plan.get("search_queries", {})
    item_text = state["item_text"]
    risk_types = plan.get("risk_types", [])
    risk_type = ", ".join(risk_types)
    max_retries = state.get("max_retries", 2)

    policy_query = _query_to_str(queries.get("policy"), item_text)

    tool_logs: list[dict] = []
    retry = 0
    law_chunks: list[dict] = []
    regulation_chunks: list[dict] = []
    guideline_chunks: list[dict] = []

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", _POLICY_GRADE_SYSTEM),
        ("human", _POLICY_GRADE_HUMAN),
    ])
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", _POLICY_REWRITE_SYSTEM),
        ("human", _POLICY_REWRITE_HUMAN),
    ])
    grade_chain = grade_prompt | _llm | _json_parser
    rewrite_chain = rewrite_prompt | _llm | _json_parser

    while True:
        # 1. Retrieve
        start = time.time()
        try:
            raw = await search_policy.ainvoke({"query": policy_query})
            raw_law = raw.get("law_chunks", [])
            raw_reg = raw.get("regulation_chunks", [])
            raw_guide = raw.get("guideline_chunks", [])
            all_policy = raw_law + raw_reg + raw_guide
        except Exception as e:
            logger.error("PolicyAgent async retrieve 실패: %s", e)
            all_policy = []

        tool_logs.append({
            "step": "policy_retrieve",
            "query": policy_query,
            "total": len(all_policy),
            "retry": retry,
            "elapsed": round(time.time() - start, 2),
        })

        if not all_policy:
            if retry >= max_retries:
                break
            policy_query = await _arewrite_policy(rewrite_chain, policy_query, item_text, risk_type, tool_logs)
            retry += 1
            continue

        # 2. Grade (배치 평가, 상위 15개)
        start = time.time()
        docs_text_parts: list[str] = []
        for i, chunk in enumerate(all_policy[:15], 1):
            meta = chunk.get("metadata", {})
            source = meta.get("source_file", "N/A")
            doc_type = meta.get("doc_type", "N/A")
            content = (chunk.get("content") or "")[:400]
            docs_text_parts.append(
                f"[문서 {i}] 유형: {doc_type} | 출처: {source}\n내용: {content}"
            )

        try:
            grade_result = await grade_chain.ainvoke({
                "item_text": item_text,
                "risk_type": risk_type,
                "docs_text": "\n\n---\n\n".join(docs_text_parts),
            })
            relevant_indices = {
                g.get("doc_index", 0) - 1
                for g in grade_result.get("grades", [])
                if g.get("relevance") == "relevant"
            }
            relevant_policy = [c for i, c in enumerate(all_policy[:15]) if i in relevant_indices]
            relevant_policy += all_policy[15:]
        except Exception as e:
            logger.error("PolicyAgent async grade 실패, 전체 포함: %s", e)
            relevant_policy = all_policy

        tool_logs.append({
            "step": "policy_grade",
            "total": len(all_policy),
            "relevant": len(relevant_policy),
            "elapsed": round(time.time() - start, 2),
        })

        if relevant_policy or retry >= max_retries:
            for chunk in relevant_policy:
                dt = chunk.get("metadata", {}).get("doc_type", "")
                if dt in {"법령", "law"}:
                    law_chunks.append(chunk)
                elif dt in {"지침", "guideline"}:
                    guideline_chunks.append(chunk)
                else:
                    regulation_chunks.append(chunk)
            break

        # 3. Rewrite
        policy_query = await _arewrite_policy(rewrite_chain, policy_query, item_text, risk_type, tool_logs)
        retry += 1

    logger.info(
        "PolicyAgent async 완료: 법령 %d건 / 규정 %d건 / 지침 %d건",
        len(law_chunks), len(regulation_chunks), len(guideline_chunks),
    )
    return {
        "law_chunks": law_chunks,
        "regulation_chunks": regulation_chunks,
        "guideline_chunks": guideline_chunks,
        "tool_logs": tool_logs,
    }


async def synthesizer_node_async(state: ReviewState) -> dict:
    """synthesizer_node의 async 버전."""
    start = time.time()
    plan = state.get("plan", {})
    risk_types = plan.get("risk_types", ["방송심의일반"])

    case_context = state.get("case_context", [])
    law_chunks = state.get("law_chunks", [])
    regulation_chunks = state.get("regulation_chunks", [])
    guideline_chunks = state.get("guideline_chunks", [])

    default_result: dict = {
        "judgment": "주의",
        "reason": "",
        "risk_type": risk_types[0] if risk_types else "",
        "related_articles": [],
        "suggested_fix": "",
        "references": [],
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATOR_SYSTEM),
        ("human", GENERATOR_HUMAN),
    ])
    chain = prompt | _llm | _json_parser

    try:
        result = await chain.ainvoke({
            "item_text": state["item_text"],
            "category": state.get("category", "미지정"),
            "risk_type": ", ".join(risk_types),
            "law_context": _format_chunks(law_chunks, "법률"),
            "regulation_context": _format_chunks(regulation_chunks, "규정"),
            "guideline_context": _format_chunks(guideline_chunks, "지침"),
            "case_context": _format_chunks(case_context, "사례"),
        })
        if isinstance(result, dict):
            default_result.update(result)
            result = default_result
        else:
            result = default_result
    except Exception as e:
        logger.error("Synthesizer async 실패: %s", e)
        default_result["reason"] = f"AI 생성 중 오류 발생: {str(e)}"
        result = default_result

    existing_ids = {r.get("chroma_id") for r in result.get("references", []) if r.get("chroma_id")}
    for ref in _context_to_refs(case_context, law_chunks, regulation_chunks, guideline_chunks):
        if ref["chroma_id"] not in existing_ids:
            result.setdefault("references", []).append(ref)
            existing_ids.add(ref["chroma_id"])

    elapsed = round(time.time() - start, 2)
    return {
        "result": result,
        "tool_logs": [{"step": "synthesizer", "judgment": result.get("judgment", ""), "elapsed": elapsed}],
    }


async def grade_answer_node_async(state: ReviewState) -> dict:
    """grade_answer_node의 async 버전."""
    start = time.time()
    result = state.get("result", {})

    all_chunks = (
        state.get("case_context", [])
        + state.get("law_chunks", [])
        + state.get("regulation_chunks", [])
        + state.get("guideline_chunks", [])
    )
    context_parts = [(c.get("content") or "")[:200] for c in all_chunks]
    context_summary = ("\n---\n".join(context_parts))[:2000] if context_parts else "(검색된 근거 없음)"

    prompt = ChatPromptTemplate.from_messages([
        ("system", GRADE_ANSWER_SYSTEM),
        ("human", GRADE_ANSWER_HUMAN),
    ])
    chain = prompt | _llm | _json_parser

    try:
        grade = await chain.ainvoke({
            "item_text": state["item_text"],
            "answer_json": json.dumps(result, ensure_ascii=False, indent=2),
            "context_summary": context_summary,
        })
        answer_grade = grade.get("grade", "pass") if isinstance(grade, dict) else "pass"
    except Exception as e:
        logger.warning("답변 async 평가 실패, pass 처리: %s", e)
        answer_grade = "pass"

    elapsed = round(time.time() - start, 2)
    retry_count = state.get("retry_count", 0) + 1

    return {
        "answer_grade": answer_grade,
        "retry_count": retry_count,
        "tool_logs": [{"step": "grade_answer", "grade": answer_grade, "elapsed": elapsed}],
    }


# ── Async 그래프 구성 ──────────────────────────────────────────────

def _build_async_graph() -> StateGraph:
    """async 노드로 구성된 그래프 — graph.astream()에서 사용."""
    workflow = StateGraph(ReviewState)

    workflow.add_node("orchestrator", orchestrator_node_async)
    workflow.add_node("case_agent", case_agent_node_async)
    workflow.add_node("policy_agent", policy_agent_node_async)
    workflow.add_node("synthesizer", synthesizer_node_async)
    workflow.add_node("grade_answer", grade_answer_node_async)

    workflow.add_edge(START, "orchestrator")
    workflow.add_edge("orchestrator", "case_agent")
    workflow.add_edge("orchestrator", "policy_agent")
    workflow.add_edge("case_agent", "synthesizer")
    workflow.add_edge("policy_agent", "synthesizer")
    workflow.add_edge("synthesizer", "grade_answer")
    workflow.add_conditional_edges(
        "grade_answer",
        route_after_grade_answer,
        {"end": END, "synthesizer": "synthesizer"},
    )

    return workflow.compile()


# ── 그래프 구성 ───────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    workflow = StateGraph(ReviewState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("case_agent", case_agent_node)
    workflow.add_node("policy_agent", policy_agent_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("grade_answer", grade_answer_node)

    # orchestrator → 두 에이전트 병렬 fan-out
    workflow.add_edge(START, "orchestrator")
    workflow.add_edge("orchestrator", "case_agent")
    workflow.add_edge("orchestrator", "policy_agent")

    # 두 에이전트 완료 후 synthesizer fan-in (LangGraph가 양쪽 완료를 대기)
    workflow.add_edge("case_agent", "synthesizer")
    workflow.add_edge("policy_agent", "synthesizer")

    workflow.add_edge("synthesizer", "grade_answer")
    workflow.add_conditional_edges(
        "grade_answer",
        route_after_grade_answer,
        {"end": END, "synthesizer": "synthesizer"},
    )

    return workflow.compile()


graph = _build_graph()
async_graph = _build_async_graph()


# ── ReviewChain (기존 인터페이스 유지) ────────────────────────────

class ReviewChain:
    """
    방송 심의 멀티 에이전트 RAG 워크플로우.

    LangGraph:
        orchestrator
            ├─► case_agent    (사례 전용: retrieve→grade→rewrite 루프)
            └─► policy_agent  (법령·규정·지침 전용: retrieve→grade→rewrite 루프)
        → synthesizer → grade_answer → end

    rag_service.py 호출 인터페이스는 변경 없음:
        ReviewChain(model_name).run(item_text, category, broadcast_type) → dict
    """

    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.model_name = model_name

    def run(
        self,
        item_text: str,
        category: str = "",
        broadcast_type: str = "",
    ) -> dict:
        initial_state: ReviewState = {
            "item_text": item_text,
            "category": category or "미지정",
            "broadcast_type": broadcast_type or "미지정",
            "plan": {},
            "case_context": [],
            "law_chunks": [],
            "regulation_chunks": [],
            "guideline_chunks": [],
            "result": {},
            "answer_grade": "",
            "retry_count": 0,
            "max_retries": 2,
            "tool_logs": [],
        }

        try:
            run_label = item_text[:40].replace("\n", " ")
            final_state = graph.invoke(
                initial_state,
                config=RunnableConfig(
                    run_name=f"review_chain:{run_label}",
                    tags=["review-chain", "langgraph", "multi-agent", broadcast_type or "미지정"],
                    metadata={
                        "item_text": item_text,
                        "category": category,
                        "broadcast_type": broadcast_type,
                    },
                ),
            )
            output = dict(final_state.get("result", {}))
            output["tool_logs"] = final_state.get("tool_logs", [])
            return output

        except Exception as e:
            logger.error("Graph 실행 실패: %s", e)
            return {
                "judgment": "주의",
                "reason": f"AI 처리 중 오류 발생: {str(e)}",
                "risk_type": "",
                "related_articles": [],
                "suggested_fix": "",
                "references": [],
                "tool_logs": [{"step": "error", "message": str(e)}],
            }

    def stream(
        self,
        item_text: str,
        category: str = "",
        broadcast_type: str = "",
    ):
        """노드별 스트리밍 실행 — 각 노드 완료 시 (node_name, state_update) yield.

        UI에서 실시간 진행 표시에 사용:
            for node_name, update in chain.stream(item_text):
                # node_name: "orchestrator", "case_agent", "policy_agent", ...
                # update: 해당 노드가 반환한 state dict
        """
        initial_state: ReviewState = {
            "item_text": item_text,
            "category": category or "미지정",
            "broadcast_type": broadcast_type or "미지정",
            "plan": {},
            "case_context": [],
            "law_chunks": [],
            "regulation_chunks": [],
            "guideline_chunks": [],
            "result": {},
            "answer_grade": "",
            "retry_count": 0,
            "max_retries": 2,
            "tool_logs": [],
        }

        try:
            run_label = item_text[:40].replace("\n", " ")
            config = RunnableConfig(
                run_name=f"review_chain:{run_label}",
                tags=["review-chain", "langgraph", "multi-agent", broadcast_type or "미지정"],
                metadata={
                    "item_text": item_text,
                    "category": category,
                    "broadcast_type": broadcast_type,
                },
            )

            # synthesizer update를 별도 추적 (retry 시 마지막 실행 결과 사용)
            # grade_answer는 "result"를 갖지 않으므로 __done__에 synthesizer 결과 전달
            result_update: dict = {}
            for chunk in graph.stream(initial_state, config=config):
                # chunk: {node_name: state_update}
                for node_name, update in chunk.items():
                    if node_name == "synthesizer":
                        result_update = update  # {"result": {...}, "tool_logs": [...]}
                    yield node_name, update

            yield "__done__", result_update

        except Exception as e:
            logger.error("Graph stream 실패: %s", e)
            yield "__error__", {"error": str(e)}

    async def astream(
        self,
        item_text: str,
        category: str = "",
        broadcast_type: str = "",
    ):
        """노드별 비동기 스트리밍 — FastAPI SSE에서 사용.

        AsyncGenerator[tuple[str, dict]]:
            ("orchestrator", {...}), ("case_agent", {...}), ..., ("__done__", {...})
        """
        initial_state: ReviewState = {
            "item_text": item_text,
            "category": category or "미지정",
            "broadcast_type": broadcast_type or "미지정",
            "plan": {},
            "case_context": [],
            "law_chunks": [],
            "regulation_chunks": [],
            "guideline_chunks": [],
            "result": {},
            "answer_grade": "",
            "retry_count": 0,
            "max_retries": 2,
            "tool_logs": [],
        }

        run_label = item_text[:40].replace("\n", " ")
        config = RunnableConfig(
            run_name=f"review_chain_async:{run_label}",
            tags=["review-chain", "langgraph", "async", broadcast_type or "미지정"],
            metadata={
                "item_text": item_text,
                "category": category,
                "broadcast_type": broadcast_type,
            },
        )

        result_update: dict = {}
        try:
            async for chunk in async_graph.astream(initial_state, config=config):
                for node_name, update in chunk.items():
                    if node_name == "synthesizer":
                        result_update = update
                    yield node_name, update
            yield "__done__", result_update
        except Exception as e:
            logger.error("Graph astream 실패: %s", e)
            yield "__error__", {"error": str(e)}