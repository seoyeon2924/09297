"""
RAG Service — AI compliance recommendation pipeline.

Real retrieval (Chroma) + LLM generation via ReviewChain.
"""

import time
from typing import Generator

from langsmith import traceable

from chains.review_chain import ReviewChain
from storage.models import ReviewStatus
from storage.repository import ReviewRepository, AuditRepository


# ── 노드별 표시 설정 ──────────────────────────────────────────────

NODE_DISPLAY = {
    "orchestrator": {"icon": "🎯", "label": "Orchestrator", "desc": "위험유형 분류 + 검색 쿼리 생성"},
    "case_agent": {"icon": "📋", "label": "CaseAgent", "desc": "사례 검색 · 평가 · 재작성 루프"},
    "policy_agent": {"icon": "📜", "label": "PolicyAgent", "desc": "법령·규정·지침 검색 · 평가 · 재작성 루프"},
    "synthesizer": {"icon": "⚖️", "label": "Synthesizer", "desc": "근거 통합 → 최종 판정"},
    "grade_answer": {"icon": "✅", "label": "GradeAnswer", "desc": "판정 품질 검증"},
}


def _summarize_node(node_name: str, update: dict) -> str:
    """노드 완료 시 요약 메시지 생성."""
    display = NODE_DISPLAY.get(node_name, {"icon": "⚙️", "label": node_name, "desc": ""})

    if node_name == "orchestrator":
        plan = update.get("plan", {})
        risk_types = plan.get("risk_types", [])
        return f"위험유형: {', '.join(risk_types)}" if risk_types else "위험유형 분류 완료"

    elif node_name == "case_agent":
        cases = update.get("case_context", [])
        logs = update.get("tool_logs", [])
        retries = sum(1 for lg in logs if lg.get("step") == "case_rewrite")
        loop_info = f" (검색 {retries + 1}회 루프)" if retries > 0 else ""
        return f"사례 {len(cases)}건 확보{loop_info}"

    elif node_name == "policy_agent":
        law = len(update.get("law_chunks", []))
        reg = len(update.get("regulation_chunks", []))
        guide = len(update.get("guideline_chunks", []))
        logs = update.get("tool_logs", [])
        retries = sum(1 for lg in logs if lg.get("step") == "policy_rewrite")
        loop_info = f" (검색 {retries + 1}회 루프)" if retries > 0 else ""
        return f"법령 {law}건 · 규정 {reg}건 · 지침 {guide}건{loop_info}"

    elif node_name == "synthesizer":
        result = update.get("result", {})
        judgment = result.get("judgment", "")
        return f"판정: {judgment}" if judgment else "판정 생성 완료"

    elif node_name == "grade_answer":
        grade = update.get("answer_grade", "")
        return f"결과: {grade}"

    return ""


class RAGService:

    @staticmethod
    @traceable(name="run_recommendation", tags=["rag-service"])
    def run_recommendation(request_id: str) -> list[dict]:
        """기존 방식 (비스트리밍) — eval, 백그라운드 실행 등에서 사용."""
        detail = ReviewRepository.get_detail(request_id)
        if not detail:
            raise ValueError(f"Request {request_id} not found")

        ReviewRepository.update_request_status(
            request_id, ReviewStatus.AI_RUNNING.value
        )

        try:
            chain = ReviewChain(model_name="gpt-4o-mini")

            results = []
            for item in detail["items"]:
                start = time.time()

                result = chain.run(
                    item_text=item["text"],
                    category=detail["request"]["category"],
                    broadcast_type=detail["request"]["broadcast_type"],
                )

                latency = int((time.time() - start) * 1000)

                try:
                    import streamlit as st
                    if "pipeline_logs" not in st.session_state:
                        st.session_state.pipeline_logs = {}
                    st.session_state.pipeline_logs[item["id"]] = {
                        "tool_logs": result.get("tool_logs", []),
                        "judgment": result.get("judgment", ""),
                        "latency_ms": latency,
                    }
                except Exception:
                    pass

                rec_id = ReviewRepository.create_ai_recommendation(
                    review_item_id=item["id"],
                    judgment=result.get("judgment", "주의"),
                    reason=result.get("reason", ""),
                    references=result.get("references", []),
                    model_name=chain.model_name,
                    prompt_version="v1.0-rag-pipeline",
                    latency_ms=latency,
                )
                results.append({"item_id": item["id"], "rec_id": rec_id})

            ReviewRepository.update_request_status(
                request_id, ReviewStatus.REVIEWING.value
            )

            AuditRepository.create_log(
                event_type="AI_RECOMMEND",
                entity_type="ReviewRequest",
                entity_id=request_id,
                actor="system",
                detail={
                    "item_count": len(detail["items"]),
                    "model": chain.model_name,
                    "pipeline": "v1.0-rag",
                },
            )

            return results

        except Exception as e:
            ReviewRepository.update_request_status(
                request_id, ReviewStatus.REQUESTED.value
            )
            raise e

    @staticmethod
    def stream_recommendation(request_id: str) -> Generator[dict, None, list[dict]]:
        """스트리밍 방식 — UI에서 실시간 진행 표시용.

        Yields:
            {"node": str, "status": "running"|"done", "summary": str, "elapsed": float}

        Returns:
            list[dict] — 최종 결과 (run_recommendation과 동일 형태)
        """
        detail = ReviewRepository.get_detail(request_id)
        if not detail:
            raise ValueError(f"Request {request_id} not found")

        ReviewRepository.update_request_status(
            request_id, ReviewStatus.AI_RUNNING.value
        )

        try:
            import streamlit as st
            has_st = True
        except Exception:
            has_st = False

        try:
            chain = ReviewChain(model_name="gpt-4o-mini")
            results = []

            for item in detail["items"]:
                all_tool_logs: list[dict] = []
                start = time.time()
                final_result = None

                for node_name, update in chain.stream(
                    item_text=item["text"],
                    category=detail["request"]["category"],
                    broadcast_type=detail["request"]["broadcast_type"],
                ):
                    if node_name == "__done__":
                        final_result = update
                        continue
                    if node_name == "__error__":
                        final_result = {
                            "result": {"judgment": "주의", "reason": f"오류: {update.get('error', '')}"},
                            "tool_logs": [{"step": "error", "message": update.get("error", "")}],
                        }
                        continue

                    node_elapsed = time.time() - start
                    summary = _summarize_node(node_name, update)

                    yield {
                        "node": node_name,
                        "status": "done",
                        "summary": summary,
                        "elapsed": round(node_elapsed, 1),
                        "item_label": item.get("label", ""),
                    }

                    # tool_logs 수집
                    if "tool_logs" in update:
                        all_tool_logs.extend(update["tool_logs"])

                latency = int((time.time() - start) * 1000)

                if final_result is None:
                    final_result = {"result": {}, "tool_logs": []}

                result = final_result.get("result", final_result)
                if "tool_logs" not in result:
                    result["tool_logs"] = all_tool_logs

                # session_state에 저장
                if has_st:
                    try:
                        if "pipeline_logs" not in st.session_state:
                            st.session_state.pipeline_logs = {}
                        st.session_state.pipeline_logs[item["id"]] = {
                            "tool_logs": all_tool_logs,
                            "judgment": result.get("judgment", ""),
                            "latency_ms": latency,
                        }
                    except Exception:
                        pass

                rec_id = ReviewRepository.create_ai_recommendation(
                    review_item_id=item["id"],
                    judgment=result.get("judgment", "주의"),
                    reason=result.get("reason", ""),
                    references=result.get("references", []),
                    model_name=chain.model_name,
                    prompt_version="v1.0-rag-pipeline",
                    latency_ms=latency,
                )
                results.append({"item_id": item["id"], "rec_id": rec_id})

            ReviewRepository.update_request_status(
                request_id, ReviewStatus.REVIEWING.value
            )

            AuditRepository.create_log(
                event_type="AI_RECOMMEND",
                entity_type="ReviewRequest",
                entity_id=request_id,
                actor="system",
                detail={
                    "item_count": len(detail["items"]),
                    "model": chain.model_name,
                    "pipeline": "v1.0-rag-stream",
                },
            )

            return results

        except Exception as e:
            ReviewRepository.update_request_status(
                request_id, ReviewStatus.REQUESTED.value
            )
            raise e

    @staticmethod
    async def async_stream_recommendation(request_id: str):
        """Async generator — FastAPI SSE 스트리밍용.

        Yields:
            {"node": str, "status": "done", "summary": str,
             "elapsed": float, "item_label": str}
        """
        detail = ReviewRepository.get_detail(request_id)
        if not detail:
            raise ValueError(f"Request {request_id} not found")

        ReviewRepository.update_request_status(
            request_id, ReviewStatus.AI_RUNNING.value
        )

        try:
            chain = ReviewChain(model_name="gpt-4o-mini")
            results = []

            for item in detail["items"]:
                all_tool_logs: list[dict] = []
                start = time.time()
                final_result = None

                async for node_name, update in chain.astream(
                    item_text=item["text"],
                    category=detail["request"]["category"],
                    broadcast_type=detail["request"]["broadcast_type"],
                ):
                    if node_name == "__done__":
                        final_result = update
                        continue
                    if node_name == "__error__":
                        final_result = {
                            "result": {
                                "judgment": "주의",
                                "reason": f"오류: {update.get('error', '')}",
                            },
                            "tool_logs": [{"step": "error", "message": update.get("error", "")}],
                        }
                        continue

                    node_elapsed = time.time() - start
                    summary = _summarize_node(node_name, update)

                    yield {
                        "node": node_name,
                        "status": "done",
                        "summary": summary,
                        "elapsed": round(node_elapsed, 1),
                        "item_label": item.get("label", ""),
                    }

                    if "tool_logs" in update:
                        all_tool_logs.extend(update["tool_logs"])

                latency = int((time.time() - start) * 1000)

                if final_result is None:
                    final_result = {"result": {}, "tool_logs": []}

                result = final_result.get("result", final_result)
                if "tool_logs" not in result:
                    result["tool_logs"] = all_tool_logs

                rec_id = ReviewRepository.create_ai_recommendation(
                    review_item_id=item["id"],
                    judgment=result.get("judgment", "주의"),
                    reason=result.get("reason", ""),
                    references=result.get("references", []),
                    model_name=chain.model_name,
                    prompt_version="v1.0-rag-astream",
                    latency_ms=latency,
                )
                results.append({"item_id": item["id"], "rec_id": rec_id})

            ReviewRepository.update_request_status(
                request_id, ReviewStatus.REVIEWING.value
            )

            AuditRepository.create_log(
                event_type="AI_RECOMMEND",
                entity_type="ReviewRequest",
                entity_id=request_id,
                actor="system",
                detail={
                    "item_count": len(detail["items"]),
                    "model": chain.model_name,
                    "pipeline": "v1.0-rag-astream",
                },
            )

        except Exception as e:
            ReviewRepository.update_request_status(
                request_id, ReviewStatus.REQUESTED.value
            )
            raise e