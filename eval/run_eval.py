"""
Evaluation Pipeline — 방송 심의 AI Agent 정확도 자동 측정.

골든 데이터셋(eval/golden_dataset.json)을 사용하여 다음 지표를 측정:
  1. Judgment Accuracy  — 판정(위반소지/주의/OK)이 기대값과 일치하는 비율
  2. Risk Type Recall   — 기대 위험유형 중 실제 검출된 비율
  3. Evidence Quality    — 근거 키워드가 references/reason에 포함된 비율
  4. Retrieval Hit Rate  — 검색된 chunks 중 관련성 있는 것의 비율
  5. Latency             — 각 항목별 / 전체 평균 처리 시간

사용법:
    python eval/run_eval.py                     # 전체 실행
    python eval/run_eval.py --tag baseline      # 태그 지정 (비교용)
    python eval/run_eval.py --ids eval_01 eval_05  # 특정 항목만

결과:
    eval/results/<tag>_<timestamp>.json    — 상세 결과
    eval/results/<tag>_<timestamp>.md      — 요약 리포트
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── 판정 매칭 로직 ────────────────────────────────────────────────

_JUDGMENT_ALIASES = {
    "위반소지": {"위반소지", "위반"},
    "주의": {"주의"},
    "OK": {"OK", "적합", "문제없음"},
}


def _normalize_judgment(j: str) -> str:
    """판정 라벨을 정규화."""
    j = j.strip()
    for canonical, aliases in _JUDGMENT_ALIASES.items():
        if j in aliases:
            return canonical
    return j


def _judgment_match(actual: str, expected: str) -> bool:
    """판정이 일치하는지 (유연한 매칭)."""
    return _normalize_judgment(actual) == _normalize_judgment(expected)


def _judgment_partial_match(actual: str, expected: str) -> float:
    """부분 점수: 정확 일치=1.0, 인접 수준=0.5, 불일치=0.0."""
    na, ne = _normalize_judgment(actual), _normalize_judgment(expected)
    if na == ne:
        return 1.0
    # 위반소지↔주의는 방향은 맞으므로 부분 점수
    adjacent = {("위반소지", "주의"), ("주의", "위반소지")}
    if (na, ne) in adjacent:
        return 0.5
    return 0.0


# ── 위험유형 매칭 ──────────────────────────────────────────────────

def _risk_type_recall(actual_risk: str, expected_types: list[str]) -> float:
    """기대 위험유형 중 실제 검출된 비율."""
    if not expected_types or expected_types == ["문제없음"]:
        return 1.0  # 기대가 없으면 만점

    actual = actual_risk.lower() if actual_risk else ""
    hits = 0
    for et in expected_types:
        et_lower = et.lower()
        if et_lower in actual or et_lower.replace("/", "") in actual.replace("/", ""):
            hits += 1
    return hits / len(expected_types) if expected_types else 1.0


# ── 근거 품질 ──────────────────────────────────────────────────────

def _evidence_quality(
    reason: str,
    references: list[dict],
    expected_keywords: list[str],
) -> dict:
    """근거 키워드 포함 여부 + reference 수 평가."""
    if not expected_keywords:
        return {"keyword_recall": 1.0, "ref_count": len(references), "has_refs": bool(references)}

    # reason + references 전체 텍스트에서 키워드 검색
    full_text = (reason or "").lower()
    for ref in references:
        full_text += " " + (ref.get("content", "") or "").lower()
        full_text += " " + (ref.get("doc_type", "") or "").lower()

    hits = sum(1 for kw in expected_keywords if kw.lower() in full_text)
    return {
        "keyword_recall": hits / len(expected_keywords),
        "ref_count": len(references),
        "has_refs": bool(references),
        "matched_keywords": [kw for kw in expected_keywords if kw.lower() in full_text],
        "missed_keywords": [kw for kw in expected_keywords if kw.lower() not in full_text],
    }


# ── 단일 항목 평가 ────────────────────────────────────────────────

def evaluate_single(
    chain,
    test_case: dict,
) -> dict:
    """단일 골든 데이터 항목을 평가."""
    item_text = test_case["item_text"]
    start = time.time()

    try:
        result = chain.run(
            item_text=item_text,
            category=test_case.get("category", ""),
            broadcast_type=test_case.get("broadcast_type", ""),
        )
        latency_sec = round(time.time() - start, 2)

        judgment = result.get("judgment", "N/A")
        reason = result.get("reason", "")
        risk_type = result.get("risk_type", "")
        references = result.get("references", [])
        tool_logs = result.get("tool_logs", [])

        # 지표 계산
        j_match = _judgment_match(judgment, test_case["expected_judgment"])
        j_partial = _judgment_partial_match(judgment, test_case["expected_judgment"])
        rt_recall = _risk_type_recall(risk_type, test_case.get("expected_risk_types", []))
        ev_quality = _evidence_quality(
            reason, references, test_case.get("expected_evidence_keywords", [])
        )

        return {
            "id": test_case["id"],
            "item_text": item_text,
            "expected_judgment": test_case["expected_judgment"],
            "actual_judgment": judgment,
            "judgment_correct": j_match,
            "judgment_partial_score": j_partial,
            "expected_risk_types": test_case.get("expected_risk_types", []),
            "actual_risk_type": risk_type,
            "risk_type_recall": rt_recall,
            "evidence": ev_quality,
            "reason": reason[:300],
            "ref_count": len(references),
            "latency_sec": latency_sec,
            "tool_logs_summary": _summarize_tool_logs(tool_logs),
            "error": None,
        }

    except Exception as e:
        latency_sec = round(time.time() - start, 2)
        logger.error("평가 실패 [%s]: %s", test_case["id"], e)
        return {
            "id": test_case["id"],
            "item_text": item_text,
            "expected_judgment": test_case["expected_judgment"],
            "actual_judgment": "ERROR",
            "judgment_correct": False,
            "judgment_partial_score": 0.0,
            "risk_type_recall": 0.0,
            "evidence": {"keyword_recall": 0.0, "ref_count": 0, "has_refs": False},
            "latency_sec": latency_sec,
            "error": str(e),
        }


def _summarize_tool_logs(tool_logs: list[dict]) -> dict:
    """tool_logs에서 핵심 요약만 추출."""
    summary = {}
    for log in tool_logs:
        step = log.get("step", "unknown")
        if step == "orchestrator":
            summary["risk_types"] = log.get("risk_types", [])
        elif step in ("case_retrieve", "policy_retrieve"):
            summary.setdefault("retrieval_queries", []).append(log.get("query", ""))
            summary.setdefault("retrieval_counts", []).append(log.get("total", 0))
        elif step in ("case_grade", "policy_grade"):
            summary.setdefault("grade_relevant", []).append(log.get("relevant", 0))
        elif step == "grade_answer":
            summary["answer_grade"] = log.get("grade", "N/A")
    return summary


# ── 전체 평가 실행 ─────────────────────────────────────────────────

def run_evaluation(
    tag: str = "default",
    filter_ids: list[str] | None = None,
) -> dict:
    """골든 데이터셋 전체를 평가하고 결과를 저장."""
    from chains.review_chain import ReviewChain

    # 골든 데이터셋 로드
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if filter_ids:
        dataset = [d for d in dataset if d["id"] in filter_ids]

    print("=" * 70)
    print(f"방송 심의 AI Evaluation Pipeline")
    print(f"태그: {tag} | 데이터셋: {len(dataset)}건")
    print("=" * 70)

    chain = ReviewChain(model_name="gpt-4o-mini")
    results: list[dict] = []

    for i, test_case in enumerate(dataset, 1):
        print(f"\n[{i}/{len(dataset)}] {test_case['id']}: {test_case['item_text'][:40]}...")
        result = evaluate_single(chain, test_case)
        results.append(result)

        # 즉시 출력
        status = "✅" if result["judgment_correct"] else ("⚠️" if result["judgment_partial_score"] > 0 else "❌")
        print(f"  {status} 판정: {result['actual_judgment']} (기대: {result['expected_judgment']}) | {result['latency_sec']}s")
        if result.get("error"):
            print(f"  ⛔ 오류: {result['error']}")

    # ── 집계 ──────────────────────────────────────────────────
    summary = _compute_summary(results)
    summary["tag"] = tag
    summary["timestamp"] = datetime.now().isoformat()
    summary["dataset_size"] = len(dataset)

    # ── 저장 ──────────────────────────────────────────────────
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{tag}_{ts}"

    # JSON 상세 결과
    json_path = results_dir / f"{filename_base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    # Markdown 요약 리포트
    md_path = results_dir / f"{filename_base}.md"
    md_content = _generate_report(summary, results, tag)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # ── 콘솔 출력 ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("평가 결과 요약")
    print("=" * 70)
    print(f"판정 정확도 (Exact)  : {summary['judgment_accuracy']:.1%}  ({summary['judgment_correct']}/{summary['dataset_size']})")
    print(f"판정 정확도 (Partial): {summary['judgment_partial_score']:.1%}")
    print(f"위험유형 재현율      : {summary['risk_type_recall']:.1%}")
    print(f"근거 키워드 재현율   : {summary['evidence_keyword_recall']:.1%}")
    print(f"근거 보유율          : {summary['evidence_has_refs_rate']:.1%}")
    print(f"평균 응답 시간       : {summary['avg_latency_sec']:.1f}s")
    print(f"오류 건수            : {summary['error_count']}")
    print(f"\n결과 저장: {json_path}")
    print(f"리포트:   {md_path}")

    return {"summary": summary, "details": results}


def _compute_summary(results: list[dict]) -> dict:
    """평가 결과 집계."""
    n = len(results)
    if n == 0:
        return {}

    correct = sum(1 for r in results if r["judgment_correct"])
    partial_sum = sum(r["judgment_partial_score"] for r in results)
    rt_recall_sum = sum(r.get("risk_type_recall", 0) for r in results)
    kw_recall_sum = sum(r.get("evidence", {}).get("keyword_recall", 0) for r in results)
    has_refs = sum(1 for r in results if r.get("evidence", {}).get("has_refs", False))
    latency_sum = sum(r["latency_sec"] for r in results)
    errors = sum(1 for r in results if r.get("error"))

    return {
        "judgment_accuracy": correct / n,
        "judgment_correct": correct,
        "judgment_partial_score": partial_sum / n,
        "risk_type_recall": rt_recall_sum / n,
        "evidence_keyword_recall": kw_recall_sum / n,
        "evidence_has_refs_rate": has_refs / n,
        "avg_latency_sec": round(latency_sum / n, 1),
        "total_latency_sec": round(latency_sum, 1),
        "error_count": errors,
    }


def _generate_report(summary: dict, results: list[dict], tag: str) -> str:
    """Markdown 요약 리포트 생성."""
    lines = [
        f"# 방송 심의 AI 평가 리포트",
        f"",
        f"- **태그**: `{tag}`",
        f"- **일시**: {summary.get('timestamp', 'N/A')}",
        f"- **데이터셋**: {summary.get('dataset_size', 0)}건",
        f"",
        f"## 종합 지표",
        f"",
        f"| 지표 | 값 |",
        f"|------|-----|",
        f"| 판정 정확도 (Exact) | {summary['judgment_accuracy']:.1%} ({summary['judgment_correct']}/{summary['dataset_size']}) |",
        f"| 판정 정확도 (Partial) | {summary['judgment_partial_score']:.1%} |",
        f"| 위험유형 재현율 | {summary['risk_type_recall']:.1%} |",
        f"| 근거 키워드 재현율 | {summary['evidence_keyword_recall']:.1%} |",
        f"| 근거 보유율 | {summary['evidence_has_refs_rate']:.1%} |",
        f"| 평균 응답 시간 | {summary['avg_latency_sec']:.1f}s |",
        f"| 오류 건수 | {summary['error_count']} |",
        f"",
        f"## 항목별 결과",
        f"",
        f"| ID | 문구 | 기대 | 실제 | 정확 | 근거 | 시간 |",
        f"|-----|------|------|------|------|------|------|",
    ]

    for r in results:
        status = "✅" if r["judgment_correct"] else ("⚠️" if r["judgment_partial_score"] > 0 else "❌")
        text = r["item_text"][:25] + "..."
        ev_kw = f"{r.get('evidence', {}).get('keyword_recall', 0):.0%}"
        lines.append(
            f"| {r['id']} | {text} | {r['expected_judgment']} | {r['actual_judgment']} | {status} | {ev_kw} | {r['latency_sec']}s |"
        )

    # 오답 분석
    wrong = [r for r in results if not r["judgment_correct"]]
    if wrong:
        lines.extend([
            f"",
            f"## 오답 분석",
            f"",
        ])
        for r in wrong:
            lines.extend([
                f"### {r['id']}: {r['item_text'][:40]}",
                f"- 기대: **{r['expected_judgment']}** → 실제: **{r['actual_judgment']}**",
                f"- 근거: {r.get('reason', 'N/A')[:150]}",
                f"- 누락 키워드: {r.get('evidence', {}).get('missed_keywords', [])}",
                f"",
            ])

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="방송 심의 AI 평가 파이프라인")
    parser.add_argument("--tag", default="default", help="실행 태그 (비교용, 예: baseline, hybrid-search)")
    parser.add_argument("--ids", nargs="*", help="평가할 항목 ID (미지정 시 전체)")
    args = parser.parse_args()

    run_evaluation(tag=args.tag, filter_ids=args.ids)


if __name__ == "__main__":
    main()
