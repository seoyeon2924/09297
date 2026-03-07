"""
grade_documents 수정 전/후 성능 비교 테스트.

LangSmith에서 트레이스를 비교하려면 LANGCHAIN_PROJECT 환경변수를 달리 설정 후 실행:
  # 수정 전 (score_threshold 방식)
  set LANGCHAIN_PROJECT=broadcast-review-BEFORE && python tests/test_grade_comparison.py

  # 수정 후 (llm_batch 방식)
  set LANGCHAIN_PROJECT=broadcast-review-AFTER && python tests/test_grade_comparison.py
"""
import json
import os
import sys
import time

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_PHRASES = [
    {
        "item_text": "다신 오지 않는 최저가 혜택",
        "category": "건강기능식품",
        "broadcast_type": "생방송",
        "expected": "위반소지",
    },
    {
        "item_text": "어린이 면역력 증진에 도움이 됨",
        "category": "건강기능식품",
        "broadcast_type": "생방송",
        "expected": "위반소지",
    },
    {
        "item_text": "감기 걸리지 않는 아이로 만들어 드립니다",
        "category": "건강기능식품",
        "broadcast_type": "생방송",
        "expected": "위반소지",
    },
    {
        "item_text": "본 제품은 식약처 인증 건강기능식품입니다",
        "category": "건강기능식품",
        "broadcast_type": "생방송",
        "expected": "적합",
    },
    {
        "item_text": "다이어트 효과까지 !!",
        "category": "건강기능식품",
        "broadcast_type": "생방송",
        "expected": "위반소지",
    },
]

_VIOLATION_LABELS = {"위반소지", "위반"}


def _is_correct(judgment: str, expected: str) -> bool:
    if expected in _VIOLATION_LABELS:
        return judgment in _VIOLATION_LABELS
    return judgment == expected


def run_comparison() -> None:
    from chains.review_chain import ReviewChain

    chain = ReviewChain()
    results = []

    langsmith_project = os.environ.get("LANGCHAIN_PROJECT", "(기본)")
    print("=" * 70)
    print("grade_documents 성능 비교 테스트")
    print(f"LangSmith 프로젝트: {langsmith_project}")
    print("=" * 70)

    for i, test in enumerate(TEST_PHRASES, 1):
        print(f"\n--- 테스트 {i}/{len(TEST_PHRASES)} ---")
        print(f"문구: {test['item_text']}")
        print(f"예상: {test['expected']}")

        start = time.time()
        try:
            result = chain.run(
                item_text=test["item_text"],
                category=test["category"],
                broadcast_type=test["broadcast_type"],
            )
            elapsed = round(time.time() - start, 2)

            judgment = result.get("judgment", "N/A")
            tool_logs = result.get("tool_logs", [])

            grade_log = next(
                (lg for lg in tool_logs if lg.get("step") == "grade_documents"), {}
            )
            grade_elapsed = grade_log.get("elapsed", "N/A")
            grade_total = grade_log.get("total_evaluated", "N/A")
            grade_relevant = grade_log.get("relevant_count", "N/A")
            grade_method = grade_log.get("method", "N/A")
            grade_error = grade_log.get("error", None)

            correct = _is_correct(judgment, test["expected"])
            print(f"판정: {judgment}  ({'O' if correct else 'X'})")
            print(f"총 소요: {elapsed}s")
            print(
                f"grade_documents: {grade_elapsed}s | 방식: {grade_method} | "
                f"평가: {grade_total}개 → 관련: {grade_relevant}개"
            )
            if grade_error:
                print(f"  경고 (fallback 동작): {grade_error}")

            results.append(
                {
                    "item_text": test["item_text"],
                    "expected": test["expected"],
                    "judgment": judgment,
                    "correct": correct,
                    "total_elapsed": elapsed,
                    "grade_elapsed": grade_elapsed,
                    "grade_total": grade_total,
                    "grade_relevant": grade_relevant,
                    "grade_method": grade_method,
                    "grade_error": grade_error,
                }
            )

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            print(f"ERROR: {e}")
            results.append(
                {
                    "item_text": test["item_text"],
                    "expected": test["expected"],
                    "judgment": "ERROR",
                    "correct": False,
                    "total_elapsed": elapsed,
                    "grade_error": str(e),
                }
            )

    # ── 요약 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    correct_count = sum(1 for r in results if r.get("correct"))
    total_count = len(results)
    avg_total = round(sum(r.get("total_elapsed", 0) for r in results) / total_count, 2)

    grade_times = [
        r["grade_elapsed"]
        for r in results
        if isinstance(r.get("grade_elapsed"), (int, float))
    ]
    avg_grade = round(sum(grade_times) / len(grade_times), 2) if grade_times else "N/A"

    error_count = sum(1 for r in results if r.get("grade_error"))

    print(f"정확도           : {correct_count}/{total_count} ({round(correct_count / total_count * 100)}%)")
    print(f"평균 총 소요시간 : {avg_total}s")
    print(f"평균 grade 소요  : {avg_grade}s")
    print(f"grade fallback   : {error_count}건")

    # ── JSON 저장 ────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "grade_comparison_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "langsmith_project": langsmith_project,
                "summary": {
                    "accuracy": f"{correct_count}/{total_count}",
                    "avg_total_time": avg_total,
                    "avg_grade_time": avg_grade,
                    "grade_errors": error_count,
                },
                "details": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n결과 저장: {out_path}")
    print("LangSmith에서 트레이스를 확인하세요!")


if __name__ == "__main__":
    run_comparison()
