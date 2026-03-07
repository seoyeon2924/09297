"""
Compare — 두 평가 결과를 비교하여 개선 효과를 시각화.

사용법:
    python eval/compare.py eval/results/baseline_20260304.json eval/results/hybrid_20260304.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def compare(path_a: str, path_b: str) -> None:
    with open(path_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    sa = data_a["summary"]
    sb = data_b["summary"]
    tag_a = sa.get("tag", "A")
    tag_b = sb.get("tag", "B")

    print("=" * 70)
    print(f"평가 비교: [{tag_a}] vs [{tag_b}]")
    print("=" * 70)

    metrics = [
        ("판정 정확도 (Exact)", "judgment_accuracy", True),
        ("판정 정확도 (Partial)", "judgment_partial_score", True),
        ("위험유형 재현율", "risk_type_recall", True),
        ("근거 키워드 재현율", "evidence_keyword_recall", True),
        ("근거 보유율", "evidence_has_refs_rate", True),
        ("평균 응답 시간", "avg_latency_sec", False),
    ]

    print(f"\n{'지표':<25s}  {tag_a:>10s}  {tag_b:>10s}  {'변화':>10s}")
    print("-" * 60)

    for label, key, higher_is_better in metrics:
        va = sa.get(key, 0)
        vb = sb.get(key, 0)
        diff = vb - va

        if key == "avg_latency_sec":
            va_str = f"{va:.1f}s"
            vb_str = f"{vb:.1f}s"
            diff_str = f"{diff:+.1f}s"
            arrow = "⬇️" if diff < 0 else ("⬆️" if diff > 0 else "➡️")
        else:
            va_str = f"{va:.1%}"
            vb_str = f"{vb:.1%}"
            diff_str = f"{diff:+.1%}"
            if higher_is_better:
                arrow = "✅" if diff > 0 else ("⚠️" if diff < 0 else "➡️")
            else:
                arrow = "✅" if diff < 0 else ("⚠️" if diff > 0 else "➡️")

        print(f"{label:<25s}  {va_str:>10s}  {vb_str:>10s}  {diff_str:>8s} {arrow}")

    # 항목별 변화
    details_a = {d["id"]: d for d in data_a.get("details", [])}
    details_b = {d["id"]: d for d in data_b.get("details", [])}

    improved = []
    regressed = []
    for eid in details_a:
        if eid not in details_b:
            continue
        da = details_a[eid]
        db = details_b[eid]
        if not da["judgment_correct"] and db["judgment_correct"]:
            improved.append(eid)
        elif da["judgment_correct"] and not db["judgment_correct"]:
            regressed.append(eid)

    if improved:
        print(f"\n✅ 개선된 항목 ({len(improved)}건): {', '.join(improved)}")
    if regressed:
        print(f"⚠️ 악화된 항목 ({len(regressed)}건): {', '.join(regressed)}")
    if not improved and not regressed:
        print(f"\n판정 변화 없음")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python eval/compare.py <결과A.json> <결과B.json>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
