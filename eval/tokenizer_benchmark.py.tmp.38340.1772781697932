"""
Tokenizer Benchmark — regex 폴백 vs kiwipiepy 형태소 분석 비교.

두 가지를 측정한다:

  [1] 정성 비교 (Qualitative)
      같은 텍스트에 대해 두 토크나이저가 생성하는 토큰 목록을 나란히 출력.
      노이즈 토큰(조사·어미 등) 제거율을 보여준다.

  [2] 정량 비교 (Quantitative) — BM25 키워드 재현율
      golden_dataset의 expected_evidence_keywords를 기준으로,
      각 쿼리에 대해 BM25 상위 10개 청크의 텍스트에
      얼마나 많은 기대 키워드가 등장하는지 측정.

      지표: Keyword Hit Rate = 기대 키워드 중 BM25 top-10 결과에서 발견된 비율

사용법:
    # ChromaDB에 문서가 인제스트된 상태에서 실행
    python eval/tokenizer_benchmark.py

    # 특정 컬렉션만
    python eval/tokenizer_benchmark.py --collection cases

    # 토크나이즈 비교만 (Chroma 불필요)
    python eval/tokenizer_benchmark.py --tokenize-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rank_bm25 import BM25Okapi

# ── 토크나이저 정의 ──────────────────────────────────────────────────

_KO_SPLIT_RE = re.compile(r"[^\w가-힣]+")
_MIN_TOKEN_LEN = 2

_KEEP_POS = {
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "XR", "SL", "SN", "MM",
}

# Kiwi 싱글톤 — 최초 1회만 초기화 (초기화 비용이 크므로 캐싱)
_kiwi_instance = None


def _get_kiwi():
    global _kiwi_instance
    if _kiwi_instance is None:
        try:
            from kiwipiepy import Kiwi
            print("  Kiwi 형태소 분석기 초기화 중...", end=" ", flush=True)
            _kiwi_instance = Kiwi()
            print("완료")
        except Exception as e:
            print(f"\n  ⚠️  kiwipiepy 초기화 실패 ({e}), regex 폴백 사용")
            _kiwi_instance = False  # 실패 표시
    return _kiwi_instance


def tokenize_regex(text: str) -> list[str]:
    """기존 간이 공백/특수문자 분리 토크나이저 (변경 前)."""
    tokens = _KO_SPLIT_RE.split(text.lower())
    return [t for t in tokens if len(t) >= _MIN_TOKEN_LEN]


def tokenize_kiwi(text: str) -> list[str]:
    """kiwipiepy 형태소 분석 기반 토크나이저 (변경 後).

    Kiwi 싱글톤을 재사용 — 대용량 배치 처리에 최적화.
    실패 시 tokenize_regex로 폴백.
    """
    kiwi = _get_kiwi()
    if not kiwi:
        return tokenize_regex(text)
    try:
        tokens = [
            tok.form.lower()
            for tok in kiwi.tokenize(text, normalize_coda=True)
            if tok.tag in _KEEP_POS and len(tok.form) >= _MIN_TOKEN_LEN
        ]
        return tokens if tokens else tokenize_regex(text)
    except Exception:
        return tokenize_regex(text)


# ── 정성 비교 ────────────────────────────────────────────────────────

_SAMPLE_TEXTS = [
    "다신 오지 않는 최저가 혜택! 오늘 방송에서만!",
    "감기 예방에 탁월한 효과! 면역력을 확실히 높여드립니다",
    "타사 제품보다 3배 효과! 경쟁사는 따라올 수 없습니다",
    "식약처 인증을 받은 안전한 제품입니다",
    "이 크림을 바르면 주름이 확실히 사라집니다",
    "지금 바로 구매하세요 - 오늘이 마지막입니다!",
    "콜라겐이 풍부하게 들어있어 피부에 좋습니다",
    "근거불확실한 완벽 프리미엄 여행 패키지",
]


def run_qualitative(texts: list[str] | None = None) -> None:
    """두 토크나이저 결과를 나란히 출력."""
    samples = texts or _SAMPLE_TEXTS

    print("\n" + "=" * 80)
    print("  [1] 정성 비교 - 토크나이즈 결과 비교")
    print("=" * 80)

    kiwi_available = bool(_get_kiwi())
    if not kiwi_available:
        print("  ⚠️  kiwipiepy 미설치 → 폴백 결과만 표시됩니다.")
        print("  설치: pip install kiwipiepy")

    total_regex_tokens = 0
    total_kiwi_tokens = 0
    noise_removed_total = 0

    for text in samples:
        r_tokens = tokenize_regex(text)
        k_tokens = tokenize_kiwi(text)

        # regex에는 있고 kiwi에는 없는 토큰 = 노이즈로 제거된 것
        removed = set(r_tokens) - set(k_tokens)
        added   = set(k_tokens) - set(r_tokens)

        total_regex_tokens += len(r_tokens)
        total_kiwi_tokens  += len(k_tokens)
        noise_removed_total += len(removed)

        print(f"\n  원문 : {text}")
        print(f"  regex: {r_tokens}")
        print(f"  kiwi : {k_tokens}")
        if removed:
            print(f"  [-]제거됨(노이즈): {sorted(removed)}")
        if added:
            print(f"  [+]추가됨(정규화): {sorted(added)}")

    print("\n" + "-" * 60)
    print(f"  총 토큰 수   regex={total_regex_tokens}  kiwi={total_kiwi_tokens}")
    if total_regex_tokens > 0:
        noise_pct = noise_removed_total / total_regex_tokens * 100
        print(f"  노이즈 제거율: {noise_removed_total}/{total_regex_tokens} = {noise_pct:.1f}%")
        print("  (조사·어미 등 불필요 토큰이 제거될수록 BM25 매칭 정밀도 향상)")


# ── 정량 비교 ────────────────────────────────────────────────────────

def _build_bm25(documents: list[str], tokenizer: Callable) -> BM25Okapi:
    tokenized = [tokenizer(doc) for doc in documents]
    return BM25Okapi(tokenized)


def _keyword_hit_rate(
    bm25: BM25Okapi,
    all_docs: list[str],
    query: str,
    expected_keywords: list[str],
    tokenizer: Callable,
    top_n: int = 10,
) -> float:
    """BM25 top_n 결과 텍스트에서 기대 키워드 재현율을 계산."""
    if not expected_keywords:
        return 1.0  # 기대 키워드가 없으면 만점

    q_tokens = tokenizer(query)
    if not q_tokens:
        return 0.0

    scores = bm25.get_scores(q_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    top_texts = " ".join(all_docs[i] for i in top_indices).lower()

    hits = sum(1 for kw in expected_keywords if kw.lower() in top_texts)
    return hits / len(expected_keywords)


def run_quantitative(collection_key: str = "cases", max_docs: int = 500) -> None:
    """golden_dataset 쿼리로 BM25 키워드 재현율을 두 토크나이저 간 비교.

    max_docs: BM25 인덱스에 사용할 문서 수 상한 (kiwipiepy 분석 시간 제한)
    """
    print("\n" + "=" * 80)
    print(f"  [2] 정량 비교 - BM25 키워드 재현율 (컬렉션: {collection_key}, 최대 {max_docs}건)")
    print("=" * 80)

    # 골든 데이터셋 로드
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 기대 키워드 없는 항목 필터링
    testable = [d for d in dataset if d.get("expected_evidence_keywords")]
    if not testable:
        print("  ⚠️  expected_evidence_keywords가 있는 항목이 없습니다.")
        return

    # ChromaDB에서 전체 문서 로드
    print(f"\n  ChromaDB '{collection_key}' 컬렉션 로드 중...")
    try:
        from storage.chroma_store import chroma_store
        coll = chroma_store.get_collection(collection_key)
        total = coll.count()
    except Exception as e:
        print(f"  ⛔ ChromaDB 접속 실패: {e}")
        print("  힌트: 먼저 문서를 인제스트하세요 (python scripts/ingest_all.py 등)")
        return

    if total == 0:
        print(f"  ⚠️  컬렉션 '{collection_key}'이 비어있습니다. 먼저 인제스트를 실행하세요.")
        return

    load_n = min(total, max_docs)
    print(f"  → {total}건 중 {load_n}건 로드 (--max-docs {max_docs})")

    all_docs: list[str] = []
    batch_size = 500
    offset = 0
    while len(all_docs) < load_n:
        fetch = min(batch_size, load_n - len(all_docs))
        batch = coll.get(limit=fetch, offset=offset, include=["documents"])
        batch_docs = batch.get("documents", []) or []
        all_docs.extend(batch_docs)
        if len(batch_docs) < fetch:
            break
        offset += fetch

    if not all_docs:
        print("  ⚠️  문서 텍스트를 가져올 수 없습니다.")
        return

    print(f"  → {len(all_docs)}건 로드 완료")

    # BM25 인덱스 2개 빌드
    print(f"\n  BM25 인덱스 빌드 중 ({len(all_docs)}건)...")
    print("    [1/2] regex 인덱스...")
    bm25_regex = _build_bm25(all_docs, tokenize_regex)
    print("    [2/2] kiwipiepy 인덱스 (형태소 분석, 잠시 소요)...")
    bm25_kiwi  = _build_bm25(all_docs, tokenize_kiwi)
    print("  → 완료")

    # 쿼리별 비교
    print(f"\n  {'ID':<10} {'쿼리':<35} {'regex':>8} {'kiwi':>8} {'변화':>8}")
    print("  " + "-" * 72)

    regex_scores: list[float] = []
    kiwi_scores:  list[float] = []

    for case in testable:
        query    = case["item_text"]
        keywords = case["expected_evidence_keywords"]

        r_score = _keyword_hit_rate(bm25_regex, all_docs, query, keywords, tokenize_regex)
        k_score = _keyword_hit_rate(bm25_kiwi,  all_docs, query, keywords, tokenize_kiwi)

        regex_scores.append(r_score)
        kiwi_scores.append(k_score)

        diff = k_score - r_score
        arrow = "✅" if diff > 0 else ("⚠️" if diff < 0 else "➡️")
        short_query = query[:33] + ".." if len(query) > 35 else query
        print(
            f"  {case['id']:<10} {short_query:<35} "
            f"{r_score:>7.1%}  {k_score:>7.1%}  {diff:>+7.1%} {arrow}"
        )

    # 집계
    avg_regex = sum(regex_scores) / len(regex_scores)
    avg_kiwi  = sum(kiwi_scores)  / len(kiwi_scores)
    avg_diff  = avg_kiwi - avg_regex
    improved  = sum(1 for r, k in zip(regex_scores, kiwi_scores) if k > r)
    regressed = sum(1 for r, k in zip(regex_scores, kiwi_scores) if k < r)
    same      = len(regex_scores) - improved - regressed

    print("\n  " + "=" * 72)
    print(f"  {'평균 키워드 재현율':<35} {avg_regex:>7.1%}  {avg_kiwi:>7.1%}  {avg_diff:>+7.1%}")
    print("  " + "=" * 72)
    print(f"\n  결과 요약:")
    print(f"    개선된 쿼리  : {improved}건")
    print(f"    악화된 쿼리  : {regressed}건")
    print(f"    변화 없음    : {same}건")
    print(f"    테스트 쿼리  : {len(testable)}건 (전체 {len(dataset)}건 중 키워드 있는 것)")

    if avg_diff > 0:
        print(f"\n  ✅ kiwipiepy 토크나이저가 BM25 키워드 재현율을 {avg_diff:+.1%} 향상시켰습니다.")
    elif avg_diff < 0:
        print(f"\n  ⚠️  kiwipiepy 토크나이저 사용 시 재현율이 {avg_diff:.1%} 하락했습니다.")
        print("     (원인: 형태소 분리 오류 또는 골든셋 키워드가 복합어인 경우)")
    else:
        print("\n  ➡️  두 토크나이저의 재현율이 동일합니다.")

    # 왜 좋아졌는지 설명
    print("\n  💡 키워드 재현율이 오르는 이유:")
    print("     regex  'ef과가'  → BM25가 '효과가' 라는 단어 전체를 토큰으로 봄")
    print("     kiwi   '효과'    → 형태소 '효과(NNG)'만 추출, 문서의 '효과' 와 정확히 매칭")
    print("     조사·어미 제거로 vocabulary mismatch 감소 → BM25 IDF 계산이 더 정확해짐")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BM25 토크나이저 벤치마크 — regex vs kiwipiepy",
    )
    parser.add_argument(
        "--collection",
        default="cases",
        choices=["cases", "regulations", "guidelines"],
        help="비교에 사용할 ChromaDB 컬렉션 (기본: cases)",
    )
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help="정성 비교만 실행 (ChromaDB 불필요)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=500,
        help="BM25 인덱스에 사용할 최대 문서 수 (기본: 500, kiwipiepy 속도 제한)",
    )
    args = parser.parse_args()

    run_qualitative()

    if not args.tokenize_only:
        run_quantitative(collection_key=args.collection, max_docs=args.max_docs)

    print("\n")


if __name__ == "__main__":
    main()
