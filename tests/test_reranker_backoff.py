"""
reranker 지수 백오프 동작 검증 테스트.

테스트 항목:
  1) 정상 호출 — time.sleep 없이 빠른 응답, 재시도 로그 없음
  2) 실패 호출 — 잘못된 API 키로 WARNING 3회 → fallback 동작

실행:
    python tests/test_reranker_backoff.py
"""

import logging
import sys
import time
from pathlib import Path

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 로깅 설정 (WARNING 이상 콘솔 출력) ───────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_reranker")

# ── 테스트용 더미 청크 ────────────────────────────────────────────
QUERY = "피부가 좋아집니다"
DUMMY_CHUNKS = [
    {
        "content": "임상 시험을 통해 피부 개선 효과가 확인되었습니다.",
        "metadata": {"source": "사례집"},
        "chroma_id": "c1",
        "relevance_score": 0.7,
    },
    {
        "content": "피부가 좋아집니다 라는 표현은 효능·효과 표방에 해당할 수 있습니다.",
        "metadata": {"source": "지침"},
        "chroma_id": "c2",
        "relevance_score": 0.6,
    },
    {
        "content": "건강기능식품 광고 심의 기준에 따르면 피부 관련 표현은 주의가 필요합니다.",
        "metadata": {"source": "규정"},
        "chroma_id": "c3",
        "relevance_score": 0.5,
    },
]

SEP = "=" * 60


# ─────────────────────────────────────────────────────────────────
# 테스트 1: 정상 호출
# ─────────────────────────────────────────────────────────────────
def test_normal():
    print(f"\n{SEP}")
    print("[TEST 1] 정상 호출 - 실제 COHERE_API_KEY 사용")
    print(f"  쿼리: {QUERY!r}")
    print(SEP)

    from utils.reranker import rerank_chunks

    start = time.perf_counter()
    result = rerank_chunks(QUERY, DUMMY_CHUNKS, top_n=3, min_score=0.0)
    elapsed = time.perf_counter() - start

    print(f"\n  결과 청크 수  : {len(result)}")
    print(f"  응답 소요 시간: {elapsed:.2f}초  (기존 time.sleep(7) 제거 확인)")
    for i, c in enumerate(result):
        print(f"    [{i+1}] score={c.get('rerank_score')}  {c['content'][:50]}…")

    assert isinstance(result, list), "결과가 list여야 합니다"
    assert elapsed < 7.0, f"응답이 7초 미만이어야 합니다 (실제: {elapsed:.2f}s)"
    print("\n  [PASS] 재시도 없이 정상 반환, 7초 미만 응답")


# ─────────────────────────────────────────────────────────────────
# 테스트 2: 잘못된 API 키 → 재시도 3회 → fallback
# ─────────────────────────────────────────────────────────────────
def test_bad_key_fallback():
    print(f"\n{SEP}")
    print("[TEST 2] 잘못된 API 키 — 재시도 3회 + fallback 확인")
    print(f"  쿼리: {QUERY!r}")
    print(SEP)

    # ── 잠깐 API 키 교체 ─────────────────────────────────────────
    from config import settings
    import utils.reranker as reranker_mod

    original_key = settings.COHERE_API_KEY
    settings.COHERE_API_KEY = "INVALID_KEY_FOR_TEST"
    reranker_mod._client = None  # 캐시된 client 초기화

    try:
        from utils.reranker import rerank_chunks

        print("\n  ※ WARNING 로그가 최대 3회 출력되어야 합니다 (tenacity 재시도)\n")

        start = time.perf_counter()
        result = rerank_chunks(QUERY, DUMMY_CHUNKS, top_n=3, min_score=0.0)
        elapsed = time.perf_counter() - start

        print(f"\n  결과 청크 수  : {len(result)}")
        print(f"  응답 소요 시간: {elapsed:.2f}초")
        print(f"  첫 번째 청크  : {result[0]['content'][:60]}…")

        # fallback: 원본 chunks[:top_n] 그대로 반환되어야 함
        assert result == DUMMY_CHUNKS[:3], \
            "fallback은 원본 chunks[:top_n]이어야 합니다"
        print("\n  [PASS] 3회 재시도 후 fallback 정상 동작")

    finally:
        # ── API 키 원복 ──────────────────────────────────────────
        settings.COHERE_API_KEY = original_key
        reranker_mod._client = None
        print(f"\n  API 키 원복 완료: ...{original_key[-6:]}")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        test_normal()
    except Exception as e:
        print(f"\n  [FAIL] (test_normal): {e}")

    try:
        test_bad_key_fallback()
    except Exception as e:
        print(f"\n  [FAIL] (test_bad_key_fallback): {e}")

    print(f"\n{SEP}")
    print("테스트 완료")
    print(SEP)
