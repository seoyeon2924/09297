"""
BM25/Vector 가중치 차등 적용 전후 검색 품질 비교.

비교 대상:
  A) 동일 가중치  — vector=1.0, bm25=1.0
  B) 차등 가중치  — vector=0.7, bm25=1.0  (cases 최적값)

평가 방법:
  - golden_dataset.json 에서 expected_evidence_keywords 가 있는 12건 사용
  - cases 컬렉션 대상 검색 (BM25 인덱스 max 500건 제한)
  - 상위 5건의 content 에 키워드가 몇 개 포함되는지 (Hit Rate)
  - 상위 5건 RRF 점수 평균 (Avg RRF)

실행:
    python eval/weight_comparison.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COLLECTION = "cases"
TOP_N = 5
MAX_DOCS = 500
EQUAL_WEIGHTS = [1.0, 1.0]
DIFF_WEIGHTS   = [0.7, 1.0]   # vector=0.7, bm25=1.0  (cases 최적값)


# ── 데이터셋 로드 (키워드 있는 12건 필터) ─────────────────────────────

def load_dataset() -> list[dict]:
    path = Path(__file__).parent / "golden_dataset.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if d.get("expected_evidence_keywords")]


# ── Hit Rate 계산 ─────────────────────────────────────────────────────

def hit_rate(chunks: list[dict], keywords: list[str]) -> float:
    """상위 chunks의 content 전체에서 keywords가 포함된 비율."""
    if not keywords:
        return 1.0
    full_text = " ".join(c.get("content", "") for c in chunks).lower()
    hits = sum(1 for kw in keywords if kw.lower() in full_text)
    return hits / len(keywords)


def avg_rrf(chunks: list[dict]) -> float:
    """상위 N건 RRF 점수 평균."""
    scores = [c.get("rrf_score", 0.0) for c in chunks]
    return sum(scores) / len(scores) if scores else 0.0


# ── 단일 쿼리 검색 (두 가지 가중치 동시 실행) ────────────────────────

def compare_query(
    query: str,
    bm25_index,           # BM25Index
    query_embedding: list[float],
    vector_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    같은 vector_results + bm25_results 를 두 가지 가중치로 RRF 병합.

    Returns:
        (equal_top5, diff_top5)
    """
    from utils.hybrid_search import reciprocal_rank_fusion

    bm25_results = bm25_index.search(query, top_n=20)

    equal_merged = reciprocal_rank_fusion(
        vector_results, bm25_results, k=60, weights=EQUAL_WEIGHTS
    )
    diff_merged = reciprocal_rank_fusion(
        vector_results, bm25_results, k=60, weights=DIFF_WEIGHTS
    )

    return equal_merged[:TOP_N], diff_merged[:TOP_N]


# ── 테이블 출력 ───────────────────────────────────────────────────────

def _trunc(s: str, n: int) -> str:
    return s[:n] + "…" if len(s) > n else s


def print_table(rows: list[dict]) -> None:
    """비교 결과를 콘솔 테이블로 출력."""
    COL_Q = 28
    COL_N = 5
    COL_R = 10

    diff_label = f"[차등 V{DIFF_WEIGHTS[0]}:B{DIFF_WEIGHTS[1]}]"
    header = (
        f"{'쿼리':<{COL_Q}} | "
        f"{'[동일 1:1]':>{COL_N+COL_R+3}} | "
        f"{diff_label:>{COL_N+COL_R+3}} | "
        f"{'승자':<6}"
    )
    sub = (
        f"{'':{COL_Q}} | "
        f"{'Hit':>{COL_N}} {'AvgRRF':>{COL_R}} | "
        f"{'Hit':>{COL_N}} {'AvgRRF':>{COL_R}} | "
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print(f"  {COLLECTION} 컬렉션 가중치 비교 (max_docs={MAX_DOCS}, top_n={TOP_N})")
    print("=" * len(header))
    print(header)
    print(sub)
    print(sep)

    eq_hits, diff_hits = [], []
    eq_rffs, diff_rffs = [], []

    for row in rows:
        q = _trunc(row["query"], COL_Q)
        eh = f"{row['equal_hit']:.2f}"
        er = f"{row['equal_rrf']:.5f}"
        dh = f"{row['diff_hit']:.2f}"
        dr = f"{row['diff_rrf']:.5f}"

        # 승자 판단 (hit_rate 우선, 같으면 rrf 비교)
        if row["diff_hit"] > row["equal_hit"]:
            winner = "차등"
        elif row["equal_hit"] > row["diff_hit"]:
            winner = "동일"
        elif row["diff_rrf"] > row["equal_rrf"]:
            winner = "차등"
        elif row["equal_rrf"] > row["diff_rrf"]:
            winner = "동일"
        else:
            winner = "동일"

        print(
            f"{q:<{COL_Q}} | "
            f"{eh:>{COL_N}} {er:>{COL_R}} | "
            f"{dh:>{COL_N}} {dr:>{COL_R}} | "
            f"{winner}"
        )

        eq_hits.append(row["equal_hit"])
        diff_hits.append(row["diff_hit"])
        eq_rffs.append(row["equal_rrf"])
        diff_rffs.append(row["diff_rrf"])

    # 평균 행
    n = len(rows)
    avg_eq_h  = sum(eq_hits) / n
    avg_diff_h = sum(diff_hits) / n
    avg_eq_r  = sum(eq_rffs) / n
    avg_diff_r = sum(diff_rffs) / n

    if avg_diff_h > avg_eq_h or (avg_diff_h == avg_eq_h and avg_diff_r > avg_eq_r):
        avg_winner = "차등"
    else:
        avg_winner = "동일"

    print(sep)
    print(
        f"{'[평균]':<{COL_Q}} | "
        f"{avg_eq_h:>{COL_N}.2f} {avg_eq_r:>{COL_R}.5f} | "
        f"{avg_diff_h:>{COL_N}.2f} {avg_diff_r:>{COL_R}.5f} | "
        f"{avg_winner}"
    )
    print("=" * len(header))
    print()

    # 개선 요약
    hit_delta = avg_diff_h - avg_eq_h
    rrf_delta = avg_diff_r - avg_eq_r
    print(f"  Hit Rate 변화 : {hit_delta:+.4f}  "
          f"({'차등 우세' if hit_delta > 0 else '동일 우세' if hit_delta < 0 else '동일'})")
    print(f"  Avg RRF 변화  : {rrf_delta:+.6f}  "
          f"({'차등 우세' if rrf_delta > 0 else '동일 우세' if rrf_delta < 0 else '동일'})")
    print()


# ── 메인 ─────────────────────────────────────────────────────────────

def main() -> None:
    from storage.chroma_store import chroma_store
    from providers.embed_openai import OpenAIEmbedProvider
    from utils.hybrid_search import BM25Index, _parse_chroma_result

    dataset = load_dataset()
    print(f"\n데이터셋 로드: {len(dataset)}건 (키워드 있는 항목만)")

    # ── BM25 인덱스 구축 (max 500건) ──────────────────────────────
    print(f"BM25 인덱스 구축 중 ({COLLECTION}, max={MAX_DOCS})...", end=" ", flush=True)
    try:
        coll = chroma_store.get_collection(COLLECTION)
        total = coll.count()
        limit = min(total, MAX_DOCS)
        if limit == 0:
            print(f"\n컬렉션 '{COLLECTION}'이 비어 있습니다. 데이터를 먼저 인덱싱하세요.")
            return

        batch = coll.get(limit=limit, include=["documents", "metadatas"])
        bm25_index = BM25Index()
        bm25_index.build(
            batch["ids"],
            batch["documents"],
            batch["metadatas"],
        )
        print(f"완료 ({bm25_index.doc_count}건 / 전체 {total}건)")
    except Exception as e:
        print(f"\nBM25 인덱스 구축 실패: {e}")
        return

    # ── 임베딩 프로바이더 ─────────────────────────────────────────
    embed_provider = OpenAIEmbedProvider()

    # ── 쿼리별 검색 + 비교 ───────────────────────────────────────
    rows: list[dict] = []
    queries = [d["item_text"] for d in dataset]

    print(f"임베딩 생성 중 ({len(queries)}건)...", end=" ", flush=True)
    try:
        embeddings = embed_provider.embed(queries)
        print("완료")
    except Exception as e:
        print(f"\n임베딩 실패: {e}")
        return

    print("검색 비교 실행 중...")
    for i, (case, emb) in enumerate(zip(dataset, embeddings), 1):
        query = case["item_text"]
        keywords = case["expected_evidence_keywords"]

        # 벡터 검색 (공통)
        try:
            raw = chroma_store.query(
                collection_key=COLLECTION,
                query_embeddings=[emb],
                n_results=20,
            )
            vector_results = _parse_chroma_result(raw)
        except Exception as e:
            logger.warning("벡터 검색 실패 [%s]: %s", case["id"], e)
            vector_results = []

        equal_top5, diff_top5 = compare_query(
            query, bm25_index, emb, vector_results
        )

        rows.append({
            "id": case["id"],
            "query": query,
            "keywords": keywords,
            "equal_hit": hit_rate(equal_top5, keywords),
            "equal_rrf": avg_rrf(equal_top5),
            "diff_hit": hit_rate(diff_top5, keywords),
            "diff_rrf": avg_rrf(diff_top5),
        })
        print(f"  [{i:>2}/{len(dataset)}] {case['id']} - eq_hit={rows[-1]['equal_hit']:.2f}, diff_hit={rows[-1]['diff_hit']:.2f}")

    print_table(rows)


if __name__ == "__main__":
    main()
