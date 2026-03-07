"""
tools/policy_tools, tools/case_tools 검색 동작 확인.

사전 조건: Chroma 서버 실행 중 (chroma run --path ./data/chroma_db --port 8000)
실행: 프로젝트 루트에서
  .venv\\Scripts\\activate
  python -m pytest tests/test_tools_chroma.py -v
  또는
  python tests/test_tools_chroma.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.policy_tools import search_policy, fetch_chunk_by_id
from tools.case_tools import search_cases


def test_search_policy():
    """규정·지침 검색: law_chunks, regulation_chunks, guideline_chunks 반환 확인."""
    policy_result = search_policy.invoke({"query": "방송 한정판매 긴급성 표현"})
    assert "law_chunks" in policy_result
    assert "regulation_chunks" in policy_result
    assert "guideline_chunks" in policy_result
    assert isinstance(policy_result["law_chunks"], list)
    assert isinstance(policy_result["regulation_chunks"], list)
    assert isinstance(policy_result["guideline_chunks"], list)
    print(f"법률: {len(policy_result['law_chunks'])}건")
    print(f"규정: {len(policy_result['regulation_chunks'])}건")
    print(f"지침: {len(policy_result['guideline_chunks'])}건")


def test_search_cases():
    """심의 사례 검색: case_chunks 반환 확인."""
    case_result = search_cases.invoke({"query": "오늘만 한정 가격"})
    assert "case_chunks" in case_result
    assert isinstance(case_result["case_chunks"], list)
    print(f"사례: {len(case_result['case_chunks'])}건")


def test_fetch_chunk_by_id():
    """search_policy 결과에서 chroma_id 하나를 골라 fetch_chunk_by_id 호출."""
    policy_result = search_policy.invoke({"query": "방송"})
    if policy_result["law_chunks"]:
        chunk, collection_key = policy_result["law_chunks"][0], "regulations"
    elif policy_result["regulation_chunks"]:
        chunk, collection_key = policy_result["regulation_chunks"][0], "regulations"
    elif policy_result["guideline_chunks"]:
        chunk, collection_key = policy_result["guideline_chunks"][0], "guidelines"
    else:
        print("(벡터DB에 규정/지침 데이터 없음 — fetch_chunk_by_id 스킵)")
        return

    chroma_id = chunk["chroma_id"]
    fetched = fetch_chunk_by_id.invoke({
        "chroma_id": chroma_id,
        "collection_key": collection_key,
    })
    assert "content" in fetched
    assert "metadata" in fetched
    assert fetched["chroma_id"] == chroma_id
    print(f"fetch_chunk_by_id OK: {chroma_id[:40]}...")


if __name__ == "__main__":
    print("=== search_policy ===")
    test_search_policy()
    print("\n=== search_cases ===")
    test_search_cases()
    print("\n=== fetch_chunk_by_id ===")
    test_fetch_chunk_by_id()
    print("\n완료.")
