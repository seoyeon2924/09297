"""
ReviewChain 단독 실행 테스트.

사전 조건: .env에 OPENAI_API_KEY 설정, Chroma 서버 실행
실행: 프로젝트 루트에서 python tests/test_review_chain.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chains.review_chain import ReviewChain

chain = ReviewChain(model_name="gpt-4o-mini")
result = chain.run(
    item_text="오늘 방송 종료 시 절대 없는 가격",
    category="건강식품",
    broadcast_type="TV홈쇼핑",
)

print("판정:", result["judgment"])
print("사유:", (result["reason"] or "")[:100])
print("참조:", len(result.get("references", [])), "건")
print("도구 로그:", result["tool_logs"])
