from openai import OpenAI
from config import settings
import httpx
import time

# 1. API Key 확인
print(f"API Key 앞 10자: {settings.OPENAI_API_KEY[:10]}...")
print()

# 2. 네트워크 연결 테스트
print("=== 네트워크 테스트 ===")
try:
    start = time.time()
    r = httpx.get("https://api.openai.com/v1/models", timeout=10)
    latency = round(time.time() - start, 2)
    print(f"OpenAI 서버 응답: {r.status_code} ({latency}초)")
except Exception as e:
    print(f"네트워크 연결 실패: {e}")
print()

# 3. 짧은 API 호출 테스트
print("=== API 호출 테스트 ===")
try:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "1+1=?"}],
        max_tokens=10,
    )
    latency = round(time.time() - start, 2)
    print(f"응답: {response.choices[0].message.content} ({latency}초)")
    print("API Key 정상!")
except Exception as e:
    print(f"API 호출 실패: {e}")
print()

# 4. 긴 컨텍스트 테스트 (Step 3 상황 시뮬레이션)
print("=== 긴 컨텍스트 테스트 ===")
try:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    long_text = "테스트 문장입니다. " * 500  # 대략 Step 3 컨텍스트 크기
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"다음을 요약해: {long_text}"}],
        max_tokens=100,
        timeout=90,
    )
    latency = round(time.time() - start, 2)
    print(f"응답 길이: {len(response.choices[0].message.content)}자 ({latency}초)")
    print("긴 컨텍스트 정상!")
except Exception as e:
    print(f"긴 컨텍스트 실패: {e}")