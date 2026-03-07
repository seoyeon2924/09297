# 방송 심의 AI Agent — MVP

방송 심의·컴플라이언스 업무를 지원하는 AI Agent MVP입니다.

## 설치

```bash
# 가상환경 생성 (권장)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

## 환경 설정

```bash
# .env 파일 생성 (.env.example 참고)
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
```

> 단계 1에서는 `MOCK_MODE=true` (기본값)로 실행하면 API 키 없이 동작합니다.

## 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속합니다.

## 화면 구성

| 페이지 | 역할 | 설명 |
|--------|------|------|
| 기준지식 관리 | 관리자 | PDF/Excel/DOCX 업로드 → 인덱싱 |
| 심의요청 등록 | PD/MD | 요청문구/강조바 입력 → 심의 요청 생성 |
| 심의요청 목록 | 심의자 | 전체 요청 목록 조회 → 상세 화면 이동 |
| 심의 상세 | 심의자 | AI 추천 확인 → 최종 판단 저장 |

## 단계 1에서 되는 것

- 앱 실행 및 4개 페이지 간 네비게이션
- SQLite DB 자동 초기화 (7개 테이블)
- 기준지식 파일 업로드 + 더미 청크 생성 (상태: INDEXED)
- 심의 요청 등록 (요청문구/강조바 독립 아이템으로 저장)
- 요청 목록 조회 + 상태 필터
- 심의 상세: AI 더미 추천 실행 (랜덤 판단 + 더미 근거)
- 상태 전이: REQUESTED → AI_RUNNING → REVIEWING → DONE/REJECTED
- 최종 심의 판단 저장
- 감사(Audit) 로그 기록

## 단계 1에서 안 되는 것 (단계 2 예정)

- 실제 PDF/Excel 파싱 (현재 더미 청크 생성)
- 실제 Chroma 임베딩/벡터 검색
- 실제 LLM 기반 AI 추천 (현재 랜덤 모의)
- LLM 기반 고급 메타데이터 생성 (section_title, keywords)
- 사용자 인증/권한 관리

## 기술 스택

- **Python** 3.11+
- **Streamlit** — UI
- **SQLAlchemy** + **SQLite** — 구조화 데이터
- **ChromaDB** — 벡터 저장 (단계 2)
- **OpenAI** — LLM / Embedding (단계 2)

## 폴더 구조

```
BroadcastComplianceAgent/
├── app.py              # Streamlit 엔트리포인트
├── config.py           # 환경 설정
├── requirements.txt
├── ui/                 # 페이지 모듈
├── services/           # 비즈니스 로직
├── storage/            # DB + Chroma 래퍼
├── providers/          # LLM/Embed/Retriever ABC
├── ingest/             # 파서·청커 (단계 2)
├── prompts/            # 프롬프트 템플릿
├── data/               # 런타임 데이터 (gitignore)
└── tests/              # 테스트
```
