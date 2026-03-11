# 방송 심의 AI Agent

LangGraph 기반 **멀티 에이전트 Self-Corrective RAG** 시스템으로,  
방송 광고 문구의 법령·규정·지침 위반 여부를 자동 판별하고 근거를 제시합니다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목적** | 방송 광고 심의 담당자가 광고 문구를 검토할 때, 관련 법령·규정·과거 사례를 자동 검색하여 판단 초안(위반소지 / 주의 / OK)과 근거를 제공 |
| **핵심 기술** | LangGraph (Multi-Agent), Self-Corrective RAG, Hybrid Search (BM25 + Vector + RRF), Cohere Reranker |
| **LLM** | OpenAI gpt-4o-mini |
| **임베딩** | OpenAI text-embedding-3-large (3072차원) |
| **벡터DB** | ChromaDB (HTTP 클라이언트 모드, 4개 컬렉션) |
| **RDB** | SQLite + SQLAlchemy ORM |
| **UI** | Streamlit (4페이지) |
| **API** | FastAPI + SSE 실시간 스트리밍 |

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI (app.py)                                      │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌───────────┐  │
│  │ 기준지식    │ │ 심의요청    │ │ 요청목록  │ │ 심의상세   │  │
│  │ 관리       │ │ 등록       │ │          │ │ (AI추천)  │  │
│  └─────┬──────┘ └─────┬──────┘ └────┬─────┘ └─────┬─────┘  │
├────────┼──────────────┼─────────────┼─────────────┼────────┤
│  FastAPI Backend (api/main.py)                              │
│  POST /reviews  │  GET /reviews  │ GET /reviews/{id}/stream │
├────────┼──────────────┼─────────────┼─────────────┼────────┤
│  Services Layer                                             │
│  ┌──────────┐ ┌──────────────┐ ┌────────────┐ ┌─────────┐  │
│  │ Ingest   │ │ Review       │ │ RAG        │ │ Audit   │  │
│  │ Service  │ │ Service      │ │ Service    │ │ Service │  │
│  └────┬─────┘ └──────────────┘ └─────┬──────┘ └─────────┘  │
├───────┼────────────────────────────────┼────────────────────┤
│  LangGraph ReviewChain                                      │
│                                                             │
│  Orchestrator ──┬── CaseAgent (사례 검색 루프)     ──┐       │
│                 └── PolicyAgent (법규 검색 루프)    ──┤       │
│                     Synthesizer ─── GradeAnswer ──── END    │
│                         ↑ fail          │ pass              │
│                         └───────────────┘                   │
├─────────────────────────────────────────────────────────────┤
│  Storage                                                    │
│  ┌────────────────────┐  ┌──────────────────────────────┐   │
│  │ SQLite             │  │ ChromaDB (HTTP, port 8000)   │   │
│  │ compliance.db      │  │ ┌────────────┐ ┌──────────┐ │   │
│  │ - documents        │  │ │regulations │ │guidelines│ │   │
│  │ - chunks           │  │ ├────────────┤ ├──────────┤ │   │
│  │ - review_requests  │  │ │cases       │ │general   │ │   │
│  │ - audit_logs       │  │ └────────────┘ └──────────┘ │   │
│  └────────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. AI 파이프라인 상세 (ReviewChain)

LangGraph 기반 **Self-Corrective RAG** 워크플로우입니다.

```
START
  │
  ▼
🎯 Orchestrator ─── 광고 문구 분석, 위험유형 분류, 검색 쿼리 생성
  │
  ├──────────────────────┐
  ▼                      ▼
📋 CaseAgent           📜 PolicyAgent         ← 병렬 실행
  │ 사례 검색 루프       │ 법규 검색 루프
  │ retrieve → grade    │ retrieve → grade
  │    ↑ rewrite ←┘     │    ↑ rewrite ←┘
  │                      │
  └──────────┬───────────┘
             ▼
⚖️ Synthesizer ─── 근거 통합 → 최종 판정 (위반소지 / 주의 / OK)
  │
  ▼
✅ GradeAnswer ─── 판정 품질 검증
  │
  ├─ pass → END
  └─ fail → Synthesizer 재시도 (최대 2회)
```

### 주요 특징

- **멀티 에이전트**: CaseAgent(사례)와 PolicyAgent(법규)가 독립적으로 검색 루프를 돌며 각각 최대 10건 근거 확보
- **Self-Corrective**: 검색 결과가 부족하면 LLM이 쿼리를 재작성하여 재검색 (최대 3회)
- **답변 품질 검증**: 생성된 판정을 GradeAnswer가 검증하고, 미달 시 Synthesizer부터 재시도
- **하이브리드 검색**: BM25(키워드) + Vector(의미) + RRF(Reciprocal Rank Fusion) 병합
- **Cohere 리랭킹**: 하이브리드 검색 결과를 Cohere rerank-multilingual-v3.0으로 재정렬

---

## 4. 검색 파이프라인 (Hybrid Search + Reranker)

```
쿼리
  │
  ├─ BM25 검색 (키워드 매칭, kiwipiepy 형태소 분석)
  │    └─ top-20
  │
  ├─ Vector 검색 (OpenAI 임베딩 코사인 유사도)
  │    └─ top-20
  │
  ▼
RRF 병합 (Reciprocal Rank Fusion)
  │  vector_weight=1.0, bm25_weight=1.0 (동일 가중치)
  │
  ▼ top-20
Cohere Reranker (rerank-multilingual-v3.0)
  │
  ▼ top-5
LLM에 전달
```

---

## 5. 폴더 구조

```
09297/
├── app.py                    # Streamlit 엔트리포인트
├── config.py                 # 환경 설정 (경로, API키, RAG 파라미터)
├── requirements.txt          # Python 패키지 의존성
│
├── ui/                       # Streamlit 페이지 모듈
│   ├── page_knowledge.py     #   기준지식 관리 (문서 업로드·인덱싱)
│   ├── page_request.py       #   심의요청 등록
│   ├── page_list.py          #   심의요청 목록
│   ├── page_review_detail.py #   심의 상세 (AI 추천 + 최종 판단)
│   ├── api_client.py         #   FastAPI httpx 클라이언트
│   └── components/           #   재사용 UI 컴포넌트
│       ├── status_badge.py   #     상태 뱃지
│       └── pipeline_viz.py   #     파이프라인 시각화
│
├── api/                      # FastAPI 백엔드
│   ├── main.py               #   FastAPI 앱 + CORS + 라우터
│   ├── schemas.py            #   Pydantic 요청/응답 스키마
│   └── routes/
│       └── review.py         #   심의 CRUD + SSE 스트리밍 API
│
├── services/                 # 비즈니스 로직
│   ├── ingest_service.py     #   문서 파싱 → 청킹 → 임베딩 → Chroma/DB 저장
│   ├── review_service.py     #   심의 요청 CRUD + 상태 관리
│   ├── rag_service.py        #   AI 추천 오케스트레이션 (ReviewChain 호출)
│   └── audit_service.py      #   감사 로그 기록
│
├── chains/                   # LangGraph 에이전트 체인
│   ├── review_chain.py       #   메인 ReviewChain (Orchestrator, Synthesizer, GradeAnswer)
│   └── case_agent.py         #   CaseAgent (사례 전용 검색 에이전트)
│
├── ingest/                   # 문서 수집·파싱·청킹
│   ├── parser_pdf.py         #   PDF 파서 (PyMuPDF)
│   ├── parser_excel.py       #   Excel 파서 (openpyxl)
│   ├── chunker.py            #   문서 유형별 구조 인식 청킹
│   └── metadata_generator.py #   LLM 기반 메타데이터 생성 (미구현)
│
├── prompts/                  # 프롬프트 템플릿
│   ├── planner.py            #   Orchestrator용 (위험유형 분류)
│   ├── grader.py             #   문서 관련성 평가 / 답변 품질 평가
│   ├── generator.py          #   Synthesizer용 (최종 판정 생성)
│   ├── compliance_review.py  #   심의 리뷰 프롬프트 (예비)
│   └── metadata_generation.py#   메타데이터 생성 프롬프트 (예비)
│
├── providers/                # LLM·임베딩 추상화
│   ├── base.py               #   ABC 인터페이스 (LLM, Embed, Retriever)
│   └── embed_openai.py       #   OpenAI 임베딩 구현 (Mock 모드 지원)
│
├── tools/                    # LangChain @tool 래퍼 (에이전트용)
│   ├── policy_tools.py       #   규정·지침 하이브리드 검색 도구
│   └── case_tools.py         #   심의 사례 하이브리드 검색 도구
│
├── utils/                    # 유틸리티
│   ├── hybrid_search.py      #   BM25 + Vector + RRF 하이브리드 검색 엔진
│   └── reranker.py           #   Cohere Reranker (지수 백오프 + fallback)
│
├── storage/                  # DB + 벡터 저장소
│   ├── models.py             #   SQLAlchemy ORM 모델 (7개 테이블)
│   ├── database.py           #   SQLite 엔진·세션 관리
│   ├── repository.py         #   CRUD 래퍼 (Document, Review, Audit)
│   └── chroma_store.py       #   ChromaDB 컬렉션 래퍼 (4개 컬렉션)
│
├── eval/                     # 평가 파이프라인
│   ├── golden_dataset.json   #   골든 데이터셋 (15건)
│   ├── run_eval.py           #   자동 평가 실행기
│   ├── compare.py            #   두 평가 결과 비교
│   ├── tokenizer_benchmark.py#   Tokenizer(regex vs kiwipiepy) 벤치마크
│   ├── weight_comparison.py  #   BM25/Vector 가중치 비교 실험
│   └── results/              #   평가 결과 (JSON + Markdown)
│
├── tests/                    # 테스트
│   ├── test_review_chain.py  #   ReviewChain 통합 테스트
│   ├── test_tools_chroma.py  #   도구 + Chroma 검색 테스트
│   ├── test_reranker_backoff.py # Reranker 백오프 동작 검증
│   ├── test_grade_comparison.py # grade_documents 전후 비교
│   ├── test_graph_visual.py  #   LangGraph 시각화 테스트
│   ├── test_prompts_import.py#   프롬프트 import 확인
│   └── test_openai_key.py    #   OpenAI API 키 검증
│
├── scripts/
│   └── reset_docs_and_chroma.py # 문서·Chroma 초기화 스크립트
│
├── docs/
│   └── review_chain_flow.md.md #   시스템흐름도
│
└── data/                     # 런타임 데이터 (gitignore)
    ├── compliance.db         #   SQLite DB
    ├── chroma_db/            #   ChromaDB 데이터
    └── uploads/              #   업로드된 문서 파일
```

---

## 6. 설치 및 실행

### 사전 요구사항

- Python 3.10+
- OpenAI API Key
- Cohere API Key (리랭커용)

### 설치

```bash
# 가상환경 생성
python -m venv .venv

# 활성화 (Windows)
.venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 환경 설정

```bash
copy .env.example .env
```

`.env` 파일에 API 키를 설정합니다:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-large
COHERE_API_KEY=your-cohere-key-here
MOCK_MODE=false
```

### 실행

```bash
# 1. ChromaDB 서버 시작 (별도 터미널)
chroma run --path ./data/chroma_db --port 8000

# 2. FastAPI 백엔드 시작 (별도 터미널)
uvicorn api.main:app --port 8001 --reload

# 3. Streamlit UI 시작
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속합니다.

---

## 7. 화면 구성

| 페이지 | 역할 | 주요 기능 |
|--------|------|-----------|
| **기준지식 관리** | 관리자 | PDF/Excel 업로드 → 파싱 → 구조 인식 청킹 → 임베딩 → Chroma 인덱싱 |
| **심의요청 등록** | PD/MD | 상품명·카테고리·방송유형 + 요청문구/강조바 입력 → 심의 요청 생성 |
| **심의요청 목록** | 심의자 | 전체 요청 목록 조회 (상태 필터) → 상세 화면 이동 |
| **심의 상세** | 심의자 | AI 추천 실행 (SSE 실시간) → 판정·사유·근거 확인 → 최종 판단 저장 |

---

## 8. 데이터 흐름

### 지식 인덱싱 경로

```
PDF/Excel 업로드
  → parser_pdf / parser_excel (텍스트 추출)
  → chunker (문서 유형별 구조 인식: 법령=조 단위, 규정=조 단위, 지침=섹션 단위, 사례=행 단위)
  → embed_openai (text-embedding-3-large, 3072차원)
  → chroma_store.upsert (배치: 50청크씩 임베딩, 30개씩 Chroma 저장)
  → SQLite 청크 레코드 저장
```

### 심의 요청 경로

```
PD가 문구 등록 (FastAPI POST /reviews)
  → 심의자가 "AI 심의 추천 실행" 클릭
  → SSE 스트리밍 (GET /reviews/{id}/stream)
  → RAGService → ReviewChain.run()
    → Orchestrator → CaseAgent ∥ PolicyAgent → Synthesizer → GradeAnswer
  → AI 추천 결과 DB 저장 (상태: REVIEWING)
  → 심의자가 최종 판단 저장 (상태: DONE / REJECTED)
```

---

## 9. 기술 스택

| 분류 | 기술 |
|------|------|
| **Language** | Python 3.10+ |
| **LLM Framework** | LangChain Core, LangGraph |
| **LLM** | OpenAI gpt-4o-mini |
| **Embedding** | OpenAI text-embedding-3-large |
| **Vector DB** | ChromaDB (HTTP 클라이언트) |
| **RDB** | SQLite + SQLAlchemy 2.0 |
| **Search** | BM25 (rank-bm25) + Vector + RRF |
| **Tokenizer** | kiwipiepy (한국어 형태소 분석) |
| **Reranker** | Cohere rerank-multilingual-v3.0 |
| **UI** | Streamlit |
| **API** | FastAPI + SSE |
| **PDF 파싱** | PyMuPDF (fitz) |
| **Excel 파싱** | openpyxl |

---

## 10. 평가 결과 요약

15건의 골든 데이터셋으로 단계적 개선을 측정했습니다.  
상세 보고서: [`eval/EVALUATION_REPORT.md`](eval/EVALUATION_REPORT.md)

| 단계 | 변경 사항 | 정확도 (Exact) | 부분점수 | 위험유형 Recall | 근거키워드 Recall | 평균 지연 |
|------|-----------|:-----------:|:-------:|:-------------:|:---------------:|:--------:|
| **Baseline** | 기본 RAG | 60.0% (9/15) | 73.3% | 90.0% | 72.9% | 46.1s |
| **+Hybrid Search** | BM25+Vector+RRF 도입 | 66.7% (10/15) | 76.7% | 90.0% | 75.1% | 55.3s |
| **+Prompt Tuning** | 프롬프트 최적화 | **73.3% (11/15)** | **83.3%** | 90.0% | 74.2% | 50.1s |

- Baseline → 최종: **정확도 60% → 73.3% (+13.3%p)**, 부분점수 73.3% → 83.3% (+10%p)

**리랭커(Cohere)**: 하이브리드 검색 결과를 Cohere rerank-multilingual-v3.0으로 재정렬해 LLM에 넘기는 상위 5건의 품질을 높이는 역할을 한다.  


