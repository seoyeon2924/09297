# PoC 보고서 — 방송 심의 AI Agent

---

## 핵심 구현 내용

이번 PoC 단계에서 실제 코드로 구현된 핵심 기능들을 **동작 원리**와 **사용 기술** 중심으로 상세히 기술합니다.

---

### 1.1 에이전트 워크플로우 (Agent Workflow)

* **구현 기능:** 방송 광고 문구의 위험 유형 분류, 사례/법규 병렬 검색, Self-Corrective 검색 루프, 최종 판정(위반소지/주의/OK) 생성 및 품질 검증

* **동작 원리:**
  * 사용자가 입력한 광고 문구(item_text)와 카테고리·방송유형을 받아 **Orchestrator** 노드에서 LLM이 위험 유형(허위/과장, 긴급성·한정판매, 부당비교 등)을 분류하고, 법규 검색용·사례 검색용 쿼리를 각각 생성한다.
  * **CaseAgent**와 **PolicyAgent**가 **병렬**로 실행된다. CaseAgent는 과거 심의 사례만, PolicyAgent는 법령·규정·지침만 검색하며, 각 에이전트 내부에서 **retrieve → grade(관련성 평가) → (부족 시) rewrite_query → retrieve** 루프를 최대 3회까지 반복하여 관련 문서를 확보한다. 두 에이전트가 독립적으로 동작하므로 사례와 법규가 서로 자리를 빼앗지 않고 각각 최대 10건씩 확보할 수 있다.
  * 두 에이전트 결과가 모이면 **Synthesizer** 노드에서 LLM이 모든 근거를 통합하여 최종 판정(위반소지/주의/OK)과 사유·인용(처리번호, 조항명)을 생성한다.
  * **GradeAnswer** 노드에서 생성된 답변이 “사례번호·조항을 실제로 인용했는지” 등 품질 기준을 통과하는지 검증한다. 통과하면 종료하고, 미달이면 Synthesizer로 돌아가 최대 2회까지 재시도한다.

* **주요 기술:** LangGraph(StateGraph, 병렬 노드, 조건부 엣지), LangChain ChatOpenAI(gpt-4o-mini), TypedDict 기반 공유 상태(ReviewState), Annotated[list, operator.add]로 병렬 노드 로그 병합, JSON 스키마 기반 출력 파싱(JsonOutputParser)

---

### 1.2 도구(Tool) 및 함수 연동

* **구현 기능:** 사례 검색(search_cases), 법규·지침 검색(search_policy), 청크 ID로 단건 조회(fetch_chunk_by_id) — 모두 LangGraph 노드에서 LLM이 “어떤 도구를 어떤 쿼리로 호출할지” 결정한 뒤 실행

* **동작 원리:**
  * **search_cases:** 질의 문자열을 받아 OpenAI Embedding으로 벡터화한 뒤, 하이브리드 검색 엔진에 `collection_key="cases"`로 요청한다. 엔진은 BM25(kiwipiepy 형태소 분석 기반 토큰)와 ChromaDB 벡터 검색을 각각 수행한 후 RRF(Reciprocal Rank Fusion)로 병합하고, 상위 20건을 Cohere rerank-multilingual-v3.0으로 재정렬해 최종 5건을 반환한다. 반환값은 `{ "case_chunks": [...] }` 형태로 에이전트 상태에 주입된다.
  * **search_policy:** 동일한 하이브리드 검색을 `regulations`·`guidelines` 컬렉션에 대해 수행하고, 리랭킹 후 법령/규정/지침 청크를 구분하여 반환한다. PolicyAgent는 이 도구를 반복 호출하며 검색 쿼리를 재작성해 관련 문서가 충분할 때까지 루프한다.
  * **fetch_chunk_by_id:** ChromaDB에 저장된 chroma_id로 단일 청크를 조회한다. LLM이 참조한 ID만 골라 상세 내용을 가져와 최종 답변의 references에 넣을 때 사용한다.

* **주요 기술:** LangChain Core `@tool` 데코레이터, OpenAI Embedding API, 자체 구현 HybridSearchEngine(BM25 + ChromaDB Vector + RRF), Cohere Reranker API, Pydantic/딕셔너리 기반 입출력

---

### 1.3 데이터 및 메모리 (RAG & Context)

* **구현 기능:** PDF/Excel 기준지식 수집, 문서 유형별 구조 인식 청킹, 벡터·키워드 이중 인덱싱, 하이브리드 검색 및 리랭킹을 통한 RAG 컨텍스트 공급

* **동작 원리:**
  * **수집·파싱:** 관리자가 업로드한 PDF는 PyMuPDF(fitz)로 페이지별 텍스트 추출, Excel은 openpyxl로 행 단위 파싱한다. 문서 유형(법령/규정/지침/사례)에 따라 **Chunker**가 서로 다른 단위로 분할한다. 법령·규정은 “제n조” 단위, 지침은 “Ⅰ. 대분류” 및 소항목 단위, 사례(Excel)는 “● 상품정보 / ● 심의의견” 등 섹션 단위로 나누고, 필요 시 RecursiveCharacterTextSplitter로 길이를 맞춘다. 각 청크에는 doc_type, page_or_row, article_number, case_number 등 메타데이터가 붙는다.
  * **임베딩·저장:** 청크 본문(content)을 OpenAI text-embedding-3-large(3072차원)로 임베딩하여 ChromaDB의 컬렉션(regulations, guidelines, cases, general)에 upsert한다. 동시에 BM25 검색을 위해 전체 문서를 kiwipiepy 형태소 분석으로 토큰화한 뒤 rank_bm25.BM25Okapi 인덱스를 메모리에 구축하고, ChromaDB 데이터 갱신 시 해당 컬렉션 인덱스를 무효화 후 재구축한다.
  * **검색 시:** 질의를 동일 임베딩 모델로 벡터화하고, BM25 토큰화(kiwipiepy)한 뒤 각각 top-20을 뽑아 RRF(가중치 1:1)로 병합한다. 이 하이브리드 결과를 Cohere Reranker로 재정렬해 상위 5건을 LLM 컨텍스트로 전달한다. 에이전트는 이 컨텍스트와 플래너가 만든 위험 유형·쿼리 로그를 공유 상태(ReviewState)에 담아 노드 간 전달한다.

* **주요 기술:** ChromaDB(HTTP 클라이언트, 4개 컬렉션), OpenAI text-embedding-3-large, rank_bm25(BM25Okapi), kiwipiepy(한국어 형태소 분석), RecursiveCharacterTextSplitter, RRF(Reciprocal Rank Fusion), Cohere rerank-multilingual-v3.0, SQLite(SQLAlchemy) 청크 메타 저장

---

## 주요 문제 해결 및 기술 리서치

구현 과정에서 마주친 기술적 문제와 이를 해결하기 위해 **찾아본 자료(리서치)** 및 **적용한 방법**을 기록합니다.

| **이슈 구분** | **문제 상황 및 원인** | **리서치 및 해결 과정 (Reference & Solution)** |
| ------------- | --------------------- | ---------------------------------------------- |
| **프롬프트** | 복잡한 광고 문구에 대해 “위반소지”와 “주의” 구분이 불안정하고, 식약처 인증 등 사실 기술을 위반으로 오탐 | • **리서치:** 판정 기준(위반소지/주의/OK) 정의 문헌 및 Few-shot 예시 사례 검토<br>• **적용:** Grader/Generator 시스템 프롬프트에 “객관적 사실 기술은 OK”, “주의는 개선 권고 수준” 등 명시적 기준 추가 후 eval_03 등 정확도 개선 |
| **도구 연동** | Cohere Reranker API 일시 장애 또는 rate limit 시 전체 심의 파이프라인 실패 | • **리서치:** tenacity 라이브러리 지수 백오프(exponential backoff) 패턴 및 fallback 설계 사례<br>• **적용:** rerank 호출에 `@retry(stop=3, wait=exponential)` 적용, 예외 시 원본 순서대로 상위 N건 반환(fallback)하여 파이프라인 중단 방지 |
| **RAG/검색** | ChromaDB만 사용 시 “처리번호”, “심의지적코드” 등 정확 키워드 매칭이 약해 사례 검색 품질 저하 | • **리서치:** BM25 + Vector 하이브리드 검색, RRF(Reciprocal Rank Fusion) 논문 및 한국어 토크나이저(kiwipiepy) 사례<br>• **적용:** BM25 인덱스를 별도 구축하고 RRF로 벡터 검색과 병합, kiwipiepy 형태소 분석으로 한국어 토큰 품질 개선 후 키워드 recall 향상 |
| **RAG/검색** | BM25와 Vector에 차등 가중치(0.7 : 1.0)를 줬을 때 유의미한 개선이 없음 | • **리서치:** weight_comparison 실험으로 Hit Rate·Avg RRF 비교<br>• **적용:** 실험 결과를 반영해 최종 시스템에는 동일 가중치(1.0 : 1.0) 유지, 추후 데이터 확충 시 재검증 예정 |
| **성능/기타** | 검색된 문서 N건에 대해 grade_documents를 N번 직렬 호출하면 지연 시간이 N배로 증가 | • **리서치:** LLM 배치 처리 및 “한 번의 호출로 여러 문서 관련성 평가” 프롬프트 패턴<br>• **적용:** grade_documents를 “문서 목록 전체를 한 번에 넘기고, JSON 배열로 각 doc_index별 relevance 반환”하도록 변경하여 호출 1회로 단축 |
| **성능/기타** | CaseAgent와 PolicyAgent가 한 retriever를 공유하면 사례와 법규가 서로 상위 자리 경쟁 | • **리서치:** LangGraph 병렬 노드와 상태 키 분리(state key per agent)<br>• **적용:** case_context와 law/regulation/guideline_chunks를 별도 상태 키로 두고, 두 에이전트를 병렬 노드로 실행해 각각 독립적으로 최대 10건씩 확보 |

---

## 핵심 동작 검증

위에서 구현한 기능이 의도대로 동작하는지 보여주는 **대표적인 실행 결과**를 첨부합니다.

---

**[검증 시나리오:** 긴급성·한정판매 유발 광고 문구에 대한 위반소지 판정 **]**

* **입력**
  * **문구:** "이 가격 마감임박! 딱 100개 한정수량"
  * **카테고리:** 생활용품
  * **방송유형:** 생방송

* **에이전트 동작:**
  1. **Orchestrator:** 위험 유형 "긴급성/한정판매" 분류, 검색 쿼리 생성 (예: "긴급성 광고 사례", "긴급성 한정판매 마감임박 한정수량", "긴급성 광고 규정" 등)
  2. **CaseAgent:** search_cases("긴급성 한정판매 광고 사례" 등) 호출 → 하이브리드 검색 후 Cohere 리랭킹 → 관련 사례 2건 확보, policy 검색으로 규정 1건 확보
  3. **PolicyAgent:** search_policy로 동일 쿼리/규정 쿼리 호출 → 법규·지침 청크 확보
  4. **Synthesizer:** 확보한 사례(처리번호 115553 등)와 규정(방송광고심의에 관한 규정 제7조)을 근거로 최종 판정 생성
  5. **GradeAnswer:** 사례번호·조항 인용 여부 검증 → pass → 종료

* **최종 결과:**
  * **판정:** 위반소지
  * **사유:** "처리번호 115553 (2025-08-27) 건에 의하면, '마감임박'과 같은 긴급성을 과도하게 조장하는 표현은 소비자에게 오인 가능성을 줄 수 있어 주의가 필요합니다. 또한 방송광고심의에 관한 규정 제7조에 따라 긴급상황으로 오인하게 할 정도의 표현은 사용해서는 안 됩니다."
  * **근거 키워드 매칭:** 마감, 한정, 긴급 (기대 키워드 100% 일치)
  * **평가 결과:** 골든 데이터셋 기준 정답(위반소지)과 일치, judgment_correct=true, evidence_keyword_recall=1.0

*(위 시나리오는 eval 골든 데이터셋 eval_07 및 평가 결과 파일 `eval/results/prompt-tuned_20260304_152301.json`의 실제 실행 로그를 요약한 것입니다.)*
