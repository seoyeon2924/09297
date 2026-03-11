# 평가 보고서 — 방송 심의 AI Agent

## 1. 평가 개요

| 항목 | 내용 |
|------|------|
| 평가 일시 | 2026-03-04 |
| 골든 데이터셋 | `eval/golden_dataset.json` (15건) |
| 평가 대상 | ReviewChain (LangGraph 멀티 에이전트 Self-Corrective RAG) |
| 평가 도구 | `eval/run_eval.py` (자동 평가), `eval/compare.py` (결과 비교) |
| LLM | OpenAI gpt-4o-mini |
| 임베딩 | OpenAI text-embedding-3-large |

### 평가 지표

| 지표 | 설명 |
|------|------|
| **Judgment Accuracy** | AI 판정(위반소지/주의/OK)이 정답과 정확히 일치하는 비율 |
| **Partial Score** | 위반소지↔주의는 0.5점, 정답 1.0점, 완전 오답 0.0점 |
| **Risk Type Recall** | AI가 올바른 위험유형(허위/과장, 긴급성 등)을 얼마나 잡아냈는지 |
| **Evidence Keyword Recall** | 판정 사유에 기대 키워드가 포함된 비율 |
| **Evidence Has Refs Rate** | AI가 근거 문서(사례번호/조항)를 제시한 비율 |
| **Avg Latency** | 건당 평균 처리 시간 |

---

## 2. 단계별 실험 결과

### 2.1 요약 비교

| 단계 | Tag | Accuracy | Partial | Risk Recall | Keyword Recall | Refs Rate | Avg Latency |
|------|-----|:--------:|:-------:|:-----------:|:--------------:|:---------:|:-----------:|
| **A. Baseline** | `baseline` | 60.0% (9/15) | 73.3% | 90.0% | 72.9% | 93.3% | 46.1s |
| **B. +Hybrid Search** | `hybrid-search` | 66.7% (10/15) | 76.7% | 90.0% | 75.1% | 93.3% | 55.3s |
| **C. +Prompt Tuning** | `prompt-tuned` | 73.3% (11/15) | 83.3% | 90.0% | 74.2% | 93.3% | 50.1s |

### 2.2 개선 추이

```
정확도(Exact Match):  60.0% ──→ 66.7% ──→ 73.3%  (+13.3%p)
부분점수(Partial):    73.3% ──→ 76.7% ──→ 83.3%  (+10.0%p)
                     [A]       [B]       [C]
```

---

## 3. 각 단계별 상세 분석

### 3.1 A → B: Hybrid Search 도입 효과

**변경 사항**: Vector-only 검색에서 **BM25 + Vector + RRF** 하이브리드 검색으로 전환

| 변화 | 값 |
|------|-----|
| 정확도 | 60.0% → 66.7% (+6.7%p, +1건) |
| 부분점수 | 73.3% → 76.7% (+3.4%p) |
| 키워드 Recall | 72.9% → 75.1% (+2.2%p) |

**새로 맞춘 항목:**

| ID | 문구 | Baseline | Hybrid |
|----|------|----------|--------|
| eval_09 | "라이선스 브랜드 캘빈클라인 정품 드로즈 세트" | 위반소지 (오답) | **주의 (정답)** |

**분석:**
- BM25가 "라이선스 브랜드"라는 **키워드를 정확히 매칭**하여 관련 사례(처리번호 119291)를 찾아냄
- Vector-only에서는 의미적 유사도만으로는 "라이선스 안내 요망"이라는 특정 지적을 잡기 어려웠음
- 대신 latency가 46.1s → 55.3s로 증가 (BM25 인덱스 구축 + 이중 검색 비용)

### 3.2 B → C: Prompt Tuning 효과

**변경 사항**: Grader/Generator 프롬프트 최적화 (판정 기준 명확화, "주의" vs "위반소지" 구분 강화)

| 변화 | 값 |
|------|-----|
| 정확도 | 66.7% → 73.3% (+6.6%p, +1건) |
| 부분점수 | 76.7% → 83.3% (+6.6%p) |
| Avg Latency | 55.3s → 50.1s (-5.2s) |

**새로 맞춘 항목:**

| ID | 문구 | Hybrid | Prompt-tuned |
|----|------|--------|-------------|
| eval_03 | "본 제품은 식약처 인증 건강기능식품입니다" | 위반소지 (오답) | **OK (정답)** |

**분석:**
- 프롬프트에 "객관적 사실 기술은 OK로 판단" 기준을 명시하여 식약처 인증 사실의 단순 기술을 오탐하지 않게 됨
- latency 감소는 검색 쿼리 생성 효율화에 의한 것으로 추정 (불필요한 재검색 감소)

### 3.3 리랭커(Cohere) 역할 및 검증 현황

**역할:** 하이브리드 검색(BM25+Vector+RRF)으로 뽑은 상위 20건을 Cohere **rerank-multilingual-v3.0**으로 재정렬한 뒤, 상위 5건만 LLM 컨텍스트로 전달한다.  
의미·쿼리 부합도 기준으로 순위를 다시 매겨, 노이즈가 섞인 상위 문서를 걸러내는 것이 목적이다.

**검증 내용:**  
`tests/test_reranker_backoff.py`에서 **(1)** 정상 호출 시 재시도 없이 빠르게 반환되는지, **(2)** Cohere API 실패(잘못된 키 등) 시 tenacity 지수 백오프 3회 후 **fallback**(원본 순서대로 상위 N건 반환)으로 파이프라인이 중단되지 않는지 검증

---

## 4. 여전히 오답인 항목 분석

최종 단계(prompt-tuned)에서 틀린 4건:

| ID | 문구 | 기대 | 실제 | 원인 분석 |
|----|------|------|------|-----------|
| eval_04 | "근거불확실한 완벽 프리미엄 여행 패키지" | 주의 | 위반소지 | "주의"와 "위반소지" 경계가 모호한 문구. 위반소지로 판단하는 것이 보수적으로 타당한 면도 있음 |
| eval_06 | "오직 방송에서만 이 구성! SK스토아 단독" | 주의 | 위반소지 | "한정판매" 관련 사례 검색은 성공했으나, 심각도 판단에서 "주의"가 아닌 "위반소지"로 과잉 판정 |
| eval_13 | "면 100% 소재로 제작된 편안한 티셔츠입니다" | OK | 위반소지 | 문제없는 문구를 위반으로 오탐. 검색된 "허위/과장" 규정을 과도 적용한 FP(False Positive) |
| eval_15 | "추석 전 완벽배송, 방송 중 결제완료 고객 한정" | 주의 | 위반소지 | eval_04, 06과 동일 패턴: "주의" 수준을 "위반소지"로 과잉 판정 |

**공통 패턴:**
- 4건 중 3건이 **"주의"를 "위반소지"로 과잉 판정** (보수적 편향)
- 1건이 **OK를 "위반소지"로 오탐** (False Positive)
- "주의"와 "위반소지"의 경계 기준이 LLM에게 여전히 어려운 판단

---

## 5. Tokenizer 벤치마크

`eval/tokenizer_benchmark.py`로 BM25 토크나이저의 한국어 처리 능력을 비교했습니다.

| 토크나이저 | 방식 | 장점 |
|-----------|------|------|
| **regex** (폴백) | `r'[가-힣]+'` 단순 분리 | 설치 불필요, 빠른 속도 |
| **kiwipiepy** | 형태소 분석 기반 | 복합어 분리, 조사 제거, 의미 단위 토크나이징 |

kiwipiepy 도입 후 BM25 검색에서 **키워드 매칭 정확도가 향상**되었으며,  
특히 "심의지적코드", "위반유형" 등 복합 한국어 용어의 검색 성능이 개선되었습니다.

### 5.1 BM25/Vector 가중치 비교

`eval/weight_comparison.py`로 RRF 병합 시 가중치 차등 적용 효과를 비교했습니다.

| 설정 | vector_weight | bm25_weight | 비고 |
|------|:------------:|:-----------:|------|
| **A. 동일 가중치** | 1.0 | 1.0 | 기본값 |
| **B. 차등 가중치** | 0.7 | 1.0 | BM25 우선 |

실험 결과, 차등 가중치(B)가 일부 쿼리에서 Hit Rate를 높였으나 **전체적으로 유의미한 차이가 없었습니다.**  
따라서 최종 시스템에는 **A(동일 가중치: vector=1.0, bm25=1.0)**를 유지했습니다.  
가중치 차등은 데이터가 더 쌓인 후 재검증할 예정입니다.

---

## 6. 테스트 목록

| 테스트 파일 | 목적 |
|------------|------|
| `tests/test_review_chain.py` | ReviewChain 단독 실행 및 결과 확인 |
| `tests/test_tools_chroma.py` | policy_tools, case_tools의 Chroma 검색 동작 확인 |
| `tests/test_reranker_backoff.py` | Cohere Reranker 지수 백오프 및 fallback 동작 검증 |
| `tests/test_grade_comparison.py` | grade_documents 배치 처리 전후 성능 비교 (LangSmith 트레이스) |
| `tests/test_graph_visual.py` | LangGraph 워크플로우 Mermaid 시각화 |
| `tests/test_prompts_import.py` | Phase 2 프롬프트 모듈 import 확인 |
| `tests/test_openai_key.py` | OpenAI API 키 및 네트워크 연결 검증 |

---

## 7. 평가 결과 파일

| 파일 | 내용 |
|------|------|
| `eval/results/baseline_20260304_140845.json` | Baseline 상세 결과 (JSON) |
| `eval/results/baseline_20260304_140845.md` | Baseline 마크다운 리포트 |
| `eval/results/hybrid-search_20260304_150558.json` | Hybrid Search 상세 결과 |
| `eval/results/hybrid-search_20260304_150558.md` | Hybrid Search 마크다운 리포트 |
| `eval/results/prompt-tuned_20260304_152301.json` | Prompt Tuning 상세 결과 |
| `eval/results/prompt-tuned_20260304_152301.md` | Prompt Tuning 마크다운 리포트 |
| `tests/grade_comparison_result.json` | grade_documents 배치 처리 비교 결과 |

---

## 8. 결론 및 향후 과제

### 달성한 것
- Baseline 대비 **정확도 13.3%p 향상** (60% → 73.3%)
- Hybrid Search + Reranker로 **키워드·의미 검색 양쪽 커버**
- kiwipiepy 형태소 분석으로 **한국어 BM25 품질 향상**
- Self-Corrective RAG로 검색 실패 시 **자동 쿼리 재작성**

### 남은 과제
- "주의" vs "위반소지" **경계 판단 정밀도** 향상 (4건 중 3건이 이 문제)
- OK 문구의 **오탐(False Positive) 감소** (eval_13)
- 사례 데이터 확충 시 검색 품질 추가 향상 기대
- LLM 모델 업그레이드(gpt-4o 등) 시 판정 정밀도 개선 가능성
