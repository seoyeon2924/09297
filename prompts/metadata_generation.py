"""
Prompt template for LLM-based chunk metadata generation.
Stage 2 implementation.
"""

METADATA_GENERATION_PROMPT = """\
당신은 방송 심의 관련 문서를 분석하는 전문가입니다.

아래 텍스트 청크를 읽고 다음 정보를 JSON으로 추출하세요:
1. section_title: 이 청크가 속한 조항명 또는 섹션 제목 (없으면 null)
2. keywords: 심의 시 핵심이 되는 키워드 3~5개 (리스트)

## 텍스트 청크:
{chunk_text}

## 출력 형식 (JSON만 출력):
{{"section_title": "...", "keywords": ["...", "..."]}}
"""
