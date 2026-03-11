"""
LLM-based metadata generator for chunks (section_title, keywords).
Stage 2 implementation — invoked via "고급 메타 생성" button.
"""

import json
import logging

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import settings
from prompts.metadata_generation import METADATA_GENERATION_PROMPT

logger = logging.getLogger(__name__)

_http_client = httpx.Client(verify=False)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY or "dummy",
        http_client=_http_client,
        request_timeout=60,
    )


class MetadataGenerator:

    _llm: ChatOpenAI | None = None

    @classmethod
    def _ensure_llm(cls) -> ChatOpenAI:
        if cls._llm is None:
            cls._llm = _get_llm()
        return cls._llm

    @staticmethod
    def generate(chunk_text: str) -> dict:
        """Call LLM to extract section_title and keywords from a chunk.

        Returns:
            {"section_title": str, "keywords": list[str]}
        """
        llm = MetadataGenerator._ensure_llm()
        prompt = METADATA_GENERATION_PROMPT.format(chunk_text=chunk_text)

        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content.strip()

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)
            return {
                "section_title": parsed.get("section_title") or "",
                "keywords": parsed.get("keywords") or [],
            }
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning("메타데이터 LLM 파싱 실패: %s", e)
            return {"section_title": "", "keywords": []}
