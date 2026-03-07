"""
LLM-based metadata generator for chunks (section_title, keywords).
Stage 2 implementation — invoked via "고급 메타 생성" button.
"""


class MetadataGenerator:
    @staticmethod
    def generate(chunk_text: str) -> dict:
        """
        Call LLM to extract:
          - section_title (str)
          - keywords (list[str])

        Returns:
            {"section_title": str, "keywords": list[str]}
        """
        # TODO: Stage 2 — call LLMProvider.generate() with metadata prompt
        raise NotImplementedError(
            "LLM metadata generation will be implemented in Stage 2"
        )
