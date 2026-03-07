"""
PDF parser using PyMuPDF (fitz).
Extracts text page-by-page, stripping common header/footer noise.
"""

from pathlib import Path

import fitz  # PyMuPDF

_NOISE_PATTERNS: list[str] = [
    "법제처",
    "국가법령정보센터",
]


class PDFParser:
    """Parse a PDF file and return per-page text with metadata."""

    @staticmethod
    def parse(file_path: str) -> list[dict]:
        """
        Returns:
            List of {"text": str, "page": int, "source_file": str}
        """
        path = Path(file_path)
        doc = fitz.open(str(path))

        results: list[dict] = []
        try:
            for page_idx in range(len(doc)):
                text = doc[page_idx].get_text()
                for pattern in _NOISE_PATTERNS:
                    text = text.replace(pattern, "")
                text = text.strip()
                if not text:
                    continue
                results.append(
                    {
                        "text": text,
                        "page": page_idx + 1,
                        "source_file": path.name,
                    }
                )
        finally:
            doc.close()
        return results

    @staticmethod
    def get_full_text(file_path: str) -> str:
        """전체 페이지 텍스트를 하나의 문자열로 합쳐 반환."""
        pages = PDFParser.parse(file_path)
        return "\n".join(p["text"] for p in pages)
