"""
Abstract base classes for pluggable LLM / Embedding / Retriever providers.

Concrete implementations live in sibling modules
(e.g. llm_openai.py, embed_openai.py, retriever_chroma.py).
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Generate text from a prompt."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...


class EmbedProvider(ABC):
    """Generate embeddings for a list of texts."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class RetrieverProvider(ABC):
    """Retrieve relevant chunks for a query."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filters: dict | None = None,
    ) -> list[dict]: ...
