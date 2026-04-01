from __future__ import annotations

from typing import ClassVar
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Pipeline: URLs ────────────────────────────────────────────────────────
    BASE_URL: str = "https://diseases.medelement.com"

    CONTENT_TYPES: ClassVar[dict[int, str]] = {4: "Kazakhstan"}
    TARGET_CONTENT_TYPES: list[int] = [4]

    # ── Pipeline: Rate limiting ───────────────────────────────────────────────
    REQUESTS_PER_SECOND: float = 1.0
    RETRY_MAX_ATTEMPTS: int = 5
    RETRY_BASE_DELAY_S: float = 2.0
    RETRY_MAX_DELAY_S: float = 60.0

    # ── Pipeline: Pagination ──────────────────────────────────────────────────
    PAGE_SIZE: int = 10

    # ── Pipeline: Concurrency ─────────────────────────────────────────────────
    CONCURRENCY: int = 3

    # ── Pipeline: Paths ───────────────────────────────────────────────────────
    STATE_DB_PATH: str = "data/state.db"
    OUTPUT_JSONL_PATH: str = "data/output/protocols.jsonl"

    # ── Pipeline: Playwright ─────────────────────────────────────────────────
    HEADLESS: bool = True
    BROWSER_TIMEOUT_MS: int = 30_000
    PLAYWRIGHT_SLOW_MO: int = 0

    # ── Pipeline: Chunking ────────────────────────────────────────────────────
    MAX_CHUNK_TOKENS: int = 512

    # ── RAG: Vector DB (Qdrant) ───────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "medprotocols"

    # ── RAG: Embeddings ───────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = "local"          # "local" | "openai"
    EMBEDDING_MODEL_LOCAL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_MODEL_OPENAI: str = "text-embedding-3-small"
    OPENAI_API_KEY: str = ""
    EMBEDDING_BATCH_SIZE: int = 64

    # ── RAG: Retrieval ────────────────────────────────────────────────────────
    TOP_K: int = 5

    # ── RAG: Answer generation — Ollama (local, default) ─────────────────────
    LLM_BACKEND: str = "ollama"           # "ollama" | "anthropic"
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # ── RAG: Answer generation — Anthropic (optional) ─────────────────────────
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-6"


settings = Settings()
