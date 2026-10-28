from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BIORAG_", extra="ignore")

    # "ollama" (free, local) or "claude" (Anthropic API, needs ANTHROPIC_API_KEY)
    llm_backend: str = "ollama"
    ollama_model: str = "qwen2.5:7b"
    ollama_judge_model: str | None = None  # defaults to ollama_model
    ollama_url: str = "http://localhost:11434"

    generator_model: str = "claude-opus-5-5"
    judge_model: str = "claude-haiku-5-5"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    data_dir: Path = Path("data")
    collection: str = "pubmed"

    chunk_size: int = 1200  # characters
    chunk_overlap: int = 200

    # Retrieval depth at each stage
    sparse_k: int = 50
    dense_k: int = 50
    fused_k: int = 30
    rerank_k: int = 8

    max_subquestions: int = 3

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


@lru_cache
def get_settings() -> Settings:
    return Settings()
