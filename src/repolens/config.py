from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from REPOLENS_* environment variables (or a local .env)."""

    model_config = SettingsConfigDict(env_prefix="REPOLENS_", env_file=".env", extra="ignore")

    data_dir: Path = Path("./data")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    max_files: int = Field(300, ge=1)
    max_file_bytes: int = Field(200_000, ge=1)
    max_total_bytes: int = Field(5_000_000, ge=1)
    max_clone_bytes: int = Field(100_000_000, ge=1)
    clone_timeout_seconds: int = Field(120, ge=1)

    chunk_lines: int = Field(40, ge=5)
    chunk_overlap: int = Field(8, ge=0)

    # Minimum cosine similarity of the best hit to consider the evidence sufficient.
    min_score: float = Field(0.25, ge=0.0, le=1.0)

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_timeout_seconds: float = 60.0
    # Optional: "minimal", "low", "medium" or "high" (sent as reasoning_effort when set).
    llm_reasoning_effort: str = ""
    llm_max_output_tokens: int = Field(2048, ge=16)
    # Paid-tier USD prices per 1M tokens, used only to ESTIMATE cost (unset -> no estimate).
    llm_input_price_per_mtok: float | None = Field(None, ge=0.0)
    llm_output_price_per_mtok: float | None = Field(None, ge=0.0)

    @field_validator("llm_reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"", "none", "minimal", "low", "medium", "high"}:
            raise ValueError("llm_reasoning_effort must be none, minimal, low, medium or high.")
        return value

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_lines:
            raise ValueError("chunk_overlap must be smaller than chunk_lines.")
        return self

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_model.strip())
