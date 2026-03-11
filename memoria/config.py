"""Memoria configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MemoriaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEMORIA_",
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore",
    )

    # Database
    db_host: str = "localhost"
    db_port: int = 6001
    db_user: str = "root"
    db_password: str = "111"
    db_name: str = "memoria"

    # Embedding
    embedding_provider: str = "openai"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = Field(default=0, description="0 = auto-infer")
    embedding_api_key: str = ""
    embedding_base_url: str | None = None

    # Auth
    master_key: str = Field(default="", description="Master API key for admin operations (min 16 chars in production)")

    # Limits
    snapshot_limit: int = Field(default=100, description="Max snapshots per user")

    @property
    def db_url(self) -> str:
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            "?charset=utf8mb4"
        )

    def warn_weak_master_key(self) -> str | None:
        """Return warning message if master_key is set but too short."""
        if self.master_key and len(self.master_key) < 16:
            return f"MEMORIA_MASTER_KEY is only {len(self.master_key)} chars — use ≥16 chars in production"
        return None


_settings: MemoriaSettings | None = None


def get_settings() -> MemoriaSettings:
    global _settings
    if _settings is None:
        _settings = MemoriaSettings()
    return _settings
