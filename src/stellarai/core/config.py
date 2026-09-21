"""集中配置与依赖注入入口配置。所有模块通过 StellarConfig 解耦具体实现。"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class StellarConfig(BaseSettings):
    """全局配置，支持环境变量覆盖（前缀 STELLAR_）。"""

    model_config = SettingsConfigDict(
        env_prefix="STELLAR_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- 推理 LLM ---
    llm_provider: str = "mock"
    llm_model: str = "mock-model"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512

    # --- 嵌入 ---
    embedder_provider: str = "mock"
    embedder_model: str = "mock-embed"
    embedder_dim: int = 64
    embedder_base_url: str | None = None
    embedder_api_key: str | None = None

    # --- 向量存储 ---
    vector_store: str = "numpy"
    vector_persist_path: str | None = None

    # --- 摄取 / 检索 ---
    chunk_size: int = 256
    chunk_overlap: int = 32
    top_k: int = 4

    # --- Agent ---
    agent_max_steps: int = 6

    # --- 服务 ---
    host: str = "0.0.0.0"
    port: int = 8000

    def model_summary(self) -> dict[str, str]:
        return {
            "llm": f"{self.llm_provider}:{self.llm_model}",
            "embedder": f"{self.embedder_provider}:{self.embedder_model}@{self.embedder_dim}",
            "vector_store": self.vector_store,
            "top_k": str(self.top_k),
        }


@lru_cache(maxsize=1)
def get_config() -> StellarConfig:
    """进程内单例配置（带缓存）。"""
    return StellarConfig()
