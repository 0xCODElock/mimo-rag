"""Configuration management for MiMo-RAG."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MiMoConfig:
    """MiMo API configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("MIMO_API_KEY", ""))
    api_base: str = field(default_factory=lambda: os.getenv("MIMO_API_BASE", "https://api.xiaomimimo.com/v1"))
    model: str = field(default_factory=lambda: os.getenv("MIMO_MODEL", "MiMo-7B-RL"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "1024")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    vector_store_path: str = field(default_factory=lambda: os.getenv("VECTOR_STORE_PATH", "./vector_store"))


@dataclass
class AppConfig:
    """Application-level configuration."""
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    mimo: MiMoConfig = field(default_factory=MiMoConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)


# Global singleton config
config = AppConfig()
