from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    drive_folder_id: str = ""
    google_service_account_json: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_rag_docs"
    manifest_path: Path = Path("manifest.json")
    audit_log_path: Path = Path("audit_log.jsonl")
    retrieval_top_k: int = 20
    rerank_top_n: int = 6


def get_settings() -> Settings:
    settings = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        drive_folder_id=os.getenv("DRIVE_FOLDER_ID", ""),
        google_service_account_json=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", ""),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "enterprise_rag_docs"),
        manifest_path=Path(os.getenv("MANIFEST_PATH", "manifest.json")),
        audit_log_path=Path(os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "20")),
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "6")),
    )
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required. Set it in .env or environment.")
    return settings
