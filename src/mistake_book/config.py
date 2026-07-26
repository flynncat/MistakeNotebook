from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


def _persistent_session_token(data_dir: Path) -> str:
    configured = os.getenv("MISTAKE_BOOK_SESSION_TOKEN")
    if configured:
        return configured
    path = data_dir / ".session-token"
    try:
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = ""
    if token:
        return token
    token = secrets.token_urlsafe(24)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(token, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    return token


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    static_dir: Path
    session_token: str
    recognition_provider: str
    local_ocr_backend: str
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    ollama_url: str
    ollama_model: str
    pipeline_version: str = "v2"

    @classmethod
    def load(cls, root_dir: Path | None = None) -> "Settings":
        root = (root_dir or Path.cwd()).resolve()
        data_dir = Path(os.getenv("MISTAKE_BOOK_DATA_DIR", root / "data")).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            root_dir=root,
            data_dir=data_dir,
            static_dir=Path(__file__).parent / "static",
            session_token=_persistent_session_token(data_dir),
            recognition_provider=os.getenv("MISTAKE_BOOK_PROVIDER", "local"),
            local_ocr_backend=os.getenv("MISTAKE_BOOK_LOCAL_OCR", "auto"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
            ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct"),
            pipeline_version=os.getenv("MISTAKE_BOOK_PIPELINE", "v2"),
        )
