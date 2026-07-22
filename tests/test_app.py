from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mistake_book.app import create_app
from mistake_book.config import Settings


def test_api_requires_session_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTAKE_BOOK_SESSION_TOKEN", "test-token")
    settings = Settings.load(tmp_path)
    client = TestClient(create_app(settings))

    root = client.get("/")
    assert root.status_code == 200
    assert "test-token" in root.text

    unauthorized = client.get("/api/batches/missing")
    assert unauthorized.status_code == 403

    authorized = client.get(
        "/api/batches/missing", headers={"X-Session-Token": "test-token"}
    )
    assert authorized.status_code == 404
