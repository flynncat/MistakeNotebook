from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Storage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.files_dir = data_dir / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "mistake_book.sqlite3"
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    pdf_path TEXT
                );
                CREATE TABLE IF NOT EXISTS problems (
                    id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'not_required',
                    selected_artifact TEXT,
                    category TEXT,
                    ocr_text TEXT,
                    confidence REAL NOT NULL DEFAULT 0,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS categories (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_batch(self) -> str:
        batch_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO batches(id,status,created_at,updated_at) VALUES(?,?,?,?)",
                (batch_id, "processing", now, now),
            )
        (self.files_dir / batch_id).mkdir(parents=True, exist_ok=True)
        return batch_id

    def add_problem(self, batch_id: str, filename: str, source: Path) -> str:
        problem_id = uuid.uuid4().hex
        suffix = source.suffix.lower() or ".img"
        target = self.files_dir / batch_id / f"{problem_id}-source{suffix}"
        shutil.copy2(source, target)
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO problems(
                    id,batch_id,filename,source_path,status,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (problem_id, batch_id, Path(filename).name, str(target), "queued", now, now),
            )
        return problem_id

    def add_uploaded_problem(self, batch_id: str, filename: str, content: bytes) -> str:
        safe_name = Path(filename).name
        digest = hashlib.sha256(content).hexdigest()[:12]
        suffix = Path(safe_name).suffix.lower() or ".img"
        temp = self.files_dir / batch_id / f"upload-{digest}{suffix}"
        temp.write_bytes(content)
        problem_id = self.add_problem(batch_id, safe_name, temp)
        temp.unlink(missing_ok=True)
        return problem_id

    def update_problem(self, problem_id: str, **values: Any) -> None:
        allowed = {
            "status",
            "review_status",
            "selected_artifact",
            "category",
            "ocr_text",
            "confidence",
            "metrics_json",
            "error",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if "metrics_json" in updates and not isinstance(updates["metrics_json"], str):
            updates["metrics_json"] = json.dumps(
                updates["metrics_json"], ensure_ascii=False, sort_keys=True
            )
        if not updates:
            return
        updates["updated_at"] = _now()
        columns = ", ".join(f"{key}=?" for key in updates)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"UPDATE problems SET {columns} WHERE id=?",
                (*updates.values(), problem_id),
            )

    def finish_batch(self, batch_id: str) -> None:
        problems = self.get_problems(batch_id)
        if not problems:
            status = "empty"
        elif any(item["status"] == "processing" for item in problems):
            status = "processing"
        elif any(item["review_status"] == "pending" for item in problems):
            status = "needs_review"
        elif any(item["status"] == "failed" for item in problems):
            status = "partial"
        else:
            status = "ready"
        with self._connect() as conn:
            conn.execute(
                "UPDATE batches SET status=?,updated_at=? WHERE id=?",
                (status, _now(), batch_id),
            )

    def set_pdf(self, batch_id: str, pdf_path: Path) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE batches SET pdf_path=?,updated_at=? WHERE id=?",
                (str(pdf_path), _now(), batch_id),
            )

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["problems"] = self.get_problems(batch_id)
        return result

    def get_problem(self, problem_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM problems WHERE id=?", (problem_id,)).fetchone()
        return self._decode_problem(row) if row else None

    def get_problems(self, batch_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM problems WHERE batch_id=? ORDER BY created_at", (batch_id,)
            ).fetchall()
        return [self._decode_problem(row) for row in rows]

    @staticmethod
    def _decode_problem(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
        return item

    def category_names(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM categories ORDER BY created_at").fetchall()
        return [row["name"] for row in rows]

    def ensure_category(self, name: str, description: str = "") -> str:
        normalized = name.strip()[:24] or "未分类"
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM categories WHERE name=?", (normalized,)
            ).fetchone()
            if row:
                return str(row["id"])
            category_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO categories(id,name,description,created_at)
                VALUES(?,?,?,?)
                """,
                (category_id, normalized, description, _now()),
            )
        return category_id
