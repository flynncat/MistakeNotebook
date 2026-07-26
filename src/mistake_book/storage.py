from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .classification import legacy_category_key


def _now() -> str:
    return datetime.now(UTC).isoformat()


def content_fingerprint(text: str, figure_kind: str = "none") -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[\s\W_]+", "", normalized, flags=re.UNICODE)
    payload = f"{normalized}|{figure_kind.strip().lower() or 'none'}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_relative_path(value: str) -> str:
    parts = [
        part
        for part in value.replace("\\", "/").split("/")
        if part not in {"", ".", ".."}
    ]
    return "/".join(parts)[-500:]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Storage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.files_dir = data_dir / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "mistake_book.sqlite3"
        self._lock = Lock()
        self._backup_before_asset_migration()
        self._initialize()

    def _backup_before_asset_migration(self) -> None:
        if not self.db_path.exists():
            return
        backup = self.db_path.with_name("mistake_book.pre-assets.sqlite3")
        if backup.exists():
            return
        with sqlite3.connect(self.db_path) as source:
            columns = {
                row[1]
                for row in source.execute("PRAGMA table_info(problems)").fetchall()
            }
            if not columns or "asset_state" in columns:
                return
            with sqlite3.connect(backup) as target:
                source.backup(target)

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
                    category TEXT NOT NULL DEFAULT '未分类',
                    category_group TEXT NOT NULL DEFAULT '未分类',
                    category_key TEXT NOT NULL DEFAULT '未分类',
                    summary TEXT NOT NULL DEFAULT '',
                    category_confidence REAL NOT NULL DEFAULT 0,
                    category_source TEXT NOT NULL DEFAULT 'automatic',
                    ocr_text TEXT,
                    confidence REAL NOT NULL DEFAULT 0,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    source_relative_path TEXT NOT NULL DEFAULT '',
                    source_sha256 TEXT NOT NULL DEFAULT '',
                    asset_state TEXT NOT NULL DEFAULT 'workbench',
                    content_fingerprint TEXT,
                    accepted_at TEXT,
                    selected_kind TEXT NOT NULL DEFAULT '',
                    parent_source_id TEXT,
                    page_index INTEGER,
                    page_total INTEGER,
                    split_label TEXT NOT NULL DEFAULT '',
                    split_ocr_text TEXT NOT NULL DEFAULT '',
                    content_blocks_version INTEGER,
                    content_blocks_json TEXT,
                    content_source_sha256 TEXT,
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
                CREATE TABLE IF NOT EXISTS exports (
                    id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._migrate_problem_columns(conn)
            self._repair_legacy_review_reasons(conn)
            self._repair_unreliable_secondary_ocr_reasons(conn)
            self._repair_tesseract_path_reasons(conn)
            obsolete = self._migrate_assets(conn)
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_problems_assets
                ON problems(asset_state,accepted_at)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_problems_published_fingerprint
                ON problems(content_fingerprint)
                WHERE asset_state='published' AND content_fingerprint IS NOT NULL
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_problems_source_sha256
                ON problems(source_sha256)
                """
            )
        for problem in obsolete:
            self._cleanup_problem_files(problem)

    @staticmethod
    def _migrate_problem_columns(conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(problems)").fetchall()
        }
        additions = {
            "category_group": "TEXT NOT NULL DEFAULT '未分类'",
            "category_key": "TEXT NOT NULL DEFAULT '未分类'",
            "summary": "TEXT NOT NULL DEFAULT ''",
            "category_confidence": "REAL NOT NULL DEFAULT 0",
            "category_source": "TEXT NOT NULL DEFAULT 'automatic'",
            "source_relative_path": "TEXT NOT NULL DEFAULT ''",
            "source_sha256": "TEXT NOT NULL DEFAULT ''",
            "asset_state": "TEXT NOT NULL DEFAULT 'workbench'",
            "content_fingerprint": "TEXT",
            "accepted_at": "TEXT",
            "selected_kind": "TEXT NOT NULL DEFAULT ''",
            "parent_source_id": "TEXT",
            "page_index": "INTEGER",
            "page_total": "INTEGER",
            "split_label": "TEXT NOT NULL DEFAULT ''",
            "split_ocr_text": "TEXT NOT NULL DEFAULT ''",
            "content_blocks_version": "INTEGER",
            "content_blocks_json": "TEXT",
            "content_source_sha256": "TEXT",
        }
        migrated = False
        for name, declaration in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE problems ADD COLUMN {name} {declaration}")
                migrated = True
        conn.execute(
            "UPDATE problems SET category='未分类' WHERE category IS NULL OR TRIM(category)=''"
        )
        if not migrated:
            return
        rows = conn.execute(
            "SELECT id,category,metrics_json FROM problems"
        ).fetchall()
        for row in rows:
            group, key = legacy_category_key(row["category"])
            try:
                metrics = json.loads(row["metrics_json"] or "{}")
            except json.JSONDecodeError:
                metrics = {}
            summary = str(metrics.get("recognition_summary", ""))
            conn.execute(
                """
                UPDATE problems
                SET category_group=?,category_key=?,summary=?,
                    category_confidence=0,category_source='migrated'
                WHERE id=?
                """,
                (group, key, summary, row["id"]),
            )

    @staticmethod
    def _repair_legacy_review_reasons(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id,ocr_text,metrics_json FROM problems
            WHERE metrics_json LIKE '%未定位到印刷题号%'
            """
        ).fetchall()
        for row in rows:
            text = str(row["ocr_text"] or "")
            if not re.search(r"[【\[\(（]?(?:例题|练习)\s*[0-9S]+", text, re.IGNORECASE):
                continue
            try:
                metrics = json.loads(row["metrics_json"] or "{}")
            except json.JSONDecodeError:
                continue

            def replace_reason(values: Any) -> list[Any]:
                if not isinstance(values, list):
                    return []
                return [
                    "第二 OCR 未识别到完整题干"
                    if value == "未定位到印刷题号"
                    else value
                    for value in values
                ]

            metrics["review_reasons"] = replace_reason(
                metrics.get("review_reasons")
            )
            structured = metrics.get("structured_problem")
            if isinstance(structured, dict):
                structured["review_reasons"] = replace_reason(
                    structured.get("review_reasons")
                )
            conn.execute(
                "UPDATE problems SET metrics_json=? WHERE id=?",
                (
                    json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                    row["id"],
                ),
            )

    @staticmethod
    def _repair_unreliable_secondary_ocr_reasons(
        conn: sqlite3.Connection,
    ) -> None:
        rows = conn.execute(
            """
            SELECT id,metrics_json FROM problems
            WHERE metrics_json LIKE '%双 OCR%'
            """
        ).fetchall()
        secondary_reasons = {
            "双 OCR 题干差异过大",
            "双 OCR 数字或比例不一致",
        }
        for row in rows:
            try:
                metrics = json.loads(row["metrics_json"] or "{}")
            except json.JSONDecodeError:
                continue
            structured = metrics.get("structured_problem")
            if not isinstance(structured, dict):
                continue
            try:
                similarity = float(structured.get("text_similarity", 0))
            except (TypeError, ValueError):
                similarity = 0
            if similarity >= 0.82:
                continue

            def without_unreliable(values: Any) -> list[Any]:
                if not isinstance(values, list):
                    return []
                return [value for value in values if value not in secondary_reasons]

            metrics["review_reasons"] = without_unreliable(
                metrics.get("review_reasons")
            )
            structured["review_reasons"] = without_unreliable(
                structured.get("review_reasons")
            )
            conn.execute(
                "UPDATE problems SET metrics_json=? WHERE id=?",
                (
                    json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                    row["id"],
                ),
            )

    @staticmethod
    def _repair_tesseract_path_reasons(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id,metrics_json FROM problems
            WHERE metrics_json LIKE '%未安装 Tesseract%'
            """
        ).fetchall()
        for row in rows:
            try:
                metrics = json.loads(row["metrics_json"] or "{}")
            except json.JSONDecodeError:
                continue
            reasons = metrics.get("review_reasons")
            if not isinstance(reasons, list):
                continue
            metrics["review_reasons"] = [
                reason for reason in reasons if reason != "未安装 Tesseract"
            ]
            conn.execute(
                "UPDATE problems SET metrics_json=? WHERE id=?",
                (
                    json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                    row["id"],
                ),
            )

    def _migrate_assets(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM problems ORDER BY updated_at DESC,id DESC"
        ).fetchall()
        published: dict[str, str] = {}
        obsolete: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            updates: dict[str, Any] = {}
            source_path = Path(str(item.get("source_path") or ""))
            if not item.get("source_relative_path"):
                updates["source_relative_path"] = str(item.get("filename") or "")
            if not item.get("source_sha256") and source_path.is_file():
                updates["source_sha256"] = _file_sha256(source_path)

            if item.get("review_status") == "accepted":
                try:
                    metrics = json.loads(item.get("metrics_json") or "{}")
                except json.JSONDecodeError:
                    metrics = {}
                structured = metrics.get("structured_problem")
                figure = (
                    str(structured.get("figure") or "none")
                    if isinstance(structured, dict)
                    else "none"
                )
                fingerprint = content_fingerprint(
                    str(item.get("ocr_text") or ""),
                    figure,
                )
                if fingerprint in published:
                    obsolete.append(item)
                    conn.execute("DELETE FROM problems WHERE id=?", (item["id"],))
                    continue
                published[fingerprint] = str(item["id"])
                selected_name = Path(
                    str(item.get("selected_artifact") or "")
                ).name
                selected_kind = {
                    "question.png": "reconstructed",
                    "normalized.png": "normalized",
                    "cleaned.png": "cleaned",
                }.get(selected_name, str(item.get("selected_kind") or "reconstructed"))
                updates.update(
                    {
                        "asset_state": "published",
                        "content_fingerprint": fingerprint,
                        "accepted_at": item.get("accepted_at")
                        or item.get("updated_at")
                        or _now(),
                        "selected_kind": selected_kind,
                    }
                )
            elif item.get("asset_state") == "published":
                updates.update(
                    {
                        "asset_state": "workbench",
                        "content_fingerprint": None,
                        "accepted_at": None,
                    }
                )

            if updates:
                assignments = ", ".join(f"{key}=?" for key in updates)
                conn.execute(
                    f"UPDATE problems SET {assignments} WHERE id=?",
                    (*updates.values(), item["id"]),
                )
        return obsolete

    def _cleanup_problem_files(self, problem: dict[str, Any]) -> None:
        source_path = Path(str(problem.get("source_path") or ""))
        artifact_dir = (
            self.files_dir
            / str(problem.get("batch_id") or "")
            / str(problem.get("id") or "")
        )
        source_path.unlink(missing_ok=True)
        if artifact_dir.is_dir():
            shutil.rmtree(artifact_dir, ignore_errors=True)

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

    def add_problem(
        self,
        batch_id: str,
        filename: str,
        source: Path,
        *,
        source_relative_path: str | None = None,
        parent_source_id: str | None = None,
        page_index: int | None = None,
        page_total: int | None = None,
        split_label: str = "",
        split_ocr_text: str = "",
    ) -> str:
        digest = _file_sha256(source)
        problem_id = uuid.uuid4().hex
        suffix = source.suffix.lower() or ".img"
        target = self.files_dir / batch_id / f"{problem_id}-source{suffix}"
        shutil.copy2(source, target)
        now = _now()
        relative_path = _safe_relative_path(source_relative_path or filename)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO problems(
                    id,batch_id,filename,source_path,status,source_relative_path,
                    source_sha256,parent_source_id,page_index,page_total,split_label,split_ocr_text,
                    created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    problem_id,
                    batch_id,
                    Path(filename).name,
                    str(target),
                    "queued",
                    relative_path,
                    digest,
                    parent_source_id,
                    page_index,
                    page_total,
                    split_label,
                    split_ocr_text,
                    now,
                    now,
                ),
            )
        return problem_id

    def add_uploaded_problem(
        self,
        batch_id: str,
        filename: str,
        content: bytes,
        *,
        source_relative_path: str | None = None,
    ) -> str:
        safe_name = Path(filename).name
        digest = hashlib.sha256(content).hexdigest()[:12]
        suffix = Path(safe_name).suffix.lower() or ".img"
        temp = self.files_dir / batch_id / f"upload-{digest}{suffix}"
        temp.write_bytes(content)
        problem_id = self.add_problem(
            batch_id,
            safe_name,
            temp,
            source_relative_path=source_relative_path or filename,
        )
        temp.unlink(missing_ok=True)
        return problem_id

    def update_problem(self, problem_id: str, **values: Any) -> None:
        allowed = {
            "status",
            "review_status",
            "selected_artifact",
            "category",
            "category_group",
            "category_key",
            "summary",
            "category_confidence",
            "category_source",
            "source_relative_path",
            "source_sha256",
            "asset_state",
            "content_fingerprint",
            "accepted_at",
            "selected_kind",
            "parent_source_id",
            "page_index",
            "page_total",
            "split_label",
            "split_ocr_text",
            "content_blocks_version",
            "content_blocks_json",
            "content_source_sha256",
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
        if "content_blocks_json" in updates and not isinstance(
            updates["content_blocks_json"], str
        ):
            updates["content_blocks_json"] = json.dumps(
                updates["content_blocks_json"], ensure_ascii=False, sort_keys=True
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

    def update_formula_blocks(
        self,
        problem_id: str,
        expected_updated_at: str,
        content_blocks: dict[str, Any],
        metrics: dict[str, Any],
        ocr_text: str,
    ) -> bool:
        updated_at = _now()
        encoded_blocks = json.dumps(
            content_blocks,
            ensure_ascii=False,
            sort_keys=True,
        )
        encoded_metrics = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE problems
                SET content_blocks_version=2,
                    content_blocks_json=?,
                    metrics_json=?,
                    ocr_text=?,
                    updated_at=?
                WHERE id=? AND updated_at=?
                """,
                (
                    encoded_blocks,
                    encoded_metrics,
                    ocr_text,
                    updated_at,
                    problem_id,
                    expected_updated_at,
                ),
            )
            return cursor.rowcount == 1

    def update_problem_cas(
        self,
        problem_id: str,
        expected_updated_at: str,
        **values: Any,
    ) -> bool:
        allowed = {
            "status",
            "review_status",
            "selected_artifact",
            "category",
            "category_group",
            "category_key",
            "summary",
            "category_confidence",
            "category_source",
            "asset_state",
            "content_fingerprint",
            "accepted_at",
            "selected_kind",
            "content_blocks_version",
            "content_blocks_json",
            "content_source_sha256",
            "ocr_text",
            "confidence",
            "metrics_json",
            "error",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        for key in ("metrics_json", "content_blocks_json"):
            if key in updates and not isinstance(updates[key], str):
                updates[key] = json.dumps(
                    updates[key],
                    ensure_ascii=False,
                    sort_keys=True,
                )
        if not updates:
            return False
        updates["updated_at"] = _now()
        columns = ", ".join(f"{key}=?" for key in updates)
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE problems SET {columns} WHERE id=? AND updated_at=?",
                (*updates.values(), problem_id, expected_updated_at),
            )
            return cursor.rowcount == 1

    def finish_batch(self, batch_id: str) -> None:
        problems = self.get_problems(batch_id)
        if not problems:
            status = "empty"
        elif any(item["status"] in {"queued", "processing"} for item in problems):
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

    def add_export(
        self,
        batch_id: str,
        kind: str,
        path: Path,
        filename: str,
    ) -> str:
        export_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO exports(id,batch_id,kind,file_path,filename,created_at)
                VALUES(?,?,?,?,?,?)
                """,
                (export_id, batch_id, kind, str(path), filename, _now()),
            )
        return export_id

    def get_export(self, export_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM exports WHERE id=?",
                (export_id,),
            ).fetchone()
        return dict(row) if row is not None else None

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
                "SELECT * FROM problems WHERE batch_id=? ORDER BY created_at,id",
                (batch_id,),
            ).fetchall()
        return [self._decode_problem(row) for row in rows]

    def list_problems(
        self,
        *,
        sort: str = "newest",
        category_group: str | None = None,
        category: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        conditions: list[str] = []
        parameters: list[Any] = []
        if category_group:
            conditions.append("category_group=?")
            parameters.append(category_group)
        if category:
            conditions.append("category_key=?")
            parameters.append(category)
        if query:
            conditions.append("(filename LIKE ? OR COALESCE(ocr_text,'') LIKE ?)")
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            value = f"%{escaped}%"
            conditions[-1] = (
                "(filename LIKE ? ESCAPE '\\' OR COALESCE(ocr_text,'') LIKE ? ESCAPE '\\')"
            )
            parameters.extend((value, value))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        direction = "DESC" if sort == "newest" else "ASC"
        with self._connect() as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM problems {where}",
                    parameters,
                ).fetchone()[0]
            )
            rows = conn.execute(
                f"""
                SELECT * FROM problems
                {where}
                ORDER BY created_at {direction}, id {direction}
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return {
            "items": [self._decode_problem(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
            "sort": sort,
        }

    def list_assets(
        self,
        *,
        sort: str = "newest",
        category_group: str | None = None,
        category: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        conditions = ["asset_state='published'", "review_status='accepted'"]
        parameters: list[Any] = []
        if category_group:
            conditions.append("category_group=?")
            parameters.append(category_group)
        if category:
            conditions.append("category_key=?")
            parameters.append(category)
        if query:
            conditions.append(
                """
                (
                    filename LIKE ? ESCAPE '\\'
                    OR source_relative_path LIKE ? ESCAPE '\\'
                    OR COALESCE(ocr_text,'') LIKE ? ESCAPE '\\'
                )
                """
            )
            escaped = (
                query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            value = f"%{escaped}%"
            parameters.extend((value, value, value))
        where = f"WHERE {' AND '.join(conditions)}"
        direction = "DESC" if sort == "newest" else "ASC"
        with self._connect() as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM problems {where}",
                    parameters,
                ).fetchone()[0]
            )
            rows = conn.execute(
                f"""
                SELECT * FROM problems
                {where}
                ORDER BY accepted_at {direction},id {direction}
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return {
            "items": [self._decode_problem(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
            "sort": sort,
        }

    def publish_asset(
        self,
        problem_id: str,
        *,
        selected_kind: str,
    ) -> dict[str, Any]:
        obsolete: list[dict[str, Any]] = []
        old_batches: set[str] = set()
        accepted_at = _now()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM problems WHERE id=?",
                (problem_id,),
            ).fetchone()
            if row is None:
                raise KeyError(problem_id)
            problem = dict(row)
            try:
                metrics = json.loads(problem.get("metrics_json") or "{}")
            except json.JSONDecodeError:
                metrics = {}
            structured = metrics.get("structured_problem")
            figure = (
                str(structured.get("figure") or "none")
                if isinstance(structured, dict)
                else "none"
            )
            fingerprint = content_fingerprint(
                str(problem.get("ocr_text") or ""),
                figure,
            )
            duplicates = conn.execute(
                """
                SELECT * FROM problems
                WHERE asset_state='published'
                  AND content_fingerprint=?
                  AND id<>?
                """,
                (fingerprint, problem_id),
            ).fetchall()
            for duplicate in duplicates:
                item = dict(duplicate)
                obsolete.append(item)
                old_batches.add(str(item["batch_id"]))
                conn.execute("DELETE FROM problems WHERE id=?", (item["id"],))
            conn.execute(
                """
                UPDATE problems
                SET asset_state='published',content_fingerprint=?,
                    accepted_at=?,selected_kind=?,status='ready',
                    review_status='accepted',updated_at=?
                WHERE id=?
                """,
                (
                    fingerprint,
                    accepted_at,
                    selected_kind,
                    accepted_at,
                    problem_id,
                ),
            )
        for item in obsolete:
            self._cleanup_problem_files(item)
        for batch_id in old_batches:
            self.finish_batch(batch_id)
        return {
            "fingerprint": fingerprint,
            "replaced_count": len(obsolete),
            "replaced_ids": [str(item["id"]) for item in obsolete],
            "accepted_at": accepted_at,
        }

    def delete_problem(self, problem_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM problems WHERE id=?",
                (problem_id,),
            ).fetchone()
            if row is None:
                return None
            problem = dict(row)
            conn.execute("DELETE FROM problems WHERE id=?", (problem_id,))
        self._cleanup_problem_files(problem)
        self.finish_batch(str(problem["batch_id"]))
        return problem

    @staticmethod
    def _decode_problem(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
        raw_blocks = item.pop("content_blocks_json", None)
        item["content_blocks"] = json.loads(raw_blocks) if raw_blocks else None
        return item

    def category_pairs(self) -> list[tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT category_group,category_key
                FROM problems
                WHERE category_group<>'' AND category_key<>''
                ORDER BY category_group,category_key
                """
            ).fetchall()
        return [
            (str(row["category_group"]), str(row["category_key"]))
            for row in rows
        ]

    def category_usage(self) -> dict[tuple[str, str], int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT category_group,category_key,COUNT(*) AS item_count
                FROM problems
                GROUP BY category_group,category_key
                """
            ).fetchall()
        return {
            (str(row["category_group"]), str(row["category_key"])): int(
                row["item_count"]
            )
            for row in rows
        }

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
