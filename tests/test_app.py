from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

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
    assert "result.items.map(item => problemCard(item, false))" in root.text
    assert "visible.map(item => problemCard(item, true))" in root.text
    assert ".map(problemCard)" not in root.text
    assert "选择文件夹" in root.text
    assert "showDirectoryPicker" in root.text
    assert "一键保存修订并接受" in root.text
    assert "async function saveAllCorrections()" in root.text
    assert "导出 Word DOCX" in root.text
    assert "分类管理" in root.text
    assert "export-docx" in root.text
    assert "处理 Sample 目录" not in root.text

    unauthorized = client.get("/api/batches/missing")
    assert unauthorized.status_code == 403

    authorized = client.get(
        "/api/batches/missing", headers={"X-Session-Token": "test-token"}
    )
    assert authorized.status_code == 404

    categories = client.get(
        "/api/categories", headers={"X-Session-Token": "test-token"}
    )
    assert categories.status_code == 200
    assert any(group["name"] == "计数" for group in categories.json()["groups"])

    taxonomy = client.get(
        "/api/taxonomy", headers={"X-Session-Token": "test-token"}
    )
    assert taxonomy.status_code == 200
    taxonomy_payload = taxonomy.json()
    taxonomy_payload["groups"].append(
        {
            "id": None,
            "name": "自定义",
            "source": "custom",
            "active": True,
            "categories": [
                {
                    "id": None,
                    "name": "新题型",
                    "source": "custom",
                    "active": True,
                }
            ],
        }
    )
    updated = client.put(
        "/api/taxonomy",
        headers={"X-Session-Token": "test-token"},
        json={
            "expected_revision": taxonomy_payload["revision"],
            "groups": taxonomy_payload["groups"],
        },
    )
    assert updated.status_code == 200
    assert any(group["name"] == "自定义" for group in updated.json()["groups"])

    history = client.get(
        "/api/problems?sort=newest&limit=10",
        headers={"X-Session-Token": "test-token"},
    )
    assert history.status_code == 200
    assert history.json()["total"] == 0

    assets = client.get(
        "/api/assets?sort=newest&limit=10",
        headers={"X-Session-Token": "test-token"},
    )
    assert assets.status_code == 200
    assert assets.json()["total"] == 0

    invalid = client.get(
        "/api/problems?sort=random",
        headers={"X-Session-Token": "test-token"},
    )
    assert invalid.status_code == 422


def test_question_preview_never_falls_back_to_dirty_cleaned_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTAKE_BOOK_SESSION_TOKEN", "preview-token")
    settings = Settings.load(tmp_path)
    app = create_app(settings)
    storage = app.state.storage
    source = tmp_path / "source.png"
    Image.new("RGB", (40, 30), "gray").save(source)
    batch_id = storage.create_batch()
    problem_id = storage.add_problem(batch_id, "source.png", source)
    artifact_dir = settings.data_dir / "files" / batch_id / problem_id
    artifact_dir.mkdir(parents=True)
    Image.new("RGB", (40, 30), "red").save(artifact_dir / "cleaned.png")
    client = TestClient(app)
    headers = {"X-Session-Token": "preview-token"}

    pending = client.get(
        f"/api/problems/{problem_id}/image/question",
        headers=headers,
    )
    assert pending.status_code == 404

    Image.new("RGB", (40, 30), "white").save(artifact_dir / "question.png")
    ready = client.get(
        f"/api/problems/{problem_id}/image/question",
        headers=headers,
    )
    assert ready.status_code == 200
    assert ready.headers["cache-control"] == "no-store"


def test_folder_upload_keeps_relative_path_and_allows_reprocessing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTAKE_BOOK_SESSION_TOKEN", "folder-token")
    settings = Settings.load(tmp_path)
    app = create_app(settings)
    app.state.processor.process_batch = lambda _batch_id: None
    client = TestClient(app)
    headers = {"X-Session-Token": "folder-token"}
    buffer = BytesIO()
    Image.new("RGB", (30, 20), "white").save(buffer, format="PNG")
    content = buffer.getvalue()

    first = client.post(
        "/api/batches",
        headers=headers,
        files={"files": ("课本/第一章/source.png", content, "image/png")},
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["import_summary"]["imported_count"] == 1
    assert first_payload["problems"][0]["source_relative_path"] == "课本/第一章/source.png"

    second = client.post(
        "/api/batches",
        headers=headers,
        files={"files": ("重拍/source.png", content, "image/png")},
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["import_summary"]["imported_count"] == 1
    assert "skipped_duplicate_count" not in second_payload["import_summary"]


def test_generated_session_token_survives_restart(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MISTAKE_BOOK_SESSION_TOKEN", raising=False)
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    first = Settings.load(tmp_path)
    second = Settings.load(tmp_path)
    assert first.session_token == second.session_token
    assert (first.data_dir / ".session-token").exists()


def test_problem_api_hides_non_actionable_secondary_ocr_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTAKE_BOOK_SESSION_TOKEN", "diagnostic-token")
    settings = Settings.load(tmp_path)
    app = create_app(settings)
    storage = app.state.storage
    source = tmp_path / "source.png"
    Image.new("RGB", (40, 30), "white").save(source)
    batch_id = storage.create_batch()
    problem_id = storage.add_problem(batch_id, "source.png", source)
    storage.update_problem(
        problem_id,
        metrics_json={
            "review_reasons": ["第二 OCR 未识别到完整题干"],
            "structured_problem": {
                "title": "【例题8】",
                "secondary_raw_text": "(Gi/zi8) 件玩具售优22元",
            },
        },
    )

    response = TestClient(app).get(
        "/api/problems?sort=newest&limit=10",
        headers={"X-Session-Token": "diagnostic-token"},
    )
    assert response.status_code == 200
    assert response.json()["items"][0]["review_diagnostics"] == []


def test_save_correction_resolves_stale_review_reasons(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTAKE_BOOK_SESSION_TOKEN", "review-token")
    settings = Settings.load(tmp_path)
    app = create_app(settings)
    storage = app.state.storage
    source = tmp_path / "source.png"
    Image.new("RGB", (40, 30), "white").save(source)
    batch_id = storage.create_batch()
    problem_id = storage.add_problem(batch_id, "source.png", source)
    storage.update_problem(
        problem_id,
        status="needs_review",
        review_status="pending",
        metrics_json={
            "review_reasons": ["尚未建立该题的人工真值，禁止自动通过"],
            "structured_problem": {
                "review_reasons": ["尚未建立该题的人工真值，禁止自动通过"]
            },
        },
    )

    response = TestClient(app).post(
        f"/api/problems/{problem_id}/review",
        headers={"X-Session-Token": "review-token"},
        json={
            "action": "set_category",
            "category_group": "计数",
            "category": "染色问题",
            "ocr_text": "【例题10】用5种颜色染色，共有多少种方法？",
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ready"
    assert result["review_status"] == "accepted"
    assert result["review_diagnostics"] == []
    assert result["metrics"]["review_reasons"] == []
    assert result["metrics"]["human_verified"] is True
    assert result["metrics"]["resolved_review_reasons"] == [
        "尚未建立该题的人工真值，禁止自动通过"
    ]
    assert result["asset_state"] == "published"
    assert result["selected_kind"] == "reconstructed"

    assets = TestClient(app).get(
        "/api/assets?sort=newest&limit=10",
        headers={"X-Session-Token": "review-token"},
    )
    assert assets.status_code == 200
    assert assets.json()["total"] == 1
    assert assets.json()["items"][0]["id"] == problem_id

    newer_source = tmp_path / "newer.png"
    Image.new("RGB", (40, 30), "gray").save(newer_source)
    newer_batch = storage.create_batch()
    newer_id = storage.add_problem(newer_batch, "newer.png", newer_source)
    assert newer_id is not None
    storage.update_problem(
        newer_id,
        status="needs_review",
        review_status="pending",
        metrics_json={
            "review_reasons": ["尚未建立该题的人工真值，禁止自动通过"]
        },
    )
    replacement = TestClient(app).post(
        f"/api/problems/{newer_id}/review",
        headers={"X-Session-Token": "review-token"},
        json={
            "action": "set_category",
            "category_group": "计数",
            "category": "染色问题",
            "ocr_text": "【例题10】用5种颜色染色，共有多少种方法？",
        },
    )
    assert replacement.status_code == 200
    assert replacement.json()["asset_result"]["replaced_count"] == 1
    assert storage.get_problem(problem_id) is None
    latest_assets = TestClient(app).get(
        "/api/assets?sort=newest&limit=10",
        headers={"X-Session-Token": "review-token"},
    ).json()
    assert latest_assets["total"] == 1
    assert latest_assets["items"][0]["id"] == newer_id


def test_exclude_deletes_problem_record_and_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTAKE_BOOK_SESSION_TOKEN", "delete-token")
    settings = Settings.load(tmp_path)
    app = create_app(settings)
    storage = app.state.storage
    source = tmp_path / "source.png"
    Image.new("RGB", (40, 30), "white").save(source)
    batch_id = storage.create_batch()
    problem_id = storage.add_problem(batch_id, "source.png", source)
    assert problem_id is not None
    problem = storage.get_problem(problem_id)
    assert problem is not None
    copied_source = Path(problem["source_path"])
    artifact_dir = settings.data_dir / "files" / batch_id / problem_id
    artifact_dir.mkdir(parents=True)
    Image.new("RGB", (40, 30), "white").save(artifact_dir / "question.png")
    storage.update_problem(
        problem_id,
        status="needs_review",
        review_status="pending",
    )

    response = TestClient(app).post(
        f"/api/problems/{problem_id}/review",
        headers={"X-Session-Token": "delete-token"},
        json={"action": "exclude"},
    )
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert storage.get_problem(problem_id) is None
    assert not copied_source.exists()
    assert not artifact_dir.exists()
