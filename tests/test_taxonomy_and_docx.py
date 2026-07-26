from __future__ import annotations

import json
import re
import threading
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mistake_book.app import create_app
from mistake_book.classification import classify_by_rules
from mistake_book.config import Settings
from mistake_book.docx_export import export_docx
from mistake_book.taxonomy import TaxonomyService


COUNTING = "\u8ba1\u6570"
COLORING = "\u67d3\u8272\u95ee\u9898"
UNCATEGORIZED = "\u672a\u5206\u7c7b"


def test_taxonomy_persists_custom_entries_and_archives_without_losing_history(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    historical_group = "\u5386\u53f2\u9886\u57df"
    historical_category = "\u65e7\u9898\u578b"
    custom_group = "\u81ea\u5b9a\u4e49\u9886\u57df"
    custom_category = "\u81ea\u5b9a\u4e49\u9898\u578b"
    service = TaxonomyService(
        data_dir,
        [(historical_group, historical_category)],
    )
    payload = service.payload({(historical_group, historical_category): 2})
    historical = next(
        group for group in payload["groups"] if group["name"] == historical_group
    )
    assert historical["active"] is False
    assert historical["categories"][0]["usage_count"] == 2

    groups = payload["groups"]
    groups.append(
        {
            "id": None,
            "name": custom_group,
            "source": "custom",
            "active": True,
            "categories": [
                {
                    "id": None,
                    "name": custom_category,
                    "source": "custom",
                    "active": True,
                }
            ],
        }
    )
    service.update({"groups": groups}, payload["revision"])
    restarted = TaxonomyService(data_dir)
    assert restarted.is_active_pair(custom_group, custom_category)
    assert restarted.is_known_pair(historical_group, historical_category)
    saved = json.loads((data_dir / "taxonomy.json").read_text(encoding="utf-8"))
    custom = next(group for group in saved["groups"] if group["name"] == custom_group)
    assert custom["id"].startswith("custom-group-")
    assert custom["categories"][0]["id"].startswith("custom-category-")
    assert "usage_count" not in custom


def test_existing_taxonomy_reconciles_new_historical_pairs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    initial = TaxonomyService(data_dir)
    old_revision = initial.payload({})["revision"]

    restarted = TaxonomyService(data_dir, [("legacy-domain", "legacy-type")])

    assert restarted.is_known_pair("legacy-domain", "legacy-type")
    payload = restarted.payload({("legacy-domain", "legacy-type"): 1})
    assert payload["revision"] == old_revision + 1
    historical = next(
        group for group in payload["groups"] if group["name"] == "legacy-domain"
    )
    assert historical["active"] is False
    assert historical["categories"][0]["usage_count"] == 1


def test_custom_names_are_nfkc_normalized_and_mutations_are_serialized(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    service = TaxonomyService(data_dir)
    payload = service.payload({})
    payload["groups"].append(
        {
            "id": None,
            "name": "\uff21\uff22",
            "source": "custom",
            "active": True,
            "categories": [
                {
                    "id": None,
                    "name": "\u9898\u578b\uff11\uff12",
                    "source": "custom",
                    "active": True,
                }
            ],
        }
    )
    service.update({"groups": payload["groups"]}, payload["revision"])
    assert service.is_active_pair("AB", "\u9898\u578b12")

    entered = threading.Event()
    finished = threading.Event()

    def wait_for_guard() -> None:
        entered.set()
        with service.mutation_guard():
            finished.set()

    with service.mutation_guard():
        worker = threading.Thread(target=wait_for_guard)
        worker.start()
        assert entered.wait(timeout=1)
        assert not finished.wait(timeout=0.05)
    worker.join(timeout=1)
    assert finished.is_set()


def test_dynamic_rules_exclude_archived_builtin_category() -> None:
    taxonomy = {
        COUNTING: ("\u6570\u4f4d\u8fdb\u4f4d",),
        "\u7ec4\u5408": ("\u62bd\u5c49\u539f\u7406",),
    }
    result = classify_by_rules(
        "\u4e09\u4e2a\u5706\u7684\u4e03\u4e2a\u533a\u57df"
        "\u6709\u591a\u5c11\u79cd\u67d3\u8272\u65b9\u6cd5",
        taxonomy,
    )
    assert (result.group, result.category) == (UNCATEGORIZED, UNCATEGORIZED)


def _docx_problem(tmp_path: Path, problem_id: str = "p1") -> dict[str, object]:
    artifact_dir = tmp_path / problem_id
    artifact_dir.mkdir()
    question = artifact_dir / "question.png"
    figure = artifact_dir / "figure-selected.png"
    Image.new("RGB", (800, 400), "white").save(question)
    Image.new("RGB", (240, 120), "white").save(figure)
    title = "\u3010\u7ec3\u4e609\u3011"
    return {
        "id": problem_id,
        "filename": f"{problem_id}.png",
        "status": "ready",
        "review_status": "accepted",
        "selected_artifact": str(question),
        "category_group": COUNTING,
        "category_key": COLORING,
        "category": COLORING,
        "ocr_text": title + "\u8ba1\u7b972\u00d710=20\uff0c\u518d\u89c2\u5bdf\u56fe\u5f62\u3002",
        "content_blocks": {
            "version": 1,
            "blocks": [
                {"type": "text", "text": title + "\u8ba1\u7b97"},
                {
                    "type": "latex",
                    "latex": r"2 \times 10=20",
                    "source_text": "2\u00d710=20",
                    "display": False,
                },
                {"type": "text", "text": "\uff0c\u518d\u89c2\u5bdf\u56fe\u5f62\u3002"},
                {
                    "type": "image",
                    "asset": "figure-selected.png",
                    "alt": "question figure",
                },
            ],
        },
    }


def test_docx_export_contains_editable_text_omml_and_figure(tmp_path: Path) -> None:
    problem = _docx_problem(tmp_path)
    result = export_docx("batch-123", [problem], tmp_path / "exports")
    assert result.path.exists()
    with zipfile.ZipFile(result.path) as archive:
        document = archive.read("word/document.xml").decode("utf-8")
        assert "m:oMath" in document
        assert "\u7ec3\u4e609" in document
        assert "Microsoft YaHei" in document
        assert any(name.startswith("word/media/") for name in archive.namelist())


def test_docx_filters_page_noise_keeps_inline_formula_and_disables_auto_update(
    tmp_path: Path,
) -> None:
    problem = _docx_problem(tmp_path)
    problem["ocr_text"] = "\u3010\u7ec3\u4e6010\u3011\u5728r\u8fdb\u5236\u4e2d\u8ba1\u7b97\uff0c\u6c42\u7ed3\u679c\u3002"
    problem["content_blocks"] = {
        "version": 2,
        "blocks": [
            {"type": "text", "text": "\u78ec\u59d1\u5854\u5751", "row_index": 0},
            {
                "type": "text",
                "text": "\u3010\u7ec3\u4e6010\u3011\u5728",
                "row_index": 1,
            },
            {
                "type": "latex",
                "latex": "r",
                "source_text": "r",
                "formula_id": "formula-r",
                "recognition_state": "auto_verified",
                "row_index": 1,
            },
            {
                "type": "text",
                "text": "\u8fdb\u5236\u4e2d\u8ba1\u7b97\uff0c",
                "row_index": 1,
            },
            {"type": "text", "text": "\u6c42\u7ed3\u679c\u3002", "row_index": 2},
            {
                "type": "text",
                "text": "2\u5e74\u7ea7V\u73ed-\u5218\u8001\u5e08\u7b2c12\u8bb2",
                "row_index": 3,
            },
        ],
    }
    result = export_docx("batch", [problem], tmp_path / "exports")
    with zipfile.ZipFile(result.path) as archive:
        document = archive.read("word/document.xml").decode("utf-8")
        settings = archive.read("word/settings.xml").decode("utf-8")

    assert "\u78ec\u59d1\u5854\u5751" not in document
    assert "2\u5e74\u7ea7" not in document
    body_paragraphs = re.findall(r"<w:p[ >].*?</w:p>", document)
    assert any(
        "\u5728" in paragraph
        and "\u8fdb\u5236\u4e2d\u8ba1\u7b97" in paragraph
        and "\u6c42\u7ed3\u679c" in paragraph
        and "m:oMath" in paragraph
        for paragraph in body_paragraphs
    )
    assert "w:updateFields" not in settings


def test_docx_exports_use_unique_paths_and_require_complete_tags(
    tmp_path: Path,
) -> None:
    problem = _docx_problem(tmp_path)
    first = export_docx("batch-123", [problem], tmp_path / "exports")
    second = export_docx("batch-123", [problem], tmp_path / "exports")
    assert first.path != second.path
    assert first.path.is_file()
    assert second.path.is_file()
    assert first.filename == second.filename

    incomplete = {**problem, "category_group": ""}
    with pytest.raises(ValueError, match="\u672a\u901a\u8fc7\u786e\u8ba4"):
        export_docx("batch-123", [incomplete], tmp_path / "exports")


def test_docx_starts_at_most_two_problems_per_page(tmp_path: Path) -> None:
    problems = [_docx_problem(tmp_path, f"p{index}") for index in range(1, 6)]
    result = export_docx("batch", problems, tmp_path / "exports")
    with zipfile.ZipFile(result.path) as archive:
        document = archive.read("word/document.xml").decode("utf-8")

    # One break follows the TOC, then one after problems 2 and 4.
    assert document.count('w:type="page"') == 3


def test_docx_text_formula_check_is_conservative(tmp_path: Path) -> None:
    problem = _docx_problem(tmp_path)
    problem["content_blocks"] = {
        "version": 1,
        "blocks": [{"type": "text", "text": "value_a^{2} remains plain text"}],
    }
    result = export_docx("batch", [problem], tmp_path / "exports")
    assert result.path.is_file()

    problem["content_blocks"] = {
        "version": 1,
        "blocks": [{"type": "text", "text": "contains \u221a2"}],
    }
    with pytest.raises(ValueError, match="\u590d\u6742\u516c\u5f0f"):
        export_docx("batch", [problem], tmp_path / "exports")


def test_docx_export_blocks_pending_unless_partial(tmp_path: Path) -> None:
    ready = _docx_problem(tmp_path, "ready")
    pending = {
        **_docx_problem(tmp_path, "pending"),
        "status": "needs_review",
        "review_status": "pending",
    }
    with pytest.raises(ValueError, match="\u672a\u901a\u8fc7\u786e\u8ba4"):
        export_docx("batch", [ready, pending], tmp_path / "exports")
    result = export_docx(
        "batch",
        [ready, pending],
        tmp_path / "exports",
        allow_partial=True,
    )
    assert result.path.is_file()


def _seed_docx_batch(settings: Settings, storage: object, tmp_path: Path) -> str:
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 40), "white").save(source)
    batch_id = storage.create_batch()
    problem_id = storage.add_problem(batch_id, "source.png", source)
    artifact_dir = settings.data_dir / "files" / batch_id / problem_id
    artifact_dir.mkdir(parents=True)
    question = artifact_dir / "question.png"
    Image.new("RGB", (800, 300), "white").save(question)
    title = "\u3010\u4f8b\u98981\u3011"
    storage.update_problem(
        problem_id,
        status="ready",
        review_status="accepted",
        selected_artifact=str(question),
        selected_kind="reconstructed",
        category_group=COUNTING,
        category=COLORING,
        category_key=COLORING,
        ocr_text=title + "2\u00d710=20\uff0c\u5171\u6709\u591a\u5c11\u79cd\u65b9\u6cd5\uff1f",
        content_blocks_json={
            "version": 1,
            "blocks": [
                {"type": "text", "text": title},
                {
                    "type": "latex",
                    "latex": r"2 \times 10=20",
                    "source_text": "2\u00d710=20",
                },
                {
                    "type": "text",
                    "text": "\uff0c\u5171\u6709\u591a\u5c11\u79cd\u65b9\u6cd5\uff1f",
                },
            ],
        },
    )
    return batch_id


def test_docx_export_api_downloads_with_word_mime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTAKE_BOOK_SESSION_TOKEN", "docx-token")
    settings = Settings.load(tmp_path)
    app = create_app(settings)
    storage = app.state.storage
    batch_id = _seed_docx_batch(settings, storage, tmp_path)
    client = TestClient(app)
    response = client.post(
        f"/api/batches/{batch_id}/export-docx",
        headers={"X-Session-Token": "docx-token"},
        json={"allow_partial": False},
    )
    assert response.status_code == 200
    download = client.get(response.json()["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    )
    assert download.content.startswith(b"PK")


def test_archived_problem_tags_remain_filterable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTAKE_BOOK_SESSION_TOKEN", "docx-token")
    settings = Settings.load(tmp_path)
    app = create_app(settings)
    storage = app.state.storage
    batch_id = _seed_docx_batch(settings, storage, tmp_path)
    batch = storage.get_batch(batch_id)
    assert batch is not None
    problem_id = batch["problems"][0]["id"]
    storage.publish_asset(problem_id, selected_kind="reconstructed")

    client = TestClient(app)
    headers = {"X-Session-Token": "docx-token"}
    taxonomy = client.get("/api/taxonomy", headers=headers).json()
    counting = next(
        group for group in taxonomy["groups"] if group["name"] == COUNTING
    )
    coloring = next(
        category
        for category in counting["categories"]
        if category["name"] == COLORING
    )
    coloring["active"] = False
    saved = client.put(
        "/api/taxonomy",
        headers=headers,
        json={
            "expected_revision": taxonomy["revision"],
            "groups": taxonomy["groups"],
        },
    )
    assert saved.status_code == 200

    filtered = client.get(
        "/api/assets",
        headers=headers,
        params={"category_group": COUNTING, "category": COLORING},
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["tags"]["active"] is False
    assert "\u5206\u7c7b\u5df2\u505c\u7528" in client.get("/").text


def test_docx_export_removes_file_when_export_record_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTAKE_BOOK_SESSION_TOKEN", "docx-token")
    settings = Settings.load(tmp_path)
    app = create_app(settings)
    storage = app.state.storage
    batch_id = _seed_docx_batch(settings, storage, tmp_path)

    def fail_add_export(*args: object, **kwargs: object) -> str:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(storage, "add_export", fail_add_export)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        f"/api/batches/{batch_id}/export-docx",
        headers={"X-Session-Token": "docx-token"},
        json={"allow_partial": False},
    )
    assert response.status_code == 500
    exports_dir = settings.data_dir / "exports"
    assert not list(exports_dir.glob("*.docx"))
