from __future__ import annotations

import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mistake_book.app import create_app
from mistake_book.config import Settings
from mistake_book.content_blocks import (
    build_content_blocks_with_fallbacks,
    split_text_and_latex,
)
from mistake_book.markdown_export import (
    build_export_basename,
    export_markdown,
    markdown_exportable,
)


def _problem(
    problem_id: str,
    image_path: Path,
    *,
    group: str = "\u8ba1\u6570",
    category: str = "\u6392\u5217\u7ec4\u5408",
    text: str = "\u3010\u4f8b\u98981\u3011\u4e00\u5171\u6709\u591a\u5c11\u79cd\u65b9\u6cd5\uff1f",
) -> dict:
    return {
        "id": problem_id,
        "filename": "source.png",
        "source_relative_path": "chapter/source.png",
        "review_status": "accepted",
        "status": "ready",
        "selected_kind": "reconstructed",
        "selected_artifact": str(image_path),
        "category_group": group,
        "category_key": category,
        "category": category,
        "ocr_text": text,
        "accepted_at": "2026-07-23T12:00:00+00:00",
        "updated_at": "2026-07-23T12:00:00+00:00",
        "content_fingerprint": "a" * 64,
        "metrics": {
            "structured_problem": {
                "title": "\u3010\u4f8b\u98981\u3011",
                "figure": "none",
            }
        },
    }


def test_markdown_export_keeps_text_searchable_without_full_question_images(
    tmp_path: Path,
) -> None:
    first_image = tmp_path / "first" / "question.png"
    second_image = tmp_path / "second" / "question.png"
    first_image.parent.mkdir()
    second_image.parent.mkdir()
    Image.new("RGB", (80, 40), "white").save(first_image)
    Image.new("RGB", (80, 40), "white").save(second_image)
    problems = [
        _problem("a" * 32, first_image),
        _problem(
            "b" * 32,
            second_image,
            group="\u5e94\u7528\u9898",
            category="\u884c\u7a0b\u95ee\u9898",
            text="\u3010\u4f8b\u98982\u3011\u706b\u8f66\u6bcf\u79d2\u884c\u591a\u5c11\u7c73\uff1f",
        ),
    ]

    result = export_markdown(
        "batch123456",
        problems,
        tmp_path / "exports",
        generated_at=datetime(2026, 7, 24, 11, 23, tzinfo=UTC),
        random_code="K7M2",
    )

    with zipfile.ZipFile(result.path) as archive:
        names = archive.namelist()
        assert len(names) == 2
        assert names[0] == result.note_name
        assert names[1] == ".obsidian/snippets/mistake-book-print.css"
        assert all(".." not in name and not name.startswith("/") for name in names)
        note = archive.read(names[0]).decode("utf-8")
        assert "type: olympiad-mistake-book" in note
        assert "problem_count: 2" in note
        assert "\u5c0f\u5965\u9519\u9898\u96c6" in note
        assert "\u6211\u7684\u89e3\u7b54" not in note
        assert "![[attachments/" not in note
        assert "[[#^mb-" in note
        assert " ^mb-" in note
        assert '<div class="mb-answer-space mb-normal"></div>' in note
        assert result.filename.startswith(
            "20260724-1123-"
        ) and result.filename.endswith("-K7M2.zip")


def test_markdown_filters_page_noise_and_joins_inline_formula_rows(
    tmp_path: Path,
) -> None:
    image = tmp_path / "problem" / "question.png"
    image.parent.mkdir()
    Image.new("RGB", (80, 40), "white").save(image)
    problem = _problem(
        "f" * 32,
        image,
        text="\u3010\u7ec3\u4e6010\u3011\u5728r\u8fdb\u5236\u4e2d\u8ba1\u7b97\uff0c\u6c42\u7ed3\u679c\u3002",
    )
    problem["metrics"]["structured_problem"]["title"] = "\u3010\u7ec3\u4e6010\u3011"
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

    result = export_markdown(
        "batch",
        [problem],
        tmp_path / "exports",
        random_code="K7M2",
    )
    with zipfile.ZipFile(result.path) as archive:
        note = archive.read(result.note_name).decode("utf-8")

    assert "\u5728$r$\u8fdb\u5236\u4e2d\u8ba1\u7b97\uff0c\u6c42\u7ed3\u679c\u3002" in note
    assert "\u78ec\u59d1\u5854\u5751" not in note
    assert "2\u5e74\u7ea7" not in note


def test_markdown_uses_attachment_for_human_verified_image_formula(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "image-formula"
    artifact_dir.mkdir()
    question = artifact_dir / "question.png"
    crop = artifact_dir / "formula-01-clean.png"
    Image.new("RGB", (80, 40), "white").save(question)
    Image.new("RGB", (40, 20), "white").save(crop)
    problem = _problem(
        "e" * 32,
        question,
        text="\u3010\u4f8b\u98981\u3011\u89c2\u5bdf\u516c\u5f0f\uff0c\u6c42\u7ed3\u679c\u3002",
    )
    problem["content_blocks"] = {
        "version": 2,
        "blocks": [
            {
                "type": "text",
                "text": "\u3010\u4f8b\u98981\u3011\u89c2\u5bdf",
                "row_index": 0,
            },
            {
                "type": "latex",
                "formula_id": "formula-image",
                "latex": r"\frac{3}{5}",
                "source_text": "3/5",
                "clean_crop_asset": crop.name,
                "recognition_state": "human_verified_image",
                "row_index": 0,
            },
            {
                "type": "text",
                "text": "\uff0c\u6c42\u7ed3\u679c\u3002",
                "row_index": 0,
            },
        ],
    }

    result = export_markdown(
        "batch",
        [problem],
        tmp_path / "exports",
        random_code="K7M2",
    )
    with zipfile.ZipFile(result.path) as archive:
        note = archive.read(result.note_name).decode("utf-8")
        attachments = [
            name for name in archive.namelist() if name.startswith("attachments/")
        ]

    assert attachments
    assert "![[attachments/" in note
    assert r"$\frac{3}{5}$" not in note


def test_markdown_export_requires_accepted_reconstructed_question(
    tmp_path: Path,
) -> None:
    image = tmp_path / "normalized.png"
    Image.new("RGB", (80, 40), "white").save(image)
    problem = _problem("c" * 32, image)
    problem["selected_kind"] = "normalized"

    assert markdown_exportable(problem) is False
    with pytest.raises(ValueError, match="Markdown"):
        export_markdown("batch", [problem], tmp_path / "exports")


def test_markdown_export_rejects_duplicate_archive_entries(tmp_path: Path) -> None:
    image = tmp_path / "question.png"
    figure = tmp_path / "figure-selected.png"
    Image.new("RGB", (80, 40), "white").save(image)
    Image.new("RGB", (40, 40), "white").save(figure)
    problem = _problem("d" * 32, image)
    problem["content_blocks"] = {
        "version": 1,
        "blocks": [
            {"type": "text", "text": problem["ocr_text"]},
            {"type": "image", "asset": "figure-selected.png"},
        ],
    }

    with pytest.raises(ValueError, match="Duplicate attachment"):
        export_markdown("batch", [problem, dict(problem)], tmp_path / "exports")


def test_formula_text_becomes_inline_latex_and_name_counts_top_three() -> None:
    blocks = split_text_and_latex(
        "\u65b9\u683c\u662f2\u00d710\uff0c\u6bd4\u662f37:25\uff0c\u6c42\u7b54\u6848\u3002"
    )
    assert [block["type"] for block in blocks] == [
        "text",
        "latex",
        "text",
        "latex",
        "text",
    ]
    assert blocks[1]["latex"] == r"2 \times 10"
    problems = [
        {"category_group": "\u8ba1\u6570"},
        {"category_group": "\u8ba1\u6570"},
        {"category_group": "\u5e94\u7528"},
        {"category_group": "\u7ec4\u5408"},
        {"category_group": "\u6570\u8bba"},
    ]
    name = build_export_basename(
        problems,
        generated_at=datetime(2026, 7, 24, 11, 23, tzinfo=UTC),
        random_code="K7M2",
    )
    assert name == "20260724-1123-\u8ba1\u65702-\u5e94\u75281-\u6570\u8bba1-\u7b491\u7c7b-K7M2"


def test_unsupported_formula_is_replaced_by_positioned_crop(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned.png"
    Image.new("RGB", (1000, 600), "white").save(cleaned)
    formula = "\u221a16=4"
    text = f"\u8ba1\u7b97\uff1a{formula}\uff0c\u6c42\u7ed3\u679c\u3002"
    model = build_content_blocks_with_fallbacks(
        text,
        [{"text": formula, "box": [0.2, 0.55, 0.3, 0.1]}],
        cleaned,
        tmp_path,
    )

    assert [block["type"] for block in model["blocks"]] == [
        "text",
        "image",
        "text",
    ]
    assert model["blocks"][0]["text"] == "\u8ba1\u7b97\uff1a"
    assert model["blocks"][2]["text"] == "\uff0c\u6c42\u7ed3\u679c\u3002"
    assert (tmp_path / model["blocks"][1]["asset"]).exists()


def test_markdown_export_api_and_frontend_button(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTAKE_BOOK_SESSION_TOKEN", "markdown-token")
    settings = Settings.load(tmp_path)
    app = create_app(settings)
    storage = app.state.storage
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 40), "white").save(source)
    batch_id = storage.create_batch()
    problem_id = storage.add_problem(batch_id, "source.png", source)
    artifact = settings.data_dir / "files" / batch_id / problem_id / "question.png"
    artifact.parent.mkdir(parents=True)
    Image.new("RGB", (800, 300), "white").save(artifact)
    storage.update_problem(
        problem_id,
        status="ready",
        review_status="accepted",
        selected_artifact=str(artifact),
        selected_kind="reconstructed",
        category_group="\u8ba1\u6570",
        category="\u6392\u5217\u7ec4\u5408",
        category_key="\u6392\u5217\u7ec4\u5408",
        ocr_text="\u3010\u4f8b\u98981\u3011\u4e00\u5171\u6709\u591a\u5c11\u79cd\u65b9\u6cd5\uff1f",
        accepted_at="2026-07-23T12:00:00+00:00",
    )
    client = TestClient(app)
    headers = {"X-Session-Token": "markdown-token"}

    page = client.get("/")
    assert 'id="export-markdown"' in page.text
    batch = client.get(f"/api/batches/{batch_id}", headers=headers)
    assert batch.json()["problems"][0]["markdown_exportable"] is True
    response = client.post(
        f"/api/batches/{batch_id}/export-markdown",
        headers=headers,
        json={"allow_partial": False},
    )
    assert response.status_code == 200
    download = client.get(response.json()["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert download.content.startswith(b"PK")
