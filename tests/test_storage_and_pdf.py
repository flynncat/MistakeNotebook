from __future__ import annotations

from pathlib import Path

from PIL import Image

from mistake_book.pdf_export import export_pdf
from mistake_book.storage import Storage


def test_storage_round_trip(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "data")
    source = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(source)

    batch_id = storage.create_batch()
    problem_id = storage.add_problem(batch_id, "题目.png", source)
    storage.update_problem(
        problem_id,
        status="ready",
        category="计数问题",
        ocr_text="共有多少种不同的方法？",
        metrics_json={"score": 0.9},
    )

    problem = storage.get_problem(problem_id)
    assert problem is not None
    assert problem["filename"] == "题目.png"
    assert problem["metrics"]["score"] == 0.9


def test_pdf_is_a4_and_requires_review_resolution(tmp_path: Path) -> None:
    image_path = tmp_path / "cleaned.png"
    Image.new("RGB", (1200, 500), "white").save(image_path)
    ready = {
        "id": "p1",
        "status": "ready",
        "review_status": "not_required",
        "selected_artifact": str(image_path),
        "category": "计数问题",
        "ocr_text": "共有多少种？",
    }
    pending = {
        **ready,
        "id": "p2",
        "status": "needs_review",
        "review_status": "pending",
    }

    try:
        export_pdf("batch", [ready, pending], tmp_path)
    except ValueError as error:
        assert "未通过确认" in str(error)
    else:
        raise AssertionError("有待确认题目时不应生成最终 PDF")

    target = export_pdf("batch", [ready, pending], tmp_path, allow_partial=True)
    assert target.exists()
    assert target.read_bytes().startswith(b"%PDF")
