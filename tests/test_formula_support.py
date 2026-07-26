from __future__ import annotations

import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from mistake_book.app import create_app
from mistake_book.config import Settings
from mistake_book.docx_export import export_docx
from mistake_book.formula_math import (
    FormulaValidationError,
    convert_latex,
    latex_to_omml,
)
from mistake_book.formula_pipeline import _merge_components, process_formulas
from mistake_book.formula_runtime import FormulaRuntime
from mistake_book.formula_runtime import FormulaRuntimeUnavailable, _verify_model


@pytest.mark.parametrize(
    "latex",
    [
        r"\frac{3}{5}",
        r"\sqrt{x^2+1}",
        r"\int_0^1 x^2\,dx",
        r"\sum_{n=1}^{\infty}\frac{1}{n^2}",
        r"\begin{bmatrix}1&2\\3&4\end{bmatrix}",
        r"\begin{cases}x^2&x>0\\0&x\le0\end{cases}",
    ],
)
def test_formula_conversion_supports_complex_printed_math(latex: str) -> None:
    converted = convert_latex(latex)
    assert converted.latex == latex
    assert converted.mathml.startswith("<math")
    assert latex_to_omml(latex).startswith("<m:oMath")


@pytest.mark.parametrize(
    "latex",
    [
        r"\input{/etc/passwd}",
        r"\href{https://example.com}{x}",
        r"\newcommand{\x}{1}",
        r"\frac{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{1}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}",
    ],
)
def test_formula_conversion_rejects_unsafe_or_excessive_latex(latex: str) -> None:
    with pytest.raises(FormulaValidationError):
        convert_latex(latex)


def test_formula_components_are_merged_in_geometric_order() -> None:
    lines = [
        {"text": "first", "box": [0.1, 0.8, 0.2, 0.05]},
        {"text": "last", "box": [0.7, 0.8, 0.2, 0.05]},
    ]
    formula = {
        "formula_id": "formula-a",
        "type": "latex",
        "latex": r"\frac{3}{5}",
        "source_box": [350, 140, 520, 220],
        "recognition_state": "auto_verified",
    }
    blocks = _merge_components(lines, [formula], 1000, 1000)
    assert [block["type"] for block in blocks] == ["text", "latex", "text"]
    assert blocks[1]["formula_id"] == "formula-a"
    assert blocks[1]["display"] is False


def test_formula_merger_keeps_single_variables_as_searchable_text() -> None:
    lines = [
        {
            "text": "\u3010\u7ec3\u4e609\u3011 \u5728r\u8fdb\u5236\u4e2d\u6709\u7b97\u5f0f\uff0c\u5176\u4e2d\u7ed3\u679c\u8f6c\u4e3a\u5341\u8fdb\u5236\uff0c",
            "box": [0.08, 0.72, 0.80, 0.12],
        },
        {"text": "\u6c42r\u7684\u503c\u3002", "box": [0.17, 0.68, 0.10, 0.05]},
    ]
    formulas = [
        {
            "formula_id": "equation",
            "type": "latex",
            "latex": r"(130)_r+(13)_r=(48)_{10}",
            "source_box": [500, 160, 850, 275],
            "recognition_state": "auto_verified",
        },
        {
            "formula_id": "answer-r",
            "type": "latex",
            "latex": "r",
            "source_box": [205, 270, 235, 325],
            "recognition_state": "auto_verified",
        },
    ]

    blocks = _merge_components(lines, formulas, 1000, 1000)

    equation = next(block for block in blocks if block.get("formula_id") == "equation")
    assert equation["row_index"] == 0
    assert not any(block.get("formula_id") == "answer-r" for block in blocks)
    assert any(
        block.get("type") == "text"
        and "\u6c42r\u7684\u503c" in str(block.get("text") or "")
        for block in blocks
    )


def test_formula_merger_replaces_only_an_exact_equation_span() -> None:
    lines = [
        {
            "text": (
                "\u3010\u7ec3\u4e607\u3011 \u5df2\u77e5\u6b63\u6574\u6570 N \u7684"
                "\u516b\u8fdb\u5236\u8868\u793a\u4e3aN=\uff0812345654321\uff09&\uff0c"
                "\u90a3\u4e48\u5728\u5341\u8fdb\u5236\u4e0b\uff0cN\u9664\u4ee57"
                "\u7684\u4f59\u6570\u4e0eN\u9664\u4ee59\u7684\u4f59\u6570\u4e4b\u548c"
                "\u662f\u591a\u5c11\uff1f"
            ),
            "box": [0.05, 0.70, 0.90, 0.10],
        }
    ]
    formulas = [
        {
            "formula_id": "variable-n",
            "type": "latex",
            "latex": "N",
            "source_box": [250, 200, 280, 260],
            "recognition_state": "auto_verified",
        },
        {
            "formula_id": "radix-equation",
            "type": "latex",
            "latex": (
                r"N \, {=} \, \big ( 1 2 3 4 5 6 5 4 3 2 1 \big )_{8}"
            ),
            "source_box": [450, 200, 700, 260],
            "recognition_state": "auto_verified",
        },
    ]

    blocks = _merge_components(lines, formulas, 1000, 1000)

    assert [block["type"] for block in blocks] == ["text", "latex", "text"]
    assert blocks[0]["text"].endswith("\u516b\u8fdb\u5236\u8868\u793a\u4e3a")
    assert blocks[1]["formula_id"] == "radix-equation"
    assert blocks[2]["text"].startswith("\uff0c\u90a3\u4e48\u5728\u5341\u8fdb\u5236\u4e0b")
    assert "variable-n" not in {
        block.get("formula_id")
        for block in blocks
    }


def test_formula_conversion_accepts_unimernet_delimiter_sizing() -> None:
    latex = (
        r"\bigl ( 1 3 0 \bigr ) _ { r } + "
        r"\bigl ( 1 3 \bigr ) _ { r } = \bigl ( 4 8 \bigr ) _ { 1 0 }"
    )
    assert convert_latex(latex).latex.endswith(r"\bigl ( 4 8 \bigr )_{1 0}")


def test_formula_preview_and_atomic_edit_api(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MISTAKE_BOOK_SESSION_TOKEN", "formula-token")
    settings = Settings.load(tmp_path)
    app = create_app(settings)
    storage = app.state.storage
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 60), "white").save(source)
    batch_id = storage.create_batch()
    problem_id = storage.add_problem(batch_id, "source.png", source)
    artifact_dir = settings.data_dir / "files" / batch_id / problem_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (30, 20), "white").save(
        artifact_dir / "formula-01-original.png"
    )
    Image.new("RGB", (30, 20), "white").save(
        artifact_dir / "formula-01-clean.png"
    )
    Image.new("RGB", (100, 60), "white").save(artifact_dir / "question.png")
    original_question = (artifact_dir / "question.png").read_bytes()
    blocks = {
        "version": 2,
        "blocks": [
            {
                "formula_id": "formula-test",
                "type": "latex",
                "latex": r"\frac{3}{8}",
                "model_latex": r"\frac{3}{8}",
                "source_text": "3/8",
                "source_box": [1, 2, 30, 20],
                "original_crop_asset": "formula-01-original.png",
                "clean_crop_asset": "formula-01-clean.png",
                "recognition_state": "needs_review",
            }
        ],
    }
    storage.update_problem(
        problem_id,
        status="needs_review",
        review_status="pending",
        selected_artifact=str(artifact_dir / "question.png"),
        category_group="group",
        category="type",
        category_key="type",
        ocr_text="formula",
        content_blocks_version=2,
        content_blocks_json=blocks,
        metrics_json={"review_reasons": ["Formula 1 requires review"]},
    )
    problem = storage.get_problem(problem_id)
    assert problem is not None
    headers = {"X-Session-Token": "formula-token"}
    client = TestClient(app)

    page = client.get("/")
    assert "\u4fdd\u6301\u56fe\u50cf\u8bc6\u522b" in page.text
    assert "\u786e\u8ba4\u4f7f\u7528\u6b64\u516c\u5f0f" in page.text
    assert "\u6062\u590d\u6a21\u578b\u8bc6\u522b\u503c" not in page.text
    assert 'data-formula-mode="image"' in page.text
    assert 'data-formula-mode="latex"' in page.text
    assert "setFormulaChoiceState" in page.text

    preview = client.post(
        "/api/formulas/preview",
        headers=headers,
        json={"latex": r"\frac{3}{5}"},
    )
    assert preview.status_code == 200
    assert "<mfrac>" in preview.json()["mathml"]

    bypass = client.post(
        f"/api/problems/{problem_id}/review",
        headers=headers,
        json={"action": "set_category", "ocr_text": "formula"},
    )
    assert bypass.status_code == 409

    updated = client.put(
        f"/api/problems/{problem_id}/formulas",
        headers=headers,
        json={
            "updated_at": problem["updated_at"],
            "formulas": [
                {"formula_id": "formula-test", "latex": r"\frac{3}{5}"}
            ],
        },
    )
    assert updated.status_code == 200
    body = updated.json()
    assert body["formulas"][0]["latex"] == r"\frac{3}{5}"
    assert body["formulas"][0]["recognition_state"] == "human_verified"
    assert r"\(\frac{3}{5}\)" in body["ocr_text"]
    assert (artifact_dir / "question.png").read_bytes() != original_question
    assert body["formulas"][0]["original_url"].endswith(
        "formula-01-original.png?token=formula-token&v="
        + body["updated_at"].replace("+", "%2B")
    )

    kept_image = client.put(
        f"/api/problems/{problem_id}/formulas",
        headers=headers,
        json={
            "updated_at": body["updated_at"],
            "formulas": [
                {
                    "formula_id": "formula-test",
                    "mode": "image",
                }
            ],
        },
    )
    assert kept_image.status_code == 200
    kept_body = kept_image.json()
    assert (
        kept_body["formulas"][0]["recognition_state"]
        == "human_verified_image"
    )
    assert "\uff3b\u516c\u5f0f\u56fe\u50cf\uff3d" in kept_body["ocr_text"]
    formula_gate = client.post(
        f"/api/problems/{problem_id}/review",
        headers=headers,
        json={"action": "accept_cleaned"},
    )
    assert formula_gate.status_code == 422
    assert "\u5206\u7c7b" in formula_gate.json()["detail"]

    stale = client.put(
        f"/api/problems/{problem_id}/formulas",
        headers=headers,
        json={
            "updated_at": problem["updated_at"],
            "formulas": [{"formula_id": "formula-test", "latex": "x"}],
        },
    )
    assert stale.status_code == 409

    active_group = app.state.taxonomy.active_payload()[0]
    storage.update_problem(
        problem_id,
        category_group=active_group["name"],
        category=active_group["categories"][0],
        category_key=active_group["categories"][0],
    )
    monkeypatch.setattr(storage, "update_problem_cas", lambda *args, **kwargs: False)
    concurrent_accept = client.post(
        f"/api/problems/{problem_id}/review",
        headers=headers,
        json={"action": "accept_cleaned"},
    )
    assert concurrent_accept.status_code == 409

    changed_text = client.post(
        f"/api/problems/{problem_id}/review",
        headers=headers,
        json={"action": "set_category", "ocr_text": "changed formula text"},
    )
    assert changed_text.status_code == 409


def test_formula_model_integrity_check_rejects_wrong_hash(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    with pytest.raises(FormulaRuntimeUnavailable):
        _verify_model(model, 5, "0" * 64)


def test_closed_formula_runtime_cannot_restart(tmp_path: Path) -> None:
    runtime = FormulaRuntime(tmp_path)
    runtime.close()
    with pytest.raises(FormulaRuntimeUnavailable):
        runtime.detect(tmp_path / "missing.png")


def test_docx_contains_native_complex_omml(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "p1"
    artifact_dir.mkdir()
    question = artifact_dir / "question.png"
    Image.new("RGB", (100, 60), "white").save(question)
    problem = {
        "id": "p1",
        "filename": "p1.png",
        "selected_artifact": str(question),
        "category_group": "group",
        "category": "type",
        "category_key": "type",
        "ocr_text": "complex formulas",
        "status": "ready",
        "review_status": "accepted",
        "metrics": {},
        "content_blocks": {
            "version": 2,
            "blocks": [
                {
                    "type": "latex",
                    "formula_id": "f1",
                    "latex": r"\frac{3}{5}",
                    "recognition_state": "human_verified",
                },
                {
                    "type": "latex",
                    "formula_id": "f2",
                    "latex": r"\sqrt{x^2+1}",
                    "recognition_state": "human_verified",
                },
                {
                    "type": "latex",
                    "formula_id": "f3",
                    "latex": r"\begin{bmatrix}1&2\\3&4\end{bmatrix}",
                    "recognition_state": "human_verified",
                },
            ],
        },
    }
    result = export_docx("batch", [problem], tmp_path / "exports")
    with zipfile.ZipFile(result.path) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "<m:f>" in document_xml
    assert "<m:rad>" in document_xml
    assert "<m:m>" in document_xml


def test_docx_uses_crop_for_human_verified_image_formula(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "image-formula"
    artifact_dir.mkdir()
    question = artifact_dir / "question.png"
    crop = artifact_dir / "formula-01-clean.png"
    Image.new("RGB", (100, 60), "white").save(question)
    Image.new("RGB", (60, 20), "white").save(crop)
    problem = {
        "id": "image-formula",
        "filename": "image-formula.png",
        "selected_artifact": str(question),
        "category_group": "group",
        "category": "type",
        "category_key": "type",
        "ocr_text": "\u3010\u7ec3\u4e601\u3011\u89c2\u5bdf\u516c\u5f0f\uff0c\u6c42\u7ed3\u679c\u3002",
        "status": "ready",
        "review_status": "accepted",
        "metrics": {"structured_problem": {"title": "\u3010\u7ec3\u4e601\u3011"}},
        "content_blocks": {
            "version": 2,
            "blocks": [
                {
                    "type": "text",
                    "text": "\u3010\u7ec3\u4e601\u3011\u89c2\u5bdf",
                    "row_index": 0,
                },
                {
                    "type": "latex",
                    "formula_id": "f1",
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
        },
    }
    result = export_docx("batch", [problem], tmp_path / "exports")
    with zipfile.ZipFile(result.path) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
        media = [name for name in archive.namelist() if name.startswith("word/media/")]
    assert media
    assert "<m:oMath" not in document_xml


def test_dsc6580_tiny_recognizes_all_four_fractions(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    sample = root / "Sample" / "Latex" / "_DSC6580.JPG"
    runtime = FormulaRuntime(root)
    if not sample.is_file() or not runtime.available:
        runtime.close()
        pytest.skip("local formula model or acceptance sample is unavailable")
    try:
        result = process_formulas(
            runtime,
            sample,
            sample,
            [],
            tmp_path,
            fallback_content_blocks={"version": 1, "blocks": []},
        )
    finally:
        runtime.close()
    formulas = [
        block["latex"]
        for block in result.content_blocks["blocks"]
        if block["type"] == "latex"
    ]
    assert formulas == [
        r"\frac{3}{5}",
        r"\frac{2}{3}",
        r"\frac{5}{8}",
        r"\frac{7}{9}",
    ]
    assert result.metrics["formula_states"]["auto_verified"] == 4
