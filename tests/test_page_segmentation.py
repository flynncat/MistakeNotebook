from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mistake_book.app import Processor
from mistake_book.config import Settings
from mistake_book.page_segmentation import (
    PageRegion,
    PageSegmenter,
    PageSplit,
    detect_question_anchors,
)
from mistake_book.recognition import MacVisionOCR
from mistake_book.reconstruction import build_structured_problem
from mistake_book.storage import Storage


def _line(text: str, y: float, *, x: float = 0.1, confidence: float = 0.8) -> dict:
    return {
        "text": text,
        "box": [x, y, 0.72, 0.035],
        "confidence": confidence,
    }


def test_anchor_detection_recovers_ocr_confused_example_number() -> None:
    lines = [
        _line("\u4f8b6\u3001\u7b2c\u4e00\u9898\uff1f", 0.86),
        _line("\u4f8b7\u3001\u7b2c\u4e8c\u9898\uff1f", 0.66),
        _line("\u52178\u3001\u7b2c\u4e09\u9898\uff1f", 0.46),
        _line("1\u3001\u7b2c\u56db\u9898\uff1f", 0.26),
    ]

    anchors = detect_question_anchors(lines, (1000, 1400))

    assert [anchor.label for anchor in anchors] == [
        "\u3010\u4f8b\u98986\u3011",
        "\u3010\u4f8b\u98987\u3011",
        "\u3010\u4f8b\u98988\u3011",
        "\u3010\u7b2c1\u9898\u3011",
    ]


def test_anchor_detection_supports_preview_exercises() -> None:
    lines = [
        _line("\u3010\u9884\u4e605\u3011 first?", 0.8),
        _line("\u3010\u9884\u4e606\u3011 second?", 0.4),
    ]
    anchors = detect_question_anchors(lines, (1000, 1400))
    assert [anchor.label for anchor in anchors] == [
        "\u3010\u9884\u4e605\u3011",
        "\u3010\u9884\u4e606\u3011",
    ]


def test_number_lists_are_not_treated_as_question_anchors() -> None:
    lines = [
        _line("\u3010\u7ec3\u4e607\u3011\u7b2c\u4e00\u9053\u9898", 0.88),
        _line("\u6c42\u7ed3\u679c\uff1f", 0.82, x=0.18),
        _line("\u3010\u7ec3\u4e608\u3011\u5c06\u6570\u5b57\u4ece\u5c0f\u5230\u5927\u6392\u5217\u53ef\u5f97", 0.58),
        _line("2\u30013\u300110\u300111\u300112\u300113\u300120\u300121\u300122\u300123\u3001", 0.52, x=0.18),
        _line("\u8bf7\u95ee123\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f", 0.46, x=0.18),
        _line("\u3010\u7ec3\u4e609\u3011\u5c06\u6570\u5b57\u4ece\u5c0f\u5230\u5927\u6392\u5217\u53ef\u5f97", 0.28),
        _line("1\u30012\u30013\u30014\u300110\u300111\u300112\u300113\u3001\u8bf7\u95ee\u7ed3\u679c\uff1f", 0.22, x=0.18),
    ]

    anchors = detect_question_anchors(lines, (1000, 1400))

    assert [anchor.label for anchor in anchors] == [
        "\u3010\u7ec3\u4e607\u3011",
        "\u3010\u7ec3\u4e608\u3011",
        "\u3010\u7ec3\u4e609\u3011",
    ]


def test_page_segmenter_keeps_three_questions_with_number_lists(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.fromarray(np.full((1400, 1000, 3), 245, dtype=np.uint8)).save(source)
    lines = [
        _line("\u3010\u7ec3\u4e607\u3011\u7b2c\u4e00\u9053\u9898", 0.88),
        _line("\u6c42\u7ed3\u679c\uff1f", 0.82, x=0.18),
        _line("\u3010\u7ec3\u4e608\u3011\u5c06\u6570\u5b57\u4ece\u5c0f\u5230\u5927\u6392\u5217\u53ef\u5f97", 0.58),
        _line("2\u30013\u300110\u300111\u300112\u300113\u300120\u300121\u300122\u300123\u3001", 0.52, x=0.18),
        _line("\u8bf7\u95ee123\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f", 0.46, x=0.18),
        _line("\u3010\u7ec3\u4e609\u3011\u5c06\u6570\u5b57\u4ece\u5c0f\u5230\u5927\u6392\u5217\u53ef\u5f97", 0.28),
        _line("1\u30012\u30013\u30014\u300110\u300111\u300112\u300113\u3001\u8bf7\u95ee\u7ed3\u679c\uff1f", 0.22, x=0.18),
    ]

    class FakeOCR:
        @staticmethod
        def available() -> bool:
            return True

        @staticmethod
        def recognize(_path: Path):
            return "", 0.9, lines

    result = PageSegmenter(FakeOCR()).split(source, tmp_path / "split", 0)

    assert result is not None
    assert [region.label for region in result.regions] == [
        "\u3010\u7ec3\u4e607\u3011",
        "\u3010\u7ec3\u4e608\u3011",
        "\u3010\u7ec3\u4e609\u3011",
    ]


def test_page_segmenter_creates_one_padded_crop_per_question(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.fromarray(np.full((1400, 1000, 3), 245, dtype=np.uint8)).save(source)
    lines = [
        _line("\u4f8b6\u3001\u7532\u8f66\u6bcf\u79d224\u7c73", 0.88),
        _line("\u8981\u884c\u591a\u5c11\u8def\u7a0b\uff1f", 0.82),
        _line("\u4f8b7\u3001\u5feb\u8f66\u6bcf\u79d217\u7c73", 0.58),
        _line("\u8f66\u957f\u5206\u522b\u662f\u591a\u5c11\uff1f", 0.52),
        _line("1\u3001\u706b\u8f66\u957f190\u7c73", 0.28),
        _line("\u6bcf\u79d2\u884c\u591a\u5c11\u7c73\uff1f", 0.22),
    ]

    class FakeOCR:
        @staticmethod
        def available() -> bool:
            return True

        @staticmethod
        def recognize(_path: Path):
            return "", 0.9, lines

    result = PageSegmenter(FakeOCR()).split(source, tmp_path / "split", 0)

    assert result is not None
    assert [region.label for region in result.regions] == [
        "\u3010\u4f8b\u98986\u3011",
        "\u3010\u4f8b\u98987\u3011",
        "\u3010\u7b2c1\u9898\u3011",
    ]
    for region in result.regions:
        assert region.source_path.exists()
        with Image.open(region.source_path) as crop:
            assert crop.height >= round(crop.width * 0.58)


def test_page_segmenter_accepts_solve_prompts_without_question_marks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.fromarray(np.full((1400, 1000, 3), 245, dtype=np.uint8)).save(source)
    lines = [
        _line("\u3010\u7ec3\u4e609\u3011\u5728r\u8fdb\u5236\u4e2d\u6709\u4e00\u4e2a\u7b97\u5f0f", 0.86),
        _line("\u6c42r\u7684\u503c\u3002", 0.80),
        _line("\u3010\u7ec3\u4e6010\u3011\u5728r\u8fdb\u5236\u4e2d\u6709\u4e00\u4e2a\u7b97\u5f0f", 0.48),
        _line("\u6c42r\u7684\u503c\u3002", 0.42),
    ]

    class FakeOCR:
        @staticmethod
        def available() -> bool:
            return True

        @staticmethod
        def recognize(_path: Path):
            return "", 0.9, lines

    result = PageSegmenter(FakeOCR()).split(source, tmp_path / "split", 0)

    assert result is not None
    assert [region.label for region in result.regions] == [
        "\u3010\u7ec3\u4e609\u3011",
        "\u3010\u7ec3\u4e6010\u3011",
    ]
    assert all(region.ocr_text.endswith("\u3002") for region in result.regions)


def test_replacing_page_record_keeps_independent_child_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    settings = Settings.load(tmp_path)
    storage = Storage(settings.data_dir)
    processor = Processor(settings, storage)
    source = tmp_path / "page.png"
    crop1 = tmp_path / "crop1.png"
    crop2 = tmp_path / "crop2.png"
    Image.new("RGB", (100, 100), "white").save(source)
    Image.new("RGB", (100, 100), "white").save(crop1)
    Image.new("RGB", (100, 100), "white").save(crop2)
    batch_id = storage.create_batch()
    parent_id = storage.add_problem(batch_id, "page.png", source)
    problem = storage.get_problem(parent_id)
    assert problem is not None
    split = PageSplit(
        regions=[
            PageRegion(
                "\u3010\u4f8b\u98986\u3011",
                crop1,
                (0, 0, 100, 50),
                200.0,
                "\u4f8b6",
                "\u4f8b6\u3001\u9898\u5e72\uff1f",
            ),
            PageRegion(
                "\u3010\u4f8b\u98987\u3011",
                crop2,
                (0, 50, 100, 100),
                180.0,
                "\u4f8b7",
                "\u4f8b7\u3001\u9898\u5e72\uff1f",
            ),
        ],
        metrics={"mode": "multi_question_page", "region_count": 2},
    )

    class FakeSegmenter:
        @staticmethod
        def split(*_args):
            return split

    processor.page_segmenter = FakeSegmenter()
    child_ids = processor._split_problem(problem)

    assert len(child_ids) == 2
    assert storage.get_problem(parent_id) is None
    children = storage.get_problems(batch_id)
    assert [child["page_index"] for child in children] == [1, 2]
    assert all(child["parent_source_id"] == parent_id for child in children)
    assert children[0]["split_ocr_text"].endswith("\uff1f")
    assert storage.get_batch(batch_id)["status"] == "processing"


def test_title_hint_supports_numbered_page_question() -> None:
    structured = build_structured_problem(
        "1\u3001\u706b\u8f66\u957f190\u7c73\uff0c\u6bcf\u79d2\u884c\u591a\u5c11\u7c73\uff1f",
        title_hint="\u3010\u7b2c1\u9898\u3011",
    )

    assert structured.title == "\u3010\u7b2c1\u9898\u3011"
    assert structured.body.startswith("\u706b\u8f66\u957f190\u7c73")


_REFERENCE = Path(__file__).parents[1] / "Sample" / "IMG_9536.HEIC"


@pytest.mark.skipif(
    sys.platform != "darwin" or not _REFERENCE.exists(),
    reason="requires the local HEIC reference and macOS Vision",
)
def test_reference_page_is_split_into_six_questions(tmp_path: Path) -> None:
    settings = Settings.load(Path(__file__).parents[1])
    result = PageSegmenter(MacVisionOCR(settings)).split(
        _REFERENCE,
        tmp_path / "reference-split",
        None,
    )

    assert result is not None
    assert [region.label for region in result.regions] == [
        "\u3010\u4f8b\u98986\u3011",
        "\u3010\u4f8b\u98987\u3011",
        "\u3010\u4f8b\u98988\u3011",
        "\u3010\u7b2c1\u9898\u3011",
        "\u3010\u7b2c2\u9898\u3011",
        "\u3010\u7b2c3\u9898\u3011",
    ]
    assert all(
        "\uff1f" in region.ocr_text or "?" in region.ocr_text
        for region in result.regions
    )
