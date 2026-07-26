from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mistake_book.figure_reconstruction import reconstruct_figure
from mistake_book.figure_preservation import structural_fidelity
from mistake_book.reconstruction import (
    build_structured_problem,
    question_text_box,
    render_problem,
)


def test_reconstruction_keeps_only_question_and_detects_figure(tmp_path: Path) -> None:
    primary = (
        "页眉\n【例题10】\n如图，三个圆圈交叠在一起，形成7个区域，"
        "用5种颜色染色，共有多少种方法？\n4×5×3=60"
    )
    secondary = "【例题10】如图，三个圆圈交叠在一起，形成7个区域，用5种颜色染色，共有多少种方法？"

    problem = build_structured_problem(primary, secondary)

    assert problem.title == "【例题10】"
    assert problem.figure == "three_overlapping_circles"
    assert "4×5×3=60" not in problem.body
    figure = tmp_path / "figure.png"
    Image.new("RGB", (300, 200), "white").save(figure)
    target = render_problem(
        problem,
        tmp_path / "question.png",
        figure_path=figure,
    )
    assert target.exists()
    with Image.open(target) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)


def test_dual_ocr_numeric_disagreement_requires_review() -> None:
    problem = build_structured_problem(
        "【例题8】一件玩具售价22元，现在钱数之比是13:7，问还剩多少元？",
        "【例题8】一件玩具售价27元，现在钱数之比是13:7，问还剩多少元？",
    )

    assert "双 OCR 数字或比例不一致" in problem.review_reasons


def test_unreliable_secondary_ocr_cannot_create_numeric_conflict() -> None:
    problem = build_structured_problem(
        "【例题10】如图，三个圆交叠形成7个区域，用5种颜色染色，每个区域有4种选择，共有多少种方法？",
        "【例题10】三个岗台一起形成7区，乱码文字4，一共有多少 ae？",
    )
    assert problem.text_similarity < 0.82
    assert problem.primary_numbers == ["10", "7", "5", "4"]
    assert problem.secondary_numbers == ["10", "7", "4"]
    assert "双 OCR 数字或比例不一致" not in problem.review_reasons


def test_secondary_ocr_failure_does_not_claim_primary_title_is_missing() -> None:
    secondary = "无法识别的第二结果"
    problem = build_structured_problem(
        "【例题8】一件玩具售价22元，小蘑与小菇原来钱数之比是37:25，问还有多少元？",
        secondary,
    )
    assert problem.title == "【例题8】"
    assert problem.secondary_raw_text == secondary
    assert "未定位到印刷题号" not in problem.review_reasons
    assert "第二 OCR 未识别到完整题干" not in problem.review_reasons


def test_title_hint_does_not_leave_a_duplicate_closing_bracket() -> None:
    problem = build_structured_problem(
        "\u3010\u7ec3\u4e607\u3011\u5df2\u77e5\u6b63\u6574\u6570N\u7684"
        "\u516b\u8fdb\u5236\u8868\u793a\uff0c\u6c42\u4f59\u6570\uff1f",
        title_hint="\u3010\u7ec3\u4e607\u3011",
    )

    assert problem.title == "\u3010\u7ec3\u4e607\u3011"
    assert not problem.body.startswith("\u3011")


def test_secondary_ocr_box_stops_before_figure_and_handwriting() -> None:
    lines = [
        {
            "text": "【例题10】如图，三个圆交叠形成7个区域，",
            "box": [0.1, 0.87, 0.72, 0.03],
        },
        {
            "text": "用5种颜色染色，其中有多少种方法？",
            "box": [0.1, 0.82, 0.62, 0.03],
        },
        {"text": "4×5×3×2×2=960", "box": [0.35, 0.45, 0.3, 0.04]},
    ]
    box = question_text_box(lines, (1000, 1000))
    assert box is not None
    assert box[1] < 100
    assert box[3] < 220


def test_secondary_ocr_box_requires_complete_printed_question() -> None:
    lines = [
        {
            "text": "【例题10】如图，三个圆交叠形成7个区域",
            "box": [0.1, 0.87, 0.72, 0.03],
        },
        {"text": "4×5×3×2×2=960", "box": [0.35, 0.45, 0.3, 0.04]},
    ]
    assert question_text_box(lines, (1000, 1000)) is None


def test_five_country_map_preserves_adjacency_and_coloring_count(
    tmp_path: Path,
) -> None:
    text = (
        "【练习9】如图，一张地图上有五个国家：A、B、C、D、E，"
        "用四种颜色染色，相邻国家不能使用同一种颜色，共有多少种染法？"
    )
    problem = build_structured_problem(text, text)

    assert problem.figure == "five_country_map"
    assert problem.figure_edges == []
    source = np.full((430, 500, 3), 255, dtype=np.uint8)
    for start, end in (
        ((20, 20), (480, 20)),
        ((20, 130), (480, 130)),
        ((20, 230), (200, 230)),
        ((200, 290), (480, 290)),
        ((20, 410), (480, 410)),
        ((20, 20), (20, 410)),
        ((200, 130), (200, 410)),
        ((480, 20), (480, 410)),
    ):
        cv2.line(source, start, end, (0, 0, 0), 4)
    cv2.ellipse(source, (155, 270), (55, 145), 0, 0, 360, (100, 100, 100), 8)
    centers = {
        "A": (250, 75),
        "B": (110, 180),
        "C": (340, 205),
        "D": (110, 320),
        "E": (340, 350),
    }
    for label, center in centers.items():
        cv2.putText(
            source,
            label,
            (center[0] - 12, center[1] + 12),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def ocr_line(label: str) -> dict:
        center_x, center_y = centers[label]
        return {
            "text": label,
            "confidence": 0.9,
            "box": [
                (center_x - 15) / 500,
                1 - (center_y + 18) / 430,
                30 / 500,
                36 / 430,
            ],
        }

    reconstructed = reconstruct_figure(
        source,
        "five_country_map",
        text,
        [ocr_line(label) for label in "ABCE"],
    )
    assert reconstructed.metrics["passed"] is True
    problem.figure_edges = [
        tuple(edge) for edge in reconstructed.metrics["adjacency_edges"]
    ]
    countries = "ABCDE"
    colorings = sum(
        all(colors[countries.index(left)] != colors[countries.index(right)] for left, right in problem.figure_edges)
        for colors in itertools.product(range(4), repeat=len(countries))
    )
    assert colorings == 96
    figure = tmp_path / "map-source.png"
    cv2.imwrite(str(figure), reconstructed.image)
    assert render_problem(problem, tmp_path / "map.png", figure_path=figure).exists()


def test_structure_gate_rejects_a_different_figure() -> None:
    reference = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(reference, (30, 30), (370, 270), (0, 0, 0), 4)
    cv2.line(reference, (30, 110), (370, 110), (0, 0, 0), 4)
    cv2.line(reference, (180, 110), (180, 270), (0, 0, 0), 4)
    cv2.line(reference, (30, 180), (180, 180), (0, 0, 0), 4)
    cv2.line(reference, (180, 205), (370, 205), (0, 0, 0), 4)
    identical = reference.copy()
    different = np.full_like(reference, 255)
    cv2.rectangle(different, (30, 30), (370, 270), (0, 0, 0), 4)
    cv2.line(different, (30, 110), (370, 110), (0, 0, 0), 4)
    cv2.line(different, (200, 110), (180, 175), (0, 0, 0), 4)
    cv2.line(different, (30, 200), (180, 175), (0, 0, 0), 4)
    cv2.line(different, (180, 175), (225, 220), (0, 0, 0), 4)
    cv2.line(different, (225, 220), (370, 200), (0, 0, 0), 4)
    cv2.line(different, (225, 220), (225, 270), (0, 0, 0), 4)

    assert structural_fidelity(reference, identical)["passed"] is True
    assert structural_fidelity(reference, different)["passed"] is False
