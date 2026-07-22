from __future__ import annotations

import cv2
import numpy as np

from mistake_book.image_pipeline import (
    conservative_remove_marks,
    extract_printed_question,
    orient_image,
)


def test_safe_cleanup_preserves_black_print_and_removes_red_on_white() -> None:
    image = np.full((320, 640, 3), 245, dtype=np.uint8)
    cv2.putText(
        image,
        "123 + 456 = ?",
        (40, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    original_print = image[50:105].copy()
    cv2.line(image, (80, 190), (540, 250), (30, 30, 220), 8, cv2.LINE_AA)

    cleaned, metrics = conservative_remove_marks(image)

    assert metrics["removed_pixels"] > 0
    assert metrics["protected_overlap_pixels"] == 0
    np.testing.assert_array_equal(cleaned[50:105], original_print)
    assert np.mean(np.abs(cleaned[180:260].astype(int) - image[180:260].astype(int))) > 0


def test_orientation_prefers_horizontal_text_rows() -> None:
    image = np.full((280, 800, 3), 255, dtype=np.uint8)
    for row in range(50, 240, 45):
        cv2.putText(
            image,
            "1234567890 1234567890",
            (30, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    sideways = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    oriented, metrics = orient_image(sideways)

    assert oriented.shape[1] > oriented.shape[0]
    assert metrics["rotation_degrees"] == 90


def test_question_extraction_removes_working_outside_text_box(tmp_path) -> None:
    image = np.full((500, 900, 3), 245, dtype=np.uint8)
    cv2.putText(image, "[1] printed question?", (80, 130), 1, 1.2, (10, 10, 10), 2)
    cv2.putText(image, "handwritten answer", (220, 360), 1, 1.4, (50, 50, 50), 3)
    source = tmp_path / "cleaned.png"
    cv2.imwrite(str(source), image)
    lines = [
        {
            "text": "【例题1】 printed question?",
            "box": [0.08, 0.70, 0.72, 0.12],
            "confidence": 0.9,
        }
    ]

    target, metrics, reasons = extract_printed_question(source, lines, tmp_path)
    extracted = cv2.imread(str(target))

    assert metrics["question_extracted"] is True
    assert reasons == []
    assert extracted.shape[0] < image.shape[0] / 2
