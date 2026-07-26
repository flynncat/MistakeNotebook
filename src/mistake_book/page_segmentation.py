from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .image_pipeline import load_image, orient_image
from .formula_runtime import FormulaRuntime, FormulaRuntimeError
from .recognition import MacVisionOCR
from .reconstruction import ordered_question_text, question_terminal_match


@dataclass(frozen=True)
class QuestionAnchor:
    label: str
    raw_text: str
    top: int
    left: int
    confidence: float
    kind: str
    number: int


@dataclass(frozen=True)
class PageRegion:
    label: str
    source_path: Path
    box: tuple[int, int, int, int]
    blur_score: float
    anchor_text: str
    ocr_text: str

    def metrics(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_path"] = str(self.source_path)
        result["box"] = list(self.box)
        return result


@dataclass(frozen=True)
class PageSplit:
    regions: list[PageRegion]
    metrics: dict[str, Any]


_EXAMPLE_ANCHOR = re.compile(
    r"^\s*[\u3010\[\(\uff08]?\s*(\u4f8b\u9898|\u4f8b|\u7ec3\u4e60|\u9884\u4e60)\s*([0-9S]{1,3})",
    re.IGNORECASE,
)
_OCR_EXAMPLE_ANCHOR = re.compile(r"^\s*\u5217\s*([0-9S]{1,2})\s*[\u3001.\uff0e]")
_NUMBERED_ANCHOR = re.compile(r"^\s*([1-9][0-9]?)\s*[\u3001.\uff0e]")
_NUMBER_LIST_CONTINUATION = re.compile(
    r"^\s*\d{1,3}\s*[\u3001.\uff0e]\s*\d{1,3}\s*[\u3001.\uff0e]"
)


def _pixel_box(
    line: dict[str, Any],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    box = line.get("box", [])
    if len(box) != 4:
        return None
    x, y, box_width, box_height = (float(value) for value in box)
    return (
        max(0, round(x * width)),
        max(0, round((1 - y - box_height) * height)),
        min(width, round((x + box_width) * width)),
        min(height, round((1 - y) * height)),
    )


def detect_question_anchors(
    lines: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> list[QuestionAnchor]:
    width, height = image_size
    candidates: list[QuestionAnchor] = []
    for line in lines:
        text = str(line.get("text", "")).strip()
        box = _pixel_box(line, width, height)
        if not text or box is None:
            continue
        left, top, _, _ = box
        if left > width * 0.24:
            continue
        example = _EXAMPLE_ANCHOR.match(text)
        ocr_example = _OCR_EXAMPLE_ANCHOR.match(text)
        numbered = (
            None
            if _NUMBER_LIST_CONTINUATION.match(text)
            else _NUMBERED_ANCHOR.match(text)
        )
        if example:
            raw_number = example.group(2).upper().replace("S", "5")
            number = int(raw_number)
            prefix = (
                example.group(1)
                if example.group(1) in {"\u7ec3\u4e60", "\u9884\u4e60"}
                else "\u4f8b\u9898"
            )
            candidates.append(
                QuestionAnchor(
                    label=f"\u3010{prefix}{number}\u3011",
                    raw_text=text,
                    top=top,
                    left=left,
                    confidence=float(line.get("confidence", 0)),
                    kind="example",
                    number=number,
                )
            )
        elif ocr_example:
            raw_number = ocr_example.group(1).upper().replace("S", "5")
            number = int(raw_number)
            candidates.append(
                QuestionAnchor(
                    label=f"\u3010\u4f8b\u9898{number}\u3011",
                    raw_text=text,
                    top=top,
                    left=left,
                    confidence=float(line.get("confidence", 0)),
                    kind="example",
                    number=number,
                )
            )
        elif numbered:
            number = int(numbered.group(1))
            candidates.append(
                QuestionAnchor(
                    label=f"\u3010\u7b2c{number}\u9898\u3011",
                    raw_text=text,
                    top=top,
                    left=left,
                    confidence=float(line.get("confidence", 0)),
                    kind="numbered",
                    number=number,
                )
            )

    candidates.sort(key=lambda item: (item.top, item.left))
    deduplicated: list[QuestionAnchor] = []
    minimum_gap = max(25, round(height * 0.018))
    for candidate in candidates:
        if deduplicated and candidate.top - deduplicated[-1].top < minimum_gap:
            continue
        if (
            candidate.kind == "numbered"
            and candidate.number >= 10
            and deduplicated
            and deduplicated[-1].kind == "example"
            and candidate.number % 10 == deduplicated[-1].number + 1
        ):
            candidate = QuestionAnchor(
                label=f"\u3010\u4f8b\u9898{candidate.number % 10}\u3011",
                raw_text=candidate.raw_text,
                top=candidate.top,
                left=candidate.left,
                confidence=candidate.confidence,
                kind="example",
                number=candidate.number % 10,
            )
        deduplicated.append(candidate)
    return deduplicated


def _has_question_mark(
    lines: list[dict[str, Any]],
    width: int,
    height: int,
    start: int,
    end: int,
) -> bool:
    for line in lines:
        box = _pixel_box(line, width, height)
        if box is None:
            continue
        center = (box[1] + box[3]) // 2
        if start <= center < end and question_terminal_match(
            str(line.get("text", ""))
        ):
            return True
    return False


def _region_ocr_text(
    lines: list[dict[str, Any]],
    anchor: QuestionAnchor,
    width: int,
    height: int,
    start: int,
    end: int,
) -> str:
    entries: list[dict[str, Any]] = []
    for line in lines:
        box = _pixel_box(line, width, height)
        text = str(line.get("text", "")).strip()
        if box is None or not text:
            continue
        if re.search(r"(?:\u7b2c.+\u90e8\u5206|\u5c0f\u8bd5\u725b\u5200)", text):
            continue
        center = (box[1] + box[3]) // 2
        confidence = float(line.get("confidence", 0))
        if not start <= center < end:
            continue
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        if confidence < 0.45 and text != anchor.raw_text and not re.search(
            r"[\uff1f?]", text
        ) and chinese_count < 8:
            continue
        normalized_box = line.get("box", [])
        x, y, box_width, _ = (float(value) for value in normalized_box)
        if x < 0.075:
            continue
        entries.append(
            {
                "x0": x,
                "x1": x + box_width,
                "width": box_width,
                "baseline": y,
                "text": text,
            }
        )
    if not entries:
        return ""

    spanning = [entry for entry in entries if entry["width"] >= 0.58]
    left = [
        entry
        for entry in entries
        if entry["width"] < 0.58 and entry["x0"] < 0.4
    ]
    right = [
        entry
        for entry in entries
        if entry["width"] < 0.58 and entry["x0"] >= 0.4
    ]

    def group_column(column: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        for entry in sorted(column, key=lambda item: -item["baseline"]):
            if groups:
                previous = statistics.median(
                    item["baseline"] for item in groups[-1]
                )
                overlaps = any(
                    min(entry["x1"], item["x1"]) - max(entry["x0"], item["x0"])
                    > 0.01
                    for item in groups[-1]
                )
                if abs(previous - entry["baseline"]) <= 0.009 and not overlaps:
                    groups[-1].append(entry)
                    continue
            groups.append([entry])
        return groups

    left_groups = group_column(left)
    right_groups = group_column(right)
    rows: list[tuple[float, str]] = []
    for index in range(max(len(left_groups), len(right_groups))):
        items = [
            *([] if index >= len(left_groups) else left_groups[index]),
            *([] if index >= len(right_groups) else right_groups[index]),
        ]
        if items:
            rows.append(
                (
                    statistics.median(item["baseline"] for item in items),
                    "".join(
                        item["text"] for item in sorted(items, key=lambda item: item["x0"])
                    ),
                )
            )
    rows.extend((entry["baseline"], entry["text"]) for entry in spanning)
    rows.sort(key=lambda item: -item[0])
    texts = [text for _, text in rows]
    start_index = next(
        (
            index
            for index, text in enumerate(texts)
            if anchor.raw_text in text
            or _EXAMPLE_ANCHOR.search(text)
            or _OCR_EXAMPLE_ANCHOR.search(text)
            or _NUMBERED_ANCHOR.search(text)
        ),
        None,
    )
    if start_index is None:
        return ordered_question_text(
            [
                line
                for line in lines
                if (
                    (box := _pixel_box(line, width, height)) is not None
                    and start <= (box[1] + box[3]) // 2 < end
                )
            ]
        )
    selected: list[str] = []
    for text in texts[start_index:]:
        terminal = question_terminal_match(text)
        selected.append(text[: terminal.end()] if terminal else text)
        if terminal:
            break
    return "\n".join(selected)


class PageSegmenter:
    def __init__(
        self,
        local_ocr: MacVisionOCR,
        formula_runtime: FormulaRuntime | None = None,
    ) -> None:
        self.local_ocr = local_ocr
        self.formula_runtime = formula_runtime

    def split(
        self,
        source_path: Path,
        output_dir: Path,
        rotation_hint: int | None,
    ) -> PageSplit | None:
        if not self.local_ocr.available():
            return None
        image = load_image(source_path)
        oriented, orientation_metrics = orient_image(image, rotation_hint)
        detail_image = load_image(source_path, maximum=4200)
        rotation = int(orientation_metrics.get("rotation_degrees", 0))
        detail_oriented = (
            orient_image(detail_image, rotation)[0] if rotation else detail_image
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        oriented_path = output_dir / "page-oriented.png"
        if not cv2.imwrite(str(oriented_path), oriented):
            raise OSError("Unable to save oriented page")
        _, confidence, lines = self.local_ocr.recognize(oriented_path)
        height, width = oriented.shape[:2]
        anchors = detect_question_anchors(lines, (width, height))
        if len(anchors) < 2:
            return None

        top_padding = max(28, round(height * 0.035))
        side_padding = max(20, round(width * 0.025))
        starts = [max(0, anchor.top - top_padding) for anchor in anchors]
        formula_boxes: list[list[int]] = []
        if self.formula_runtime is not None:
            try:
                formula_boxes = [
                    [int(value) for value in item["box"]]
                    for item in self.formula_runtime.detect(oriented_path).get(
                        "formulas",
                        [],
                    )
                ]
            except FormulaRuntimeError:
                formula_boxes = []
        boundary_padding = max(4, round(height * 0.003))
        for index in range(1, len(starts)):
            boundary = starts[index]
            crossing = [
                box for box in formula_boxes if box[1] < boundary < box[3]
            ]
            if not crossing:
                continue
            candidates = []
            for box in crossing:
                candidates.extend(
                    [
                        max(starts[index - 1] + 1, box[1] - boundary_padding),
                        min(height - 1, box[3] + boundary_padding),
                    ]
                )
            starts[index] = min(candidates, key=lambda value: abs(value - boundary))
        regions: list[PageRegion] = []
        rejected_labels: list[str] = []
        for index, anchor in enumerate(anchors):
            start = starts[index]
            end = starts[index + 1] if index + 1 < len(starts) else height
            if end - start < height * 0.055:
                rejected_labels.append(anchor.label)
                continue
            region_ocr_text = _region_ocr_text(
                lines,
                anchor,
                width,
                height,
                start,
                end,
            )
            meaningful_text = re.sub(
                r"[^\u4e00-\u9fffA-Za-z0-9]",
                "",
                region_ocr_text,
            )
            if (
                not question_terminal_match(region_ocr_text)
                and len(meaningful_text) < 18
            ):
                rejected_labels.append(anchor.label)
                continue
            x0 = max(0, round(width * 0.075) - side_padding)
            x1 = min(width, round(width * 0.96) + side_padding)
            detail_height, detail_width = detail_oriented.shape[:2]
            scale_x = detail_width / width
            scale_y = detail_height / height
            detail_box = (
                max(0, round(x0 * scale_x)),
                max(0, round(start * scale_y)),
                min(detail_width, round(x1 * scale_x)),
                min(detail_height, round(end * scale_y)),
            )
            crop = detail_oriented[
                detail_box[1] : detail_box[3],
                detail_box[0] : detail_box[2],
            ]
            if crop.size == 0:
                rejected_labels.append(anchor.label)
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            minimum_height = round(crop.shape[1] * 0.58)
            if crop.shape[0] < minimum_height:
                canvas = np.full(
                    (minimum_height, crop.shape[1], 3),
                    255,
                    dtype=np.uint8,
                )
                canvas[: crop.shape[0], : crop.shape[1]] = crop
                crop = canvas
            crop_path = output_dir / f"question-{index + 1:02d}.png"
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Unable to save question crop {index + 1}")
            regions.append(
                PageRegion(
                    label=anchor.label,
                    source_path=crop_path,
                    box=detail_box,
                    blur_score=round(blur_score, 2),
                    anchor_text=anchor.raw_text,
                    ocr_text=region_ocr_text,
                )
            )

        required = max(2, len(anchors) - 1)
        if len(regions) < required:
            raise ValueError(
                f"Detected {len(anchors)} question anchors but validated only "
                f"{len(regions)} regions"
            )
        return PageSplit(
            regions=regions,
            metrics={
                "mode": "multi_question_page",
                "line_count": len(lines),
                "ocr_confidence": round(confidence, 4),
                "anchor_count": len(anchors),
                "region_count": len(regions),
                "formula_boundary_boxes": formula_boxes,
                "rejected_labels": rejected_labels,
                "orientation": orientation_metrics,
            },
        )
