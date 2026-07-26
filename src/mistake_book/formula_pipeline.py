from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any
import unicodedata

import cv2
import numpy as np

from .formula_math import FormulaValidationError, convert_latex, normalize_latex
from .formula_runtime import (
    FormulaRuntime,
    FormulaRuntimeError,
    FormulaRuntimeUnavailable,
)


_MATH_CHAR = re.compile(r"[\dA-Za-z+\-*/=<>()[\]{}^_\u00d7\u00f7\u221a\u2211\u222b]")


@dataclass(frozen=True)
class FormulaPipelineResult:
    content_blocks: dict[str, Any]
    metrics: dict[str, Any]
    review_reasons: list[str]


def _line_box(line: dict[str, Any], width: int, height: int) -> list[int] | None:
    box = line.get("box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x, y, box_width, box_height = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    return [
        max(0, round(x * width)),
        max(0, round((1 - y - box_height) * height)),
        min(width, round((x + box_width) * width)),
        min(height, round((1 - y) * height)),
    ]


def _intersection(first: list[int], second: list[int]) -> int:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _expanded_box(box: list[int], width: int, height: int) -> list[int]:
    horizontal_padding = max(2, round(height * 0.0005))
    vertical_padding = max(4, round(height * 0.001))
    return [
        max(0, box[0] - horizontal_padding),
        max(0, box[1] - vertical_padding),
        min(width, box[2] + horizontal_padding),
        min(height, box[3] + vertical_padding),
    ]


def _save_formula_variants(
    normalized: np.ndarray,
    cleaned: np.ndarray,
    box: list[int],
    artifact_dir: Path,
    index: int,
) -> tuple[Path, Path, Path]:
    original_path = artifact_dir / f"formula-{index:02d}-original.png"
    clean_path = artifact_dir / f"formula-{index:02d}-clean.png"
    contrast_path = artifact_dir / f"formula-{index:02d}-contrast.png"
    x1, y1, x2, y2 = box
    original = normalized[y1:y2, x1:x2]
    clean = cleaned[y1:y2, x1:x2]
    if original.size == 0 or clean.size == 0:
        raise ValueError("empty formula crop")
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    _, contrast = cv2.threshold(
        clahe,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    cv2.imwrite(str(original_path), original, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    cv2.imwrite(str(clean_path), binary, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    cv2.imwrite(str(contrast_path), contrast, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    return original_path, clean_path, contrast_path


def _validated_latex(result: dict[str, Any]) -> str | None:
    latex = normalize_latex(str(result.get("latex") or ""))
    if not latex:
        return None
    try:
        return convert_latex(latex).latex
    except FormulaValidationError:
        return None


def _formula_id(clean_path: Path, box: list[int]) -> str:
    digest = hashlib.sha256(clean_path.read_bytes()).hexdigest()[:12]
    return f"formula-{digest}-{box[0]}-{box[1]}"


def _math_ratio(text: str) -> float:
    compact = "".join(text.split())
    if not compact:
        return 0.0
    return len(_MATH_CHAR.findall(compact)) / len(compact)


def _visible_formula_text(latex: str) -> str:
    value = latex
    value = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)[lr]?\b", "", value)
    value = re.sub(r"\\(?:[,;:!]|quad|qquad)", "", value)
    value = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"\1/\2",
        value,
    )
    value = re.sub(
        r"\\sqrt\s*\{([^{}]+)\}",
        lambda match: f"\u221a({match.group(1)})",
        value,
    )
    replacements = {
        r"\times": "\u00d7",
        r"\div": "\u00f7",
        r"\leq": "\u2264",
        r"\le": "\u2264",
        r"\geq": "\u2265",
        r"\ge": "\u2265",
        r"\neq": "\u2260",
        r"\approx": "\u2248",
        r"\cdot": "\u00b7",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(r"\\[A-Za-z]+", "", value)
    value = value.replace("_", "").replace("^", "")
    return value.replace("{", "").replace("}", "")


def _normalized_text_with_positions(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text):
        for candidate in unicodedata.normalize("NFKC", character).lower():
            if candidate.isspace():
                continue
            normalized.append(candidate)
            positions.append(index)
    return "".join(normalized), positions


_FORMULA_CONFUSIONS = {
    "0": {"o"},
    "1": {"i", "l"},
    "5": {"s"},
    "6": {"b"},
    "8": {"&", "b"},
    "9": {"g"},
}


def _safe_formula_variant(expected: str, actual: str) -> bool:
    if len(expected) != len(actual):
        return False
    differences = [
        (expected_character, actual_character)
        for expected_character, actual_character in zip(expected, actual)
        if expected_character != actual_character
    ]
    if not differences or len(differences) > max(1, round(len(expected) * 0.08)):
        return False
    return all(
        actual_character in _FORMULA_CONFUSIONS.get(expected_character, set())
        for expected_character, actual_character in differences
    )


def _exact_formula_span(
    text: str,
    formula: dict[str, Any],
    line_box: list[int],
) -> tuple[int, int] | None:
    visible = _visible_formula_text(
        str(formula.get("latex") or formula.get("model_latex") or "")
    )
    needle, _ = _normalized_text_with_positions(visible)
    haystack, positions = _normalized_text_with_positions(text)
    if not needle or not haystack or not positions:
        return None
    matches = [
        match.start()
        for match in re.finditer(re.escape(needle), haystack)
    ]
    if not matches and len(haystack) >= len(needle):
        matches = [
            start
            for start in range(len(haystack) - len(needle) + 1)
            if _safe_formula_variant(
                needle,
                haystack[start : start + len(needle)],
            )
        ]
    if not matches:
        return None
    formula_box = formula["source_box"]
    line_width = max(1, line_box[2] - line_box[0])
    expected = min(
        1.0,
        max(
            0.0,
            (
                ((formula_box[0] + formula_box[2]) / 2)
                - line_box[0]
            )
            / line_width,
        ),
    )
    start = min(
        matches,
        key=lambda candidate: abs(
            ((candidate + len(needle) / 2) / max(1, len(haystack)))
            - expected
        ),
    )
    end = start + len(needle)
    return positions[start], positions[end - 1] + 1


def _is_simple_text_formula(formula: dict[str, Any]) -> bool:
    visible, _ = _normalized_text_with_positions(
        _visible_formula_text(
            str(formula.get("latex") or formula.get("model_latex") or "")
        )
    )
    return bool(re.fullmatch(r"[a-z0-9]", visible))


def _semantic_formula_span(
    text: str,
    formula: dict[str, Any],
    estimated_start: int,
    estimated_end: int,
) -> tuple[int, int] | None:
    latex = str(formula.get("latex") or formula.get("model_latex") or "")
    if "=" in latex:
        equal_at = text.find("=")
        operators = [
            index
            for index, character in enumerate(text[: max(0, equal_at)])
            if character in "+-\u00d7\u00f7*/"
        ]
        if equal_at >= 0 and operators:
            opening = max(
                (
                    text.rfind(character, 0, operators[-1])
                    for character in "(\uff08[\u3010"
                ),
                default=-1,
            )
            closing_candidates = [
                index
                for character in ")\uff09]\u3011"
                if (index := text.find(character, equal_at + 1)) >= 0
            ]
            if opening >= 0 and closing_candidates:
                return opening, min(closing_candidates) + 1
    compact = re.sub(r"[\s{}]", "", latex)
    if re.fullmatch(r"[A-Za-z]", compact):
        candidates: list[tuple[int, int]] = []
        for marker in ("\u8fdb\u5236",):
            start = text.find(marker)
            if start > 0:
                candidates.append((start - 1, start))
        for match in re.finditer(
            rf"(?:\u6c42|\u95ee)\s*([A-Za-z0-9]?)\s*\u7684\u503c",
            text,
        ):
            group_start, group_end = match.span(1)
            if group_start < group_end:
                candidates.append((group_start, group_end))
            else:
                candidates.append((group_start, group_start))
        if candidates:
            return min(
                candidates,
                key=lambda span: abs(((span[0] + span[1]) / 2) - estimated_start),
            )
    if estimated_end > estimated_start:
        return estimated_start, estimated_end
    return None


def _merge_components(
    lines: list[dict[str, Any]],
    formulas: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    line_height_values: list[int] = []
    for line_index, line in enumerate(lines):
        text = str(line.get("text") or "").strip()
        box = _line_box(line, width, height)
        if not text or box is None:
            continue
        line_height_values.append(max(1, box[3] - box[1]))
        overlapping = [
            formula
            for formula in formulas
            if _intersection(box, formula["source_box"]) > 0
        ]
        if not overlapping:
            components.append(
                {
                    "kind": "text",
                    "box": box,
                    "text": text,
                    "stable_id": f"text-{line_index}",
                }
            )
            continue
        overlapping.sort(key=lambda item: item["source_box"][0])
        current = 0
        current_x = box[0]
        spans: list[tuple[int, int, dict[str, Any]]] = []
        for formula in overlapping:
            if _is_simple_text_formula(formula):
                formula["alignment_state"] = "text_preserved"
                continue
            exact = _exact_formula_span(text, formula, box)
            if exact is None:
                formula.setdefault("alignment_state", "unaligned")
                continue
            formula["alignment_state"] = "aligned"
            spans.append((*exact, formula))
        spans.sort(key=lambda item: (item[0], item[1]))
        for left, right, formula in spans:
            formula_box = formula["source_box"]
            left = max(current, left)
            right = max(left, right)
            if left > current:
                components.append(
                    {
                        "kind": "text",
                        "box": [current_x, box[1], formula_box[0], box[3]],
                        "text": text[current:left],
                        "stable_id": f"text-{line_index}-{current}",
                    }
                )
            current = right
            current_x = formula_box[2]
        if current < len(text):
            remaining = text[current:]
            if spans and "=" in str(
                spans[-1][2].get("latex")
                or spans[-1][2].get("model_latex")
                or ""
            ):
                remaining = re.sub(
                    r"^[.\u3002\uff0eoO]+(?=[,\uff0c])",
                    "",
                    remaining,
                )
            components.append(
                {
                    "kind": "text",
                    "box": [current_x, box[1], box[2], box[3]],
                    "text": remaining,
                    "stable_id": f"text-{line_index}-{current}",
                }
            )

    median_height = (
        float(np.median(line_height_values)) if line_height_values else 20.0
    )
    for formula in formulas:
        if not formula.get("alignment_state"):
            formula["alignment_state"] = "standalone"
        if formula.get("alignment_state") == "text_preserved":
            continue
        components.append(
            {
                "kind": "formula",
                "box": formula["source_box"],
                "formula": formula,
                "stable_id": formula["formula_id"],
            }
        )
    components.sort(key=lambda item: (item["box"][1], item["box"][0], item["stable_id"]))
    rows: list[list[dict[str, Any]]] = []
    for component in components:
        box = component["box"]
        target: list[dict[str, Any]] | None = None
        for row in reversed(rows):
            row_top = min(item["box"][1] for item in row)
            row_bottom = max(item["box"][3] for item in row)
            row_start = float(np.median([item["box"][1] for item in row]))
            overlap = max(0, min(row_bottom, box[3]) - max(row_top, box[1]))
            smaller = max(1, min(row_bottom - row_top, box[3] - box[1]))
            center_delta = abs((row_top + row_bottom) / 2 - (box[1] + box[3]) / 2)
            top_delta = abs(row_start - box[1])
            if (
                top_delta <= 0.8 * median_height
                or (
                    overlap / smaller >= 0.5
                    and center_delta <= 0.45 * median_height
                )
            ):
                target = row
                break
        if target is None:
            target = []
            rows.append(target)
        target.append(component)
    rows.sort(
        key=lambda row: (
            min(item["box"][1] for item in row),
            min(item["box"][0] for item in row),
        )
    )
    blocks: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row.sort(key=lambda item: (item["box"][0], item["box"][1], item["stable_id"]))
        for item in row:
            if item["kind"] == "text":
                text = item["text"].strip()
                if text:
                    blocks.append(
                        {
                            "type": "text",
                            "text": text,
                            "row_index": row_index,
                            "source_box": list(item["box"]),
                        }
                    )
                continue
            formula = dict(item["formula"])
            formula["row_index"] = row_index
            formula["baseline_y"] = max(entry["box"][3] for entry in row)
            formula["display"] = (
                len(row) == 1
                or (item["box"][3] - item["box"][1]) > 2.5 * median_height
            )
            blocks.append(formula)
    return blocks


def process_formulas(
    runtime: FormulaRuntime,
    normalized_path: Path,
    cleaned_path: Path,
    lines: list[dict[str, Any]],
    artifact_dir: Path,
    *,
    fallback_content_blocks: dict[str, Any],
    figure_asset: str | None = None,
    figure_box: list[int] | None = None,
) -> FormulaPipelineResult:
    fallback_blocks = fallback_content_blocks
    reasons: list[str] = []
    try:
        detected = runtime.detect(normalized_path)
    except (FormulaRuntimeUnavailable, FormulaRuntimeError) as error:
        reasons.append(f"Formula OCR unavailable: {error}")
        return FormulaPipelineResult(
            content_blocks=fallback_blocks,
            metrics={"formula_runtime": runtime.status(), "error": str(error)},
            review_reasons=reasons,
        )
    normalized = cv2.imread(str(normalized_path), cv2.IMREAD_COLOR)
    cleaned = cv2.imread(str(cleaned_path), cv2.IMREAD_COLOR)
    if normalized is None or cleaned is None:
        raise OSError("unable to read formula source image")
    height, width = normalized.shape[:2]
    crops: list[tuple[dict[str, Any], Path, Path, Path]] = []
    detected_formulas = list(detected.get("formulas") or [])
    if len(detected_formulas) > 64:
        reasons.append("Formula count exceeds 64; lowest-confidence boxes were omitted")
        detected_formulas = sorted(
            detected_formulas,
            key=lambda item: float(item.get("score") or 0),
            reverse=True,
        )[:64]
        detected_formulas.sort(
            key=lambda item: (
                int(item["box"][1]),
                int(item["box"][0]),
            )
        )
    for item in detected_formulas:
        box = _expanded_box(
            [int(value) for value in item["box"]],
            width,
            height,
        )
        if figure_box and _intersection(box, figure_box) > _area(box) * 0.25:
            continue
        if _area(box) < 64:
            continue
        original_path, clean_path, contrast_path = _save_formula_variants(
            normalized,
            cleaned,
            box,
            artifact_dir,
            len(crops) + 1,
        )
        crops.append((dict(item, box=box), original_path, clean_path, contrast_path))
    if not crops:
        return FormulaPipelineResult(
            content_blocks=fallback_blocks,
            metrics={
                "formula_runtime": runtime.status(),
                "detector_elapsed_ms": detected.get("elapsed_ms"),
                "formula_count": 0,
            },
            review_reasons=[],
        )
    recognition_paths = [
        path
        for _, _, clean_path, contrast_path in crops
        for path in (clean_path, contrast_path)
    ]
    try:
        recognized = runtime.recognize(recognition_paths)
    except (FormulaRuntimeUnavailable, FormulaRuntimeError) as error:
        reasons.append(f"Formula recognition unavailable: {error}")
        recognized = {"results": [{} for _ in recognition_paths], "error": str(error)}
    results = list(recognized.get("results") or [])
    formulas: list[dict[str, Any]] = []
    for index, (crop, original_path, clean_path, _) in enumerate(crops):
        first = results[index * 2] if index * 2 < len(results) else {}
        second = results[index * 2 + 1] if index * 2 + 1 < len(results) else {}
        first_latex = _validated_latex(first)
        second_latex = _validated_latex(second)
        preferred = max(
            (first, second),
            key=lambda item: float(item.get("mean_token_probability") or 0),
        )
        preferred_latex = _validated_latex(preferred)
        auto_verified = (
            first_latex is not None
            and first_latex == second_latex
            and bool(first.get("eos_reached"))
            and bool(second.get("eos_reached"))
            and not bool(first.get("unk_reached"))
            and not bool(second.get("unk_reached"))
            and min(
                float(first.get("mean_token_probability") or 0),
                float(second.get("mean_token_probability") or 0),
            )
            >= 0.80
            and min(
                float(first.get("p05_token_probability") or 0),
                float(second.get("p05_token_probability") or 0),
            )
            >= 0.35
        )
        if auto_verified:
            state = "auto_verified"
            latex = first_latex
        elif preferred_latex:
            state = "needs_review"
            latex = preferred_latex
            reasons.append(f"Formula {index + 1} requires review")
        else:
            state = "image_fallback"
            latex = ""
            reasons.append(f"Formula {index + 1} has no safe LaTeX result")
        box = crop["box"]
        formulas.append(
            {
                "formula_id": _formula_id(clean_path, box),
                "type": "latex",
                "latex": latex,
                "model_latex": preferred_latex or "",
                "source_text": str(preferred.get("latex") or ""),
                "source_box": box,
                "original_crop_asset": original_path.name,
                "clean_crop_asset": clean_path.name,
                "detector_confidence": round(float(crop.get("score") or 0), 6),
                "mean_token_probability": round(
                    float(preferred.get("mean_token_probability") or 0),
                    6,
                ),
                "p05_token_probability": round(
                    float(preferred.get("p05_token_probability") or 0),
                    6,
                ),
                "eos_reached": bool(preferred.get("eos_reached")),
                "recognition_state": state,
                "recognizer": "unimernet-tiny",
                "edited_at": None,
            }
        )
    blocks = _merge_components(lines, formulas, width, height)
    unaligned = [
        formula
        for formula in formulas
        if formula.get("alignment_state") == "unaligned"
    ]
    if unaligned:
        reasons.append(
            f"{len(unaligned)} \u4e2a\u516c\u5f0f\u65e0\u6cd5\u4e0e\u539f\u9898\u6587\u5b57\u7cbe\u786e\u5bf9\u9f50\uff0c"
            "\u5df2\u4fdd\u7559\u539f\u59cb\u6587\u5b57\u6216\u516c\u5f0f\u884c\u56fe\u50cf\uff0c"
            "\u672a\u6267\u884c\u63a8\u6d4b\u6027\u66ff\u6362"
        )
        fallback_items = list(fallback_blocks.get("blocks") or [])
        if fallback_items:
            blocks = fallback_items
            content_version = int(fallback_blocks.get("version") or 1)
        else:
            content_version = 2
    else:
        content_version = 2
    if figure_asset:
        if not any(
            block.get("type") == "image"
            and block.get("asset") == figure_asset
            for block in blocks
        ):
            blocks.append(
                {
                    "type": "image",
                    "asset": figure_asset,
                    "alt": "question figure",
                    "source_box": list(figure_box or []),
                }
            )
    return FormulaPipelineResult(
        content_blocks={
            "version": content_version,
            "blocks": blocks,
            "formula_runtime": runtime.status(),
        },
        metrics={
            "formula_runtime": runtime.status(),
            "detector_elapsed_ms": detected.get("elapsed_ms"),
            "recognizer_elapsed_ms": recognized.get("elapsed_ms"),
            "recognizer_round_trip_ms": recognized.get("round_trip_ms"),
            "recognizer_device": recognized.get("device"),
            "formula_count": len(formulas),
            "formula_alignment": {
                state: sum(
                    formula.get("alignment_state") == state
                    for formula in formulas
                )
                for state in (
                    "aligned",
                    "text_preserved",
                    "standalone",
                    "unaligned",
                )
            },
            "formula_states": {
                state: sum(
                    formula["recognition_state"] == state for formula in formulas
                )
                for state in (
                    "auto_verified",
                    "needs_review",
                    "human_verified",
                    "image_fallback",
                )
            },
        },
        review_reasons=reasons,
    )
