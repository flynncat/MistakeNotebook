from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .reconstruction import build_structured_problem, question_content_blocks


_OPERAND = r"(?:\d+(?:\.\d+)?|[A-Za-z])"
_OPERATOR = r"(?:[+\-=:\/]|[xX*]|\u00d7|\u00f7|\u2264|\u2265)"
_FORMULA = re.compile(
    rf"(?<![\dA-Za-z]){_OPERAND}(?:\s*{_OPERATOR}\s*{_OPERAND})+"
)


def _latex_for(expression: str) -> str | None:
    compact = re.sub(r"\s+", "", expression)
    if not compact or not _FORMULA.fullmatch(compact):
        return None
    replacements = {
        "\u00d7": r" \times ",
        "x": r" \times ",
        "X": r" \times ",
        "*": r" \times ",
        "\u00f7": r" \div ",
        "\u2264": r" \le ",
        "\u2265": r" \ge ",
        "<=": r" \le ",
        ">=": r" \ge ",
    }
    latex = compact
    for source, target in replacements.items():
        latex = latex.replace(source, target)
    return latex.strip()


def split_text_and_latex(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    cursor = 0
    for match in _FORMULA.finditer(text):
        if match.start() > cursor:
            blocks.append({"type": "text", "text": text[cursor : match.start()]})
        source = match.group(0)
        latex = _latex_for(source)
        if latex:
            blocks.append(
                {
                    "type": "latex",
                    "latex": latex,
                    "display": False,
                    "source_text": source,
                }
            )
        else:
            blocks.append({"type": "text", "text": source})
        cursor = match.end()
    if cursor < len(text):
        blocks.append({"type": "text", "text": text[cursor:]})
    return [block for block in blocks if block.get("text") or block.get("latex")]


def build_content_blocks(
    text: str,
    *,
    figure_asset: str | None = None,
    figure_box: list[int] | None = None,
) -> dict[str, Any]:
    blocks = split_text_and_latex(text.strip())
    if figure_asset:
        blocks.append(
            {
                "type": "image",
                "asset": figure_asset,
                "alt": "question figure",
                "source_box": list(figure_box or []),
            }
        )
    return {"version": 1, "blocks": blocks}


def build_content_blocks_with_fallbacks(
    text: str,
    lines: list[dict[str, Any]],
    cleaned_path: Path,
    artifact_dir: Path,
    *,
    figure_asset: str | None = None,
    figure_box: list[int] | None = None,
) -> dict[str, Any]:
    unsupported = re.compile(
        r"[\u221a\u2211\u222b\u2248\u2260{}]"
        r"|[A-Za-z0-9][\^_][A-Za-z0-9]"
        r"|[\(\uff08][^\)\uff09]*[+\-\u00d7\u00f7=:/\u2264\u2265]"
        r"[^\)\uff09]*[\)\uff09]"
        r"|\\(?:frac|sqrt|sum|int)"
    )
    events: list[tuple[int, int, dict[str, Any]]] = []
    with Image.open(cleaned_path) as image:
        width, height = image.size
        for line in lines:
            line_text = str(line.get("text") or "").strip()
            box = line.get("box", [])
            if not line_text or not unsupported.search(line_text) or len(box) != 4:
                continue
            start = text.find(line_text)
            if start < 0:
                continue
            x, y, box_width, box_height = (float(value) for value in box)
            padding = max(8, round(height * 0.004))
            crop_box = (
                max(0, round(x * width) - padding),
                max(0, round((1 - y - box_height) * height) - padding),
                min(width, round((x + box_width) * width) + padding),
                min(height, round((1 - y) * height) + padding),
            )
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                continue
            asset = f"formula-fallback-{len(events) + 1:02d}.png"
            image.crop(crop_box).save(artifact_dir / asset)
            events.append(
                (
                    start,
                    start + len(line_text),
                    {
                        "type": "image",
                        "asset": asset,
                        "alt": "formula",
                        "source_box": list(crop_box),
                        "source_text": line_text,
                    },
                )
            )
    if not events:
        return build_content_blocks(
            text,
            figure_asset=figure_asset,
            figure_box=figure_box,
        )
    blocks: list[dict[str, Any]] = []
    cursor = 0
    for start, end, image_block in sorted(events, key=lambda item: item[0]):
        if start < cursor:
            continue
        blocks.extend(split_text_and_latex(text[cursor:start]))
        blocks.append(image_block)
        cursor = end
    blocks.extend(split_text_and_latex(text[cursor:]))
    if figure_asset:
        blocks.append(
            {
                "type": "image",
                "asset": figure_asset,
                "alt": "question figure",
                "source_box": list(figure_box or []),
            }
        )
    return {"version": 1, "blocks": blocks}


def content_blocks_for_problem(problem: dict[str, Any]) -> dict[str, Any]:
    existing = problem.get("content_blocks")
    if isinstance(existing, dict) and existing.get("version") in {1, 2}:
        blocks = existing.get("blocks")
        if isinstance(blocks, list):
            if existing.get("version") == 2:
                metrics = problem.get("metrics") or {}
                structured_data = metrics.get("structured_problem") or {}
                title_hint = (
                    str(structured_data.get("title") or "")
                    if isinstance(structured_data, dict)
                    else ""
                )
                structured = build_structured_problem(
                    str(problem.get("ocr_text") or ""),
                    title_hint=title_hint or problem.get("split_label") or None,
                )
                if title_hint:
                    structured.title = title_hint
                return question_content_blocks(structured, existing)
            return existing
    metrics = problem.get("metrics") or {}
    structured = metrics.get("structured_problem") or {}
    figure = structured.get("figure") if isinstance(structured, dict) else "none"
    artifact_dir = Path(str(problem.get("selected_artifact") or "")).parent
    figure_path = artifact_dir / "figure-selected.png"
    figure_metrics = metrics.get("figure_preservation") or {}
    figure_asset = (
        figure_path.name
        if figure and figure != "none" and figure_path.is_file()
        else None
    )
    figure_box = (
        figure_metrics.get("box")
        if isinstance(figure_metrics, dict)
        else None
    )
    return build_content_blocks(
        str(problem.get("ocr_text") or ""),
        figure_asset=figure_asset,
        figure_box=figure_box if isinstance(figure_box, list) else None,
    )


def content_block_rows(model: dict[str, Any]) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_row: int | None = None

    def flush() -> None:
        nonlocal current, current_row
        if current:
            rows.append(current)
        current = []
        current_row = None

    for original in model.get("blocks") or []:
        if not isinstance(original, dict):
            continue
        block = dict(original)
        block_type = block.get("type")
        if block_type == "image" or (
            block_type == "latex" and bool(block.get("display"))
        ):
            flush()
            rows.append([block])
            continue
        row_index = block.get("row_index")
        if (
            isinstance(row_index, int)
            and current
            and current_row is not None
            and row_index != current_row
        ):
            flush()
        if isinstance(row_index, int):
            current_row = row_index
        if block_type != "text" or "\n" not in str(block.get("text") or ""):
            current.append(block)
            continue
        pieces = str(block.get("text") or "").split("\n")
        for index, piece in enumerate(pieces):
            if piece:
                current.append({**block, "text": piece})
            if index < len(pieces) - 1:
                flush()
    flush()
    return rows


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
