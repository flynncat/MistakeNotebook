from __future__ import annotations

import difflib
import io
import os
import re
import shutil
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .font_selection import load_print_font


FigureKind = Literal["none", "three_overlapping_circles", "five_country_map", "grid_2x10"]
CountryEdge = tuple[str, str]

_QUESTION_TERMINAL = re.compile(
    r"(?:[\uff1f?]|"
    r"(?:\u6c42|\u8bf7\u95ee|\u95ee|\u8ba1\u7b97|\u8bc1\u660e|\u5224\u65ad|"
    r"\u662f\u5426|\u591a\u5c11|\u51e0(?:\u4e2a|\u540d|\u79cd|\u679a|\u672c|\u6b21)?)"
    r"[^\u3002\uff01!?\uff1f\n]{0,48}[\u3002\uff01!.])"
)


def question_terminal_match(text: str) -> re.Match[str] | None:
    return _QUESTION_TERMINAL.search(text)


@dataclass
class StructuredProblem:
    title: str
    body: str
    figure: FigureKind
    primary_text: str
    secondary_text: str
    secondary_raw_text: str
    text_similarity: float
    primary_numbers: list[str]
    secondary_numbers: list[str]
    figure_edges: list[CountryEdge]
    review_reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def tesseract_ocr(image_path: Path) -> str:
    configured = os.getenv("TESSERACT_CMD", "").strip()
    candidates = [
        configured,
        shutil.which("tesseract") or "",
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    executable = next(
        (candidate for candidate in candidates if candidate and Path(candidate).is_file()),
        None,
    )
    if not executable:
        raise RuntimeError("未找到 Tesseract 可执行文件")
    result = subprocess.run(
        [
            executable,
            str(image_path),
            "stdout",
            "-l",
            "chi_sim+eng",
            "--psm",
            "6",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Tesseract OCR 失败")
    return result.stdout.strip()


def question_text_box(
    lines: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    start: int | None = None
    for index, line in enumerate(lines):
        if re.search(
            r"(?:[\u3010\[\(\uff08]?(?:\u4f8b\u9898|\u4f8b|\u7ec3\u4e60)\s*[0-9S]+|^\s*\u5217\s*[0-9S]+\s*[\u3001.\uff0e]|^\s*[1-9][0-9]?\s*[\u3001.\uff0e])",
            str(line.get("text", "")),
            re.IGNORECASE,
        ):
            start = index
            break
    if start is None:
        return None

    selected: list[dict[str, Any]] = []
    found_question_end = False
    for line in lines[start:]:
        selected.append(line)
        if question_terminal_match(str(line.get("text", ""))):
            found_question_end = True
            break
    if not found_question_end:
        return None

    width, height = image_size
    boxes: list[tuple[int, int, int, int]] = []
    for line in selected:
        box = line.get("box", [])
        if len(box) != 4:
            continue
        x, y, box_width, box_height = (float(value) for value in box)
        boxes.append(
            (
                max(0, round(x * width)),
                max(0, round((1 - y - box_height) * height)),
                min(width, round((x + box_width) * width)),
                min(height, round((1 - y) * height)),
            )
        )
    if not boxes:
        return None
    horizontal_padding = max(12, round(width * 0.012))
    vertical_padding = max(8, round(height * 0.008))
    return (
        max(0, min(box[0] for box in boxes) - horizontal_padding),
        max(0, min(box[1] for box in boxes) - vertical_padding),
        min(width, max(box[2] for box in boxes) + horizontal_padding),
        min(height, max(box[3] for box in boxes) + vertical_padding),
    )


def tesseract_question_ocr(
    image_path: Path,
    lines: list[dict[str, Any]],
    crop_path: Path,
) -> tuple[str, list[int] | None]:
    with Image.open(image_path) as image:
        box = question_text_box(lines, image.size)
        if box is None:
            return "", None
        question = ImageOps.autocontrast(ImageOps.grayscale(image.crop(box)))
        question = question.resize(
            (question.width * 2, question.height * 2),
            Image.Resampling.LANCZOS,
        )
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        question.save(crop_path)
    return tesseract_ocr(crop_path), list(box)


def ordered_question_text(lines: list[dict[str, Any]]) -> str:
    entries: list[dict[str, Any]] = []
    for line in lines:
        box = line.get("box", [])
        text = str(line.get("text", "")).strip()
        if len(box) != 4 or not text:
            continue
        x, y, width, height = (float(value) for value in box)
        entries.append(
            {
                "x0": x,
                "x1": x + width,
                "top": 1 - y - height,
                "bottom": 1 - y,
                "baseline": y,
                "text": text,
                "confidence": float(line.get("confidence", 0)),
            }
        )
    entries.sort(key=lambda item: (item["top"], item["x0"]))
    rows: list[list[dict[str, Any]]] = []
    for entry in entries:
        best_row: list[dict[str, Any]] | None = None
        best_overlap = 0.0
        for row in rows:
            row_top = min(item["top"] for item in row)
            row_bottom = max(item["bottom"] for item in row)
            vertical = max(
                0.0,
                min(entry["bottom"], row_bottom) - max(entry["top"], row_top),
            )
            denominator = max(
                0.001,
                min(entry["bottom"] - entry["top"], row_bottom - row_top),
            )
            vertical_ratio = vertical / denominator
            horizontal = max(
                0.0,
                min(entry["x1"], max(item["x1"] for item in row))
                - max(entry["x0"], min(item["x0"] for item in row)),
            )
            entry_width = max(0.001, entry["x1"] - entry["x0"])
            row_width = max(item["x1"] for item in row) - min(
                item["x0"] for item in row
            )
            horizontal_ratio = horizontal / max(0.001, min(entry_width, row_width))
            if (
                vertical_ratio >= 0.28
                and horizontal_ratio < 0.18
                and vertical_ratio > best_overlap
            ):
                best_row = row
                best_overlap = vertical_ratio
        if best_row is None:
            rows.append([entry])
        else:
            best_row.append(entry)

    rows.sort(
        key=lambda row: -statistics.median(item["baseline"] for item in row)
    )
    row_texts = [
        "".join(item["text"] for item in sorted(row, key=lambda item: item["x0"]))
        for row in rows
    ]
    start = next(
        (index for index, text in enumerate(row_texts) if _QUESTION_START.search(text)),
        None,
    )
    if start is None:
        return ""
    question_index = next(
        (
            index
            for index in range(start, len(row_texts))
            if question_terminal_match(row_texts[index])
        ),
        None,
    )
    if question_index is None:
        return "\n".join(row_texts[start:])
    trailing: list[str] = []
    previous_baseline = statistics.median(
        item["baseline"] for item in rows[question_index]
    )
    for index in range(question_index + 1, len(rows)):
        baseline = statistics.median(item["baseline"] for item in rows[index])
        confidence = max(item["confidence"] for item in rows[index])
        if previous_baseline - baseline > 0.045 or confidence < 0.45:
            break
        trailing.append(row_texts[index])
        previous_baseline = baseline
    selected = [
        *row_texts[start:question_index],
        *trailing,
        row_texts[question_index],
    ]
    return "\n".join(selected)


_QUESTION_START = re.compile(
    r"(?:[\u3010\[\(\uff08]?(?:\u4f8b\u9898|\u4f8b|\u7ec3\u4e60)\s*[0-9S]+"
    r"[\u3011\]\)\uff09]?|"
    r"\u5217\s*[0-9S]+\s*[\u3001.\uff0e]|"
    r"(?:^|\n)\s*[1-9][0-9]?\s*[\u3001.\uff0e])",
    re.IGNORECASE,
)


def _question_fragment(
    text: str,
    title_hint: str | None = None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    match = _QUESTION_START.search(text)
    if not match:
        if title_hint and text.strip():
            fragment = f"{title_hint} {text.strip()}"
        else:
            return "", ["未定位到印刷题号"]
    elif title_hint:
        fragment = f"{title_hint} {text[match.end():]}"
    else:
        fragment = text[match.start() :]
    end = question_terminal_match(fragment)
    if end:
        fragment = fragment[: end.end()]
    else:
        reasons.append("未定位到题干问号")
    fragment = re.sub(r"[ \t]+", " ", fragment)
    fragment = re.sub(r"\s*\n\s*", "", fragment)
    return fragment.strip(), reasons


def _normalized_for_compare(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text).lower()


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?::\d+)?", text)


def _figure_kind(body: str) -> FigureKind:
    compact = re.sub(r"\s+", "", body)
    if "三个圆圈" in compact and ("交叠" in compact or "重叠" in compact):
        return "three_overlapping_circles"
    if "地图" in compact and "五个国家" in compact:
        return "five_country_map"
    if re.search(r"2\s*[×xX*]\s*10\s*方格", compact):
        return "grid_2x10"
    return "none"


def _figure_edges(figure: FigureKind) -> list[CountryEdge]:
    return []


def build_structured_problem(
    primary_text: str,
    secondary_text: str = "",
    *,
    title_hint: str | None = None,
) -> StructuredProblem:
    primary, reasons = _question_fragment(primary_text, title_hint)
    secondary, _ = _question_fragment(secondary_text, title_hint)
    title_match = re.match(
        r"[【\[\(（]?((?:例题|练习)\s*([0-9S]+)|第\s*([0-9]+)\s*题)[】\]］\)）]?\s*",
        primary,
        re.IGNORECASE,
    )
    if title_match:
        if title_match.group(3):
            title = f"【第{title_match.group(3)}题】"
        else:
            number = title_match.group(2).upper().replace("S", "5")
            prefix = "练习" if title_match.group(1).startswith("练习") else "例题"
            title = f"【{prefix}{number}】"
        body = primary[title_match.end() :].strip().lstrip("\u3001\uff0c,. \t")
    else:
        title = "【题目】"
        body = primary
        reasons.append("题号格式未可靠恢复")
    primary_compare = _normalized_for_compare(primary)
    secondary_compare = _normalized_for_compare(secondary)
    similarity = (
        difflib.SequenceMatcher(None, primary_compare, secondary_compare).ratio()
        if primary_compare and secondary_compare
        else 0.0
    )
    primary_numbers = _numbers(primary)
    secondary_numbers = _numbers(secondary)
    if secondary and similarity >= 0.82 and primary_numbers != secondary_numbers:
        reasons.append("双 OCR 数字或比例不一致")
    if len(body) < 12:
        reasons.append("重建题干过短")
    figure = _figure_kind(body)
    return StructuredProblem(
        title=title,
        body=body,
        figure=figure,
        primary_text=primary,
        secondary_text=secondary,
        secondary_raw_text=secondary_text.strip(),
        text_similarity=round(similarity, 4),
        primary_numbers=primary_numbers,
        secondary_numbers=secondary_numbers,
        figure_edges=_figure_edges(figure),
        review_reasons=list(dict.fromkeys(reasons)),
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    font, _ = load_print_font(size, bold=bold)
    return font


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for character in text:
        if character == "\n":
            if current:
                lines.append(current)
                current = ""
            continue
        candidate = current + character
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def render_problem(
    problem: StructuredProblem,
    target: Path,
    *,
    width: int = 1800,
    figure_path: Path | None = None,
) -> Path:
    margin = 110
    title_font = _font(52, bold=True)
    body_font = _font(48)
    line_height = 76
    probe = Image.new("RGB", (width, 100), "white")
    probe_draw = ImageDraw.Draw(probe)
    lines = _wrap_text(probe_draw, problem.body, body_font, width - 2 * margin)
    figure_image: Image.Image | None = None
    figure_width = 0
    figure_height = 0
    if problem.figure != "none":
        if figure_path is None or not figure_path.exists():
            raise ValueError("题目需要配图，但没有通过保真校验的原图裁剪")
        with Image.open(figure_path) as source:
            figure_image = source.convert("RGB")
        scale = min(
            (width - 2 * margin) / figure_image.width,
            430 / figure_image.height,
        )
        figure_width = max(1, round(figure_image.width * scale))
        figure_height = max(1, round(figure_image.height * scale))
        figure_image = figure_image.resize(
            (figure_width, figure_height),
            Image.Resampling.LANCZOS,
        )
    height = margin * 2 + 90 + len(lines) * line_height
    if figure_height:
        height += figure_height + 90
    canvas = Image.new("RGB", (width, max(500, height)), "white")
    draw = ImageDraw.Draw(canvas)
    y = margin
    draw.text((margin, y), problem.title, font=title_font, fill="black")
    y += 90
    for line in lines:
        draw.text((margin, y), line, font=body_font, fill="black")
        y += line_height
    if figure_image is not None:
        canvas.paste(figure_image, ((width - figure_width) // 2, y + 30))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
    return target


def _question_content_blocks(
    problem: StructuredProblem,
    content_blocks: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = [
        dict(block)
        for block in content_blocks.get("blocks") or []
        if isinstance(block, dict)
    ]
    selected: list[dict[str, Any]] = []
    figures = [block for block in blocks if block.get("type") == "image"]
    started = False
    accumulated = ""
    title_pattern = re.compile(
        r"[\u3010\[\(\uff08]?(?:\u4f8b\u9898|\u7ec3\u4e60|\u7b2c)\s*"
        r"[0-9S]+\s*(?:\u9898)?[\u3011\]\)\uff09]?",
        re.IGNORECASE,
    )
    for block in blocks:
        if block.get("type") == "image":
            continue
        if block.get("type") == "text":
            text = str(block.get("text") or "")
            if not started:
                title = problem.title.strip()
                title_at = text.find(title)
                match = title_pattern.search(text)
                if title_at >= 0:
                    text = text[title_at + len(title) :]
                    started = True
                elif match:
                    text = text[match.end() :]
                    started = True
                else:
                    continue
            if text:
                block["text"] = text
                selected.append(block)
                accumulated += text
        elif block.get("type") == "latex" and started:
            selected.append(block)
            accumulated += " FORMULA "
        if started and question_terminal_match(accumulated):
            break
    if not started:
        selected = [
            block
            for block in blocks
            if block.get("type") in {"text", "latex"}
        ]
    return selected, figures


def question_content_blocks(
    problem: StructuredProblem,
    content_blocks: dict[str, Any],
) -> dict[str, Any]:
    selected, figures = _question_content_blocks(problem, content_blocks)
    return {
        **content_blocks,
        "blocks": [*selected, *figures],
    }


def content_blocks_to_text(
    problem: StructuredProblem,
    content_blocks: dict[str, Any],
) -> str:
    blocks, _ = _question_content_blocks(problem, content_blocks)
    pieces: list[str] = []
    previous_row: int | None = None
    for block in blocks:
        row = block.get("row_index")
        if isinstance(row, int) and previous_row is not None and row != previous_row:
            pieces.append("\n")
        if isinstance(row, int):
            previous_row = row
        if block.get("type") == "text":
            pieces.append(str(block.get("text") or ""))
        elif block.get("type") == "latex":
            if block.get("recognition_state") == "human_verified_image":
                pieces.append("\uff3b\u516c\u5f0f\u56fe\u50cf\uff3d")
                continue
            latex = str(block.get("latex") or "").strip()
            if latex:
                pieces.append(
                    f"\\[{latex}\\]" if block.get("display") else f"\\({latex}\\)"
                )
    body = "".join(pieces).strip()
    body = re.sub(r"(\\\))[\s.\u3002\uff0eoO]+(?=[,\uff0c])", r"\1", body)
    end = question_terminal_match(body)
    if end:
        body = body[: end.end()].strip()
    return f"{problem.title} {body}".strip()


def _formula_image(
    block: dict[str, Any],
    artifact_dir: Path,
    maximum_height: int,
) -> Image.Image:
    fallback_name = str(block.get("clean_crop_asset") or "")
    fallback = artifact_dir / fallback_name if Path(fallback_name).name == fallback_name else None
    state = str(block.get("recognition_state") or "")
    latex = str(block.get("latex") or "").strip()
    if latex and state in {"auto_verified", "human_verified"}:
        try:
            from matplotlib.mathtext import math_to_image

            render_latex = re.sub(
                r"\\(?:big|Big|bigg|Bigg)l\b",
                r"\\left",
                latex,
            )
            render_latex = re.sub(
                r"\\(?:big|Big|bigg|Bigg)r\b",
                r"\\right",
                render_latex,
            )
            render_latex = re.sub(
                r"\\(?:big|Big|bigg|Bigg)\b",
                "",
                render_latex,
            )
            output = io.BytesIO()
            math_to_image(
                f"${render_latex}$",
                output,
                dpi=180,
                format="png",
                color="black",
            )
            output.seek(0)
            image = Image.open(output).convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            image = background.convert("RGB")
        except Exception:
            image = None
    else:
        image = None
    if image is None:
        if fallback is None or not fallback.is_file() or fallback.is_symlink():
            raise ValueError("公式无法渲染，且没有可用的白底公式截图")
        image = Image.open(fallback).convert("RGB")
    target_height = min(maximum_height, max(54, image.height))
    if image.height != target_height:
        scale = target_height / image.height
        image = image.resize(
            (max(1, round(image.width * scale)), target_height),
            Image.Resampling.LANCZOS,
        )
    return image


def render_problem_content_blocks(
    problem: StructuredProblem,
    content_blocks: dict[str, Any],
    artifact_dir: Path,
    target: Path,
    *,
    width: int = 1800,
) -> Path:
    margin = 110
    title_font = _font(52, bold=True)
    body_font = _font(48)
    line_height = 86
    maximum_formula_height = 76
    blocks, figures = _question_content_blocks(problem, content_blocks)
    canvas = Image.new("RGB", (width, 6000), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), problem.title, font=title_font, fill="black")
    x = margin
    y = margin + 96
    right = width - margin
    previous_row: int | None = None

    def next_line() -> None:
        nonlocal x, y
        x = margin
        y += line_height

    for block in blocks:
        row = block.get("row_index")
        if isinstance(row, int) and previous_row is not None and row != previous_row:
            next_line()
        if isinstance(row, int):
            previous_row = row
        if block.get("type") == "text":
            for character in str(block.get("text") or ""):
                if character == "\n":
                    next_line()
                    continue
                character_width = max(1, round(draw.textlength(character, font=body_font)))
                if x + character_width > right and x > margin:
                    next_line()
                draw.text((x, y), character, font=body_font, fill="black")
                x += character_width
            continue
        if block.get("type") != "latex":
            continue
        formula = _formula_image(block, artifact_dir, maximum_formula_height)
        if block.get("display"):
            if x > margin:
                next_line()
            formula_x = max(margin, (width - formula.width) // 2)
            canvas.paste(formula, (formula_x, y))
            next_line()
            continue
        if x + formula.width > right and x > margin:
            next_line()
        formula_y = y + max(0, (line_height - formula.height) // 2)
        canvas.paste(formula, (x, formula_y))
        x += formula.width

    if x > margin:
        next_line()
    for figure in figures:
        asset = str(figure.get("asset") or "")
        if not asset or Path(asset).name != asset:
            continue
        path = artifact_dir / asset
        if not path.is_file() or path.is_symlink():
            continue
        image = Image.open(path).convert("RGB")
        scale = min((width - 2 * margin) / image.width, 430 / image.height)
        image = image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        y += 30
        canvas.paste(image, ((width - image.width) // 2, y))
        y += image.height + 30
    final_height = max(500, min(canvas.height, y + margin))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.crop((0, 0, width, final_height)).save(
        target,
        format="PNG",
        optimize=True,
    )
    return target
