from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

FONT_NAME = "STSong-Light"
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 15 * mm
PROBLEM_SLOT_HEIGHT = 118 * mm
MAX_QUESTION_HEIGHT = 62 * mm


def _group_problems(
    problems: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for problem in problems:
        group = str(problem.get("category_group") or "未分类")
        category = str(problem.get("category_key") or problem["category"])
        grouped.setdefault(group, {}).setdefault(category, []).append(problem)
    return grouped


def _register_font() -> None:
    try:
        pdfmetrics.getFont(FONT_NAME)
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))


def _eligible(problem: dict[str, Any]) -> bool:
    return bool(
        problem.get("selected_artifact")
        and problem.get("category")
        and problem.get("ocr_text")
        and (
            problem.get("status") == "ready"
            or problem.get("review_status") == "accepted"
        )
    )


def export_pdf(
    batch_id: str,
    problems: list[dict[str, Any]],
    output_dir: Path,
    *,
    allow_partial: bool = False,
) -> Path:
    eligible = [problem for problem in problems if _eligible(problem)]
    skipped = [problem for problem in problems if not _eligible(problem)]
    if skipped and not allow_partial:
        raise ValueError(f"仍有 {len(skipped)} 道题未通过确认，不能生成最终 PDF")
    if not eligible:
        raise ValueError("没有可导出的题目")

    _register_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"错题集-{batch_id[:8]}.pdf"
    grouped = _group_problems(eligible)

    document = canvas.Canvas(str(target), pagesize=A4, pageCompression=1)
    document.setTitle("小奥错题集")
    document.setAuthor("错题整理工具")

    document.setFont(FONT_NAME, 22)
    document.drawString(MARGIN, PAGE_HEIGHT - 32 * mm, "小奥错题集")
    document.setFont(FONT_NAME, 11)
    document.drawString(MARGIN, PAGE_HEIGHT - 42 * mm, f"共 {len(eligible)} 道题")
    document.setFont(FONT_NAME, 15)
    document.drawString(MARGIN, PAGE_HEIGHT - 58 * mm, "目录")
    y = PAGE_HEIGHT - 70 * mm
    for group, categories in grouped.items():
        if y < 32 * mm:
            document.showPage()
            y = PAGE_HEIGHT - 25 * mm
        document.setFont(FONT_NAME, 13)
        document.drawString(MARGIN, y, group)
        y -= 7 * mm
        for category, items in categories.items():
            if y < 25 * mm:
                document.showPage()
                y = PAGE_HEIGHT - 25 * mm
                document.setFont(FONT_NAME, 11)
                document.drawString(MARGIN, y, f"{group}（续）")
                y -= 7 * mm
            document.setFont(FONT_NAME, 11)
            document.drawString(MARGIN + 8 * mm, y, category)
            document.drawRightString(PAGE_WIDTH - MARGIN, y, f"{len(items)} 题")
            y -= 7 * mm
    if skipped:
        y -= 4 * mm
        document.setFont(FONT_NAME, 9)
        document.drawString(MARGIN, y, f"本次部分导出跳过 {len(skipped)} 道待确认题目")
    document.setFillColorRGB(0.45, 0.45, 0.45)
    document.setFont(FONT_NAME, 8)
    document.drawRightString(PAGE_WIDTH - MARGIN, 7 * mm, "1")
    document.setFillColorRGB(0, 0, 0)
    document.showPage()

    page_number = 2
    y = PAGE_HEIGHT - MARGIN
    number = 1
    items_on_page = 0
    for group, categories in grouped.items():
        for category, items in categories.items():
            section = f"{group} / {category}"
            if items_on_page >= 2:
                _draw_page_footer(document, page_number)
                document.showPage()
                page_number += 1
                y = PAGE_HEIGHT - MARGIN
                items_on_page = 0
            elif y < PAGE_HEIGHT - MARGIN:
                if y < 42 * mm:
                    _draw_page_footer(document, page_number)
                    document.showPage()
                    page_number += 1
                    y = PAGE_HEIGHT - MARGIN
                    items_on_page = 0
                else:
                    y -= 5 * mm
            document.setFont(FONT_NAME, 15)
            document.drawString(MARGIN, y - 5 * mm, section)
            y -= 14 * mm
            for problem in items:
                image_path = Path(str(problem["selected_artifact"]))
                with Image.open(image_path) as image:
                    pixel_width, pixel_height = image.size
                available_width = PAGE_WIDTH - 2 * MARGIN
                scale = min(
                    available_width / pixel_width,
                    MAX_QUESTION_HEIGHT / pixel_height,
                )
                draw_width = pixel_width * scale
                draw_height = pixel_height * scale
                block_height = PROBLEM_SLOT_HEIGHT
                if items_on_page >= 2 or y - block_height < MARGIN:
                    _draw_page_footer(document, page_number)
                    document.showPage()
                    page_number += 1
                    y = PAGE_HEIGHT - MARGIN
                    items_on_page = 0
                    document.setFont(FONT_NAME, 10)
                    document.setFillColorRGB(0.4, 0.4, 0.4)
                    document.drawString(MARGIN, y - 4 * mm, section)
                    document.setFillColorRGB(0, 0, 0)
                    y -= 10 * mm
                document.setFont(FONT_NAME, 12)
                document.drawString(MARGIN, y - 5 * mm, f"{number}.")
                x = (PAGE_WIDTH - draw_width) / 2
                image_top = y - 10 * mm
                image_y = image_top - draw_height
                document.drawImage(
                    str(image_path),
                    x,
                    image_y,
                    width=draw_width,
                    height=draw_height,
                    preserveAspectRatio=True,
                    mask="auto",
                )
                y -= block_height
                number += 1
                items_on_page += 1

    _draw_page_footer(document, page_number)
    document.save()
    return target


def _draw_page_footer(document: canvas.Canvas, page_number: int) -> None:
    document.setFillColorRGB(0.45, 0.45, 0.45)
    document.setFont(FONT_NAME, 8)
    document.drawCentredString(PAGE_WIDTH / 2, 7 * mm, str(page_number))
    document.setFillColorRGB(0, 0, 0)
