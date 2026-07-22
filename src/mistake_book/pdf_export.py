from __future__ import annotations

from collections import defaultdict
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
CORRECTION_HEIGHT = 45 * mm


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
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for problem in eligible:
        grouped[str(problem["category"])].append(problem)

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
    page_number = 2
    for category, items in grouped.items():
        if y < 25 * mm:
            document.showPage()
            y = PAGE_HEIGHT - 25 * mm
            page_number += 1
        document.setFont(FONT_NAME, 11)
        document.drawString(MARGIN, y, category)
        document.drawRightString(PAGE_WIDTH - MARGIN, y, str(page_number))
        y -= 7 * mm
        page_number += len(items)
    if skipped:
        y -= 4 * mm
        document.setFont(FONT_NAME, 9)
        document.drawString(MARGIN, y, f"本次部分导出跳过 {len(skipped)} 道待确认题目")
    document.showPage()

    number = 1
    for category, items in grouped.items():
        for problem in items:
            document.setFont(FONT_NAME, 13)
            document.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 5 * mm, f"{number}. {category}")
            image_path = Path(str(problem["selected_artifact"]))
            with Image.open(image_path) as image:
                pixel_width, pixel_height = image.size
            available_width = PAGE_WIDTH - 2 * MARGIN
            available_height = PAGE_HEIGHT - 2 * MARGIN - CORRECTION_HEIGHT - 15 * mm
            scale = min(available_width / pixel_width, available_height / pixel_height)
            draw_width = pixel_width * scale
            draw_height = pixel_height * scale
            x = (PAGE_WIDTH - draw_width) / 2
            image_top = PAGE_HEIGHT - MARGIN - 12 * mm
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
            line_top = max(MARGIN + CORRECTION_HEIGHT, image_y - 7 * mm)
            document.setFont(FONT_NAME, 9)
            document.drawString(MARGIN, line_top, "订正：")
            line_y = line_top - 7 * mm
            while line_y >= MARGIN:
                document.setStrokeColorRGB(0.82, 0.82, 0.82)
                document.line(MARGIN, line_y, PAGE_WIDTH - MARGIN, line_y)
                line_y -= 8 * mm
            document.setFillColorRGB(0.45, 0.45, 0.45)
            document.setFont(FONT_NAME, 8)
            document.drawRightString(
                PAGE_WIDTH - MARGIN,
                7 * mm,
                f"{number} / {len(eligible)}",
            )
            document.setFillColorRGB(0, 0, 0)
            document.showPage()
            number += 1

    document.save()
    return target
