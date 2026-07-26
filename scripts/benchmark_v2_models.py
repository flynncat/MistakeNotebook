from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mistake_book.config import Settings
from mistake_book.figure_preservation import preserve_figure
from mistake_book.image_pipeline import load_image, orient_image
from mistake_book.recognition import RecognitionService
from mistake_book.reconstruction import (
    build_structured_problem,
    render_problem,
    tesseract_ocr,
)
from mistake_book.v2_models import (
    DISHandwritingAdapter,
    UVDocAdapter,
    erase_handwriting,
    load_manifest,
    make_print_layer,
)


def _save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"无法保存：{path}")


def _residual_red_fraction(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = (
        (hsv[:, :, 1] > 55)
        & ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 168))
        & (bgr[:, :, 2].astype(np.int16) - bgr[:, :, 1].astype(np.int16) > 12)
    )
    return float(red.mean())


def run_benchmark(root: Path, output: Path, limit: int | None = None) -> dict:
    settings = Settings.load(root)
    recognizer = RecognitionService(settings)
    uvdoc = UVDocAdapter(root)
    handwriting = DISHandwritingAdapter(root)
    sample_paths = sorted((root / "Sample").glob("*"))
    sample_paths = [
        path
        for path in sample_paths
        if path.suffix.lower() in {".heic", ".jpg", ".jpeg", ".png", ".webp"}
    ]
    if limit:
        sample_paths = sample_paths[:limit]
    if not sample_paths:
        raise RuntimeError("Sample 中没有测试图片")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    started_all = time.perf_counter()

    for source in sample_paths:
        item_dir = output / source.stem
        item_dir.mkdir(parents=True, exist_ok=True)
        original = load_image(source)
        rotation_hint = recognizer.rotation_hint(source)
        oriented, orientation_metrics = orient_image(original, rotation_hint)
        _save(item_dir / "01-oriented.png", oriented)

        dewarped = uvdoc.unwarp(oriented)
        _save(item_dir / "02-uvdoc.png", dewarped.image)
        if handwriting.available():
            mask_run = handwriting.predict_mask(dewarped.image)
            mask_image = mask_run.image
            erased = erase_handwriting(dewarped.image, mask_image)
            dis_seconds = round(mask_run.seconds, 3)
            dis_device = mask_run.device
        else:
            mask_image = np.zeros(dewarped.image.shape[:2], dtype=np.uint8)
            erased = dewarped.image.copy()
            dis_seconds = 0.0
            dis_device = "skipped_rejected_model"
        _save(item_dir / "03-handwriting-mask.png", mask_image)
        _save(item_dir / "04-erased.png", erased)
        print_layer = make_print_layer(erased)
        _save(item_dir / "05-print-layer.png", print_layer)

        original_text, original_confidence, original_lines = recognizer.local_ocr.recognize(
            item_dir / "02-uvdoc.png"
        )
        print_text, print_confidence, _ = recognizer.local_ocr.recognize(
            item_dir / "05-print-layer.png"
        )
        try:
            secondary_text = tesseract_ocr(item_dir / "02-uvdoc.png")
        except RuntimeError as error:
            secondary_text = ""
            secondary_error = str(error)
        else:
            secondary_error = ""
        structured = build_structured_problem(original_text, secondary_text)
        figure = (
            preserve_figure(
                item_dir / "02-uvdoc.png",
                original_lines,
                structured.primary_text,
                item_dir,
                recognizer.local_ocr,
            )
            if structured.figure != "none"
            else None
        )
        if figure is not None and not figure.metrics.get("passed", False):
            raise RuntimeError(
                f"{source.name} 配图重建未通过质量门禁："
                + "；".join(figure.review_reasons)
            )
        if figure is not None and structured.figure == "five_country_map":
            structured.figure_edges = [
                tuple(edge)
                for edge in figure.metrics.get("adjacency_edges", [])
                if len(edge) == 2
            ]
        render_problem(
            structured,
            item_dir / "06-reconstructed.png",
            figure_path=figure.selected_path if figure else None,
        )
        mask_probability = mask_image.astype(np.float32) / 255
        result = {
            "file": source.name,
            "rotation": orientation_metrics,
            "uvdoc_seconds": round(dewarped.seconds, 3),
            "uvdoc_device": dewarped.device,
            "dis_seconds": dis_seconds,
            "dis_device": dis_device,
            "mask_fraction_030": round(float((mask_probability >= 0.3).mean()), 6),
            "mask_fraction_050": round(float((mask_probability >= 0.5).mean()), 6),
            "mask_fraction_070": round(float((mask_probability >= 0.7).mean()), 6),
            "residual_red_fraction": round(_residual_red_fraction(erased), 6),
            "print_white_mean": round(float(cv2.cvtColor(print_layer, cv2.COLOR_BGR2GRAY).mean()), 3),
            "ocr_original": original_text,
            "ocr_original_confidence": round(original_confidence, 4),
            "ocr_print": print_text,
            "ocr_print_confidence": round(print_confidence, 4),
            "ocr_secondary": secondary_text,
            "ocr_secondary_error": secondary_error,
            "structured": structured.to_dict(),
            "figure_preservation": figure.metrics if figure else None,
            "images": {
                "oriented": f"{source.stem}/01-oriented.png",
                "uvdoc": f"{source.stem}/02-uvdoc.png",
                "mask": f"{source.stem}/03-handwriting-mask.png",
                "erased": f"{source.stem}/04-erased.png",
                "print": f"{source.stem}/05-print-layer.png",
                "reconstructed": f"{source.stem}/06-reconstructed.png",
            },
        }
        results.append(result)

    figure_results = [
        item["figure_preservation"]
        for item in results
        if item["figure_preservation"] is not None
    ]
    all_figure_gates_passed = bool(figure_results) and all(
        item.get("passed", False) for item in figure_results
    )
    report = {
        "schema_version": 1,
        "manifest": load_manifest(root),
        "sample_count": len(results),
        "total_seconds": round(time.perf_counter() - started_all, 3),
        "results": results,
        "quality_status": "sample_figure_gates_passed"
        if all_figure_gates_passed
        else "figure_gate_failed",
        "figure_gate_count": len(figure_results),
        "figure_gate_passed_count": sum(
            bool(item.get("passed", False)) for item in figure_results
        ),
        "text_review_count": sum(
            bool(item["structured"]["review_reasons"]) for item in results
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_html(output / "report.html", report)
    _write_confirmation_pdf(output / "clean-confirmation.pdf", output, report)
    return report


def _write_html(path: Path, report: dict) -> None:
    cards = []
    for item in report["results"]:
        images = "".join(
            f"<figure><figcaption>{label}</figcaption><img src='{html.escape(url)}'></figure>"
            for key, label in (
                ("oriented", "方向校正"),
                ("uvdoc", "UVDoc 展平"),
                ("mask", "DIS 手写概率"),
                ("erased", "模型擦除"),
                ("print", "印刷层"),
                ("reconstructed", "结构化重建"),
            )
            for url in [item["images"][key]]
        )
        cards.append(
            f"""
<article>
  <h2>{html.escape(item['file'])}</h2>
  <p>UVDoc {item['uvdoc_seconds']}s / {item['uvdoc_device']}；
     DIS {item['dis_seconds']}s / {item['dis_device']}；
     mask≥0.5 {item['mask_fraction_050']:.2%}；
     残余红色 {item['residual_red_fraction']:.2%}</p>
  <div class="images">{images}</div>
  <details><summary>OCR 对比</summary>
    <h3>展平图</h3><pre>{html.escape(item['ocr_original'])}</pre>
    <h3>印刷层</h3><pre>{html.escape(item['ocr_print'])}</pre>
  </details>
</article>"""
        )
    path.write_text(
        f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>V2 模型基准</title><style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;background:#eeeae1;margin:24px;color:#222}}
main{{max-width:1500px;margin:auto}}article{{background:#fff;padding:18px;margin:18px 0;border-radius:12px}}
.images{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}figure{{margin:0;min-width:0}}
figcaption{{font-size:12px;color:#666}}img{{width:100%;height:320px;object-fit:contain;background:#ddd}}
pre{{white-space:pre-wrap}}@media(max-width:900px){{.images{{grid-template-columns:1fr 1fr}}}}
</style><main><h1>V2 模型基准</h1>
<p>状态：{report['quality_status']}；样例 {report['sample_count']}；总耗时 {report['total_seconds']}s。</p>
{''.join(cards)}</main></html>""",
        encoding="utf-8",
    )


def _write_confirmation_pdf(path: Path, output: Path, report: dict) -> None:
    pages: list[Image.Image] = []
    font_path = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
    font = (
        ImageFont.truetype(str(font_path), 30)
        if font_path.exists()
        else ImageFont.load_default(size=30)
    )
    for item in report["results"]:
        source = output / item["images"]["reconstructed"]
        with Image.open(source) as image:
            reconstructed = image.convert("RGB")
        page = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(page)
        draw.text((70, 55), item["file"], font=font, fill="black")
        scale = min(1100 / reconstructed.width, 1500 / reconstructed.height)
        size = (
            max(1, round(reconstructed.width * scale)),
            max(1, round(reconstructed.height * scale)),
        )
        reconstructed = reconstructed.resize(size, Image.Resampling.LANCZOS)
        page.paste(reconstructed, ((1240 - size[0]) // 2, 130))
        pages.append(page)
    if pages:
        pages[0].save(
            path,
            "PDF",
            resolution=150,
            save_all=True,
            append_images=pages[1:],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("output/v2-benchmark"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = run_benchmark(args.root.resolve(), args.output.resolve(), args.limit)
    print(
        json.dumps(
            {
                "sample_count": report["sample_count"],
                "total_seconds": report["total_seconds"],
                "quality_status": report["quality_status"],
            },
            ensure_ascii=False,
        )
    )
