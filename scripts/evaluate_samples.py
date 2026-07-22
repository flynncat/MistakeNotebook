from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import cv2

from mistake_book.config import Settings
from mistake_book.image_pipeline import (
    extract_printed_question,
    load_image,
    process_image,
)
from mistake_book.pdf_export import export_pdf
from mistake_book.recognition import RecognitionService


def evaluate(root: Path, output: Path) -> int:
    sample_dir = root / "Sample"
    images = sorted(
        path
        for path in sample_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    )
    if not images:
        raise SystemExit("Sample 目录中没有图片")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    settings = Settings.load(root)
    recognizer = RecognitionService(settings)
    categories: list[str] = []
    results: list[dict[str, object]] = []
    pdf_problems: list[dict[str, object]] = []

    for source in images:
        item_dir = output / source.stem
        item_dir.mkdir()
        original_preview = item_dir / "source.png"
        cv2.imwrite(str(original_preview), load_image(source))
        try:
            rotation_hint = recognizer.rotation_hint(source)
            pipeline = process_image(source, item_dir, rotation_hint)
            recognition = recognizer.recognize(pipeline.normalized_path, categories)
            if recognition.category and recognition.category not in categories:
                categories.append(recognition.category)
            extraction_reasons: list[str] = []
            if recognition.lines:
                selected, extraction_metrics, extraction_reasons = extract_printed_question(
                    pipeline.cleaned_path,
                    recognition.lines,
                    item_dir,
                )
                pipeline.selected_path = selected
                pipeline.metrics.update(extraction_metrics)
            reasons = [
                *pipeline.review_reasons,
                *recognition.review_reasons,
                *extraction_reasons,
            ]
            passed = (
                pipeline.cleaned_path.exists()
                and pipeline.metrics["output_width"] >= 300
                and pipeline.metrics["output_height"] >= 150
                and pipeline.metrics["protected_overlap_pixels"] == 0
                and bool(recognition.text)
                and recognition.category != "未分类"
                and not reasons
            )
            result = {
                "file": source.name,
                "passed": passed,
                "category": recognition.category,
                "ocr_confidence": round(recognition.confidence, 4),
                "ocr_text": recognition.text,
                "summary": recognition.summary,
                "provider": recognition.provider,
                "review_reasons": reasons,
                "metrics": pipeline.metrics,
                "source_preview": str(original_preview.relative_to(output)),
                "normalized": str(pipeline.normalized_path.relative_to(output)),
                "cleaned": str(pipeline.selected_path.relative_to(output)),
            }
            pdf_problems.append(
                {
                    "id": source.stem,
                    "status": "ready" if passed else "needs_review",
                    "review_status": "not_required" if passed else "pending",
                    "selected_artifact": str(pipeline.selected_path),
                    "category": recognition.category,
                    "ocr_text": recognition.text,
                }
            )
        except Exception as error:
            result = {
                "file": source.name,
                "passed": False,
                "category": "未分类",
                "ocr_confidence": 0,
                "ocr_text": "",
                "summary": "",
                "provider": "none",
                "review_reasons": [f"{type(error).__name__}: {error}"],
                "metrics": {},
                "source_preview": str(original_preview.relative_to(output)),
            }
        results.append(result)

    passed_count = sum(bool(item["passed"]) for item in results)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sample_count": len(results),
        "passed_count": passed_count,
        "review_count": len(results) - passed_count,
        "results": results,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_html(output, report)
    try:
        export_pdf("sample", pdf_problems, output, allow_partial=True)
    except ValueError:
        pass
    print(json.dumps({key: report[key] for key in ("sample_count", "passed_count", "review_count")}, ensure_ascii=False))
    return 0 if all("metrics" in item for item in results) else 1


def _write_html(output: Path, report: dict[str, object]) -> None:
    cards = []
    for item in report["results"]:  # type: ignore[index]
        assert isinstance(item, dict)
        images = "".join(
            f'<figure><figcaption>{label}</figcaption><img src="{html.escape(str(item.get(key, "")))}"></figure>'
            for key, label in (
                ("source_preview", "原图"),
                ("normalized", "校正"),
                ("cleaned", "清理"),
            )
            if item.get(key)
        )
        reasons = "".join(
            f"<li>{html.escape(str(reason))}</li>"
            for reason in item.get("review_reasons", [])
        )
        cards.append(
            f"""
            <article>
              <h2>{html.escape(str(item['file']))}
                <span class="{'pass' if item['passed'] else 'review'}">
                  {'通过' if item['passed'] else '需确认'}
                </span>
              </h2>
              <div class="images">{images}</div>
              <p><b>分类：</b>{html.escape(str(item.get('category', '')))}
                　<b>OCR：</b>{float(item.get('ocr_confidence', 0)):.0%}</p>
              <pre>{html.escape(str(item.get('ocr_text', '')))}</pre>
              <ul>{reasons}</ul>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>Sample 自动回归</title>
<style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;margin:24px;background:#f4f1ea;color:#222}}
main{{max-width:1400px;margin:auto}}article{{background:white;border-radius:12px;padding:16px;margin:16px 0}}
h1,h2{{margin:0 0 12px}}h2 span{{font-size:13px;padding:3px 9px;border-radius:20px}}
.pass{{background:#d9f0de;color:#246235}}.review{{background:#f8dfd8;color:#963e29}}
.images{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}figure{{margin:0}}
figcaption{{font-size:12px;color:#666}}img{{width:100%;height:330px;object-fit:contain;background:#eee}}
pre{{white-space:pre-wrap;max-height:140px;overflow:auto}}li{{color:#963e29}}
</style><main><h1>Sample 自动回归：{report['passed_count']} / {report['sample_count']} 自动通过</h1>
{''.join(cards)}</main></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("output/sample-evaluation")
    )
    arguments = parser.parse_args()
    raise SystemExit(evaluate(arguments.root.resolve(), arguments.output.resolve()))
