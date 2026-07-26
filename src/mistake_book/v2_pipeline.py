from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .config import Settings
from .content_blocks import build_content_blocks_with_fallbacks, source_sha256
from .font_selection import default_font_metrics
from .formula_pipeline import process_formulas
from .formula_runtime import FormulaRuntime
from .figure_preservation import preserve_figure
from .image_pipeline import (
    conservative_remove_marks,
    load_image,
    orient_image,
    process_image,
)
from .recognition import RecognitionResult, RecognitionService
from .reconstruction import (
    StructuredProblem,
    build_structured_problem,
    content_blocks_to_text,
    ordered_question_text,
    question_content_blocks,
    render_problem,
    render_problem_content_blocks,
    tesseract_question_ocr,
)
from .v2_models import UVDocAdapter


def _ocr_candidate_score(text: str) -> tuple[int, int, int]:
    numbers = set(re.findall(r"\d+(?::\d+)?", text))
    punctuation_evidence = (
        1
        if re.search(r"(?:\u2026\u2026|\u22ef|\.{3,})", text)
        else -1
        if re.search(r"[\u00b7\u2022]", text)
        else 0
    )
    return (
        1 if re.search(r"[\uff1f?]", text) else 0,
        len(numbers),
        punctuation_evidence,
    )


def _restore_missed_radix_variable(
    content_blocks: dict[str, Any],
    source_text: str,
) -> None:
    match = re.search(r"\u5728\s*([A-Za-z])\s*\u8fdb\u5236", source_text)
    if not match:
        return
    variable = match.group(1)
    blocks = content_blocks.get("blocks") or []
    for index, block in enumerate(blocks):
        if block.get("type") != "text":
            continue
        text = str(block.get("text") or "")
        if f"\u5728{variable}\u8fdb\u5236" in text:
            return
        compact = text.rstrip()
        if not compact.endswith("\u5728") or index + 1 >= len(blocks):
            continue
        following = blocks[index + 1]
        if (
            following.get("type") == "latex"
            and str(following.get("latex") or "").strip() == variable
        ):
            return
        if (
            following.get("type") == "text"
            and str(following.get("text") or "").lstrip().startswith("\u8fdb\u5236")
        ):
            block["text"] = f"{text}{variable}"
            return


@dataclass
class V2PipelineResult:
    normalized_path: Path
    cleaned_path: Path
    selected_path: Path
    recognition: RecognitionResult
    structured: StructuredProblem
    metrics: dict[str, Any]
    review_reasons: list[str]


class V2Processor:
    def __init__(self, settings: Settings, recognition: RecognitionService) -> None:
        self.settings = settings
        self.recognition = recognition
        self.uvdoc = UVDocAdapter(settings.root_dir)
        self.formulas = FormulaRuntime(settings.root_dir)

    @staticmethod
    def _save(path: Path, image) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
            raise OSError(f"无法保存 V2 产物：{path}")

    def process(
        self,
        source_path: Path,
        artifact_dir: Path,
        existing_categories: list[str],
        rotation_hint: int | None,
        title_hint: str | None = None,
        primary_text_hint: str | None = None,
    ) -> V2PipelineResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        reasons: list[str] = []
        metrics: dict[str, Any] = {"pipeline_version": "v2-reconstruction"}
        image = load_image(source_path)
        oriented, orientation_metrics = orient_image(image, rotation_hint)
        metrics.update(orientation_metrics)

        normalized_path = artifact_dir / "normalized.png"
        if title_hint:
            self._save(normalized_path, oriented)
            metrics["dewarp_model"] = "page-segment-crop"
        elif self.uvdoc.available():
            try:
                dewarped = self.uvdoc.unwarp(oriented)
            except Exception as error:
                reasons.append(f"UVDoc 展平失败，已回退传统校正：{type(error).__name__}")
                fallback = process_image(source_path, artifact_dir, rotation_hint)
                normalized_path = fallback.normalized_path
                metrics.update(fallback.metrics)
                metrics["uvdoc_error"] = f"{type(error).__name__}: {error}"
            else:
                self._save(normalized_path, dewarped.image)
                metrics.update(
                    {
                        "dewarp_model": "uvdoc",
                        "dewarp_seconds": round(dewarped.seconds, 3),
                        "dewarp_device": dewarped.device,
                        "dewarp_metadata": dewarped.metadata,
                    }
                )
        else:
            reasons.append("UVDoc 模型不可用，已回退传统校正")
            fallback = process_image(source_path, artifact_dir, rotation_hint)
            normalized_path = fallback.normalized_path
            metrics.update(fallback.metrics)

        normalized = cv2.imread(str(normalized_path), cv2.IMREAD_COLOR)
        if normalized is None:
            raise OSError("无法读取 V2 展平产物")
        cleaned, cleaning_metrics = conservative_remove_marks(normalized)
        cleaned_path = artifact_dir / "cleaned.png"
        self._save(cleaned_path, cleaned)
        metrics.update(cleaning_metrics)

        recognition = self.recognition.recognize(normalized_path, existing_categories)
        try:
            secondary_text, secondary_box = tesseract_question_ocr(
                normalized_path,
                recognition.lines,
                artifact_dir / "secondary_ocr_question.png",
            )
            metrics["secondary_ocr_text_box"] = secondary_box
            metrics["secondary_ocr_scope"] = "printed_question_only"
        except RuntimeError as error:
            secondary_text = ""
            reasons.append(str(error))
        crop_ordered_text = (
            ordered_question_text(recognition.lines) if title_hint else ""
        )
        candidates = [
            text
            for text in (primary_text_hint, crop_ordered_text)
            if text
        ]
        if not candidates:
            candidates.append(recognition.text)

        primary_text = max(candidates, key=_ocr_candidate_score)
        metrics["primary_ocr_scope"] = (
            "full_page_region"
            if primary_text_hint and primary_text == primary_text_hint
            else "question_crop"
        )
        structured = build_structured_problem(
            primary_text,
            secondary_text,
            title_hint=title_hint,
        )
        figure_path: Path | None = None
        if structured.figure != "none":
            figure = preserve_figure(
                normalized_path,
                recognition.lines,
                structured.primary_text,
                artifact_dir,
                self.recognition.local_ocr,
            )
            if figure is None:
                reasons.append("题目需要配图，但未能可靠定位原图配图")
                raise ValueError("拒绝在缺少原图裁剪时猜测生成配图")
            if not figure.metrics.get("passed", False):
                reasons.extend(figure.review_reasons)
                raise ValueError("配图未通过原图像素支撑校验，拒绝输出带笔记的原图")
            figure_path = figure.selected_path
            metrics["figure_preservation"] = figure.metrics
            if structured.figure == "five_country_map":
                structured.figure_edges = [
                    tuple(edge)
                    for edge in figure.metrics.get("adjacency_edges", [])
                    if len(edge) == 2
                ]
            reasons.extend(figure.review_reasons)
        selected_path = render_problem(
            structured,
            artifact_dir / "question.png",
            figure_path=figure_path,
        )
        figure_box = (
            metrics.get("figure_preservation", {}).get("box")
            if figure_path
            else None
        )
        fallback_content_blocks = build_content_blocks_with_fallbacks(
            structured.primary_text,
            recognition.lines,
            cleaned_path,
            artifact_dir,
            figure_asset=figure_path.name if figure_path else None,
            figure_box=figure_box,
        )
        if os.getenv("MISTAKE_BOOK_FORMULA_OCR", "1") == "1":
            formula_result = process_formulas(
                self.formulas,
                normalized_path,
                cleaned_path,
                recognition.lines,
                artifact_dir,
                fallback_content_blocks=fallback_content_blocks,
                figure_asset=figure_path.name if figure_path else None,
                figure_box=figure_box,
            )
            content_blocks = formula_result.content_blocks
            metrics["formula_recognition"] = formula_result.metrics
            reasons.extend(formula_result.review_reasons)
        else:
            content_blocks = fallback_content_blocks
            metrics["formula_recognition"] = {"disabled": True}
        if content_blocks.get("version") == 2:
            _restore_missed_radix_variable(
                content_blocks,
                structured.primary_text,
            )
            content_blocks = question_content_blocks(structured, content_blocks)
            formula_aware_text = content_blocks_to_text(structured, content_blocks)
            structured.primary_text = formula_aware_text
            structured.body = formula_aware_text.removeprefix(structured.title).strip()
            selected_path = render_problem_content_blocks(
                structured,
                content_blocks,
                artifact_dir,
                artifact_dir / "question.png",
            )
        metrics.update(
            {
                "structured_problem": structured.to_dict(),
                "font": default_font_metrics(),
                "output_mode": "typeset_reconstruction",
                "quality_gate": "manual_review_required"
                if structured.review_reasons
                else "dual_ocr_consistent",
                "content_blocks": content_blocks,
                "content_source_sha256": source_sha256(cleaned_path),
            }
        )
        reasons.extend(recognition.review_reasons)
        reasons.extend(structured.review_reasons)
        if os.getenv("MISTAKE_BOOK_V2_ALLOW_UNVERIFIED") != "1":
            reasons.append("尚未建立该题的人工真值，禁止自动通过")
        reasons = list(dict.fromkeys(reasons))
        return V2PipelineResult(
            normalized_path=normalized_path,
            cleaned_path=cleaned_path,
            selected_path=selected_path,
            recognition=recognition,
            structured=structured,
            metrics=metrics,
            review_reasons=reasons,
        )
