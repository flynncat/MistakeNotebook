from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from .classification import (
    TAXONOMY,
    Classification,
    classify_by_rules,
    valid_category,
)
from .config import Settings
from .paddle_ocr_runtime import PaddleOCRRuntime, PaddleOCRRuntimeError


_ELLIPSIS_SOURCE = re.compile(
    r"(?:[.\u2026\u22ef\u00b7\u2022]{2,}|\u2026)"
)
_SUSPICIOUS_ELLIPSIS = re.compile(
    r"[\u00b7\u2022]"
    r"|\u3001\s*[\uff0c,]\s*\u90a3\u4e48"
)
_PUNCTUATION_ONLY = re.compile(
    r"[\s\u00b7\u2022\u2026\u22ef.\uff0e\u3002,\uff0c\u3001]"
)


def _canonicalize_ellipsis(text: str) -> str:
    return _ELLIPSIS_SOURCE.sub("\u2026\u2026", text)


def _select_punctuation_candidate(line: dict[str, Any]) -> str:
    primary = str(line.get("text") or "")
    if _ELLIPSIS_SOURCE.search(primary):
        return _canonicalize_ellipsis(primary)
    if not _SUSPICIOUS_ELLIPSIS.search(primary):
        return primary
    primary_confidence = float(line.get("confidence") or 0)
    primary_skeleton = _PUNCTUATION_ONLY.sub("", primary)
    for alternative in line.get("alternatives") or []:
        candidate = str(alternative.get("text") or "")
        confidence = float(alternative.get("confidence") or 0)
        if (
            not _ELLIPSIS_SOURCE.search(candidate)
            or confidence < primary_confidence - 0.08
            or _PUNCTUATION_ONLY.sub("", candidate) != primary_skeleton
        ):
            continue
        return _canonicalize_ellipsis(candidate)
    return primary


@dataclass
class RecognitionResult:
    text: str
    confidence: float
    category_group: str
    category: str
    category_confidence: float
    category_source: str
    summary: str
    provider: str
    review_reasons: list[str]
    lines: list[dict[str, Any]]


class MacVisionOCR:
    provider = "macos-vision"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.source = Path(__file__).parent / "helpers" / "vision_ocr.swift"
        self.binary = settings.data_dir / "bin" / "vision_ocr"

    def available(self) -> bool:
        return sys.platform == "darwin" and self.source.exists()

    def _ensure_binary(self) -> None:
        if self.binary.exists() and self.binary.stat().st_mtime >= self.source.stat().st_mtime:
            return
        self.binary.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["xcrun", "swiftc", str(self.source), "-o", str(self.binary)],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "无法编译 macOS Vision OCR")

    def _run(self, image_path: Path) -> list[dict[str, Any]]:
        self._ensure_binary()
        result = subprocess.run(
            [str(self.binary), str(image_path)],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "macOS Vision OCR 失败")
        payload = json.loads(result.stdout)
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        lines = list(payload.get("lines", []))
        for line in lines:
            selected = _select_punctuation_candidate(line)
            if selected != line.get("text"):
                line["primary_text"] = line.get("text")
                line["text"] = selected
                line["punctuation_consensus"] = True
        return lines

    def detect_rotation(self, image_path: Path) -> int | None:
        if not self.available():
            return None
        lines = self._run(image_path)
        ratios: list[tuple[float, int]] = []
        for line in lines:
            box = line.get("box", [])
            if len(box) != 4 or float(box[3]) <= 0:
                continue
            weight = max(1, len(str(line.get("text", ""))))
            ratios.append((float(box[2]) / float(box[3]), weight))
        if not ratios:
            return None
        weighted_ratio = sum(ratio * weight for ratio, weight in ratios) / sum(
            weight for _, weight in ratios
        )
        return -90 if weighted_ratio < 1.2 else None

    def recognize(
        self, image_path: Path
    ) -> tuple[str, float, list[dict[str, Any]]]:
        lines = self._run(image_path)
        text = "\n".join(str(line.get("text", "")).strip() for line in lines).strip()
        weighted = [
            (float(line.get("confidence", 0)), max(1, len(str(line.get("text", "")))))
            for line in lines
        ]
        total = sum(weight for _, weight in weighted)
        raw_confidence = (
            sum(score * weight for score, weight in weighted) / total if total else 0
        )
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        structure_bonus = 0.1 if re.search(r"[【\[]?(?:例题|练习)\d+", text) else 0
        confidence = min(
            0.95,
            raw_confidence + min(0.3, chinese_count / 240) + structure_bonus,
        )
        return text, confidence, lines


class PaddleOCRLocal:
    provider = "paddleocr"

    def __init__(self, settings: Settings) -> None:
        self.runtime = PaddleOCRRuntime(settings.root_dir, settings.data_dir)

    def available(self) -> bool:
        return self.runtime.available

    def detect_rotation(self, _image_path: Path) -> int | None:
        return None

    def recognize(
        self, image_path: Path
    ) -> tuple[str, float, list[dict[str, Any]]]:
        lines = self.runtime.recognize(image_path)
        for line in lines:
            line["text"] = _select_punctuation_candidate(line)
        text = "\n".join(
            str(line.get("text") or "").strip() for line in lines
        ).strip()
        weighted = [
            (
                float(line.get("confidence") or 0),
                max(1, len(str(line.get("text") or ""))),
            )
            for line in lines
        ]
        total = sum(weight for _, weight in weighted)
        confidence = (
            sum(score * weight for score, weight in weighted) / total
            if total
            else 0.0
        )
        return text, max(0.0, min(0.95, confidence)), lines

    def close(self) -> None:
        self.runtime.close()


class OpenAICompatibleVision:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def recognize(
        self,
        image_path: Path,
        taxonomy: dict[str, tuple[str, ...]],
    ) -> RecognitionResult:
        if not self.settings.openai_api_key:
            raise RuntimeError("未配置 OPENAI_API_KEY")
        media_type = "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        prompt = (
            "识别这张小学奥数题照片。忽略所有手写笔记、计算过程、红笔批改和答案，"
            "只读取印刷题目。返回 JSON：text 为完整印刷题干，summary 为一句话知识点，"
            "category_group 为一级领域，category 为二级题型，"
            "confidence 为 OCR 置信度，category_confidence 为分类置信度，均为 0 到 1。"
            f"必须从以下启用中的内置分类选择："
            f"{json.dumps(taxonomy, ensure_ascii=False)}。不要解题。"
        )
        response = httpx.post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={
                "model": self.settings.openai_model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                            },
                        ],
                    }
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        payload = _parse_json(content)
        confidence = float(payload.get("confidence", 0))
        group = _normalize_category(str(payload.get("category_group", "")))
        category = _normalize_category(str(payload.get("category", "")))
        category_confidence = float(payload.get("category_confidence", 0))
        valid = valid_category(group, category, taxonomy)
        return RecognitionResult(
            text=str(payload.get("text", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
            category_group=group if valid else "未分类",
            category=category if valid else "未分类",
            category_confidence=max(0.0, min(1.0, category_confidence)) if valid else 0.0,
            category_source="cloud" if valid else "cloud_invalid",
            summary=str(payload.get("summary", "")).strip(),
            provider=f"openai-compatible:{self.settings.openai_model}",
            review_reasons=[] if valid else ["云端模型未返回合法题型"],
            lines=[],
        )


def _parse_json(value: str) -> dict[str, Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, re.DOTALL)
        if not match:
            raise ValueError("模型未返回 JSON")
        return json.loads(match.group())


def _normalize_category(value: str) -> str:
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9·]", "", value).strip()
    return cleaned[:12] or "未分类"


def _rule_category(
    text: str,
    taxonomy: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, str]:
    result = classify_by_rules(text, taxonomy)
    return result.category, result.summary


def _ollama_category(
    settings: Settings,
    text: str,
    taxonomy: dict[str, tuple[str, ...]],
) -> Classification | None:
    if os.getenv("MISTAKE_BOOK_DISABLE_OLLAMA") == "1" or not text:
        return None
    prompt = f"""
你是小学奥数题目分类器。忽略题目中的手写过程，只根据以下 OCR 题干归类。
启用中的内置分类词表：{json.dumps(taxonomy, ensure_ascii=False)}
必须从固定词表选择合法的一级领域和二级题型。
只输出 JSON：{{"category_group":"一级领域","category":"二级题型",
"summary":"一句话知识点","confidence":0.0}}，不要解题。

题干：
{text[:5000]}
""".strip()
    try:
        response = httpx.post(
            f"{settings.ollama_url.rstrip('/')}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = _parse_json(response.json().get("response", "{}"))
        group = _normalize_category(str(payload.get("category_group", "")))
        category = _normalize_category(str(payload.get("category", "")))
        summary = str(payload.get("summary", "")).strip()
        confidence = float(payload.get("confidence", 0))
        if valid_category(group, category, taxonomy) and summary and 0 <= confidence <= 1:
            return Classification(
                group=group,
                category=category,
                summary=summary,
                confidence=confidence,
                source="ollama",
            )
    except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return None


class RecognitionService:
    def __init__(
        self,
        settings: Settings,
        taxonomy_provider: Callable[[], dict[str, tuple[str, ...]]] | None = None,
    ) -> None:
        self.settings = settings
        self.vision_ocr = MacVisionOCR(settings)
        self.paddle_ocr = PaddleOCRLocal(settings)
        backend = settings.local_ocr_backend.strip().lower()
        if backend not in {"auto", "vision", "paddle"}:
            raise ValueError(
                "MISTAKE_BOOK_LOCAL_OCR must be auto, vision, or paddle"
            )
        if backend == "vision":
            self.local_ocr = self.vision_ocr
        elif backend == "paddle":
            self.local_ocr = self.paddle_ocr
        elif self.vision_ocr.available():
            self.local_ocr = self.vision_ocr
        else:
            self.local_ocr = self.paddle_ocr
        self.cloud = OpenAICompatibleVision(settings)
        self._taxonomy_provider = taxonomy_provider or (lambda: TAXONOMY)

    def close(self) -> None:
        self.paddle_ocr.close()

    def rotation_hint(self, image_path: Path) -> int | None:
        if self.settings.recognition_provider != "local":
            return None
        try:
            return self.local_ocr.detect_rotation(image_path)
        except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError):
            return None

    def classify_text(
        self,
        text: str,
        existing_categories: list[str] | None = None,
    ) -> Classification:
        taxonomy = self._taxonomy_provider()
        rule_category = classify_by_rules(text, taxonomy)
        model_category = (
            _ollama_category(self.settings, text, taxonomy)
            if rule_category.confidence < 0.75
            else None
        )
        if (
            model_category
            and model_category.confidence >= rule_category.confidence + 0.05
        ):
            return model_category
        return rule_category

    def recognize(self, image_path: Path, existing_categories: list[str]) -> RecognitionResult:
        if self.settings.recognition_provider == "cloud":
            result = self.cloud.recognize(image_path, self._taxonomy_provider())
            if result.confidence < 0.75 or len(re.sub(r"\W", "", result.text)) < 8:
                result.review_reasons.append("云端题干识别置信度不足")
            return result
        if not self.local_ocr.available():
            return RecognitionResult(
                text="",
                confidence=0,
                category_group="未分类",
                category="未分类",
                category_confidence=0,
                category_source="none",
                summary="本机没有可用的 OCR",
                provider="none",
                review_reasons=["本机没有可用的 OCR"],
                lines=[],
            )
        try:
            text, confidence, lines = self.local_ocr.recognize(image_path)
        except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            return RecognitionResult(
                text="",
                confidence=0,
                category_group="未分类",
                category="未分类",
                category_confidence=0,
                category_source="none",
                summary="OCR 执行失败",
                provider="macos-vision",
                review_reasons=[str(error)],
                lines=[],
            )
        classification = self.classify_text(text, existing_categories)
        reasons: list[str] = []
        if confidence < 0.75:
            reasons.append("题干 OCR 置信度较低")
        if len(re.sub(r"[\W_]", "", text)) < 8:
            reasons.append("识别出的有效题干过短")
        reasons.extend(classification.review_reasons)
        return RecognitionResult(
            text=text,
            confidence=confidence,
            category_group=classification.group,
            category=classification.category,
            category_confidence=classification.confidence,
            category_source=classification.source,
            summary=classification.summary,
            provider=f"{self.local_ocr.provider}+ollama"
            if classification.source == "ollama"
            else f"{self.local_ocr.provider}+rules",
            review_reasons=list(dict.fromkeys(reasons)),
            lines=lines,
        )
