from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Settings


@dataclass
class RecognitionResult:
    text: str
    confidence: float
    category: str
    summary: str
    provider: str
    review_reasons: list[str]
    lines: list[dict[str, Any]]


class MacVisionOCR:
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
        return list(payload.get("lines", []))

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


class OpenAICompatibleVision:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def recognize(self, image_path: Path, categories: list[str]) -> RecognitionResult:
        if not self.settings.openai_api_key:
            raise RuntimeError("未配置 OPENAI_API_KEY")
        media_type = "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        prompt = (
            "识别这张小学奥数题照片。忽略所有手写笔记、计算过程、红笔批改和答案，"
            "只读取印刷题目。返回 JSON：text 为完整印刷题干，summary 为一句话知识点，"
            "category 为简短题型分类（2-12 个汉字），confidence 为 0 到 1。"
            f"现有分类可优先复用：{categories or ['暂无']}。不要解题。"
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
        return RecognitionResult(
            text=str(payload.get("text", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
            category=_normalize_category(str(payload.get("category", ""))),
            summary=str(payload.get("summary", "")).strip(),
            provider=f"openai-compatible:{self.settings.openai_model}",
            review_reasons=[],
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


def _rule_category(text: str) -> tuple[str, str]:
    compact = re.sub(r"\s+", "", text)
    rules = [
        (r"至少有多少.*(?:相同|完全相同)|至少.*列.*相同", "抽屉原理", "利用抽屉原理保证至少若干列颜色相同"),
        (r"染色|涂色|颜色", "计数·染色问题", "利用分类、乘法原理或容斥计算染色方案"),
        (r"进位|不进位|相加至少发生", "计数·数位进位", "分析数位相加与进位条件"),
        (r"组成.*(?:位数|数字)|三位数|四位数|卡片.*数字", "计数·组数问题", "按数位限制组成整数并计数"),
        (r"之比|钱数比|比例", "比例问题", "根据前后数量之比建立方程"),
        (r"排列|排成|站队|顺序", "计数·排列问题", "研究有顺序的排列方案"),
        (r"组合|选出|取出|选法", "计数·组合问题", "研究无顺序的选择方案"),
        (r"共有多少|有多少种|多少种不同|几种", "计数问题", "使用分类计数或乘法原理"),
        (r"整除|余数|质数|合数|因数|倍数|约数", "数论问题", "研究整数性质、整除或余数"),
        (r"面积|周长|角度|三角形|正方形|长方形|圆", "几何问题", "分析图形数量或几何量"),
        (r"路程|速度|相遇|追及", "行程问题", "根据路程、速度和时间关系求解"),
        (r"浓度|利润|工程|工作效率|鸡兔同笼", "应用题", "建立数量关系解决实际问题"),
    ]
    for pattern, category, summary in rules:
        if re.search(pattern, compact):
            return category, summary
    return "未分类", "暂未识别出稳定的小奥知识点"


def _ollama_category(
    settings: Settings, text: str, existing_categories: list[str]
) -> tuple[str, str] | None:
    if os.getenv("MISTAKE_BOOK_DISABLE_OLLAMA") == "1" or not text:
        return None
    prompt = f"""
你是小学奥数题目分类器。忽略题目中的手写过程，只根据以下 OCR 题干归类。
已有分类：{json.dumps(existing_categories, ensure_ascii=False)}
优先复用准确的已有分类，否则创建 2-12 个汉字的简短分类。
只输出 JSON：{{"category":"分类","summary":"一句话知识点"}}，不要解题。

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
        category = _normalize_category(str(payload.get("category", "")))
        summary = str(payload.get("summary", "")).strip()
        if category != "未分类" and summary:
            return category, summary
    except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return None


class RecognitionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local_ocr = MacVisionOCR(settings)
        self.cloud = OpenAICompatibleVision(settings)

    def rotation_hint(self, image_path: Path) -> int | None:
        if self.settings.recognition_provider != "local":
            return None
        try:
            return self.local_ocr.detect_rotation(image_path)
        except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError):
            return None

    def recognize(self, image_path: Path, existing_categories: list[str]) -> RecognitionResult:
        if self.settings.recognition_provider == "cloud":
            result = self.cloud.recognize(image_path, existing_categories)
            if result.confidence < 0.75 or len(re.sub(r"\W", "", result.text)) < 8:
                result.review_reasons.append("云端题干识别置信度不足")
            return result
        if not self.local_ocr.available():
            return RecognitionResult(
                text="",
                confidence=0,
                category="未分类",
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
                category="未分类",
                summary="OCR 执行失败",
                provider="macos-vision",
                review_reasons=[str(error)],
                lines=[],
            )
        rule_category = _rule_category(text)
        model_category = (
            _ollama_category(self.settings, text, existing_categories)
            if rule_category[0] == "未分类"
            else None
        )
        category, summary = model_category or rule_category
        reasons: list[str] = []
        if confidence < 0.75:
            reasons.append("题干 OCR 置信度较低")
        if len(re.sub(r"[\W_]", "", text)) < 8:
            reasons.append("识别出的有效题干过短")
        if category == "未分类":
            reasons.append("未识别出稳定题型")
        return RecognitionResult(
            text=text,
            confidence=confidence,
            category=category,
            summary=summary,
            provider="macos-vision+ollama" if model_category else "macos-vision+rules",
            review_reasons=reasons,
            lines=lines,
        )
