#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from PIL import Image


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _result_dict(result: Any) -> dict[str, Any]:
    value = getattr(result, "res", None)
    if isinstance(value, dict):
        return value
    value = getattr(result, "json", None)
    if callable(value):
        value = value()
    if isinstance(value, dict):
        nested = value.get("res")
        return nested if isinstance(nested, dict) else value
    return {}


def _serializable(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


class PaddleOCRWorker:
    def __init__(self, language: str) -> None:
        from paddleocr import PaddleOCR

        self.engine = PaddleOCR(
            lang=language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image_path: Path) -> dict[str, Any]:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        results = list(
            self.engine.predict(
                input=str(image_path),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        )
        lines: list[dict[str, Any]] = []
        for result in results:
            payload = _result_dict(result)
            texts = _serializable(payload.get("rec_texts"))
            scores = _serializable(payload.get("rec_scores"))
            raw_polygons = payload.get("rec_polys")
            if raw_polygons is None:
                raw_polygons = payload.get("dt_polys")
            polygons = _serializable(raw_polygons)
            for index, raw_text in enumerate(texts):
                text = str(raw_text or "").strip()
                if not text or index >= len(polygons):
                    continue
                points = polygons[index]
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                if not xs or not ys:
                    continue
                x0, x1 = max(0.0, min(xs)), min(float(width), max(xs))
                y0, y1 = max(0.0, min(ys)), min(float(height), max(ys))
                if x1 <= x0 or y1 <= y0:
                    continue
                confidence = (
                    float(scores[index]) if index < len(scores) else 0.0
                )
                lines.append(
                    {
                        "text": text,
                        "confidence": max(0.0, min(1.0, confidence)),
                        "box": [
                            x0 / width,
                            1.0 - (y1 / height),
                            (x1 - x0) / width,
                            (y1 - y0) / height,
                        ],
                    }
                )
        lines.sort(
            key=lambda line: (
                -float(line["box"][1]),
                float(line["box"][0]),
            )
        )
        return {"lines": lines}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    worker = PaddleOCRWorker(args.lang)
    if args.warmup:
        _emit({"ready": True, "warmup": True})
        return 0
    _emit({"ready": True})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            image_path = Path(str(request["image_path"])).expanduser().resolve()
            _emit({"ok": True, **worker.recognize(image_path)})
        except Exception as error:
            _emit(
                {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=8),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
