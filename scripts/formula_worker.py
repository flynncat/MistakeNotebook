#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _safe_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class DetectorWorker:
    def __init__(self, model_path: Path) -> None:
        with contextlib.redirect_stdout(sys.stderr):
            from pix2text.formula_detector import MathFormulaDetector

            self.detector = MathFormulaDetector(
                model_path=model_path,
                model_backend="onnx",
                device="cpu",
            )

    def handle(self, request: dict) -> dict:
        image_path = _safe_path(str(request["image_path"]))
        resized_shape = int(request.get("resized_shape", 1280))
        confidence = float(request.get("confidence", 0.35))
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            detected = self.detector.detect(
                str(image_path),
                resized_shape=resized_shape,
                conf=confidence,
                iou=0.45,
                verbose=False,
            )
        formulas = []
        for item in detected:
            points = item["box"].tolist()
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            formulas.append(
                {
                    "box": [
                        max(0, round(min(xs))),
                        max(0, round(min(ys))),
                        max(0, round(max(xs))),
                        max(0, round(max(ys))),
                    ],
                    "score": float(item["score"]),
                    "formula_type": str(item["type"]),
                }
            )
        return {
            "formulas": formulas,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }


class RecognizerWorker:
    def __init__(self, model_dir: Path) -> None:
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from omegaconf import OmegaConf
            from unimernet.models.unimernet.unimernet import UniMERModel
            from unimernet.processors.formula_processor import FormulaImageEvalProcessor

            device_name = "mps" if torch.backends.mps.is_available() else "cpu"
            self.torch = torch
            self.device = torch.device(device_name)
            config = OmegaConf.create(
                {
                    "model_name": "unimernet",
                    "model_config": {
                        "model_name": str(model_dir),
                        "max_seq_len": 1536,
                    },
                    "tokenizer_name": "nougat",
                    "tokenizer_config": {"path": str(model_dir)},
                    "load_pretrained": False,
                    "load_finetuned": False,
                }
            )
            self.model = UniMERModel.from_config(config)
            checkpoint_path = model_dir / "unimernet_tiny.pth"
            self.model.load_checkpoint(str(checkpoint_path))
            self.model = self.model.to(self.device).eval()
            self.processor = FormulaImageEvalProcessor([192, 672])

    def _batch(self, image_paths: list[Path]) -> dict:
        from PIL import Image

        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensors.append(self.processor(image.convert("RGB")))
        pixels = self.torch.stack(tensors).to(self.device)
        if pixels.shape[1] == 1:
            pixels = pixels.repeat(1, 3, 1, 1)
        core = self.model.model.model
        tokenizer = self.model.tokenizer.tokenizer
        with self.torch.inference_mode():
            generated = core.generate(
                pixel_values=pixels,
                max_new_tokens=1536,
                decoder_start_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                temperature=1.0,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        sequences = generated.sequences
        generated_tokens = sequences[:, -len(generated.scores) :]
        probability_rows: list[list[float]] = [
            [] for _ in range(sequences.shape[0])
        ]
        for step, logits in enumerate(generated.scores):
            selected = logits.log_softmax(dim=-1).gather(
                1,
                generated_tokens[:, step].unsqueeze(-1),
            ).squeeze(-1).exp().detach().float().cpu().tolist()
            for row_index, probability in enumerate(selected):
                probability_rows[row_index].append(float(probability))
        decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        results = []
        for index, latex in enumerate(decoded):
            ids = sequences[index].detach().cpu().tolist()
            generated_ids = generated_tokens[index].detach().cpu().tolist()
            probabilities = probability_rows[index]
            if tokenizer.eos_token_id in generated_ids:
                eos_index = generated_ids.index(tokenizer.eos_token_id)
                probabilities = probabilities[: eos_index + 1]
            eos_reached = tokenizer.eos_token_id in ids
            unk_reached = (
                tokenizer.unk_token_id is not None
                and tokenizer.unk_token_id in ids
            )
            if probabilities:
                sorted_probabilities = sorted(float(value) for value in probabilities)
                p05_index = max(
                    0,
                    int((len(sorted_probabilities) - 1) * 0.05),
                )
                mean_probability = sum(sorted_probabilities) / len(sorted_probabilities)
                p05_probability = sorted_probabilities[p05_index]
            else:
                mean_probability = 0.0
                p05_probability = 0.0
            results.append(
                {
                    "latex": latex.strip(),
                    "token_count": len(probabilities),
                    "mean_token_probability": round(mean_probability, 6),
                    "p05_token_probability": round(p05_probability, 6),
                    "eos_reached": eos_reached,
                    "unk_reached": unk_reached,
                }
            )
        return {"results": results}

    def handle(self, request: dict) -> dict:
        image_paths = [_safe_path(str(item)) for item in request["image_paths"]]
        if not image_paths or len(image_paths) > 128:
            raise ValueError("image_paths must contain 1 to 128 files")
        started = time.perf_counter()
        try:
            results: list[dict] = []
            for offset in range(0, len(image_paths), 1):
                results.extend(
                    self._batch(image_paths[offset : offset + 1])["results"]
                )
        except RuntimeError:
            if self.device.type != "mps":
                raise
            self.device = self.torch.device("cpu")
            self.model = self.model.to(self.device)
            results = []
            for offset in range(0, len(image_paths), 1):
                results.extend(
                    self._batch(image_paths[offset : offset + 1])["results"]
                )
        return {
            "results": results,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "device": str(self.device),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("detect", "recognize"))
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    try:
        model_path = Path(args.model_path).expanduser().resolve()
        if args.mode == "detect":
            worker = DetectorWorker(model_path)
        else:
            worker = RecognizerWorker(model_path)
    except Exception as error:
        _emit(
            {
                "ready": False,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    _emit({"ready": True, "mode": args.mode})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = worker.handle(request)
            _emit({"ok": True, **response})
        except Exception as error:
            _emit(
                {
                    "ok": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
