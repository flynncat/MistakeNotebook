from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class ModelRun:
    image: np.ndarray
    seconds: float
    device: str
    metadata: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "models" / "v2_manifest.json").read_text(encoding="utf-8"))


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"无法加载模型源码：{path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _torch_device(prefer_mps: bool = True) -> str:
    import torch

    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class UVDocAdapter:
    def __init__(self, root: Path, *, prefer_mps: bool = True) -> None:
        self.root = root
        self.source_dir = root / ".models" / "sources" / "uvdoc"
        self.weight_path = self.source_dir / "model" / "best_model.pkl"
        self.device = _torch_device(prefer_mps)
        self.model = None
        self._model_module = None

    def available(self) -> bool:
        return (self.source_dir / "model.py").exists() and self.weight_path.exists()

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.available():
            raise FileNotFoundError("UVDoc 源码或权重不存在")
        import torch

        self._model_module = _load_module("mistake_book_uvdoc_model", self.source_dir / "model.py")
        model = self._model_module.UVDocnet(num_filter=32, kernel_size=5)
        checkpoint = torch.load(
            self.weight_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()
        self.model = model

    def unwarp(self, bgr: np.ndarray) -> ModelRun:
        self.load()
        import torch
        import torch.nn.functional as functional

        assert self.model is not None
        started = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = cv2.resize(rgb, (488, 712), interpolation=cv2.INTER_AREA)
        tensor = (
            torch.from_numpy(resized.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )
        original = (
            torch.from_numpy(rgb.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            positions, _ = self.model(tensor)
            grid = functional.interpolate(
                positions[0].unsqueeze(0),
                size=rgb.shape[:2],
                mode="bilinear",
                align_corners=True,
            )
            output = functional.grid_sample(
                original,
                grid.transpose(1, 2).transpose(2, 3),
                align_corners=True,
            )
        result = np.clip(
            output[0].detach().cpu().numpy().transpose(1, 2, 0) * 255,
            0,
            255,
        ).astype(np.uint8)
        return ModelRun(
            image=cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
            seconds=time.perf_counter() - started,
            device=self.device,
            metadata={
                "model": "uvdoc",
                "weight_sha256": sha256(self.weight_path),
                "input_size": [488, 712],
            },
        )


class DISHandwritingAdapter:
    def __init__(self, root: Path, *, prefer_mps: bool = True) -> None:
        self.root = root
        self.source_path = (
            root / ".models" / "sources" / "dis-base" / "IS-Net" / "models" / "isnet.py"
        )
        self.weight_path = (
            root / ".models" / "weights" / "handwriting-dis" / "isnet.pth"
        )
        self.device = _torch_device(prefer_mps)
        self.model = None

    def available(self) -> bool:
        return self.source_path.exists() and self.weight_path.exists()

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.available():
            raise FileNotFoundError("DIS 源码或手写分割权重不存在")
        import torch

        module = _load_module("mistake_book_dis_isnet", self.source_path)
        model = module.ISNetDIS()
        state = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

    def predict_mask(
        self,
        bgr: np.ndarray,
        *,
        tile_size: int = 1024,
        overlap: int = 192,
    ) -> ModelRun:
        self.load()
        import torch

        assert self.model is not None
        started = time.perf_counter()
        height, width = bgr.shape[:2]
        probability = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        stride = tile_size - overlap
        y_starts = list(range(0, max(1, height - tile_size + 1), stride))
        x_starts = list(range(0, max(1, width - tile_size + 1), stride))
        if not y_starts or y_starts[-1] != max(0, height - tile_size):
            y_starts.append(max(0, height - tile_size))
        if not x_starts or x_starts[-1] != max(0, width - tile_size):
            x_starts.append(max(0, width - tile_size))
        window_y = np.hanning(tile_size) if len(y_starts) > 1 else np.ones(tile_size)
        window_x = np.hanning(tile_size) if len(x_starts) > 1 else np.ones(tile_size)
        blend = np.maximum(np.outer(window_y, window_x).astype(np.float32), 0.05)

        with torch.inference_mode():
            for y in y_starts:
                for x in x_starts:
                    tile = bgr[y : y + tile_size, x : x + tile_size]
                    tile_height, tile_width = tile.shape[:2]
                    padded = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
                    padded[:tile_height, :tile_width] = tile
                    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
                    tensor = (tensor - 0.5).to(self.device)
                    predictions, _ = self.model(tensor)
                    tile_probability = (
                        predictions[0][0, 0].detach().cpu().numpy()[:tile_height, :tile_width]
                    )
                    tile_blend = blend[:tile_height, :tile_width]
                    probability[y : y + tile_height, x : x + tile_width] += (
                        tile_probability * tile_blend
                    )
                    weights[y : y + tile_height, x : x + tile_width] += tile_blend
        probability /= np.maximum(weights, 1e-6)
        mask = np.clip(probability * 255, 0, 255).astype(np.uint8)
        return ModelRun(
            image=mask,
            seconds=time.perf_counter() - started,
            device=self.device,
            metadata={
                "model": "handwriting-dis",
                "weight_sha256": sha256(self.weight_path),
                "tile_size": tile_size,
                "overlap": overlap,
                "min_probability": float(probability.min()),
                "max_probability": float(probability.max()),
                "mean_probability": float(probability.mean()),
            },
        )


def erase_handwriting(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float = 0.55,
    dilation: int = 2,
) -> np.ndarray:
    binary = mask >= round(threshold * 255)
    if dilation:
        binary = cv2.dilate(
            binary.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation * 2 + 1, dilation * 2 + 1),
            ),
        ).astype(bool)
    result = bgr.copy()
    result[binary] = 255
    return result


def make_print_layer(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(15, gray.shape[1] / 35))
    flattened = cv2.divide(gray, np.maximum(background, 1), scale=255)
    binary = cv2.adaptiveThreshold(
        flattened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        17,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
