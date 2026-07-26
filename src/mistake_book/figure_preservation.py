from __future__ import annotations

import re
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .figure_reconstruction import FigureKind, reconstruct_figure


@dataclass
class FigureResult:
    original_path: Path
    cleaned_path: Path
    selected_path: Path
    box: list[int]
    metrics: dict[str, Any]
    review_reasons: list[str]


def _pixel_box(
    line: dict[str, Any],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    box = line.get("box", [])
    if len(box) != 4:
        return None
    x, y, box_width, box_height = (float(value) for value in box)
    return (
        max(0, round(x * width)),
        max(0, round((1 - y - box_height) * height)),
        min(width, round((x + box_width) * width)),
        min(height, round((1 - y) * height)),
    )


def _question_bottom(
    lines: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[int, int | None]:
    anchor = next(
        (
            index
            for index, line in enumerate(lines)
            if re.search(r"(?:例题|练习)\s*[0-9S]+", str(line.get("text", "")))
        ),
        None,
    )
    if anchor is None:
        return round(height * 0.25), None
    bottom = 0
    end_index = anchor
    for index, line in enumerate(lines[anchor : anchor + 8], start=anchor):
        pixel_box = _pixel_box(line, width, height)
        if pixel_box:
            bottom = max(bottom, pixel_box[3])
        end_index = index
        if re.search(r"[？?]", str(line.get("text", ""))):
            break
    return bottom or round(height * 0.25), end_index


def _map_search_box(
    lines: list[dict[str, Any]],
    width: int,
    height: int,
    question_bottom: int,
    question_end_index: int | None,
) -> tuple[int, int, int, int] | None:
    if question_end_index is None:
        return None
    labels = []
    for line in lines[question_end_index + 1 :]:
        if not re.fullmatch(r"[A-E]", str(line.get("text", "")).strip()):
            continue
        box = _pixel_box(line, width, height)
        if box and question_bottom - 10 <= box[1] <= question_bottom + height * 0.42:
            labels.append(box)
    if len(labels) < 2:
        return None
    x0 = min(box[0] for box in labels)
    y0 = min(box[1] for box in labels)
    x1 = max(box[2] for box in labels)
    y1 = max(box[3] for box in labels)
    span_x, span_y = max(1, x1 - x0), max(1, y1 - y0)
    margin_x = max(90, round(span_x * 1.4))
    margin_y = max(35, round(span_y * 0.2))
    return (
        max(0, x0 - margin_x),
        max(question_bottom, y0 - margin_y),
        min(width, x1 + margin_x),
        min(height, y1 + margin_y),
    )


def _refine_line_frame(
    image: np.ndarray,
    search_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = search_box
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 31)
    flattened = cv2.divide(gray, np.maximum(background, 1), scale=255)
    edges = cv2.Canny(flattened, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(25, round(min(region.shape[:2]) * 0.1)),
        minLineLength=max(35, round(min(region.shape[:2]) * 0.2)),
        maxLineGap=20,
    )
    if lines is None:
        return None
    horizontal: list[tuple[int, int, int, int]] = []
    vertical: list[tuple[int, int, int, int]] = []
    for line in lines[:, 0]:
        lx0, ly0, lx1, ly1 = (int(value) for value in line)
        if abs(ly1 - ly0) <= 7:
            horizontal.append((lx0, ly0, lx1, ly1))
        if abs(lx1 - lx0) <= 7:
            vertical.append((lx0, ly0, lx1, ly1))
    if len(horizontal) < 2 or len(vertical) < 2:
        return None
    frame_x0 = min(min(line[0], line[2]) for line in vertical)
    frame_x1 = max(max(line[0], line[2]) for line in vertical)
    frame_y0 = min(min(line[1], line[3]) for line in horizontal)
    frame_y1 = max(max(line[1], line[3]) for line in horizontal)
    if frame_x1 - frame_x0 < 80 or frame_y1 - frame_y0 < 60:
        return None
    padding = 14
    return (
        max(0, x0 + frame_x0 - padding),
        max(0, y0 + frame_y0 - padding),
        min(image.shape[1], x0 + frame_x1 + padding),
        min(image.shape[0], y0 + frame_y1 + padding),
    )


def _fallback_figure_box(
    image: np.ndarray,
    question_bottom: int,
) -> tuple[int, int, int, int] | None:
    height, width = image.shape[:2]
    search_bottom = min(height, question_bottom + round(height * 0.48))
    region = image[question_bottom:search_bottom]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 31)
    flattened = cv2.divide(gray, np.maximum(background, 1), scale=255)
    foreground = cv2.adaptiveThreshold(
        flattened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )
    joined = cv2.dilate(
        foreground,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 11)),
        iterations=2,
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    candidates = []
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        if (
            width * 0.08 <= box_width <= width * 0.65
            and height * 0.05 <= box_height <= height * 0.4
            and area >= 1200
        ):
            density = area / max(1, box_width * box_height)
            position_bonus = 1.2 if x < width * 0.55 else 1.0
            candidates.append(
                (
                    area * density * position_bonus,
                    x,
                    y + question_bottom,
                    box_width,
                    box_height,
                )
            )
    if not candidates:
        return None
    _, x, y, box_width, box_height = max(candidates)
    padding = 24
    return (
        max(0, x - padding),
        max(question_bottom, y - padding),
        min(width, x + box_width + padding),
        min(search_bottom, y + box_height + padding),
    )


def _circle_figure_box(
    image: np.ndarray,
    question_bottom: int,
) -> tuple[int, int, int, int] | None:
    height, width = image.shape[:2]
    search_bottom = min(height, question_bottom + round(height * 0.45))
    search_right = round(width * 0.62)
    region = image[question_bottom:search_bottom, :search_right]
    gray = cv2.medianBlur(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), 7)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(45, round(height * 0.04)),
        param1=120,
        param2=35,
        minRadius=max(25, round(height * 0.035)),
        maxRadius=max(60, round(height * 0.14)),
    )
    if circles is None:
        return None
    candidates = [
        (int(round(x)), int(round(y)) + question_bottom, int(round(radius)))
        for x, y, radius in circles[0][:30]
    ]
    best: tuple[float, tuple[tuple[int, int, int], ...]] | None = None
    for group in combinations(candidates, 3):
        radii = np.array([circle[2] for circle in group], dtype=np.float32)
        mean_radius = float(radii.mean())
        if float(radii.max() - radii.min()) > mean_radius * 0.35:
            continue
        centers = np.array([(circle[0], circle[1]) for circle in group], dtype=np.float32)
        distances = [
            float(np.linalg.norm(centers[left] - centers[right]))
            for left, right in ((0, 1), (0, 2), (1, 2))
        ]
        if min(distances) < mean_radius * 0.55 or max(distances) > mean_radius * 1.8:
            continue
        order = np.argsort(centers[:, 1])
        top = centers[order[0]]
        bottoms = centers[order[1:]]
        if float(bottoms[:, 1].mean() - top[1]) < mean_radius * 0.45:
            continue
        score = (
            float(radii.std()) * 2
            + abs(float(bottoms[0, 1] - bottoms[1, 1]))
            + abs(float(bottoms[:, 0].mean() - top[0]))
        )
        if best is None or score < best[0]:
            best = (score, group)
    if best is None:
        return None
    group = best[1]
    padding = 28
    return (
        max(0, min(x - radius for x, _, radius in group) - padding),
        max(question_bottom, min(y - radius for _, y, radius in group) - padding),
        min(width, max(x + radius for x, _, radius in group) + padding),
        min(search_bottom, max(y + radius for _, y, radius in group) + padding),
    )


def locate_figure(
    image: np.ndarray,
    lines: list[dict[str, Any]],
    question_text: str,
) -> tuple[int, int, int, int] | None:
    height, width = image.shape[:2]
    question_bottom, end_index = _question_bottom(lines, width, height)
    if "三个圆圈" in question_text:
        circle_box = _circle_figure_box(image, question_bottom)
        if circle_box:
            return circle_box
    if "地图" in question_text and re.search(r"[A-E]", question_text):
        search_box = _map_search_box(
            lines,
            width,
            height,
            question_bottom,
            end_index,
        )
        if search_box:
            refined = _refine_line_frame(image, search_box)
            if refined:
                return refined
    return _fallback_figure_box(image, question_bottom)


def _red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = bgr[:, :, 0].astype(np.int16)
    green = bgr[:, :, 1].astype(np.int16)
    red_channel = bgr[:, :, 2].astype(np.int16)
    red = (
        (red_channel > 100)
        & (hsv[:, :, 1] > 70)
        & ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 168))
        & (red_channel - green > 35)
        & (red_channel - blue > 25)
    )
    return cv2.dilate(
        red.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)


def _clean_colored_marks(bgr: np.ndarray) -> np.ndarray:
    red = _red_mask(bgr)
    if not red.any():
        return bgr.copy()
    return cv2.inpaint(
        bgr,
        red.astype(np.uint8) * 255,
        3,
        cv2.INPAINT_TELEA,
    )


def _ink_mask(bgr: np.ndarray, *, ignore_red: bool) -> np.ndarray:
    working = _clean_colored_marks(bgr) if ignore_red else bgr.copy()
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), max(9, gray.shape[1] / 18))
    flattened = cv2.divide(gray, np.maximum(background, 1), scale=255)
    return cv2.adaptiveThreshold(
        flattened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )


def structural_fidelity(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape[:2] != candidate.shape[:2]:
        candidate = cv2.resize(
            candidate,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    reference_ink = _ink_mask(reference, ignore_red=True)
    candidate_ink = _ink_mask(candidate, ignore_red=False)
    reference_edges = cv2.Canny(reference_ink, 50, 150) > 0
    candidate_edges = cv2.Canny(candidate_ink, 50, 150) > 0
    if not reference_edges.any() or not candidate_edges.any():
        return {
            "edge_recall": 0.0,
            "edge_precision": 0.0,
            "chamfer_distance": float("inf"),
            "endpoint_delta": 999,
            "junction_delta": 999,
            "passed": False,
        }
    reference_distance = cv2.distanceTransform(
        (~reference_edges).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    candidate_distance = cv2.distanceTransform(
        (~candidate_edges).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    recall = float((candidate_distance[reference_edges] <= 3).mean())
    precision = float((reference_distance[candidate_edges] <= 3).mean())
    chamfer = float(
        (
            candidate_distance[reference_edges].mean()
            + reference_distance[candidate_edges].mean()
        )
        / 2
    )

    def topology(mask: np.ndarray) -> tuple[int, int]:
        remaining = (mask > 0).astype(np.uint8) * 255
        skeleton = np.zeros_like(remaining)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(remaining):
            eroded = cv2.erode(remaining, element)
            opened = cv2.dilate(eroded, element)
            skeleton = cv2.bitwise_or(
                skeleton,
                cv2.subtract(remaining, opened),
            )
            remaining = eroded
        skeleton_bool = skeleton > 0
        neighbors = cv2.filter2D(
            skeleton_bool.astype(np.uint8),
            cv2.CV_16U,
            np.ones((3, 3), dtype=np.uint8),
            borderType=cv2.BORDER_CONSTANT,
        ) - skeleton_bool.astype(np.uint8)
        endpoints = int(np.count_nonzero(skeleton_bool & (neighbors == 1)))
        junctions = int(np.count_nonzero(skeleton_bool & (neighbors >= 3)))
        return endpoints, junctions

    reference_endpoints, reference_junctions = topology(reference_ink)
    candidate_endpoints, candidate_junctions = topology(candidate_ink)
    endpoint_delta = abs(reference_endpoints - candidate_endpoints)
    junction_delta = abs(reference_junctions - candidate_junctions)
    endpoint_limit = max(6, round(reference_endpoints * 0.25))
    junction_limit = max(8, round(reference_junctions * 0.25))
    passed = (
        recall >= 0.88
        and precision >= 0.82
        and chamfer <= 2.8
        and endpoint_delta <= endpoint_limit
        and junction_delta <= junction_limit
    )
    return {
        "edge_recall": round(recall, 4),
        "edge_precision": round(precision, 4),
        "chamfer_distance": round(chamfer, 4),
        "reference_endpoints": reference_endpoints,
        "candidate_endpoints": candidate_endpoints,
        "endpoint_delta": endpoint_delta,
        "reference_junctions": reference_junctions,
        "candidate_junctions": candidate_junctions,
        "junction_delta": junction_delta,
        "passed": passed,
    }


def preserve_figure(
    image_path: Path,
    lines: list[dict[str, Any]],
    question_text: str,
    artifact_dir: Path,
    local_ocr: Any | None = None,
) -> FigureResult | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    box = locate_figure(image, lines, question_text)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    original = image[y0:y1, x0:x1]
    if original.size == 0:
        return None
    artifact_dir.mkdir(parents=True, exist_ok=True)
    original_path = artifact_dir / "figure-original.png"
    cv2.imwrite(str(original_path), original)

    if "三个圆圈" in question_text:
        figure_kind: FigureKind = "three_overlapping_circles"
    elif "地图" in question_text:
        figure_kind = "five_country_map"
    elif re.search(r"2\s*[×xX*]\s*10\s*方格", question_text):
        figure_kind = "grid_2x10"
    else:
        return None

    local_lines: list[dict[str, Any]] = []
    local_ocr_error = ""
    if figure_kind == "five_country_map":
        try:
            if local_ocr is None:
                raise RuntimeError("未提供局部 OCR")
            _, _, local_lines = local_ocr.recognize(original_path)
        except Exception as error:
            local_ocr_error = f"{type(error).__name__}: {error}"

    reconstruction = reconstruct_figure(
        original,
        figure_kind,
        question_text,
        local_lines,
    )
    cleaned = reconstruction.image
    metrics = dict(reconstruction.metrics)
    metrics.update(
        {
            "box": [x0, y0, x1, y1],
            "selection": "pixel_grounded_reconstruction"
            if metrics["passed"]
            else "reconstruction_rejected",
            "local_ocr_error": local_ocr_error,
        }
    )
    reasons = list(reconstruction.review_reasons)
    cleaned_path = artifact_dir / "figure-cleaned.png"
    cv2.imwrite(str(cleaned_path), cleaned)
    selected = cleaned
    selected_path = artifact_dir / "figure-selected.png"
    cv2.imwrite(str(selected_path), selected)
    return FigureResult(
        original_path=original_path,
        cleaned_path=cleaned_path,
        selected_path=selected_path,
        box=[x0, y0, x1, y1],
        metrics=metrics,
        review_reasons=reasons,
    )
