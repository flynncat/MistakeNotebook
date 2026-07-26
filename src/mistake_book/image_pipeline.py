from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()


@dataclass
class PipelineResult:
    normalized_path: Path
    cleaned_path: Path
    selected_path: Path
    metrics: dict[str, Any]
    review_reasons: list[str]


def _resize_for_processing(image: np.ndarray, maximum: int = 2600) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1:
        return image
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def load_image(path: Path, maximum: int = 2600) -> np.ndarray:
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        array = np.asarray(rgb)
    return _resize_for_processing(
        cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
        maximum=maximum,
    )


def _text_horizontal_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (600, max(200, round(600 * gray.shape[0] / gray.shape[1]))))
    binary = cv2.adaptiveThreshold(
        small,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    def joined_line_score(horizontal: bool) -> float:
        kernel = (
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
            if horizontal
            else cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
        )
        joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        score = 0.0
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            long_side, short_side = (width, height) if horizontal else (height, width)
            if short_side < 2 or long_side / short_side < 2.8:
                continue
            if width * height > binary.size * 0.08:
                continue
            score += min(long_side, 180) * min(long_side / short_side, 8)
        return score

    horizontal_score = joined_line_score(True)
    vertical_score = joined_line_score(False)
    return math.log1p(horizontal_score) - math.log1p(vertical_score)


def orient_image(
    image: np.ndarray, rotation_override: int | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    if rotation_override in {-90, 90, 180}:
        rotations = {
            -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
        }
        return cv2.rotate(image, rotations[rotation_override]), {
            "rotation_degrees": rotation_override,
            "orientation_confidence": 0.9,
            "orientation_scores": [],
            "orientation_source": "ocr-boxes",
        }
    candidates = [image, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)]
    scores = [_text_horizontal_score(candidate) for candidate in candidates]
    rotate_90 = scores[1] > scores[0]
    oriented = candidates[1] if rotate_90 else candidates[0]
    confidence = min(1.0, 0.5 + abs(scores[1] - scores[0]) / 4)
    return oriented, {
        "rotation_degrees": 90 if rotate_90 else 0,
        "orientation_confidence": round(confidence, 4),
        "orientation_scores": [round(score, 4) for score in scores],
        "orientation_source": "image-heuristic",
    }


def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _perspective_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = _order_points(points)
    width = int(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    height = int(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    if width < 100 or height < 100:
        return image
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32),
        target,
    )
    return cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def rectify_page(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    scale = min(1.0, 1200 / max(height, width))
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(preview, cv2.COLOR_BGR2LAB)
    light = cv2.GaussianBlur(lab[:, :, 0], (9, 9), 0)
    threshold_value, paper = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = cv2.morphologyEx(
        paper,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)),
    )
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, np.ndarray] | None = None
    preview_area = preview.shape[0] * preview.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < preview_area * 0.25:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            continue
        rectangularity = area / max(cv2.contourArea(cv2.convexHull(contour)), 1)
        score = min(1.0, area / preview_area) * rectangularity
        if best is None or score > best[0]:
            best = (score, approximation.reshape(4, 2).astype(np.float32) / scale)
    if best is None:
        return image, {
            "page_confidence": 0.35,
            "page_threshold": round(float(threshold_value), 2),
            "perspective_applied": False,
        }
    score, points = best
    transformed = _perspective_transform(image, points)
    return transformed, {
        "page_confidence": round(float(score), 4),
        "page_threshold": round(float(threshold_value), 2),
        "perspective_applied": True,
    }


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=80,
        minLineLength=max(80, image.shape[1] // 5),
        maxLineGap=30,
    )
    if lines is None:
        return image, 0.0
    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180
        if abs(angle) <= 15:
            angles.append(angle)
    if not angles:
        return image, 0.0
    angle = float(np.median(angles))
    if abs(angle) < 0.15:
        return image, angle
    center = (image.shape[1] / 2, image.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    corrected = cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, angle


def _dewarp_text_baseline(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, min(180, int(np.percentile(gray, 35))), 255, cv2.THRESH_BINARY_INV)[1]
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    row_density = np.count_nonzero(binary, axis=1)
    if row_density.max(initial=0) < image.shape[1] * 0.08:
        return image, {"dewarp_applied": False, "baseline_amplitude": 0.0}
    center_y = int(np.argmax(row_density))
    radius = max(20, image.shape[0] // 80)
    xs: list[float] = []
    ys: list[float] = []
    bins = 36
    for index in range(bins):
        left = round(index * image.shape[1] / bins)
        right = round((index + 1) * image.shape[1] / bins)
        y0, y1 = max(0, center_y - radius), min(image.shape[0], center_y + radius + 1)
        locations = np.argwhere(binary[y0:y1, left:right] > 0)
        if len(locations) >= 5:
            xs.append((left + right) / 2)
            ys.append(float(np.median(locations[:, 0] + y0)))
    if len(xs) < bins * 0.55:
        return image, {"dewarp_applied": False, "baseline_amplitude": 0.0}
    coefficients = np.polyfit(np.asarray(xs), np.asarray(ys), 2)
    all_x = np.arange(image.shape[1], dtype=np.float32)
    curve = np.polyval(coefficients, all_x)
    linear = np.linspace(curve[0], curve[-1], image.shape[1])
    displacement = curve - linear
    amplitude = float(displacement.max() - displacement.min())
    residual = float(
        np.sqrt(np.mean((np.asarray(ys) - np.polyval(coefficients, np.asarray(xs))) ** 2))
    )
    if amplitude < max(4.0, image.shape[0] * 0.003) or amplitude > image.shape[0] * 0.04 or residual > 8:
        return image, {
            "dewarp_applied": False,
            "baseline_amplitude": round(amplitude, 3),
            "baseline_residual": round(residual, 3),
        }
    map_x, map_y = np.meshgrid(
        np.arange(image.shape[1], dtype=np.float32),
        np.arange(image.shape[0], dtype=np.float32),
    )
    map_y += displacement[np.newaxis, :].astype(np.float32)
    corrected = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, {
        "dewarp_applied": True,
        "baseline_amplitude": round(amplitude, 3),
        "baseline_residual": round(residual, 3),
    }


def normalize_background(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, channel_a, channel_b = cv2.split(lab)
    background = cv2.GaussianBlur(light, (0, 0), sigmaX=max(15, image.shape[1] / 45))
    target = int(np.percentile(background, 75))
    normalized = cv2.addWeighted(light, 1.0, background, -1.0, target)
    normalized = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(12, 12)).apply(normalized)
    return cv2.cvtColor(cv2.merge((normalized, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def crop_to_content(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 21))
    content = (gray < 205) & (local > 7)
    content_u8 = (content.astype(np.uint8) * 255)
    content_u8 = cv2.morphologyEx(
        content_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(content_u8, 8)
    keep = np.zeros_like(content_u8)
    max_component = image.shape[0] * image.shape[1] * 0.08
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if 5 <= area <= max_component and width < image.shape[1] * 0.95 and height < image.shape[0] * 0.95:
            keep[labels == index] = 255
    points = cv2.findNonZero(keep)
    if points is None:
        return image, {"content_crop_applied": False}
    x, y, width, height = cv2.boundingRect(points)
    pad_x = max(24, round(width * 0.04))
    pad_y = max(24, round(height * 0.08))
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(image.shape[1], x + width + pad_x), min(image.shape[0], y + height + pad_y)
    if (x1 - x0) * (y1 - y0) < image.shape[0] * image.shape[1] * 0.08:
        return image, {"content_crop_applied": False}
    return image[y0:y1, x0:x1], {
        "content_crop_applied": True,
        "content_box": [x0, y0, x1, y1],
    }


def conservative_remove_marks(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    red_raw = (
        (hsv[:, :, 1] > 65)
        & ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 168))
        & (image[:, :, 2].astype(np.int16) - image[:, :, 1].astype(np.int16) > 12)
    )
    red_labels_count, red_labels, red_stats, _ = cv2.connectedComponentsWithStats(
        red_raw.astype(np.uint8), 8
    )
    red = np.zeros_like(red_raw)
    image_area = image.shape[0] * image.shape[1]
    for index in range(1, red_labels_count):
        _, _, width, height, area = red_stats[index]
        box_area = max(1, width * height)
        if (
            3 <= area <= max(10000, image_area * 0.008)
            and (box_area <= image_area * 0.04 or area / box_area < 0.35)
        ):
            red[red_labels == index] = True
    low_saturation = hsv[:, :, 1] < 38
    print_core = ((gray < 125) & low_saturation) | ((gray < 92) & ~red)
    protection = cv2.dilate(
        print_core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)
    local_background = cv2.GaussianBlur(gray, (0, 0), 13)
    white_fraction = cv2.blur((gray > 188).astype(np.float32), (17, 17))
    pencil = (
        low_saturation
        & (gray >= 138)
        & (gray < 205)
        & (local_background > 188)
        & ((local_background.astype(np.int16) - gray.astype(np.int16)) > 11)
        & (white_fraction > 0.72)
    )
    safe_red = red & (gray > 70) & ~protection
    safe_pencil = pencil & ~protection
    safe = safe_red | safe_pencil
    safe = cv2.morphologyEx(
        safe.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
    ).astype(bool)
    safe &= ~protection
    background = cv2.GaussianBlur(image, (0, 0), 18)
    cleaned = image.copy()
    cleaned[safe] = background[safe]
    changed = np.any(cleaned != image, axis=2)
    overlap = int(np.count_nonzero(changed & protection))
    metrics = {
        "red_candidate_pixels": int(np.count_nonzero(red)),
        "pencil_candidate_pixels": int(np.count_nonzero(pencil)),
        "removed_pixels": int(np.count_nonzero(changed)),
        "protected_overlap_pixels": overlap,
        "removed_fraction": round(float(np.count_nonzero(changed) / changed.size), 6),
    }
    return cleaned, metrics


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise OSError(f"无法写入图片：{path}")


def extract_printed_question(
    image_path: Path,
    ocr_lines: list[dict[str, Any]],
    artifact_dir: Path,
) -> tuple[Path, dict[str, Any], list[str]]:
    anchor = next(
        (
            index
            for index, line in enumerate(ocr_lines)
            if re.search(r"(?:例题|练习)\s*\d+", str(line.get("text", "")))
        ),
        None,
    )
    if anchor is None:
        return image_path, {"question_extracted": False}, ["未定位到印刷题号"]
    selected: list[dict[str, Any]] = []
    for line in ocr_lines[anchor : anchor + 7]:
        selected.append(line)
        if re.search(r"[？?]", str(line.get("text", ""))):
            break
    selected_text = "".join(str(line.get("text", "")) for line in selected)
    if not re.search(r"[？?]", selected_text):
        return image_path, {"question_extracted": False}, ["未定位到题干结束位置"]

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return image_path, {"question_extracted": False}, ["无法读取清理产物"]
    height, width = image.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for line in selected:
        box = line.get("box", [])
        if len(box) != 4:
            continue
        x, y, box_width, box_height = (float(value) for value in box)
        boxes.append(
            (
                max(0, round(x * width)),
                max(0, round((1 - y - box_height) * height)),
                min(width, round((x + box_width) * width)),
                min(height, round((1 - y) * height)),
            )
        )
    if not boxes:
        return image_path, {"question_extracted": False}, ["题干没有有效位置框"]
    x0 = max(0, min(box[0] for box in boxes) - 24)
    y0 = max(0, min(box[1] for box in boxes) - 18)
    x1 = min(width, max(box[2] for box in boxes) + 24)
    y1 = min(height, max(box[3] for box in boxes) + 18)
    text_crop = image[y0:y1, x0:x1]

    figure_crop: np.ndarray | None = None
    figure_box: list[int] | None = None
    needs_figure = bool(re.search(r"如图|见下图|图中|方格图|地图", selected_text))
    if needs_figure:
        search_top = min(height - 1, y1)
        search_bottom = min(height, y1 + round(height * 0.48))
        region = image[search_top:search_bottom]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        saturation = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[:, :, 1]
        contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            bx, by, bw, bh = cv2.boundingRect(contour)
            contour_mask = np.zeros_like(gray)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            mean_saturation = cv2.mean(saturation, mask=contour_mask)[0]
            if (
                area >= 450
                and bw >= 38
                and bh >= 38
                and bw <= width * 0.45
                and bh <= height * 0.42
                and 0.25 <= area / max(1, bw * bh) <= 1.1
                and mean_saturation < 55
            ):
                candidates.append((bx, by + search_top, bx + bw, by + search_top + bh))
        if candidates:
            clusters: list[list[tuple[int, int, int, int]]] = []
            for candidate in candidates:
                placed = False
                for cluster in clusters:
                    cx0 = min(box[0] for box in cluster) - 45
                    cy0 = min(box[1] for box in cluster) - 45
                    cx1 = max(box[2] for box in cluster) + 45
                    cy1 = max(box[3] for box in cluster) + 45
                    if not (
                        candidate[2] < cx0
                        or candidate[0] > cx1
                        or candidate[3] < cy0
                        or candidate[1] > cy1
                    ):
                        cluster.append(candidate)
                        placed = True
                        break
                if not placed:
                    clusters.append([candidate])
            cluster = max(
                clusters,
                key=lambda group: (
                    max(box[2] for box in group) - min(box[0] for box in group)
                )
                * (max(box[3] for box in group) - min(box[1] for box in group)),
            )
            fx0 = max(0, min(box[0] for box in cluster) - 30)
            fy0 = max(search_top, min(box[1] for box in cluster) - 30)
            fx1 = min(width, max(box[2] for box in cluster) + 30)
            fy1 = min(search_bottom, max(box[3] for box in cluster) + 30)
            figure_crop = image[fy0:fy1, fx0:fx1]
            figure_box = [fx0, fy0, fx1, fy1]

    paper_pixels = image[
        max(0, height // 20) : min(height, height // 5),
        max(0, width // 20) : min(width, width // 5),
    ]
    background = np.median(paper_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
    padding, gap = 24, 18
    content_width = text_crop.shape[1]
    content_height = text_crop.shape[0]
    if figure_crop is not None:
        content_width = max(content_width, figure_crop.shape[1])
        content_height += gap + figure_crop.shape[0]
    canvas = np.full(
        (content_height + 2 * padding, content_width + 2 * padding, 3),
        background,
        dtype=np.uint8,
    )
    canvas[padding : padding + text_crop.shape[0], padding : padding + text_crop.shape[1]] = (
        text_crop
    )
    if figure_crop is not None:
        figure_y = padding + text_crop.shape[0] + gap
        canvas[
            figure_y : figure_y + figure_crop.shape[0],
            padding : padding + figure_crop.shape[1],
        ] = figure_crop
    target = artifact_dir / "question.png"
    _save_png(target, canvas)
    reasons = ["题目提到配图但未可靠定位配图"] if needs_figure and figure_crop is None else []
    return target, {
        "question_extracted": True,
        "question_text_box": [x0, y0, x1, y1],
        "figure_required": needs_figure,
        "figure_box": figure_box,
    }, reasons


def process_image(
    source: Path, artifact_dir: Path, rotation_override: int | None = None
) -> PipelineResult:
    image = load_image(source)
    oriented, orientation_metrics = orient_image(image, rotation_override)
    rectified, page_metrics = rectify_page(oriented)
    straightened, deskew_angle = deskew(rectified)
    dewarped, dewarp_metrics = _dewarp_text_baseline(straightened)
    normalized = normalize_background(dewarped)
    cropped, crop_metrics = crop_to_content(normalized)
    cleaned, cleanup_metrics = conservative_remove_marks(cropped)

    normalized_path = artifact_dir / "normalized.png"
    cleaned_path = artifact_dir / "cleaned.png"
    _save_png(normalized_path, cropped)
    _save_png(cleaned_path, cleaned)

    metrics: dict[str, Any] = {
        **orientation_metrics,
        **page_metrics,
        "deskew_angle": round(deskew_angle, 4),
        **dewarp_metrics,
        **crop_metrics,
        **cleanup_metrics,
        "output_width": int(cleaned.shape[1]),
        "output_height": int(cleaned.shape[0]),
    }
    review_reasons: list[str] = []
    if metrics["orientation_confidence"] < 0.55:
        review_reasons.append("方向判断置信度较低")
    if metrics["page_confidence"] < 0.30:
        review_reasons.append("未可靠检测到纸面边界")
    if cleanup_metrics["protected_overlap_pixels"] > 0:
        review_reasons.append("清理区域与印刷内容保护区冲突")
    if cleanup_metrics["removed_fraction"] > 0.08:
        review_reasons.append("清理面积异常偏大")
    selected = cleaned_path if cleanup_metrics["removed_pixels"] else normalized_path
    return PipelineResult(
        normalized_path=normalized_path,
        cleaned_path=cleaned_path,
        selected_path=selected,
        metrics=metrics,
        review_reasons=review_reasons,
    )
