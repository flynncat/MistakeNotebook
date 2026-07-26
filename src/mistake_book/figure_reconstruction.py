from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FigureKind = Literal["three_overlapping_circles", "five_country_map", "grid_2x10"]
AxisSegment = tuple[int, int, int, int]


@dataclass
class ReconstructedFigure:
    image: np.ndarray
    metrics: dict[str, Any]
    review_reasons: list[str]


def _red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = bgr[:, :, 0].astype(np.int16)
    green = bgr[:, :, 1].astype(np.int16)
    red = bgr[:, :, 2].astype(np.int16)
    return (
        (red > 95)
        & (hsv[:, :, 1] > 45)
        & ((hsv[:, :, 0] < 18) | (hsv[:, :, 0] > 165))
        & (red - green > 22)
        & (red - blue > 15)
    )


def _flattened_gray(bgr: np.ndarray) -> np.ndarray:
    working = bgr.copy()
    working[_red_mask(working)] = 255
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    sigma = max(9.0, min(gray.shape[:2]) / 18)
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.divide(gray, np.maximum(background, 1), scale=255)


def _strict_ink(bgr: np.ndarray) -> np.ndarray:
    flattened = _flattened_gray(bgr)
    dark_values = flattened[flattened < 220]
    if dark_values.size:
        threshold, _ = cv2.threshold(
            dark_values.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        cutoff = int(np.clip(threshold + 18, 105, 155))
    else:
        cutoff = 130
    return (flattened < cutoff).astype(np.uint8) * 255


def _merge_axis_segments(
    segments: list[AxisSegment],
    *,
    horizontal: bool,
    coordinate_tolerance: int = 7,
    gap_tolerance: int = 18,
) -> list[AxisSegment]:
    if not segments:
        return []
    groups: list[list[AxisSegment]] = []
    for segment in sorted(segments, key=lambda item: item[1] if horizontal else item[0]):
        coordinate = segment[1] if horizontal else segment[0]
        for group in groups:
            group_coordinate = int(
                round(
                    np.median(
                        [item[1] if horizontal else item[0] for item in group]
                    )
                )
            )
            if abs(coordinate - group_coordinate) <= coordinate_tolerance:
                group.append(segment)
                break
        else:
            groups.append([segment])

    merged: list[AxisSegment] = []
    for group in groups:
        coordinate = int(
            round(np.median([item[1] if horizontal else item[0] for item in group]))
        )
        intervals = sorted(
            (
                (min(item[0], item[2]), max(item[0], item[2]))
                if horizontal
                else (min(item[1], item[3]), max(item[1], item[3]))
            )
            for item in group
        )
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end + gap_tolerance:
                end = max(end, next_end)
                continue
            merged.append(
                (start, coordinate, end, coordinate)
                if horizontal
                else (coordinate, start, coordinate, end)
            )
            start, end = next_start, next_end
        merged.append(
            (start, coordinate, end, coordinate)
            if horizontal
            else (coordinate, start, coordinate, end)
        )
    return merged


def _axis_segments(
    bgr: np.ndarray,
    *,
    min_horizontal_fraction: float = 0.22,
    min_vertical_fraction: float = 0.22,
) -> tuple[list[AxisSegment], list[AxisSegment], np.ndarray]:
    ink = _strict_ink(bgr)
    height, width = ink.shape
    flattened = _flattened_gray(bgr)
    line_ink = (flattened < 190).astype(np.uint8) * 255
    horizontal_length = max(25, round(width * 0.075))
    vertical_length = max(25, round(height * 0.075))
    horizontal_mask = cv2.morphologyEx(
        line_ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1)),
    )
    vertical_mask = cv2.morphologyEx(
        line_ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length)),
    )
    horizontal_mask = cv2.morphologyEx(
        horizontal_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)),
    )
    vertical_mask = cv2.morphologyEx(
        vertical_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11)),
    )

    horizontal: list[AxisSegment] = []
    contours, _ = cv2.findContours(
        horizontal_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if (
            box_width >= width * min_horizontal_fraction
            and box_height <= max(18, height * 0.05)
        ):
            coordinate = y + box_height // 2
            horizontal.append((x, coordinate, x + box_width - 1, coordinate))

    vertical: list[AxisSegment] = []
    contours, _ = cv2.findContours(
        vertical_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if (
            box_height >= height * min_vertical_fraction
            and box_width <= max(18, width * 0.05)
        ):
            coordinate = x + box_width // 2
            vertical.append((coordinate, y, coordinate, y + box_height - 1))

    hough = cv2.HoughLinesP(
        line_ink,
        1,
        np.pi / 360,
        threshold=max(20, round(min(height, width) * 0.045)),
        minLineLength=max(28, round(min(height, width) * 0.09)),
        maxLineGap=max(16, round(min(height, width) * 0.04)),
    )
    if hough is not None:
        for x0, y0, x1, y1 in hough[:, 0]:
            line_width = abs(int(x1) - int(x0))
            line_height = abs(int(y1) - int(y0))
            if (
                line_height <= 5
                and line_width >= width * min_horizontal_fraction * 0.7
            ):
                coordinate = round((int(y0) + int(y1)) / 2)
                horizontal.append((int(x0), coordinate, int(x1), coordinate))
            elif (
                line_width <= 5
                and line_height >= height * min_vertical_fraction * 0.55
            ):
                coordinate = round((int(x0) + int(x1)) / 2)
                vertical.append((coordinate, int(y0), coordinate, int(y1)))

    horizontal = _merge_axis_segments(horizontal, horizontal=True)
    vertical = _merge_axis_segments(vertical, horizontal=False)
    horizontal = [
        segment
        for segment in horizontal
        if _line_support(
            _line_mask((height, width), [segment], [], thickness=3),
            ink,
        )
        >= 0.78
    ]
    vertical = [
        segment
        for segment in vertical
        if _line_support(
            _line_mask((height, width), [], [segment], thickness=3),
            ink,
        )
        >= 0.78
    ]
    return horizontal, vertical, ink


def _snap_segments(
    horizontal: list[AxisSegment],
    vertical: list[AxisSegment],
    *,
    tolerance: int = 14,
) -> tuple[list[AxisSegment], list[AxisSegment]]:
    x_coordinates = [segment[0] for segment in vertical]
    y_coordinates = [segment[1] for segment in horizontal]

    def nearest(value: int, coordinates: list[int]) -> int:
        if not coordinates:
            return value
        candidate = min(coordinates, key=lambda coordinate: abs(coordinate - value))
        return candidate if abs(candidate - value) <= tolerance else value

    snapped_horizontal = [
        (nearest(x0, x_coordinates), y0, nearest(x1, x_coordinates), y1)
        for x0, y0, x1, y1 in horizontal
    ]
    snapped_vertical = [
        (x0, nearest(y0, y_coordinates), x1, nearest(y1, y_coordinates))
        for x0, y0, x1, y1 in vertical
    ]
    return snapped_horizontal, snapped_vertical


def _line_mask(
    shape: tuple[int, int],
    horizontal: list[AxisSegment],
    vertical: list[AxisSegment],
    *,
    thickness: int = 5,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for x0, y0, x1, y1 in [*horizontal, *vertical]:
        cv2.line(mask, (x0, y0), (x1, y1), 255, thickness, cv2.LINE_8)
    return mask


def _line_support(line_mask: np.ndarray, ink: np.ndarray) -> float:
    candidate = line_mask > 0
    if not candidate.any():
        return 0.0
    distance = cv2.distanceTransform(
        (ink == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    return float((distance[candidate] <= 3.5).mean())


def _interior_regions(boundary_mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    free = (boundary_mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, 8)
    height, width = free.shape
    regions: list[int] = []
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        touches_border = (
            x == 0 or y == 0 or x + box_width >= width or y + box_height >= height
        )
        if not touches_border and area >= height * width * 0.015:
            regions.append(index)
    return labels, regions


def _box_center(
    line: dict[str, Any],
    width: int,
    height: int,
) -> tuple[int, int] | None:
    box = line.get("box", [])
    if len(box) != 4:
        return None
    x, y, box_width, box_height = (float(value) for value in box)
    return (
        round((x + box_width / 2) * width),
        round((1 - y - box_height / 2) * height),
    )


def _country_labels(question_text: str) -> list[str]:
    match = re.search(
        r"(?:国家|区域)[：:，,\s]*([A-Z](?:[、,，\s]*[A-Z]){2,})",
        question_text,
    )
    if not match:
        return []
    return list(dict.fromkeys(re.findall(r"[A-Z]", match.group(1))))


def _italic_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"),
        Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _render_lines_and_labels(
    shape: tuple[int, int],
    horizontal: list[AxisSegment],
    vertical: list[AxisSegment],
    labels: dict[str, tuple[int, int]],
    *,
    scale: int = 2,
) -> np.ndarray:
    height, width = shape
    canvas = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(canvas)
    line_width = max(3, round(min(height, width) * 0.006 * scale))
    for x0, y0, x1, y1 in [*horizontal, *vertical]:
        draw.line(
            (x0 * scale, y0 * scale, x1 * scale, y1 * scale),
            fill="black",
            width=line_width,
        )
    font = _italic_font(max(22, round(height * 0.09)) * scale)
    for text, (center_x, center_y) in labels.items():
        box = draw.textbbox((0, 0), text, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        draw.text(
            (
                center_x * scale - text_width / 2,
                center_y * scale - text_height / 2 - box[1],
            ),
            text,
            font=font,
            fill="black",
        )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def _region_center(labels: np.ndarray, region: int) -> tuple[int, int]:
    ys, xs = np.where(labels == region)
    return int(round(float(np.median(xs)))), int(round(float(np.median(ys))))


def _map_adjacency(
    labels: np.ndarray,
    region_names: dict[int, str],
    horizontal: list[AxisSegment],
    vertical: list[AxisSegment],
    *,
    offset: int = 9,
) -> list[tuple[str, str]]:
    height, width = labels.shape
    pair_counts: dict[tuple[str, str], int] = {}

    def add_pairs(first: np.ndarray, second: np.ndarray) -> None:
        for left_region, right_region in zip(first.tolist(), second.tolist()):
            if (
                left_region not in region_names
                or right_region not in region_names
                or left_region == right_region
            ):
                continue
            pair = tuple(
                sorted((region_names[left_region], region_names[right_region]))
            )
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    for x0, y0, x1, _ in horizontal:
        start, end = sorted((x0, x1))
        xs = np.arange(start + offset, end - offset + 1)
        if xs.size < 5 or y0 - offset < 0 or y0 + offset >= height:
            continue
        add_pairs(labels[y0 - offset, xs], labels[y0 + offset, xs])
    for x0, y0, _, y1 in vertical:
        start, end = sorted((y0, y1))
        ys = np.arange(start + offset, end - offset + 1)
        if ys.size < 5 or x0 - offset < 0 or x0 + offset >= width:
            continue
        add_pairs(labels[ys, x0 - offset], labels[ys, x0 + offset])
    return sorted(
        pair for pair, count in pair_counts.items() if count >= max(8, offset)
    )


def _reconstruct_map(
    bgr: np.ndarray,
    question_text: str,
    ocr_lines: list[dict[str, Any]],
) -> ReconstructedFigure:
    height, width = bgr.shape[:2]
    horizontal, vertical, ink = _axis_segments(bgr)
    horizontal, vertical = _snap_segments(horizontal, vertical, tolerance=42)
    boundary = _line_mask((height, width), horizontal, vertical)
    labels_image, regions = _interior_regions(boundary)
    expected = _country_labels(question_text)
    assigned: dict[int, str] = {}
    grounded_centers: dict[str, tuple[int, int]] = {}
    recognized: list[str] = []

    for line in ocr_lines:
        text = str(line.get("text", "")).strip()
        if text not in expected or text in recognized:
            continue
        center = _box_center(line, width, height)
        if center is None:
            continue
        center_x, center_y = center
        if not (0 <= center_x < width and 0 <= center_y < height):
            continue
        region = int(labels_image[center_y, center_x])
        if region not in regions or region in assigned:
            continue
        assigned[region] = text
        grounded_centers[text] = center
        recognized.append(text)

    missing_names = [name for name in expected if name not in recognized]
    missing_regions = [region for region in regions if region not in assigned]
    inferred: list[str] = []
    if len(missing_names) == 1 and len(missing_regions) == 1:
        name = missing_names[0]
        region = missing_regions[0]
        assigned[region] = name
        grounded_centers[name] = _region_center(labels_image, region)
        inferred.append(name)

    support = _line_support(boundary, ink)
    region_names = {region: name for region, name in assigned.items()}
    adjacency = _map_adjacency(
        labels_image,
        region_names,
        horizontal,
        vertical,
    )
    passed = (
        len(horizontal) >= 5
        and len(vertical) >= 3
        and len(expected) >= 3
        and len(regions) == len(expected)
        and len(assigned) == len(expected)
        and len(inferred) <= 1
        and support >= 0.82
        and len(adjacency) >= len(expected)
    )
    reasons = [] if passed else ["地图几何或标签未通过原图像素支撑校验"]
    image = _render_lines_and_labels(
        (height, width),
        horizontal,
        vertical,
        grounded_centers,
    )
    metrics = {
        "reconstruction": "pixel_grounded_vector",
        "kind": "five_country_map",
        "horizontal_segments": [list(segment) for segment in horizontal],
        "vertical_segments": [list(segment) for segment in vertical],
        "line_support": round(support, 4),
        "interior_region_count": len(regions),
        "expected_labels": expected,
        "recognized_labels": recognized,
        "inferred_labels": inferred,
        "adjacency_edges": [list(edge) for edge in adjacency],
        "background": "pure_white",
        "residual_handwriting_pixels": 0,
        "passed": passed,
    }
    return ReconstructedFigure(image=image, metrics=metrics, review_reasons=reasons)


def _select_three_circles(bgr: np.ndarray) -> list[tuple[int, int, int]]:
    height, width = bgr.shape[:2]
    gray = cv2.medianBlur(_flattened_gray(bgr), 7)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(45, round(min(height, width) * 0.15)),
        param1=120,
        param2=35,
        minRadius=max(25, round(min(height, width) * 0.16)),
        maxRadius=max(60, round(min(height, width) * 0.43)),
    )
    if circles is None:
        return []
    candidates = [
        (int(round(x)), int(round(y)), int(round(radius)))
        for x, y, radius in circles[0][:40]
    ]
    best: tuple[float, tuple[tuple[int, int, int], ...]] | None = None
    for group in combinations(candidates, 3):
        radii = np.array([circle[2] for circle in group], dtype=np.float32)
        mean_radius = float(radii.mean())
        if float(radii.max() - radii.min()) > mean_radius * 0.25:
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
    return list(best[1]) if best else []


def _circle_mask(
    shape: tuple[int, int],
    circles: list[tuple[int, int, int]],
    *,
    thickness: int = 4,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y, radius in circles:
        cv2.circle(mask, (x, y), radius, 255, thickness, cv2.LINE_8)
    return mask


def _glyph_components(
    ink: np.ndarray,
    structure_mask: np.ndarray,
    region_labels: np.ndarray,
    regions: list[int],
) -> list[tuple[int, int, int, int, int, int]]:
    residual = cv2.bitwise_and(
        ink,
        cv2.bitwise_not(
            cv2.dilate(
                structure_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            )
        ),
    )
    count, components, stats, _ = cv2.connectedComponentsWithStats(residual, 8)
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if not (5 <= width <= 55 and 14 <= height <= 60 and area >= 28):
            continue
        center_x, center_y = x + width // 2, y + height // 2
        region = int(region_labels[center_y, center_x])
        if region in regions:
            candidates.append((region, x, y, width, height, area))
    return candidates


def _reconstruct_circles(bgr: np.ndarray) -> ReconstructedFigure:
    height, width = bgr.shape[:2]
    ink = _strict_ink(bgr)
    circles = _select_three_circles(bgr)
    boundary = _circle_mask((height, width), circles)
    _, regions = _interior_regions(boundary)

    support = _line_support(boundary, ink)
    passed = (
        len(circles) == 3
        and len(regions) == 7
        and support >= 0.62
    )
    scale = 2
    output = np.full((height * scale, width * scale, 3), 255, dtype=np.uint8)
    line_width = max(3, round(min(height, width) * 0.006 * scale))
    for x, y, radius in circles:
        cv2.circle(
            output,
            (x * scale, y * scale),
            radius * scale,
            (0, 0, 0),
            line_width,
            cv2.LINE_AA,
        )

    reasons = [] if passed else ["圆形配图未通过三圆和七区域结构校验"]
    metrics = {
        "reconstruction": "pixel_grounded_vector",
        "kind": "three_overlapping_circles",
        "circles": [list(circle) for circle in circles],
        "line_support": round(support, 4),
        "interior_region_count": len(regions),
        "removed_non_structural_ink": True,
        "background": "pure_white",
        "residual_handwriting_pixels": 0,
        "passed": passed,
    }
    return ReconstructedFigure(image=output, metrics=metrics, review_reasons=reasons)


def _cluster_numbers(values: list[int], *, tolerance: int) -> list[list[int]]:
    groups: list[list[int]] = []
    for value in sorted(values):
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _reconstruct_grid(bgr: np.ndarray) -> ReconstructedFigure:
    height, width = bgr.shape[:2]
    horizontal, vertical, ink = _axis_segments(
        bgr,
        min_horizontal_fraction=0.45,
        min_vertical_fraction=0.14,
    )
    if len(horizontal) >= 3:
        row_coordinates = sorted(segment[1] for segment in horizontal)
        frame_y0, frame_y1 = row_coordinates[0], row_coordinates[-1]
        frame_height = max(1, frame_y1 - frame_y0)
        vertical = [
            segment
            for segment in vertical
            if max(
                0,
                min(segment[3], frame_y1) - max(segment[1], frame_y0),
            )
            >= frame_height * 0.45
        ]
        flattened = _flattened_gray(bgr)
        line_ink = (flattened < 205).astype(np.uint8) * 255
        hough = cv2.HoughLinesP(
            line_ink,
            1,
            np.pi / 720,
            threshold=18,
            minLineLength=max(45, round(frame_height * 0.3)),
            maxLineGap=max(24, round(frame_height * 0.22)),
        )
        x_candidates = [segment[0] for segment in vertical]
        if hough is not None:
            for x0, y0, x1, y1 in hough[:, 0]:
                if (
                    abs(int(x1) - int(x0)) <= 8
                    and min(int(y0), int(y1)) <= frame_y1
                    and max(int(y0), int(y1)) >= frame_y0
                ):
                    x_candidates.append(round((int(x0) + int(x1)) / 2))
        approximate_left = min(segment[0] for segment in horizontal)
        approximate_right = max(segment[2] for segment in horizontal)
        x_candidates = [
            coordinate
            for coordinate in x_candidates
            if approximate_left - 12 <= coordinate <= approximate_right + 12
        ]
        x_candidates = sorted(
            int(round(float(np.median(group))))
            for group in _cluster_numbers(x_candidates, tolerance=7)
        )
        if len(x_candidates) >= 11:
            frame_x0, frame_x1 = x_candidates[0], x_candidates[-1]
            expected_x = np.linspace(frame_x0, frame_x1, 11)
            selected_x: list[int] = []
            for target_x in expected_x:
                available = [
                    coordinate
                    for coordinate in x_candidates
                    if coordinate not in selected_x
                ]
                candidate = min(
                    available,
                    key=lambda coordinate: abs(coordinate - target_x),
                )
                if abs(candidate - target_x) <= max(28, (frame_x1 - frame_x0) / 30):
                    selected_x.append(candidate)
            if len(selected_x) == 11:
                vertical = [
                    (coordinate, frame_y0, coordinate, frame_y1)
                    for coordinate in sorted(selected_x)
                ]
                horizontal = [
                    (frame_x0, coordinate, frame_x1, coordinate)
                    for coordinate in row_coordinates[:3]
                ]
        horizontal, vertical = _snap_segments(
            horizontal,
            vertical,
            tolerance=10,
        )
        frame_x0 = min(segment[0] for segment in horizontal)
        frame_x1 = max(segment[2] for segment in horizontal)
    else:
        frame_x0, frame_y0, frame_x1, frame_y1 = 0, 0, width - 1, height - 1
    boundary = _line_mask((height, width), horizontal, vertical, thickness=4)
    support = _line_support(boundary, ink)

    crop_x0 = max(0, frame_x0 - round(width * 0.18))
    crop_x1 = min(width, frame_x1 + 8)
    crop_y0 = max(0, frame_y0 - round(height * 0.24))
    crop_y1 = min(height, frame_y1 + 8)
    print_pixels = ink.copy()
    inside = np.zeros_like(print_pixels)
    cv2.rectangle(
        inside,
        (frame_x0 + 3, frame_y0 + 3),
        (frame_x1 - 3, frame_y1 - 3),
        255,
        -1,
    )
    print_pixels[inside > 0] = 0
    print_pixels = cv2.bitwise_or(print_pixels, boundary)
    cropped = print_pixels[crop_y0:crop_y1, crop_x0:crop_x1]
    output = np.full((*cropped.shape, 3), 255, dtype=np.uint8)
    output[cropped > 0] = 0
    output = cv2.resize(
        output,
        (output.shape[1] * 2, output.shape[0] * 2),
        interpolation=cv2.INTER_NEAREST,
    )
    passed = len(horizontal) == 3 and len(vertical) == 11 and support >= 0.82
    reasons = [] if passed else ["方格配图未通过 2×10 网格结构校验"]
    metrics = {
        "reconstruction": "pixel_grounded_binary",
        "kind": "grid_2x10",
        "horizontal_segments": [list(segment) for segment in horizontal],
        "vertical_segments": [list(segment) for segment in vertical],
        "line_support": round(support, 4),
        "grid_rows": max(0, len(horizontal) - 1),
        "grid_columns": max(0, len(vertical) - 1),
        "crop": [crop_x0, crop_y0, crop_x1, crop_y1],
        "background": "pure_white",
        "residual_handwriting_pixels_inside_grid": 0,
        "passed": passed,
    }
    return ReconstructedFigure(image=output, metrics=metrics, review_reasons=reasons)


def reconstruct_figure(
    bgr: np.ndarray,
    figure_kind: FigureKind,
    question_text: str,
    ocr_lines: list[dict[str, Any]] | None = None,
) -> ReconstructedFigure:
    if figure_kind == "five_country_map":
        return _reconstruct_map(bgr, question_text, ocr_lines or [])
    if figure_kind == "three_overlapping_circles":
        return _reconstruct_circles(bgr)
    if figure_kind == "grid_2x10":
        return _reconstruct_grid(bgr)
    raise ValueError(f"不支持的配图类型：{figure_kind}")
