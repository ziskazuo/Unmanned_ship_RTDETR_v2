#!/usr/bin/env python3
"""
Build single-tower training assets for the sealand dataset.

This script prepares:
1. A per-frame radar training image generated from `radar_pcd`
2. Rotated-box annotations fitted from `gt_filter/*_tosensor_filter.json`
3. RGB 2D camera supervision from `gt_filter_only_yaw/*_tosensor_filter.json`
4. A JSONL sample index that ties cameras, radar, and labels together

The radar image generated here is a geometry-friendly Cartesian radar view for
training. A more realistic polar PPI can be rendered later for visualization.
"""

from __future__ import annotations

import argparse
import os
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import yaml
from PIL import Image


DEFAULT_DATASET_ROOT = Path("/data1/liziao/USV/dataset/sealand_data/dataset")
DEFAULT_OUTPUT_ROOT = Path("./prepared/sealand_single_tower_4km_super4_960_route_roi")
CAMERA_CANONICAL_ORDER = (
    ("Back", "CamBack"),
    ("Front", "CamFront"),
    ("Left", "CamLeft"),
    ("Right", "CamRight"),
)
CAMERA_NAMES = tuple(raw_name for _, raw_name in CAMERA_CANONICAL_ORDER)
CANONICAL_CAMERA_KEYS = tuple(camera_key for camera_key, _ in CAMERA_CANONICAL_ORDER)
CANONICAL_CAMERA_TO_ID = {
    camera_key: idx for idx, camera_key in enumerate(CANONICAL_CAMERA_KEYS)
}
RAW_CAMERA_TO_CANONICAL = {
    raw_name: camera_key for camera_key, raw_name in CAMERA_CANONICAL_ORDER
}
NONE_CAMERA_ID = len(CANONICAL_CAMERA_KEYS)
CAMERA_IMAGE_WIDTH = 1024.0
CAMERA_IMAGE_HEIGHT = 512.0
FRAME_COUNT = 800
SUPER_CATEGORY_ORDER = ["CargoShip", "CruiseShip", "FishingVessel", "RecreationalBoat"]
SUPER_CATEGORY_MAP = {
    "Containership": "CargoShip",
    "libertyship": "FishingVessel",
    "smallcargo": "CargoShip",
    "suppliervessel": "CargoShip",
    "queenmarry": "CruiseShip",
    "ramonasteam": "CruiseShip",
    "fishingboat": "FishingVessel",
    "Yacht": "RecreationalBoat",
    "Sailboat": "RecreationalBoat",
    "HouseBoat": "CruiseShip",
    "Motorboat": "RecreationalBoat",
    "Boataaa": "FishingVessel",
    "CoastGuard": "RecreationalBoat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare single-tower radar/camera training assets for sealand_data."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the sealand dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where prepared assets will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Valid", "Test"],
        choices=["Train", "Valid", "Test", "StressTest"],
        help="Splits to prepare.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=960,
        help="Radar image size in pixels.",
    )
    parser.add_argument(
        "--range-max",
        type=float,
        default=4000.0,
        help="Effective radar range in meters used for training image generation.",
    )
    parser.add_argument(
        "--ref-range",
        type=float,
        default=800.0,
        help="Reference range for gain compensation.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="Percentile used to normalize the radar image.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on total samples to generate, useful for smoke tests.",
    )
    parser.add_argument(
        "--class-scheme",
        type=str,
        default="super4",
        choices=["super4", "fine"],
        help="Class grouping used for exported categories.",
    )
    parser.add_argument(
        "--one-class",
        action="store_true",
        help="Export COCO annotations as a single 'ship' category regardless of class scheme.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Number of worker processes used for per-sample generation.",
    )
    return parser.parse_args()


def split_scene_prefix(split_name: str) -> str:
    return {
        "Train": "T",
        "Valid": "V",
        "Test": "S",
        "StressTest": "H",
    }[split_name]


def iter_scenes(split_dir: Path, split_name: str) -> Iterable[Path]:
    prefix = split_scene_prefix(split_name)
    for path in sorted(split_dir.iterdir()):
        if path.is_dir() and path.name.startswith(prefix):
            yield path


def normalize_category_name(object_name: str) -> str:
    name = object_name
    if name.startswith("USV_"):
        name = name[4:]
    if "_C_" in name:
        name = name.split("_C_", 1)[0]
    while name and name[-1].isdigit():
        name = name[:-1]
    return name or object_name


def map_super_category_name(category_name: str) -> str:
    if category_name not in SUPER_CATEGORY_MAP:
        raise KeyError(f"Unmapped fine-grained category: {category_name}")
    return SUPER_CATEGORY_MAP[category_name]


def yaw_from_bev_entry(bev_entry: Dict[str, object]) -> float:
    return float(bev_entry["yaw"])


def polygon_center(poly: Sequence[Sequence[float]]) -> List[float]:
    return [
        sum(float(pt[0]) for pt in poly) / max(len(poly), 1),
        sum(float(pt[1]) for pt in poly) / max(len(poly), 1),
    ]


def point_inside_image(point: Sequence[float], resolution: int) -> bool:
    max_coord = float(resolution - 1)
    return 0.0 <= float(point[0]) <= max_coord and 0.0 <= float(point[1]) <= max_coord


def polygon_inside_image(poly: Sequence[Sequence[float]], resolution: int) -> bool:
    return all(point_inside_image(point, resolution) for point in poly)


def clamp_point_to_image(point: Sequence[float], resolution: int) -> List[float]:
    max_coord = float(resolution - 1)
    return [
        min(max(float(point[0]), 0.0), max_coord),
        min(max(float(point[1]), 0.0), max_coord),
    ]


def order_polygon_clockwise(points: Sequence[Sequence[float]]) -> List[List[float]]:
    cx = sum(float(pt[0]) for pt in points) / len(points)
    cy = sum(float(pt[1]) for pt in points) / len(points)
    return [
        [float(pt[0]), float(pt[1])]
        for pt in sorted(
            points,
            key=lambda pt: math.atan2(float(pt[1]) - cy, float(pt[0]) - cx),
        )
    ]


def project_metric_xy_to_pixels(
    points_xy: Sequence[Sequence[float]], resolution: int, range_max: float
) -> List[List[float]]:
    cx = resolution / 2.0
    cy = resolution / 2.0
    scale = (resolution / 2.0) / max(range_max, 1e-6)
    return [
        [cx + float(y) * scale, cy - float(x) * scale] for x, y in points_xy
    ]


def project_pixels_to_metric_xy(
    points_uv: Sequence[Sequence[float]], resolution: int, range_max: float
) -> List[List[float]]:
    cx = resolution / 2.0
    cy = resolution / 2.0
    scale = (resolution / 2.0) / max(range_max, 1e-6)
    return [
        [(cy - float(v)) / scale, (float(u) - cx) / scale] for u, v in points_uv
    ]


def cross_product(o: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    return (float(a[0]) - float(o[0])) * (float(b[1]) - float(o[1])) - (
        float(a[1]) - float(o[1])
    ) * (float(b[0]) - float(o[0]))


def convex_hull_xy(points_xy: Sequence[Sequence[float]]) -> List[List[float]]:
    points = sorted({(float(x), float(y)) for x, y in points_xy})
    if len(points) <= 1:
        return [[pt[0], pt[1]] for pt in points]

    lower: List[Tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: List[Tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return [[pt[0], pt[1]] for pt in hull]


def rotate_xy(point: Sequence[float], angle: float) -> Tuple[float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = float(point[0])
    y = float(point[1])
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def fit_min_area_rect_xy(points_xy: Sequence[Sequence[float]]) -> List[List[float]]:
    hull = convex_hull_xy(points_xy)
    if not hull:
        return []
    if len(hull) == 1:
        x, y = hull[0]
        return [[x, y], [x, y], [x, y], [x, y]]
    if len(hull) == 2:
        return [hull[0], hull[1], hull[1], hull[0]]

    best_area = None
    best_corners: List[List[float]] = []
    for idx, point in enumerate(hull):
        nxt = hull[(idx + 1) % len(hull)]
        edge_angle = math.atan2(
            float(nxt[1]) - float(point[1]),
            float(nxt[0]) - float(point[0]),
        )
        rotated = [rotate_xy(pt, -edge_angle) for pt in hull]
        xs = [pt[0] for pt in rotated]
        ys = [pt[1] for pt in rotated]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        area = (max_x - min_x) * (max_y - min_y)
        if best_area is None or area < best_area:
            rect_rot = [
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
                (min_x, min_y),
            ]
            best_corners = [list(rotate_xy(pt, edge_angle)) for pt in rect_rot]
            best_area = area
    return order_polygon_clockwise(best_corners)


def intersect_with_vertical(p1: Sequence[float], p2: Sequence[float], x_value: float) -> List[float]:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if abs(x2 - x1) < 1e-9:
        return [x_value, y1]
    t = (x_value - x1) / (x2 - x1)
    return [x_value, y1 + t * (y2 - y1)]


def intersect_with_horizontal(p1: Sequence[float], p2: Sequence[float], y_value: float) -> List[float]:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if abs(y2 - y1) < 1e-9:
        return [x1, y_value]
    t = (y_value - y1) / (y2 - y1)
    return [x1 + t * (x2 - x1), y_value]


def clip_polygon_with_boundary(
    polygon: Sequence[Sequence[float]],
    inside_fn,
    intersect_fn,
) -> List[List[float]]:
    if not polygon:
        return []
    output: List[List[float]] = []
    prev = [float(v) for v in polygon[-1]]
    prev_inside = inside_fn(prev)
    for curr_raw in polygon:
        curr = [float(curr_raw[0]), float(curr_raw[1])]
        curr_inside = inside_fn(curr)
        if curr_inside:
            if not prev_inside:
                output.append(intersect_fn(prev, curr))
            output.append(curr)
        elif prev_inside:
            output.append(intersect_fn(prev, curr))
        prev = curr
        prev_inside = curr_inside
    return output


def clip_polygon_to_image(
    polygon: Sequence[Sequence[float]], resolution: int
) -> List[List[float]]:
    max_coord = float(resolution - 1)
    clipped = [[float(pt[0]), float(pt[1])] for pt in polygon]
    clipped = clip_polygon_with_boundary(
        clipped,
        lambda pt: pt[0] >= 0.0,
        lambda p1, p2: intersect_with_vertical(p1, p2, 0.0),
    )
    clipped = clip_polygon_with_boundary(
        clipped,
        lambda pt: pt[0] <= max_coord,
        lambda p1, p2: intersect_with_vertical(p1, p2, max_coord),
    )
    clipped = clip_polygon_with_boundary(
        clipped,
        lambda pt: pt[1] >= 0.0,
        lambda p1, p2: intersect_with_horizontal(p1, p2, 0.0),
    )
    clipped = clip_polygon_with_boundary(
        clipped,
        lambda pt: pt[1] <= max_coord,
        lambda p1, p2: intersect_with_horizontal(p1, p2, max_coord),
    )
    return clipped


def clip_polygon_to_rect(
    polygon: Sequence[Sequence[float]], width: float, height: float
) -> List[List[float]]:
    max_x = float(width - 1.0)
    max_y = float(height - 1.0)
    clipped = [[float(pt[0]), float(pt[1])] for pt in polygon]
    clipped = clip_polygon_with_boundary(
        clipped,
        lambda pt: pt[0] >= 0.0,
        lambda p1, p2: intersect_with_vertical(p1, p2, 0.0),
    )
    clipped = clip_polygon_with_boundary(
        clipped,
        lambda pt: pt[0] <= max_x,
        lambda p1, p2: intersect_with_vertical(p1, p2, max_x),
    )
    clipped = clip_polygon_with_boundary(
        clipped,
        lambda pt: pt[1] >= 0.0,
        lambda p1, p2: intersect_with_horizontal(p1, p2, 0.0),
    )
    clipped = clip_polygon_with_boundary(
        clipped,
        lambda pt: pt[1] <= max_y,
        lambda p1, p2: intersect_with_horizontal(p1, p2, max_y),
    )
    return clipped


def metric_rbox_from_polygon(points_xy: Sequence[Sequence[float]]) -> List[float]:
    ordered = order_polygon_clockwise(points_xy)
    edges = []
    for idx, point in enumerate(ordered):
        nxt = ordered[(idx + 1) % len(ordered)]
        dx = float(nxt[0]) - float(point[0])
        dy = float(nxt[1]) - float(point[1])
        edges.append((math.hypot(dx, dy), dx, dy))
    long_edge = max(edges, key=lambda item: item[0])
    short_edge = min(edges, key=lambda item: item[0])
    center_x = sum(float(pt[0]) for pt in ordered) / len(ordered)
    center_y = sum(float(pt[1]) for pt in ordered) / len(ordered)
    yaw = math.atan2(long_edge[2], long_edge[1])
    return [center_x, center_y, long_edge[0], short_edge[0], yaw]


def build_training_polygon_from_raw_poly(
    raw_poly: Sequence[Sequence[float]], resolution: int, range_max: float
) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    if not raw_poly:
        raise KeyError("empty raw polygon")

    if not point_inside_image(polygon_center(raw_poly), resolution):
        raise KeyError("polygon center outside image")

    visible_polygon = clip_polygon_to_image(raw_poly, resolution)
    if len(visible_polygon) < 3:
        raise KeyError("visible polygon too small after clipping")

    train_poly = fit_min_area_rect_xy(visible_polygon)
    for _ in range(4):
        if polygon_inside_image(train_poly, resolution):
            break
        visible_polygon = clip_polygon_to_image(train_poly, resolution)
        if len(visible_polygon) < 3:
            break
        train_poly = fit_min_area_rect_xy(visible_polygon)

    if not polygon_inside_image(train_poly, resolution):
        train_poly = [clamp_point_to_image(point, resolution) for point in train_poly]

    metric_poly = project_pixels_to_metric_xy(train_poly, resolution, range_max)
    rbox_xywhr = metric_rbox_from_polygon(metric_poly)
    return order_polygon_clockwise(train_poly), order_polygon_clockwise(visible_polygon), rbox_xywhr


def polygon_from_corners3d_pixels(
    object_entry: Dict[str, object], resolution: int, range_max: float
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[float], str]:
    radar_proj = object_entry.get("radar_proj", {})
    corners_3d = radar_proj.get("corners_3d")
    if corners_3d:
        points_xy = [[float(point[0]), float(point[1])] for point in corners_3d]
        ordered_xy = fit_min_area_rect_xy(points_xy)
        raw_poly = project_metric_xy_to_pixels(ordered_xy, resolution, range_max)
        train_poly, visible_polygon, rbox_xywhr = build_training_polygon_from_raw_poly(
            raw_poly, resolution, range_max
        )
        return train_poly, visible_polygon, raw_poly, rbox_xywhr, "radar_proj.corners_3d"

    bev_entry = object_entry.get("bev_rot_only_yaw")
    if bev_entry:
        raw_poly = rotated_box_corners_pixels(bev_entry, resolution, range_max)
        train_poly, visible_polygon, rbox_xywhr = build_training_polygon_from_raw_poly(
            raw_poly, resolution, range_max
        )
        return train_poly, visible_polygon, raw_poly, rbox_xywhr, "bev_rot_only_yaw_fallback"

    raise KeyError("Neither radar_proj.corners_3d nor bev_rot_only_yaw is available")


def rotated_box_corners_pixels(
    bev_entry: Dict[str, object], resolution: int, range_max: float
) -> List[List[float]]:
    center = bev_entry["center"]
    extent = bev_entry["extent"]
    yaw = yaw_from_bev_entry(bev_entry)

    xc = float(center["x"])
    yc = float(center["y"])
    half_l = float(extent["x"])
    half_w = float(extent["y"])

    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    cx = resolution / 2.0
    cy = resolution / 2.0
    scale = (resolution / 2.0) / max(range_max, 1e-6)

    corners: List[List[float]] = []
    for dx, dy in (
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, -half_w),
        (-half_l, half_w),
    ):
        x = xc + dx * cos_y - dy * sin_y
        y = yc + dx * sin_y + dy * cos_y
        u = cx + y * scale
        v = cy - x * scale
        corners.append([u, v])
    return corners


def polygon_bounds(poly: Sequence[Sequence[float]]) -> List[float]:
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for idx, point in enumerate(poly):
        nxt = poly[(idx + 1) % len(poly)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(area) / 2.0


def box_xyxy_from_box2d_dict(box_2d: Dict[str, object]) -> List[float]:
    return [
        float(box_2d["xmin"]),
        float(box_2d["ymin"]),
        float(box_2d["xmax"]),
        float(box_2d["ymax"]),
    ]


def clip_box_xyxy_to_rect(
    box_xyxy: Sequence[float], width: float, height: float
) -> List[float] | None:
    x1 = min(max(float(box_xyxy[0]), 0.0), float(width - 1.0))
    y1 = min(max(float(box_xyxy[1]), 0.0), float(height - 1.0))
    x2 = min(max(float(box_xyxy[2]), 0.0), float(width - 1.0))
    y2 = min(max(float(box_xyxy[3]), 0.0), float(height - 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def polygon_from_corners2d(corners_2d: Sequence[Sequence[float]]) -> List[List[float]]:
    points = []
    for point in corners_2d:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        points.append([float(point[0]), float(point[1])])
    return order_polygon_clockwise(points) if len(points) >= 3 else []


def build_camera_supervision(
    object_entry: Dict[str, object],
    image_width: float = CAMERA_IMAGE_WIDTH,
    image_height: float = CAMERA_IMAGE_HEIGHT,
) -> Dict[str, object]:
    camera_boxes: List[List[float]] = [[0.0, 0.0, 0.0, 0.0] for _ in CANONICAL_CAMERA_KEYS]
    camera_poly_area: List[float] = [0.0 for _ in CANONICAL_CAMERA_KEYS]
    visible_cameras: List[int] = [0 for _ in CANONICAL_CAMERA_KEYS]
    has_camera_box: List[int] = [0 for _ in CANONICAL_CAMERA_KEYS]
    primary_camera = NONE_CAMERA_ID
    best_area = -1.0

    cams = object_entry.get("cams", {})
    for camera_id, raw_camera_name in enumerate(CAMERA_NAMES):
        camera_key = RAW_CAMERA_TO_CANONICAL[raw_camera_name]
        camera_entry = cams.get(raw_camera_name) or {}

        clipped_polygon: List[List[float]] = []
        polygon_area_value = 0.0
        corners_2d = camera_entry.get("corners_2d")
        if corners_2d:
            polygon_2d = polygon_from_corners2d(corners_2d)
            if polygon_2d:
                clipped_polygon = clip_polygon_to_rect(
                    polygon_2d, image_width, image_height
                )
                if len(clipped_polygon) >= 3:
                    polygon_area_value = polygon_area(clipped_polygon)

        clipped_box_xyxy = None
        box_2d = camera_entry.get("box_2d")
        if box_2d:
            clipped_box_xyxy = clip_box_xyxy_to_rect(
                box_xyxy_from_box2d_dict(box_2d), image_width, image_height
            )

        if clipped_box_xyxy is None and clipped_polygon:
            clipped_box_xyxy = polygon_bounds(clipped_polygon)
            clipped_box_xyxy = [
                clipped_box_xyxy[0],
                clipped_box_xyxy[1],
                clipped_box_xyxy[0] + clipped_box_xyxy[2],
                clipped_box_xyxy[1] + clipped_box_xyxy[3],
            ]

        if clipped_box_xyxy is not None:
            has_camera_box[camera_id] = 1
            camera_boxes[camera_id] = [float(v) for v in clipped_box_xyxy]

        if polygon_area_value <= 0.0 and clipped_box_xyxy is not None:
            polygon_area_value = max(
                0.0,
                (clipped_box_xyxy[2] - clipped_box_xyxy[0]) *
                (clipped_box_xyxy[3] - clipped_box_xyxy[1]),
            )

        camera_poly_area[camera_id] = float(polygon_area_value)

        # Treat any clipped on-image footprint as visible supervision for the
        # RouteROI branch, even if the raw simulator visibility flag is false.
        # This keeps boundary-truncated targets from becoming "none" while
        # still preserving the clipped geometry that the camera branch sees.
        is_visible = polygon_area_value > 0.0
        visible_cameras[camera_id] = 1 if is_visible else 0
        if is_visible and polygon_area_value > best_area:
            best_area = polygon_area_value
            primary_camera = CANONICAL_CAMERA_TO_ID[camera_key]

    return {
        "gt_primary_camera": primary_camera,
        "gt_visible_cameras": visible_cameras,
        "gt_camera_box_2d": camera_boxes,
        "gt_camera_poly_area": camera_poly_area,
        "gt_has_camera_box": has_camera_box,
    }


def read_pcd_points(pcd_path: Path) -> List[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = []
    with pcd_path.open("r", encoding="utf-8", errors="ignore") as handle:
        data_section = False
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if data_section:
                fields = stripped.split()
                if len(fields) < 3:
                    continue
                try:
                    x, y, z = map(float, fields[:3])
                except ValueError:
                    continue
                points.append((x, y, z))
            elif stripped.startswith("DATA "):
                data_section = True
    return points


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 1.0
    clamped = max(0.0, min(100.0, pct))
    idx = int((clamped / 100.0) * (len(values) - 1))
    return sorted(values)[idx]


def build_radar_image(
    points: Sequence[Tuple[float, float, float]],
    resolution: int,
    range_max: float,
    ref_range: float,
    clip_percentile: float,
) -> Image.Image:
    grid = [[0.0] * resolution for _ in range(resolution)]
    cx = resolution / 2.0
    cy = resolution / 2.0
    scale = (resolution / 2.0) / max(range_max, 1e-6)

    for x, y, _z in points:
        radius = math.hypot(x, y)
        if radius > range_max or radius < 1e-6:
            continue
        gain = min(max(radius / max(ref_range, 1e-6), 0.001), 1000.0)
        u = int(cx + y * scale)
        v = int(cy - x * scale)
        if 0 <= u < resolution and 0 <= v < resolution:
            grid[v][u] += gain

    flat = [math.log1p(val) for row in grid for val in row]
    denom = max(percentile(flat, clip_percentile), 1e-6)

    image = Image.new("L", (resolution, resolution), 0)
    pixels = image.load()
    index = 0
    for v in range(resolution):
        for u in range(resolution):
            value = int(max(0, min(255, round(flat[index] / denom * 255.0))))
            pixels[u, v] = value
            index += 1
    return image


def build_annotations(
    gt_filter_path: Path,
    gt_filter_only_yaw_path: Path,
    tower_name: str,
    resolution: int,
    range_max: float,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, List[List[float]]]]]:
    with gt_filter_path.open("r", encoding="utf-8") as handle:
        gt = json.load(handle)
    if not gt_filter_only_yaw_path.is_file():
        raise FileNotFoundError(
            f"Missing RGB 2D GT source file: {gt_filter_only_yaw_path}")
    with gt_filter_only_yaw_path.open("r", encoding="utf-8") as handle:
        gt_rgb = json.load(handle)

    tower = gt["towers"][tower_name]
    tower_rgb = gt_rgb["towers"].get(tower_name, {})
    tower_rgb_objects = tower_rgb.get("objects", {})
    annotations: List[Dict[str, object]] = []
    yaml_polygons: Dict[str, Dict[str, List[List[float]]]] = {}

    for object_name, object_entry in sorted(tower.get("objects", {}).items()):
        try:
            poly, visible_polygon, raw_poly, rbox_xywhr, label_source = polygon_from_corners3d_pixels(
                object_entry=object_entry,
                resolution=resolution,
                range_max=range_max,
            )
        except KeyError:
            continue

        bbox = polygon_bounds(poly)
        area = polygon_area(poly)
        if area <= 0.0 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
            continue

        fine_category_name = normalize_category_name(object_name)
        super_category_name = SUPER_CATEGORY_MAP.get(fine_category_name)
        camera_source_entry = tower_rgb_objects.get(object_name, object_entry)
        camera_supervision = build_camera_supervision(camera_source_entry)
        annotation = {
            "instance_name": object_name,
            "category_name": super_category_name or fine_category_name,
            "fine_category_name": fine_category_name,
            "super_category_name": super_category_name,
            "bbox": bbox,
            "segmentation": [sum(([pt[0], pt[1]] for pt in poly), [])],
            "poly": poly,
            "visible_polygon": visible_polygon,
            "raw_poly": raw_poly,
            "area": area,
            "rbox_xywhr": rbox_xywhr,
            "label_source": label_source,
            **camera_supervision,
        }
        annotations.append(annotation)
        yaml_polygons[object_name] = {"poly": poly}

    return annotations, yaml_polygons


def camera_paths_for_frame(tower_dir: Path, frame_id: str) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for camera_name in CAMERA_NAMES:
        image_path = tower_dir / "cams" / camera_name / "rgb" / f"{frame_id}.png"
        paths[camera_name] = str(image_path)
    return paths


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, content: object) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False)


def write_yaml(path: Path, content: object) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, allow_unicode=False, sort_keys=True)


def write_label_list(path: Path, categories: Sequence[Dict[str, object]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for category in categories:
            handle.write(f"{category['name']}\n")


def build_coco_scaffold(description: str) -> Dict[str, object]:
    return {
        "info": {
            "description": description,
            "version": "1.0",
            "year": 2026,
            "contributor": "Unmanned Ship RTDETR",
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [],
    }


class CategoryRegistry:
    def __init__(self, one_class: bool, class_scheme: str) -> None:
        self.one_class = one_class
        self.class_scheme = class_scheme
        if self.one_class:
            initial_names = ["ship"]
        elif self.class_scheme == "super4":
            initial_names = list(SUPER_CATEGORY_ORDER)
        else:
            initial_names = []
        self.name_to_id: Dict[str, int] = {
            name: idx + 1 for idx, name in enumerate(initial_names)
        }

    def category_name(self, raw_name: str) -> str:
        fine_name = normalize_category_name(raw_name)
        if self.one_class:
            return "ship"
        if self.class_scheme == "super4":
            return map_super_category_name(fine_name)
        return fine_name

    def get_id(self, raw_name: str) -> int:
        cat_name = self.category_name(raw_name)
        if cat_name not in self.name_to_id:
            self.name_to_id[cat_name] = len(self.name_to_id) + 1
        return self.name_to_id[cat_name]

    def export_categories(self) -> List[Dict[str, object]]:
        return [
            {"id": cat_id, "name": name, "supercategory": "ship"}
            for name, cat_id in sorted(self.name_to_id.items(), key=lambda item: item[1])
        ]


def build_sample_record(
    split_name: str,
    scene_name: str,
    tower_name: str,
    frame_id: str,
    dataset_root: Path,
    output_root: Path,
    resolution: int,
    range_max: float,
    ref_range: float,
    clip_percentile: float,
) -> Dict[str, object]:
    scene_dir = dataset_root / split_name / scene_name
    tower_dir = scene_dir / tower_name

    pcd_path = tower_dir / "radar_pcd" / f"radar_{frame_id}.pcd"
    gt_filter_path = scene_dir / "gt_filter" / f"gt_{frame_id}_tosensor_filter.json"
    gt_filter_only_yaw_path = (
        scene_dir / "gt_filter_only_yaw" / f"gt_{frame_id}_tosensor_filter.json"
    )
    gt_sensor_path = scene_dir / "gt_sensor" / f"gt_{frame_id}_tosensor.json"
    gt_world_path = scene_dir / "gt" / f"gt_{frame_id}.json"
    opv2v_yaml_path = scene_dir / "opv2v_yaml" / tower_name / f"{frame_id}.yaml"

    radar_image_out = (
        output_root
        / split_name
        / "radar_bev"
        / scene_name
        / tower_name
        / f"{frame_id}.png"
    )
    ann_json_out = (
        output_root
        / split_name
        / "annotations"
        / scene_name
        / tower_name
        / f"{frame_id}.json"
    )
    ann_yaml_out = (
        output_root
        / split_name
        / "annotations_yaml"
        / scene_name
        / tower_name
        / f"{frame_id}.yaml"
    )

    points = read_pcd_points(pcd_path)
    radar_image = build_radar_image(
        points=points,
        resolution=resolution,
        range_max=range_max,
        ref_range=ref_range,
        clip_percentile=clip_percentile,
    )
    ensure_parent(radar_image_out)
    radar_image.save(radar_image_out)

    annotations, yaml_polygons = build_annotations(
        gt_filter_path=gt_filter_path,
        gt_filter_only_yaw_path=gt_filter_only_yaw_path,
        tower_name=tower_name,
        resolution=resolution,
        range_max=range_max,
    )

    ann_payload = {
        "sample_id": f"{split_name}/{scene_name}/{tower_name}/{frame_id}",
        "split": split_name,
        "scene_id": scene_name,
        "tower_id": tower_name,
        "frame_id": frame_id,
        "resolution": resolution,
        "range_max": range_max,
        "annotation_type": "rotated_box_polygon",
        "annotations": annotations,
    }
    write_json(ann_json_out, ann_payload)
    write_yaml(ann_yaml_out, yaml_polygons)

    return {
        "sample_id": ann_payload["sample_id"],
        "split": split_name,
        "scene_id": scene_name,
        "tower_id": tower_name,
        "frame_id": frame_id,
        "camera_paths": camera_paths_for_frame(tower_dir, frame_id),
        "radar_pcd_path": str(pcd_path),
        "radar_bev_path": str(radar_image_out),
        "annotation_json_path": str(ann_json_out),
        "annotation_yaml_path": str(ann_yaml_out),
        "gt_filter_path": str(gt_filter_path),
        "gt_filter_only_yaw_path": str(gt_filter_only_yaw_path),
        "gt_sensor_path": str(gt_sensor_path),
        "gt_world_path": str(gt_world_path),
        "opv2v_yaml_path": str(opv2v_yaml_path),
    }


def build_sample_record_from_spec(
    spec: Tuple[str, str, str, str, str, str, int, float, float, float]
) -> Dict[str, object]:
    (
        split_name,
        scene_name,
        tower_name,
        frame_id,
        dataset_root,
        output_root,
        resolution,
        range_max,
        ref_range,
        clip_percentile,
    ) = spec
    return build_sample_record(
        split_name=split_name,
        scene_name=scene_name,
        tower_name=tower_name,
        frame_id=frame_id,
        dataset_root=Path(dataset_root),
        output_root=Path(output_root),
        resolution=resolution,
        range_max=range_max,
        ref_range=ref_range,
        clip_percentile=clip_percentile,
    )


def append_coco_sample(
    coco: Dict[str, object],
    sample: Dict[str, object],
    annotation_payload_path: Path,
    category_registry: CategoryRegistry,
    image_id: int,
    ann_id_start: int,
    output_root: Path,
) -> int:
    radar_bev_path = Path(sample["radar_bev_path"])
    file_name = str(radar_bev_path.relative_to(output_root / sample["split"] / "radar_bev"))
    radar_im_file = str(radar_bev_path.relative_to(output_root))
    camera_im_file = [sample["camera_paths"][camera_name] for camera_name in CAMERA_NAMES]
    with annotation_payload_path.open("r", encoding="utf-8") as handle:
        ann_payload = json.load(handle)

    coco["images"].append(
        {
            "id": image_id,
            "width": ann_payload["resolution"],
            "height": ann_payload["resolution"],
            "file_name": file_name,
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "scene_id": sample["scene_id"],
            "tower_id": sample["tower_id"],
            "frame_id": sample["frame_id"],
            "radar_im_file": radar_im_file,
            "camera_im_file": camera_im_file,
            "camera_names": list(CANONICAL_CAMERA_KEYS),
            "license": 1,
        }
    )

    next_ann_id = ann_id_start
    for ann in ann_payload["annotations"]:
        category_id = category_registry.get_id(ann["instance_name"])
        coco["annotations"].append(
            {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": ann["bbox"],
                "segmentation": ann["segmentation"],
                "area": ann["area"],
                "iscrowd": 0,
                "gt_primary_camera": ann["gt_primary_camera"],
                "gt_visible_cameras": ann["gt_visible_cameras"],
                "gt_camera_box_2d": ann["gt_camera_box_2d"],
                "gt_camera_poly_area": ann["gt_camera_poly_area"],
                "gt_has_camera_box": ann["gt_has_camera_box"],
            }
        )
        next_ann_id += 1
    return next_ann_id


def valid_tower_dirs(scene_dir: Path) -> Iterable[Path]:
    for path in sorted(scene_dir.iterdir()):
        if path.is_dir() and path.name.startswith("CoastGuard"):
            yield path


def iter_sample_specs(
    split_name: str, split_dir: Path, dataset_root: Path, output_root: Path
) -> Iterator[Tuple[str, str, str, str, str, str]]:
    for scene_dir in iter_scenes(split_dir, split_name):
        for tower_dir in valid_tower_dirs(scene_dir):
            for frame_number in range(FRAME_COUNT):
                frame_id = f"{frame_number:06d}"
                pcd_path = tower_dir / "radar_pcd" / f"radar_{frame_id}.pcd"
                gt_filter_path = scene_dir / "gt_filter" / f"gt_{frame_id}_tosensor_filter.json"
                gt_filter_only_yaw_path = (
                    scene_dir / "gt_filter_only_yaw" /
                    f"gt_{frame_id}_tosensor_filter.json")
                if not pcd_path.is_file() or not gt_filter_path.is_file() or \
                        not gt_filter_only_yaw_path.is_file():
                    continue
                yield (
                    split_name,
                    scene_dir.name,
                    tower_dir.name,
                    frame_id,
                    str(dataset_root),
                    str(output_root),
                )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    category_registry = CategoryRegistry(one_class=args.one_class, class_scheme=args.class_scheme)
    split_coco: Dict[str, Dict[str, object]] = {
        split_name: build_coco_scaffold(
            f"Sealand single-tower radar dataset ({split_name})"
        )
        for split_name in args.splits
    }
    split_image_ids: Dict[str, int] = {split_name: 0 for split_name in args.splits}
    split_ann_ids: Dict[str, int] = {split_name: 0 for split_name in args.splits}

    for split_name in args.splits:
        split_dir = dataset_root / split_name
        if not split_dir.is_dir():
            print(f"[WARN] Split not found, skipping: {split_dir}")
            continue

        sample_specs: List[Tuple[str, str, str, str, str, str, int, float, float, float]] = []
        for base_spec in iter_sample_specs(split_name, split_dir, dataset_root, output_root):
            if args.limit and total_samples + len(sample_specs) >= args.limit:
                break
            sample_specs.append(
                base_spec
                + (
                    args.resolution,
                    args.range_max,
                    args.ref_range,
                    args.clip_percentile,
                )
            )

        split_index_path = output_root / f"{split_name.lower()}_samples.jsonl"
        ensure_parent(split_index_path)
        with split_index_path.open("w", encoding="utf-8") as split_index:
            if sample_specs:
                max_workers = max(1, args.workers)
                chunksize = max(1, min(32, len(sample_specs) // max_workers // 4 or 1))
                if max_workers == 1:
                    sample_iter = map(build_sample_record_from_spec, sample_specs)
                else:
                    executor = ProcessPoolExecutor(max_workers=max_workers)
                    sample_iter = executor.map(
                        build_sample_record_from_spec, sample_specs, chunksize=chunksize
                    )
                try:
                    for sample in sample_iter:
                        split_index.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        split_ann_ids[split_name] = append_coco_sample(
                            coco=split_coco[split_name],
                            sample=sample,
                            annotation_payload_path=Path(sample["annotation_json_path"]),
                            category_registry=category_registry,
                            image_id=split_image_ids[split_name],
                            ann_id_start=split_ann_ids[split_name],
                            output_root=output_root,
                        )
                        split_image_ids[split_name] += 1
                        total_samples += 1
                        if total_samples % 50 == 0:
                            print(f"[INFO] Prepared {total_samples} samples...")
                finally:
                    if max_workers != 1:
                        executor.shutdown(wait=True)

        print(f"[INFO] Wrote split index: {split_index_path}")
        if args.limit and total_samples >= args.limit:
            print(f"[INFO] Reached limit={args.limit}, stopping sample generation early.")
            break

    categories = category_registry.export_categories()
    for split_name in args.splits:
        if split_name not in split_coco:
            continue
        split_coco[split_name]["categories"] = categories
        coco_path = output_root / f"{split_name.lower()}_coco.json"
        write_json(coco_path, split_coco[split_name])
        print(f"[INFO] Wrote COCO annotation: {coco_path}")

    category_map_path = output_root / "category_mapping.json"
    write_json(
        category_map_path,
        {
            "one_class": args.one_class,
            "class_scheme": args.class_scheme,
            "categories": categories,
        },
    )
    print(f"[INFO] Wrote category mapping: {category_map_path}")
    label_list_path = output_root / "label_list.txt"
    write_label_list(label_list_path, categories)
    print(f"[INFO] Wrote label list: {label_list_path}")

    print(f"[DONE] Prepared {total_samples} samples in {output_root}")


if __name__ == "__main__":
    main()
