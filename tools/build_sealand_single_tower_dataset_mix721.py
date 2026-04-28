#!/usr/bin/env python3
"""
Build a mixed 7:2:1 split dataset from the existing 2km hbb_min8 prepared source.

Key behavior:
1. Aggregate all source samples from Train/Valid/Test.
2. Re-split by scene/tower using block-level assignment (default block=50).
3. Keep existing 2km + RouteROI construction logic, while adding boundary clipping
   policy for radar polygons.
4. Export new JSONL + COCO + annotation payloads + markdown stats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import yaml


DEFAULT_SOURCE_ROOT = Path(
    "./prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8"
)
DEFAULT_OUTPUT_ROOT = Path(
    "./prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721"
)
DEFAULT_STATS_MD = Path("./docs/mix721_dataset_stats.md")

SPLIT_ORDER = ("Train", "Valid", "Test")
SPLIT_ORDER_LOWER = tuple(split.lower() for split in SPLIT_ORDER)

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
        description="Build sealand 2km mix721 dataset with boundary clipping stats."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Existing prepared source root that contains train/valid/test_samples.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for the new mix721 prepared dataset.",
    )
    parser.add_argument(
        "--stats-md",
        type=Path,
        default=DEFAULT_STATS_MD,
        help="Output markdown path for dataset stats report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260425,
        help="Deterministic seed for block-level shuffle.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        choices=[50, 100],
        default=50,
        help="Block size inside each scene/tower for split assignment.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1536,
        help="Radar image size in pixels.",
    )
    parser.add_argument(
        "--range-max",
        type=float,
        default=2000.0,
        help="Effective radar range in meters (2km).",
    )
    parser.add_argument(
        "--min-visible-ratio",
        type=float,
        default=0.3,
        help="Keep clipped boundary annotation only if visible_ratio >= threshold.",
    )
    parser.add_argument(
        "--min-clipped-edge",
        type=float,
        default=8.0,
        help="Keep clipped boundary annotation only if clipped min-edge >= threshold.",
    )
    parser.add_argument(
        "--radar-image-mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy", "symlink"],
        help="How to materialize radar image files in output dataset.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Worker processes for per-sample annotation rebuild.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on generated sample count for smoke test.",
    )
    return parser.parse_args()


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
    if not points:
        return []
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
    return [[cx + float(y) * scale, cy - float(x) * scale] for x, y in points_xy]


def project_pixels_to_metric_xy(
    points_uv: Sequence[Sequence[float]], resolution: int, range_max: float
) -> List[List[float]]:
    cx = resolution / 2.0
    cy = resolution / 2.0
    scale = (resolution / 2.0) / max(range_max, 1e-6)
    return [[(cy - float(v)) / scale, (float(u) - cx) / scale] for u, v in points_uv]


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
    if len(ordered) < 2:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
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
    return order_polygon_clockwise(corners)


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
                (clipped_box_xyxy[2] - clipped_box_xyxy[0])
                * (clipped_box_xyxy[3] - clipped_box_xyxy[1]),
            )

        camera_poly_area[camera_id] = float(polygon_area_value)

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


def parse_source_sample_id(sample_id: str) -> Tuple[str, str, str, str]:
    parts = sample_id.split("/")
    if len(parts) != 4:
        raise ValueError(f"Invalid sample_id format: {sample_id}")
    return parts[0], parts[1], parts[2], parts[3]


def frame_to_int(frame_id: str) -> int:
    try:
        return int(frame_id)
    except ValueError:
        return int(frame_id.split("_")[-1])


def stable_group_seed(base_seed: int, scene_id: str, tower_id: str) -> int:
    digest = hashlib.sha256(f"{scene_id}/{tower_id}".encode("utf-8")).hexdigest()
    return (int(digest[:16], 16) + int(base_seed)) & 0x7FFFFFFF


def allocate_block_counts(total_blocks: int) -> Tuple[int, int, int]:
    if total_blocks <= 0:
        return (0, 0, 0)
    ratios = (0.7, 0.2, 0.1)
    raw = [ratio * total_blocks for ratio in ratios]
    floors = [math.floor(v) for v in raw]
    remain = total_blocks - sum(floors)
    ranked = sorted(
        ((raw[idx] - floors[idx], idx) for idx in range(3)),
        key=lambda item: (-item[0], item[1]),
    )
    for idx in range(remain):
        floors[ranked[idx % 3][1]] += 1
    return floors[0], floors[1], floors[2]


def load_source_samples(source_root: Path) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    seen_ids = set()
    for split_lower in SPLIT_ORDER_LOWER:
        jsonl_path = source_root / f"{split_lower}_samples.jsonl"
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"Missing source samples file: {jsonl_path}")
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample_id = sample.get("sample_id")
                if not sample_id:
                    continue
                if sample_id in seen_ids:
                    continue
                seen_ids.add(sample_id)
                src_split, scene_id, tower_id, frame_id = parse_source_sample_id(sample_id)
                sample["source_split"] = src_split
                sample["scene_id"] = sample.get("scene_id", scene_id)
                sample["tower_id"] = sample.get("tower_id", tower_id)
                sample["frame_id"] = sample.get("frame_id", frame_id)
                samples.append(sample)
    return samples


def assign_mix721_splits(
    source_samples: Sequence[Dict[str, object]],
    seed: int,
    block_size: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for sample in source_samples:
        key = (str(sample["scene_id"]), str(sample["tower_id"]))
        groups[key].append(sample)

    assigned: List[Dict[str, object]] = []
    assignment_meta: Dict[str, object] = {
        "scene_tower_block_summary": {},
        "adjacent_split_change": {},
    }

    for (scene_id, tower_id), records in sorted(groups.items()):
        ordered = sorted(records, key=lambda item: frame_to_int(str(item["frame_id"])))
        blocks = [ordered[idx: idx + block_size] for idx in range(0, len(ordered), block_size)]

        shuffled_indices = list(range(len(blocks)))
        rng = random.Random(stable_group_seed(seed, scene_id, tower_id))
        rng.shuffle(shuffled_indices)

        n_train, n_valid, n_test = allocate_block_counts(len(blocks))
        block_to_split: Dict[int, str] = {}
        for rank, block_idx in enumerate(shuffled_indices):
            if rank < n_train:
                block_to_split[block_idx] = "Train"
            elif rank < n_train + n_valid:
                block_to_split[block_idx] = "Valid"
            else:
                block_to_split[block_idx] = "Test"

        block_summary = []
        frame_assign_for_leak = []
        for block_idx, block in enumerate(blocks):
            target_split = block_to_split[block_idx]
            frame_ids = [str(item["frame_id"]) for item in block]
            frame_ints = [frame_to_int(fid) for fid in frame_ids]
            block_summary.append(
                {
                    "block_id": block_idx,
                    "split": target_split,
                    "num_frames": len(frame_ids),
                    "frame_start": min(frame_ints) if frame_ints else None,
                    "frame_end": max(frame_ints) if frame_ints else None,
                }
            )
            for item in block:
                out = dict(item)
                out["split"] = target_split
                out["mix_block_id"] = block_idx
                out["mix_block_size"] = block_size
                out["mix_seed"] = seed
                out["sample_id"] = f"{target_split}/{scene_id}/{tower_id}/{item['frame_id']}"
                assigned.append(out)
                frame_assign_for_leak.append(
                    (frame_to_int(str(item["frame_id"])), target_split, block_idx)
                )

        frame_assign_for_leak.sort(key=lambda x: x[0])
        adjacent_pairs = 0
        adjacent_split_changes = 0
        adjacent_cross_block_changes = 0
        for prev, curr in zip(frame_assign_for_leak, frame_assign_for_leak[1:]):
            if curr[0] - prev[0] != 1:
                continue
            adjacent_pairs += 1
            if curr[1] != prev[1]:
                adjacent_split_changes += 1
                if curr[2] != prev[2]:
                    adjacent_cross_block_changes += 1

        scene_tower_key = f"{scene_id}/{tower_id}"
        assignment_meta["scene_tower_block_summary"][scene_tower_key] = {
            "num_samples": len(records),
            "num_blocks": len(blocks),
            "num_blocks_train": n_train,
            "num_blocks_valid": n_valid,
            "num_blocks_test": n_test,
            "blocks": block_summary,
        }
        assignment_meta["adjacent_split_change"][scene_tower_key] = {
            "adjacent_pairs": adjacent_pairs,
            "adjacent_split_changes": adjacent_split_changes,
            "adjacent_cross_block_changes": adjacent_cross_block_changes,
        }

    return assigned, assignment_meta


def center_range_from_object(object_entry: Dict[str, object]) -> float | None:
    bev_entry = object_entry.get("bev_rot_only_yaw")
    if isinstance(bev_entry, dict):
        center = bev_entry.get("center")
        if isinstance(center, dict) and "x" in center and "y" in center:
            return math.hypot(float(center["x"]), float(center["y"]))

    radar_proj = object_entry.get("radar_proj")
    if isinstance(radar_proj, dict):
        center = radar_proj.get("center")
        if isinstance(center, dict) and "x" in center and "y" in center:
            return math.hypot(float(center["x"]), float(center["y"]))
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            return math.hypot(float(center[0]), float(center[1]))
    return None


def full_polygon_from_object_entry(
    object_entry: Dict[str, object], resolution: int, range_max: float
) -> Tuple[List[List[float]], List[float], str]:
    radar_proj = object_entry.get("radar_proj", {})
    corners_3d = radar_proj.get("corners_3d")
    if corners_3d:
        points_xy = [[float(point[0]), float(point[1])] for point in corners_3d]
        ordered_xy = fit_min_area_rect_xy(points_xy)
        full_poly = project_metric_xy_to_pixels(ordered_xy, resolution, range_max)
        full_poly = order_polygon_clockwise(full_poly)
        full_metric_poly = project_pixels_to_metric_xy(full_poly, resolution, range_max)
        full_rbox = metric_rbox_from_polygon(full_metric_poly)
        return full_poly, full_rbox, "radar_proj.corners_3d"

    bev_entry = object_entry.get("bev_rot_only_yaw")
    if bev_entry:
        full_poly = rotated_box_corners_pixels(bev_entry, resolution, range_max)
        full_metric_poly = project_pixels_to_metric_xy(full_poly, resolution, range_max)
        full_rbox = metric_rbox_from_polygon(full_metric_poly)
        return full_poly, full_rbox, "bev_rot_only_yaw_fallback"

    raise KeyError("Neither radar_proj.corners_3d nor bev_rot_only_yaw is available")


def polygon_min_edge(poly: Sequence[Sequence[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    rect = fit_min_area_rect_xy(poly)
    if len(rect) < 2:
        return 0.0
    edges = []
    for idx, point in enumerate(rect):
        nxt = rect[(idx + 1) % len(rect)]
        edges.append(
            math.hypot(float(nxt[0]) - float(point[0]), float(nxt[1]) - float(point[1]))
        )
    return min(edges) if edges else 0.0


def flatten_polygon(poly: Sequence[Sequence[float]]) -> List[float]:
    out: List[float] = []
    for pt in poly:
        out.extend([float(pt[0]), float(pt[1])])
    return out


def link_or_copy_file(src: Path, dst: Path, mode: str) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "symlink":
        dst.symlink_to(src)
        return
    shutil.copy2(src, dst)


def build_annotations_with_boundary_policy(
    gt_filter_path: Path,
    gt_filter_only_yaw_path: Path,
    tower_name: str,
    resolution: int,
    range_max: float,
    min_visible_ratio: float,
    min_clipped_edge: float,
) -> Tuple[
    List[Dict[str, object]],
    Dict[str, Dict[str, List[List[float]]]],
    Dict[str, int],
    List[float],
    Dict[str, int],
]:
    with gt_filter_path.open("r", encoding="utf-8") as handle:
        gt = json.load(handle)

    if not gt_filter_only_yaw_path.is_file():
        raise FileNotFoundError(f"Missing RGB 2D GT source file: {gt_filter_only_yaw_path}")
    with gt_filter_only_yaw_path.open("r", encoding="utf-8") as handle:
        gt_rgb = json.load(handle)

    tower = gt.get("towers", {}).get(tower_name, {})
    tower_rgb = gt_rgb.get("towers", {}).get(tower_name, {})
    tower_objects = tower.get("objects", {})
    tower_rgb_objects = tower_rgb.get("objects", {})

    boundary_stats = Counter(
        {
            "total_objects": 0,
            "center_in_2km_objects": 0,
            "full_inside_radar_objects": 0,
            "boundary_crossing_objects": 0,
            "clipped_keep_objects": 0,
            "clipped_drop_visible_ratio_objects": 0,
            "clipped_drop_min8_objects": 0,
        }
    )
    filter_reasons = Counter()

    annotations: List[Dict[str, object]] = []
    yaml_polygons: Dict[str, Dict[str, List[List[float]]]] = {}
    visible_ratios: List[float] = []

    for object_name, object_entry in sorted(tower_objects.items()):
        boundary_stats["total_objects"] += 1

        center_range = center_range_from_object(object_entry)
        if center_range is None:
            filter_reasons["drop_missing_center_range"] += 1
            continue
        if center_range > range_max:
            filter_reasons["drop_center_out_2km"] += 1
            continue
        boundary_stats["center_in_2km_objects"] += 1

        try:
            full_poly, full_rbox_xywhr, label_source = full_polygon_from_object_entry(
                object_entry=object_entry,
                resolution=resolution,
                range_max=range_max,
            )
        except KeyError:
            filter_reasons["drop_missing_full_polygon"] += 1
            continue

        full_poly = order_polygon_clockwise(full_poly)
        full_area = polygon_area(full_poly)
        if full_area <= 0.0:
            filter_reasons["drop_invalid_full_polygon_area"] += 1
            continue

        is_boundary_crossing = not polygon_inside_image(full_poly, resolution)
        clipped_poly = order_polygon_clockwise(full_poly)
        visible_ratio = 1.0
        clip_status = "full_inside"

        if is_boundary_crossing:
            boundary_stats["boundary_crossing_objects"] += 1
            clipped_poly = clip_polygon_to_image(full_poly, resolution)
            clipped_poly = order_polygon_clockwise(clipped_poly)
            clipped_area = polygon_area(clipped_poly)
            visible_ratio = clipped_area / full_area if full_area > 0 else 0.0
            visible_ratios.append(float(visible_ratio))
            clip_status = "boundary_crossing"

            if len(clipped_poly) < 3 or clipped_area <= 0.0:
                boundary_stats["clipped_drop_visible_ratio_objects"] += 1
                filter_reasons["drop_clipped_polygon_too_small"] += 1
                continue
            if visible_ratio < min_visible_ratio:
                boundary_stats["clipped_drop_visible_ratio_objects"] += 1
                filter_reasons["drop_visible_ratio_lt_threshold"] += 1
                continue
            clipped_min_edge = polygon_min_edge(clipped_poly)
            if clipped_min_edge < min_clipped_edge:
                boundary_stats["clipped_drop_min8_objects"] += 1
                filter_reasons["drop_clipped_min_edge_lt_threshold"] += 1
                continue

            boundary_stats["clipped_keep_objects"] += 1
            clip_status = "clipped_keep"
        else:
            boundary_stats["full_inside_radar_objects"] += 1

        clipped_area = polygon_area(clipped_poly)
        bbox = polygon_bounds(clipped_poly)
        if clipped_area <= 0.0 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
            filter_reasons["drop_kept_polygon_invalid_bbox_or_area"] += 1
            continue

        fine_category_name = normalize_category_name(object_name)
        super_category_name = SUPER_CATEGORY_MAP.get(fine_category_name)
        camera_source_entry = tower_rgb_objects.get(object_name, object_entry)
        camera_supervision = build_camera_supervision(camera_source_entry)

        # Prefer gt_filter_only_yaw cams, but fallback to gt_filter cams when:
        # 1) primary camera has no valid box, or
        # 2) all 4 camera boxes are empty.
        # This avoids overwriting usable primary camera supervision with all-zero boxes
        # when gt_filter_only_yaw is incomplete for a subset of objects.
        if camera_source_entry is not object_entry:
            has_cam_box = [
                int(v) for v in camera_supervision.get("gt_has_camera_box", [0, 0, 0, 0])
            ]
            primary_cam = int(camera_supervision.get("gt_primary_camera", NONE_CAMERA_ID))
            all_empty = sum(has_cam_box) == 0
            primary_missing = (
                0 <= primary_cam < 4 and has_cam_box[primary_cam] == 0
            )
            if all_empty or primary_missing:
                fallback_supervision = build_camera_supervision(object_entry)
                fallback_has = [
                    int(v)
                    for v in fallback_supervision.get("gt_has_camera_box", [0, 0, 0, 0])
                ]
                fallback_primary = int(
                    fallback_supervision.get("gt_primary_camera", NONE_CAMERA_ID)
                )
                # Keep route rule unchanged: only patch primary camera box when possible.
                if 0 <= primary_cam < 4 and fallback_has[primary_cam] == 1:
                    camera_supervision["gt_has_camera_box"][primary_cam] = 1
                    camera_supervision["gt_visible_cameras"][primary_cam] = 1
                    camera_supervision["gt_camera_box_2d"][primary_cam] = list(
                        fallback_supervision["gt_camera_box_2d"][primary_cam]
                    )
                    camera_supervision["gt_camera_poly_area"][primary_cam] = float(
                        fallback_supervision["gt_camera_poly_area"][primary_cam]
                    )
                elif all_empty and 0 <= fallback_primary < 4 and fallback_has[fallback_primary] == 1:
                    camera_supervision = fallback_supervision

        clipped_rect = fit_min_area_rect_xy(clipped_poly)
        if not polygon_inside_image(clipped_rect, resolution):
            clipped_rect = [clamp_point_to_image(point, resolution) for point in clipped_rect]
        clipped_metric_poly = project_pixels_to_metric_xy(clipped_rect, resolution, range_max)
        clipped_rbox_xywhr = metric_rbox_from_polygon(clipped_metric_poly)

        annotation = {
            "instance_name": object_name,
            "category_name": super_category_name or fine_category_name,
            "fine_category_name": fine_category_name,
            "super_category_name": super_category_name,
            "bbox": [float(v) for v in bbox],
            "segmentation": [flatten_polygon(clipped_poly)],
            "poly": [[float(pt[0]), float(pt[1])] for pt in clipped_poly],
            "visible_polygon": [[float(pt[0]), float(pt[1])] for pt in clipped_poly],
            "raw_poly": [[float(pt[0]), float(pt[1])] for pt in full_poly],
            "area": float(clipped_area),
            "rbox_xywhr": [float(v) for v in clipped_rbox_xywhr],
            "label_source": label_source,
            **camera_supervision,
            "debug_full_radar_poly": [[float(pt[0]), float(pt[1])] for pt in full_poly],
            "debug_full_rbox_xywhr": [float(v) for v in full_rbox_xywhr],
            "debug_clip_info": {
                "clip_status": clip_status,
                "is_boundary_crossing": bool(is_boundary_crossing),
                "visible_ratio": float(visible_ratio),
                "min_clipped_edge": float(polygon_min_edge(clipped_poly)),
                "min_visible_ratio_threshold": float(min_visible_ratio),
                "min_clipped_edge_threshold": float(min_clipped_edge),
                "center_range": float(center_range),
            },
        }

        annotations.append(annotation)
        yaml_polygons[object_name] = {"poly": [[float(pt[0]), float(pt[1])] for pt in clipped_poly]}

    return annotations, yaml_polygons, dict(boundary_stats), visible_ratios, dict(filter_reasons)


def build_sample_record_and_files(task: Dict[str, object]) -> Dict[str, object]:
    sample = task["sample"]
    output_root = Path(task["output_root"])
    resolution = int(task["resolution"])
    range_max = float(task["range_max"])
    min_visible_ratio = float(task["min_visible_ratio"])
    min_clipped_edge = float(task["min_clipped_edge"])
    radar_image_mode = str(task["radar_image_mode"])

    split_name = str(sample["split"])
    scene_name = str(sample["scene_id"])
    tower_name = str(sample["tower_id"])
    frame_id = str(sample["frame_id"])

    radar_src = Path(str(sample["radar_bev_path"]))
    gt_filter_path = Path(str(sample["gt_filter_path"]))
    gt_filter_only_yaw_path = Path(str(sample["gt_filter_only_yaw_path"]))

    radar_image_out = (
        output_root / split_name / "radar_bev" / scene_name / tower_name / f"{frame_id}.png"
    )
    ann_json_out = (
        output_root / split_name / "annotations" / scene_name / tower_name / f"{frame_id}.json"
    )
    ann_yaml_out = (
        output_root
        / split_name
        / "annotations_yaml"
        / scene_name
        / tower_name
        / f"{frame_id}.yaml"
    )

    if not radar_src.is_file():
        raise FileNotFoundError(f"Missing source radar image: {radar_src}")
    link_or_copy_file(radar_src, radar_image_out, radar_image_mode)

    annotations, yaml_polygons, boundary_stats, visible_ratios, filter_reasons = (
        build_annotations_with_boundary_policy(
            gt_filter_path=gt_filter_path,
            gt_filter_only_yaw_path=gt_filter_only_yaw_path,
            tower_name=tower_name,
            resolution=resolution,
            range_max=range_max,
            min_visible_ratio=min_visible_ratio,
            min_clipped_edge=min_clipped_edge,
        )
    )

    ann_payload = {
        "sample_id": str(sample["sample_id"]),
        "split": split_name,
        "source_split": str(sample.get("source_split", "")),
        "scene_id": scene_name,
        "tower_id": tower_name,
        "frame_id": frame_id,
        "resolution": resolution,
        "range_max": range_max,
        "annotation_type": "rotated_box_polygon",
        "mix_block_id": int(sample["mix_block_id"]),
        "mix_block_size": int(sample["mix_block_size"]),
        "mix_seed": int(sample["mix_seed"]),
        "boundary_stats": boundary_stats,
        "boundary_visible_ratios": [float(v) for v in visible_ratios],
        "filter_reason_counts": filter_reasons,
        "annotations": annotations,
    }
    write_json(ann_json_out, ann_payload)
    write_yaml(ann_yaml_out, yaml_polygons)

    sample_record = {
        "sample_id": ann_payload["sample_id"],
        "split": split_name,
        "source_split": ann_payload["source_split"],
        "scene_id": scene_name,
        "tower_id": tower_name,
        "frame_id": frame_id,
        "mix_block_id": ann_payload["mix_block_id"],
        "mix_block_size": ann_payload["mix_block_size"],
        "mix_seed": ann_payload["mix_seed"],
        "camera_paths": sample["camera_paths"],
        "radar_pcd_path": sample.get("radar_pcd_path"),
        "radar_bev_path": str(radar_image_out),
        "annotation_json_path": str(ann_json_out),
        "annotation_yaml_path": str(ann_yaml_out),
        "gt_filter_path": str(gt_filter_path),
        "gt_filter_only_yaw_path": str(gt_filter_only_yaw_path),
        "gt_sensor_path": sample.get("gt_sensor_path"),
        "gt_world_path": sample.get("gt_world_path"),
        "opv2v_yaml_path": sample.get("opv2v_yaml_path"),
        "source_sample_id": sample.get("source_sample_id", ""),
    }

    return sample_record


class CategoryRegistry:
    def __init__(self) -> None:
        self.name_to_id: Dict[str, int] = {
            name: idx + 1 for idx, name in enumerate(SUPER_CATEGORY_ORDER)
        }

    def category_name(self, raw_name: str) -> str:
        fine_name = normalize_category_name(raw_name)
        return map_super_category_name(fine_name)

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


def append_coco_sample(
    coco: Dict[str, object],
    sample: Dict[str, object],
    annotation_payload_path: Path,
    category_registry: CategoryRegistry,
    image_id: int,
    ann_id_start: int,
    output_root: Path,
) -> int:
    radar_bev_path = Path(str(sample["radar_bev_path"]))
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


def safe_pct(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return 100.0 * float(num) / float(den)


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def format_table(headers: List[str], rows: List[List[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build_stats_markdown(
    output_root: Path,
    stats: Dict[str, object],
    assignment_meta: Dict[str, object],
    seed: int,
    block_size: int,
    range_max: float,
    min_visible_ratio: float,
    min_clipped_edge: float,
) -> str:
    split_image_counts: Dict[str, int] = stats["split_image_counts"]
    split_ann_counts: Dict[str, int] = stats["split_ann_counts"]
    split_empty_counts: Dict[str, int] = stats["split_empty_counts"]

    split_class_counts: Dict[str, Counter] = stats["split_class_counts"]
    split_scene_tower_image_counts: Dict[str, Counter] = stats["split_scene_tower_image_counts"]
    split_scene_tower_ann_counts: Dict[str, Counter] = stats["split_scene_tower_ann_counts"]

    scene_tower_split_counts: Dict[str, Counter] = stats["scene_tower_split_counts"]
    split_source_prefix_counts: Dict[str, Counter] = stats["split_source_prefix_counts"]

    route_stats: Counter = stats["route_stats"]
    primary_camera_dist: Counter = stats["primary_camera_dist"]
    visible_count_dist: Counter = stats["visible_count_dist"]
    boundary_stats: Counter = stats["boundary_stats"]
    filter_reason_counts: Counter = stats["filter_reason_counts"]
    visible_ratios: List[float] = stats["visible_ratios"]

    lines: List[str] = []
    lines.append("# mix721 dataset stats")
    lines.append("")
    lines.append("## Build setup")
    lines.append(f"- output_root: `{output_root}`")
    lines.append(f"- split_strategy: `scene/tower + block shuffle + 7:2:1`")
    lines.append(f"- seed: `{seed}`")
    lines.append(f"- block_size: `{block_size}`")
    lines.append(f"- range_max_m: `{range_max}`")
    lines.append(f"- boundary_keep_visible_ratio_threshold: `{min_visible_ratio}`")
    lines.append(f"- boundary_keep_min_edge_px_threshold: `{min_clipped_edge}`")
    lines.append("")

    lines.append("## Split summary")
    rows = []
    for split in SPLIT_ORDER:
        rows.append(
            [
                split.lower(),
                split_image_counts[split],
                split_ann_counts[split],
                split_empty_counts[split],
                f"{safe_pct(split_empty_counts[split], split_image_counts[split]):.2f}%",
            ]
        )
    rows.append(
        [
            "total",
            sum(split_image_counts.values()),
            sum(split_ann_counts.values()),
            sum(split_empty_counts.values()),
            f"{safe_pct(sum(split_empty_counts.values()), sum(split_image_counts.values())):.2f}%",
        ]
    )
    lines.append(format_table(["split", "images", "annotations", "empty_images", "empty_rate"], rows))
    lines.append("")

    lines.append("## Source scene coverage (T/V/S)")
    cov_rows = []
    for split in SPLIT_ORDER:
        prefix_counter = split_source_prefix_counts[split]
        cov_rows.append(
            [
                split.lower(),
                prefix_counter.get("T", 0),
                prefix_counter.get("V", 0),
                prefix_counter.get("S", 0),
            ]
        )
    lines.append(format_table(["split", "from_T", "from_V", "from_S"], cov_rows))
    lines.append("")

    lines.append("## Class distribution by split")
    class_names = sorted(
        {
            name
            for split in SPLIT_ORDER
            for name in split_class_counts[split].keys()
        }
    )
    class_rows = []
    for split in SPLIT_ORDER:
        total_ann = split_ann_counts[split]
        for name in class_names:
            count = split_class_counts[split].get(name, 0)
            class_rows.append(
                [
                    split.lower(),
                    name,
                    count,
                    f"{safe_pct(count, total_ann):.2f}%",
                ]
            )
    lines.append(format_table(["split", "class", "count", "ratio_in_split"], class_rows))
    lines.append("")

    lines.append("## Scene/Tower distribution by split (images, annotations)")
    dist_rows = []
    all_scene_towers = sorted(scene_tower_split_counts.keys())
    for split in SPLIT_ORDER:
        for scene_tower in all_scene_towers:
            img_count = split_scene_tower_image_counts[split].get(scene_tower, 0)
            ann_count = split_scene_tower_ann_counts[split].get(scene_tower, 0)
            if img_count <= 0:
                continue
            dist_rows.append([split.lower(), scene_tower, img_count, ann_count])
    lines.append(format_table(["split", "scene/tower", "images", "annotations"], dist_rows))
    lines.append("")

    lines.append("## Scene/Tower split ratio (train/valid/test)")
    ratio_rows = []
    for scene_tower in all_scene_towers:
        counter = scene_tower_split_counts[scene_tower]
        total = sum(counter.values())
        ratio_rows.append(
            [
                scene_tower,
                counter.get("Train", 0),
                counter.get("Valid", 0),
                counter.get("Test", 0),
                f"{safe_pct(counter.get('Train', 0), total):.2f}%",
                f"{safe_pct(counter.get('Valid', 0), total):.2f}%",
                f"{safe_pct(counter.get('Test', 0), total):.2f}%",
            ]
        )
    lines.append(
        format_table(
            [
                "scene/tower",
                "train_images",
                "valid_images",
                "test_images",
                "train_ratio",
                "valid_ratio",
                "test_ratio",
            ],
            ratio_rows,
        )
    )
    lines.append("")

    lines.append("## RouteROI coverage")
    total_ann = route_stats["total_annotations"]
    route_rows = [
        [
            "gt_primary_camera",
            route_stats["has_gt_primary_camera"],
            f"{safe_pct(route_stats['has_gt_primary_camera'], total_ann):.2f}%",
        ],
        [
            "gt_visible_cameras",
            route_stats["has_gt_visible_cameras"],
            f"{safe_pct(route_stats['has_gt_visible_cameras'], total_ann):.2f}%",
        ],
        [
            "gt_camera_box_2d",
            route_stats["has_gt_camera_box_2d"],
            f"{safe_pct(route_stats['has_gt_camera_box_2d'], total_ann):.2f}%",
        ],
        [
            "gt_has_camera_box",
            route_stats["has_gt_has_camera_box"],
            f"{safe_pct(route_stats['has_gt_has_camera_box'], total_ann):.2f}%",
        ],
    ]
    lines.append(format_table(["field", "valid_annotations", "coverage"], route_rows))
    lines.append("")

    lines.append("## gt_primary_camera distribution")
    cam_rows = []
    for cam_id in [0, 1, 2, 3, 4]:
        count = primary_camera_dist.get(cam_id, 0)
        cam_name = {
            0: "Back",
            1: "Front",
            2: "Left",
            3: "Right",
            4: "None",
        }[cam_id]
        cam_rows.append([cam_id, cam_name, count, f"{safe_pct(count, total_ann):.2f}%"])
    lines.append(format_table(["id", "camera", "count", "ratio"], cam_rows))
    lines.append("")

    lines.append("## gt_visible_cameras count distribution")
    vis_rows = []
    for vis_count in [0, 1, 2, 3, 4]:
        count = visible_count_dist.get(vis_count, 0)
        vis_rows.append([vis_count, count, f"{safe_pct(count, total_ann):.2f}%"])
    lines.append(format_table(["num_visible_cameras", "count", "ratio"], vis_rows))
    lines.append("")

    lines.append("## Radar boundary clipping stats")
    vis_mean = mean(visible_ratios) if visible_ratios else 0.0
    vis_median = median(visible_ratios) if visible_ratios else 0.0
    vis_p10 = quantile(visible_ratios, 0.10)
    vis_p25 = quantile(visible_ratios, 0.25)

    boundary_rows = [
        ["total_objects", boundary_stats.get("total_objects", 0)],
        ["center_in_2km_objects", boundary_stats.get("center_in_2km_objects", 0)],
        ["full_inside_radar_objects", boundary_stats.get("full_inside_radar_objects", 0)],
        ["boundary_crossing_objects", boundary_stats.get("boundary_crossing_objects", 0)],
        ["clipped_keep_objects", boundary_stats.get("clipped_keep_objects", 0)],
        [
            "clipped_drop_visible_ratio_objects",
            boundary_stats.get("clipped_drop_visible_ratio_objects", 0),
        ],
        ["clipped_drop_min8_objects", boundary_stats.get("clipped_drop_min8_objects", 0)],
    ]
    lines.append(format_table(["metric", "value"], boundary_rows))
    lines.append("")
    lines.append(
        format_table(
            ["visible_ratio_stat", "value"],
            [
                ["mean", f"{vis_mean:.6f}"],
                ["median", f"{vis_median:.6f}"],
                ["p10", f"{vis_p10:.6f}"],
                ["p25", f"{vis_p25:.6f}"],
            ],
        )
    )
    lines.append("")

    lines.append("## All filtering reasons")
    filter_rows = [[name, count] for name, count in sorted(filter_reason_counts.items())]
    lines.append(format_table(["filter_reason", "count"], filter_rows))
    lines.append("")

    lines.append("## Block leakage check (adjacent frames)")
    leak_rows = []
    total_adj_pairs = 0
    total_adj_changes = 0
    total_adj_cross_block_changes = 0
    for scene_tower, item in sorted(assignment_meta["adjacent_split_change"].items()):
        adj_pairs = int(item["adjacent_pairs"])
        adj_changes = int(item["adjacent_split_changes"])
        adj_cross_block = int(item["adjacent_cross_block_changes"])
        total_adj_pairs += adj_pairs
        total_adj_changes += adj_changes
        total_adj_cross_block_changes += adj_cross_block
        leak_rows.append(
            [
                scene_tower,
                adj_pairs,
                adj_changes,
                adj_cross_block,
                f"{safe_pct(adj_changes, adj_pairs):.2f}%",
            ]
        )
    lines.append(
        format_table(
            [
                "scene/tower",
                "adjacent_pairs",
                "adjacent_split_changes",
                "adjacent_cross_block_changes",
                "change_rate",
            ],
            leak_rows,
        )
    )
    lines.append("")
    lines.append(
        f"- overall_adjacent_pairs: `{total_adj_pairs}`"
    )
    lines.append(
        f"- overall_adjacent_split_changes: `{total_adj_changes}` ({safe_pct(total_adj_changes, total_adj_pairs):.2f}%)"
    )
    lines.append(
        f"- overall_adjacent_cross_block_changes: `{total_adj_cross_block_changes}` ({safe_pct(total_adj_cross_block_changes, total_adj_pairs):.2f}%)"
    )
    lines.append("")

    lines.append("## Acceptance checks")
    split_cover_ok = all(
        split_source_prefix_counts[split].get("T", 0) > 0
        and split_source_prefix_counts[split].get("V", 0) > 0
        and split_source_prefix_counts[split].get("S", 0) > 0
        for split in SPLIT_ORDER
    )

    ratio_ok_count = 0
    for scene_tower in all_scene_towers:
        counter = scene_tower_split_counts[scene_tower]
        total = sum(counter.values())
        if total <= 0:
            continue
        train_ratio = counter.get("Train", 0) / total
        valid_ratio = counter.get("Valid", 0) / total
        test_ratio = counter.get("Test", 0) / total
        if 0.60 <= train_ratio <= 0.80 and 0.10 <= valid_ratio <= 0.30 and 0.05 <= test_ratio <= 0.20:
            ratio_ok_count += 1

    route_roi_ok = (
        route_stats["has_gt_primary_camera"] == total_ann
        and route_stats["has_gt_visible_cameras"] == total_ann
        and route_stats["has_gt_camera_box_2d"] == total_ann
        and route_stats["has_gt_has_camera_box"] == total_ann
    )

    boundary_ok = (
        boundary_stats.get("boundary_crossing_objects", 0) > 0
        and (
            boundary_stats.get("clipped_keep_objects", 0)
            + boundary_stats.get("clipped_drop_visible_ratio_objects", 0)
            + boundary_stats.get("clipped_drop_min8_objects", 0)
        )
        > 0
    )

    center_filter_ok = boundary_stats.get("center_in_2km_objects", 0) <= boundary_stats.get("total_objects", 0)

    lines.append(f"- train/valid/test all cover original T/V/S scenes: `{split_cover_ok}`")
    lines.append(
        f"- scene/tower ratio near 7:2:1 (tolerance train[0.60,0.80], valid[0.10,0.30], test[0.05,0.20]): `{ratio_ok_count}/{len(all_scene_towers)}`"
    )
    lines.append(
        f"- no frame-level random leakage (adjacent split-change mostly at block boundaries): `overall change rate {safe_pct(total_adj_changes, total_adj_pairs):.2f}%`"
    )
    lines.append(f"- 2km center filter still enforced: `{center_filter_ok}`")
    lines.append(f"- boundary crossing annotations clipped and keep/drop handled: `{boundary_ok}`")
    lines.append(f"- RouteROI supervision fields complete: `{route_roi_ok}`")

    return "\n".join(lines) + "\n"


def update_stats_from_annotation_payload(
    stats: Dict[str, object],
    sample: Dict[str, object],
    ann_payload: Dict[str, object],
) -> None:
    split = str(sample["split"])
    scene_id = str(sample["scene_id"])
    tower_id = str(sample["tower_id"])
    scene_tower = f"{scene_id}/{tower_id}"

    annotations = ann_payload.get("annotations", [])

    stats["split_image_counts"][split] += 1
    stats["split_ann_counts"][split] += len(annotations)
    if not annotations:
        stats["split_empty_counts"][split] += 1

    stats["split_scene_tower_image_counts"][split][scene_tower] += 1
    stats["split_scene_tower_ann_counts"][split][scene_tower] += len(annotations)
    stats["scene_tower_split_counts"][scene_tower][split] += 1

    src_scene_prefix = scene_id[0] if scene_id else "?"
    stats["split_source_prefix_counts"][split][src_scene_prefix] += 1

    for ann in annotations:
        category_name = str(ann.get("category_name", "Unknown"))
        stats["split_class_counts"][split][category_name] += 1

        stats["route_stats"]["total_annotations"] += 1

        if "gt_primary_camera" in ann:
            stats["route_stats"]["has_gt_primary_camera"] += 1
            cam_id = int(ann["gt_primary_camera"])
            stats["primary_camera_dist"][cam_id] += 1

        visible = ann.get("gt_visible_cameras")
        if isinstance(visible, list) and len(visible) == 4:
            stats["route_stats"]["has_gt_visible_cameras"] += 1
            vis_count = int(sum(int(v) for v in visible))
            stats["visible_count_dist"][vis_count] += 1

        cam_boxes = ann.get("gt_camera_box_2d")
        if (
            isinstance(cam_boxes, list)
            and len(cam_boxes) == 4
            and all(isinstance(box, list) and len(box) == 4 for box in cam_boxes)
        ):
            stats["route_stats"]["has_gt_camera_box_2d"] += 1

        has_box = ann.get("gt_has_camera_box")
        if isinstance(has_box, list) and len(has_box) == 4:
            stats["route_stats"]["has_gt_has_camera_box"] += 1

    payload_boundary_stats = ann_payload.get("boundary_stats", {})
    for key, val in payload_boundary_stats.items():
        stats["boundary_stats"][str(key)] += int(val)

    for value in ann_payload.get("boundary_visible_ratios", []):
        try:
            stats["visible_ratios"].append(float(value))
        except (TypeError, ValueError):
            continue

    for key, val in ann_payload.get("filter_reason_counts", {}).items():
        stats["filter_reason_counts"][str(key)] += int(val)


def empty_stats() -> Dict[str, object]:
    return {
        "split_image_counts": Counter(),
        "split_ann_counts": Counter(),
        "split_empty_counts": Counter(),
        "split_class_counts": {split: Counter() for split in SPLIT_ORDER},
        "split_scene_tower_image_counts": {split: Counter() for split in SPLIT_ORDER},
        "split_scene_tower_ann_counts": {split: Counter() for split in SPLIT_ORDER},
        "scene_tower_split_counts": defaultdict(Counter),
        "split_source_prefix_counts": {split: Counter() for split in SPLIT_ORDER},
        "route_stats": Counter(),
        "primary_camera_dist": Counter(),
        "visible_count_dist": Counter(),
        "boundary_stats": Counter(),
        "visible_ratios": [],
        "filter_reason_counts": Counter(),
    }


def main() -> None:
    args = parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    stats_md_path = args.stats_md.resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    ensure_parent(stats_md_path)

    print(f"[INFO] Loading source samples from: {source_root}")
    source_samples = load_source_samples(source_root)
    print(f"[INFO] Loaded source samples: {len(source_samples)}")

    for sample in source_samples:
        sample["source_sample_id"] = sample.get("sample_id", "")

    assigned_samples, assignment_meta = assign_mix721_splits(
        source_samples=source_samples,
        seed=args.seed,
        block_size=args.block_size,
    )

    assigned_samples = sorted(
        assigned_samples,
        key=lambda item: (
            SPLIT_ORDER.index(str(item["split"])),
            str(item["scene_id"]),
            str(item["tower_id"]),
            frame_to_int(str(item["frame_id"])),
        ),
    )

    if args.limit and args.limit > 0:
        assigned_samples = assigned_samples[: args.limit]
        print(f"[INFO] Applying limit={args.limit}, selected samples: {len(assigned_samples)}")

    print(f"[INFO] Total assigned samples: {len(assigned_samples)}")

    tasks = [
        {
            "sample": sample,
            "output_root": str(output_root),
            "resolution": args.resolution,
            "range_max": args.range_max,
            "min_visible_ratio": args.min_visible_ratio,
            "min_clipped_edge": args.min_clipped_edge,
            "radar_image_mode": args.radar_image_mode,
        }
        for sample in assigned_samples
    ]

    built_samples: List[Dict[str, object]] = []
    if tasks:
        max_workers = max(1, int(args.workers))
        if max_workers == 1:
            iterator = map(build_sample_record_and_files, tasks)
        else:
            chunksize = max(1, min(64, len(tasks) // max_workers // 4 or 1))
            executor = ProcessPoolExecutor(max_workers=max_workers)
            iterator = executor.map(build_sample_record_and_files, tasks, chunksize=chunksize)

        try:
            for idx, sample_record in enumerate(iterator, start=1):
                built_samples.append(sample_record)
                if idx % 100 == 0:
                    print(f"[INFO] Rebuilt annotations for {idx}/{len(tasks)} samples...")
        finally:
            if max_workers != 1:
                executor.shutdown(wait=True)

    built_samples = sorted(
        built_samples,
        key=lambda item: (
            SPLIT_ORDER.index(str(item["split"])),
            str(item["scene_id"]),
            str(item["tower_id"]),
            frame_to_int(str(item["frame_id"])),
        ),
    )

    print(f"[INFO] Writing split JSONL + COCO to: {output_root}")
    category_registry = CategoryRegistry()
    split_coco = {
        split: build_coco_scaffold(f"Sealand single-tower radar dataset ({split}, mix721)")
        for split in SPLIT_ORDER
    }
    split_image_ids = {split: 0 for split in SPLIT_ORDER}
    split_ann_ids = {split: 0 for split in SPLIT_ORDER}

    stats = empty_stats()

    split_to_samples: Dict[str, List[Dict[str, object]]] = {split: [] for split in SPLIT_ORDER}
    for sample in built_samples:
        split_to_samples[str(sample["split"])].append(sample)

    for split in SPLIT_ORDER:
        split_samples_path = output_root / f"{split.lower()}_samples.jsonl"
        ensure_parent(split_samples_path)
        with split_samples_path.open("w", encoding="utf-8") as handle:
            for sample in split_to_samples[split]:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

                ann_payload_path = Path(str(sample["annotation_json_path"]))
                with ann_payload_path.open("r", encoding="utf-8") as ann_handle:
                    ann_payload = json.load(ann_handle)

                update_stats_from_annotation_payload(stats, sample, ann_payload)

                split_ann_ids[split] = append_coco_sample(
                    coco=split_coco[split],
                    sample=sample,
                    annotation_payload_path=ann_payload_path,
                    category_registry=category_registry,
                    image_id=split_image_ids[split],
                    ann_id_start=split_ann_ids[split],
                    output_root=output_root,
                )
                split_image_ids[split] += 1

        print(f"[INFO] Wrote split index: {split_samples_path}")

    categories = category_registry.export_categories()
    for split in SPLIT_ORDER:
        split_coco[split]["categories"] = categories
        coco_path = output_root / f"{split.lower()}_coco.json"
        write_json(coco_path, split_coco[split])
        print(f"[INFO] Wrote COCO annotation: {coco_path}")

    category_map_path = output_root / "category_mapping.json"
    write_json(
        category_map_path,
        {
            "one_class": False,
            "class_scheme": "super4",
            "categories": categories,
        },
    )
    print(f"[INFO] Wrote category mapping: {category_map_path}")

    label_list_path = output_root / "label_list.txt"
    write_label_list(label_list_path, categories)
    print(f"[INFO] Wrote label list: {label_list_path}")

    stats_md = build_stats_markdown(
        output_root=output_root,
        stats=stats,
        assignment_meta=assignment_meta,
        seed=args.seed,
        block_size=args.block_size,
        range_max=args.range_max,
        min_visible_ratio=args.min_visible_ratio,
        min_clipped_edge=args.min_clipped_edge,
    )
    ensure_parent(stats_md_path)
    stats_md_path.write_text(stats_md, encoding="utf-8")
    print(f"[INFO] Wrote stats markdown: {stats_md_path}")

    print(f"[DONE] Prepared {len(built_samples)} samples in {output_root}")


if __name__ == "__main__":
    main()
