#!/usr/bin/env python3
"""
Sanity visualization for mix721 dataset samples.

For each split (train/valid/test), randomly sample N frames and export debug images
that include:
- Radar image with clipped radar polygon + fitted rbox
- Primary RGB camera with gt_camera_box_2d overlays
- Text info: split/scene/tower/frame + filtering/clip stats
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET_ROOT = Path(
    "./prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721"
)
DEFAULT_OUTPUT_ROOT = Path("./output/mix721_dataset_debug")

CAMERA_CANONICAL_ORDER = (
    ("Back", "CamBack"),
    ("Front", "CamFront"),
    ("Left", "CamLeft"),
    ("Right", "CamRight"),
)
CAMERA_NAMES = tuple(raw_name for _, raw_name in CAMERA_CANONICAL_ORDER)
CANONICAL_CAMERA_KEYS = tuple(camera_key for camera_key, _ in CAMERA_CANONICAL_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual sanity check for mix721 dataset")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="mix721 prepared dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="output dir for debug visualizations",
    )
    parser.add_argument(
        "--per-split",
        type=int,
        default=50,
        help="number of random samples per split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260425,
        help="deterministic random seed",
    )
    return parser.parse_args()


def frame_to_int(frame_id: str) -> int:
    try:
        return int(frame_id)
    except ValueError:
        return int(frame_id.split("_")[-1])


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


def parse_sample_jsonl(path: Path) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    samples.sort(
        key=lambda item: (
            str(item.get("scene_id", "")),
            str(item.get("tower_id", "")),
            frame_to_int(str(item.get("frame_id", "0"))),
        )
    )
    return samples


def choose_primary_camera_for_sample(annotations: List[Dict[str, object]]) -> int:
    votes: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for ann in annotations:
        cam_id = int(ann.get("gt_primary_camera", 4))
        if cam_id in votes:
            votes[cam_id] += 1
    best_cam = max(votes.items(), key=lambda item: (item[1], -item[0]))[0]
    return best_cam


def draw_polygon(
    draw: ImageDraw.ImageDraw,
    poly: Sequence[Sequence[float]],
    offset: Tuple[float, float],
    scale: Tuple[float, float],
    color: Tuple[int, int, int],
    width: int = 2,
) -> None:
    if len(poly) < 2:
        return
    ox, oy = offset
    sx, sy = scale
    points = [(ox + float(pt[0]) * sx, oy + float(pt[1]) * sy) for pt in poly]
    points.append(points[0])
    draw.line(points, fill=color, width=width)


def draw_box_xyxy(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    offset: Tuple[float, float],
    scale: Tuple[float, float],
    color: Tuple[int, int, int],
    width: int = 2,
) -> None:
    ox, oy = offset
    sx, sy = scale
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle(
        [ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy],
        outline=color,
        width=width,
    )


def safe_open_image(path: Path, fallback_size: Tuple[int, int]) -> Image.Image:
    if path.is_file():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", fallback_size, (32, 32, 32))


def render_sample_debug(
    sample: Dict[str, object],
    split: str,
    sample_idx: int,
    output_dir: Path,
) -> None:
    ann_path = Path(str(sample["annotation_json_path"]))
    with ann_path.open("r", encoding="utf-8") as handle:
        ann_payload = json.load(handle)

    annotations: List[Dict[str, object]] = ann_payload.get("annotations", [])
    boundary_stats: Dict[str, int] = ann_payload.get("boundary_stats", {})
    filter_reasons: Dict[str, int] = ann_payload.get("filter_reason_counts", {})

    radar_path = Path(str(sample["radar_bev_path"]))
    radar_img = safe_open_image(radar_path, (1536, 1536))

    primary_cam_id = choose_primary_camera_for_sample(annotations)
    primary_cam_name = CAMERA_NAMES[primary_cam_id]
    camera_paths = sample.get("camera_paths", {})
    primary_cam_path = Path(str(camera_paths.get(primary_cam_name, "")))
    camera_img = safe_open_image(primary_cam_path, (1024, 512))

    radar_target_w, radar_target_h = 1024, 1024
    cam_target_w, cam_target_h = 1024, 512

    radar_vis = radar_img.resize((radar_target_w, radar_target_h), Image.BILINEAR)
    cam_vis = camera_img.resize((cam_target_w, cam_target_h), Image.BILINEAR)

    canvas_w = 2160
    canvas_h = 1240
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    radar_offset = (40.0, 40.0)
    cam_offset = (1096.0, 40.0)
    text_offset = (1096.0, 580.0)

    canvas.paste(radar_vis, (int(radar_offset[0]), int(radar_offset[1])))
    canvas.paste(cam_vis, (int(cam_offset[0]), int(cam_offset[1])))

    radar_scale = (
        radar_target_w / max(radar_img.width, 1),
        radar_target_h / max(radar_img.height, 1),
    )
    cam_scale = (
        cam_target_w / max(camera_img.width, 1),
        cam_target_h / max(camera_img.height, 1),
    )

    for ann in annotations:
        poly = ann.get("poly", [])
        if isinstance(poly, list) and len(poly) >= 3:
            draw_polygon(draw, poly, radar_offset, radar_scale, color=(0, 255, 120), width=2)
            rbox = fit_min_area_rect_xy(poly)
            if len(rbox) >= 4:
                draw_polygon(draw, rbox, radar_offset, radar_scale, color=(255, 220, 0), width=2)

        has_box = ann.get("gt_has_camera_box", [0, 0, 0, 0])
        cam_boxes = ann.get("gt_camera_box_2d", [[0, 0, 0, 0] for _ in range(4)])
        if (
            isinstance(has_box, list)
            and len(has_box) == 4
            and isinstance(cam_boxes, list)
            and len(cam_boxes) == 4
            and int(has_box[primary_cam_id]) == 1
        ):
            box = cam_boxes[primary_cam_id]
            if isinstance(box, list) and len(box) == 4:
                draw_box_xyxy(draw, box, cam_offset, cam_scale, color=(0, 255, 120), width=2)

    draw.rectangle(
        [
            radar_offset[0],
            radar_offset[1],
            radar_offset[0] + radar_target_w,
            radar_offset[1] + radar_target_h,
        ],
        outline=(120, 120, 120),
        width=2,
    )
    draw.rectangle(
        [
            cam_offset[0],
            cam_offset[1],
            cam_offset[0] + cam_target_w,
            cam_offset[1] + cam_target_h,
        ],
        outline=(120, 120, 120),
        width=2,
    )

    header_lines = [
        f"sample: {sample.get('sample_id', '')}",
        f"split={split} scene={sample.get('scene_id', '')} tower={sample.get('tower_id', '')} frame={sample.get('frame_id', '')}",
        f"primary_camera_panel={CANONICAL_CAMERA_KEYS[primary_cam_id]} ({primary_cam_name})",
        f"radar_path={radar_path}",
        f"camera_path={primary_cam_path}",
    ]

    boundary_lines = [
        "boundary stats:",
        f"  total_objects={boundary_stats.get('total_objects', 0)}",
        f"  center_in_2km={boundary_stats.get('center_in_2km_objects', 0)}",
        f"  full_inside={boundary_stats.get('full_inside_radar_objects', 0)}",
        f"  crossing={boundary_stats.get('boundary_crossing_objects', 0)}",
        f"  clipped_keep={boundary_stats.get('clipped_keep_objects', 0)}",
        f"  drop_visible_ratio={boundary_stats.get('clipped_drop_visible_ratio_objects', 0)}",
        f"  drop_min8={boundary_stats.get('clipped_drop_min8_objects', 0)}",
    ]

    filter_lines = ["filter reasons:"]
    for key, value in sorted(filter_reasons.items()):
        filter_lines.append(f"  {key}={value}")
    if len(filter_lines) == 1:
        filter_lines.append("  (none)")

    ann_lines = [
        f"kept_annotations={len(annotations)}",
        "legend: green=clipped poly, yellow=fitted rbox",
    ]

    all_lines = header_lines + [""] + ann_lines + [""] + boundary_lines + [""] + filter_lines

    y = text_offset[1]
    for line in all_lines:
        draw.text((text_offset[0], y), line, fill=(220, 220, 220), font=font)
        y += 18

    split_dir = output_dir / split.lower()
    split_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"{sample_idx:03d}_{sample.get('scene_id', '')}_{sample.get('tower_id', '')}_{sample.get('frame_id', '')}.png"
    )
    out_path = split_dir / file_name
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    split_to_jsonl = {
        "Train": dataset_root / "train_samples.jsonl",
        "Valid": dataset_root / "valid_samples.jsonl",
        "Test": dataset_root / "test_samples.jsonl",
    }

    split_samples: Dict[str, List[Dict[str, object]]] = {}
    for split, path in split_to_jsonl.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing split samples file: {path}")
        split_samples[split] = parse_sample_jsonl(path)

    for split in ("Train", "Valid", "Test"):
        samples = split_samples[split]
        rng = random.Random(args.seed + {"Train": 0, "Valid": 1, "Test": 2}[split])
        n = min(args.per_split, len(samples))
        chosen = rng.sample(samples, n) if n < len(samples) else list(samples)
        chosen.sort(
            key=lambda item: (
                str(item.get("scene_id", "")),
                str(item.get("tower_id", "")),
                frame_to_int(str(item.get("frame_id", "0"))),
            )
        )

        print(f"[INFO] Rendering {len(chosen)} samples for split={split} ...")
        for idx, sample in enumerate(chosen, start=1):
            render_sample_debug(sample=sample, split=split, sample_idx=idx, output_dir=output_root)

    print(f"[DONE] Saved sanity check images to: {output_root}")


if __name__ == "__main__":
    main()
