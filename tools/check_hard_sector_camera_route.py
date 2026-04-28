#!/usr/bin/env python3
"""Hard sector camera route check and debug visualization.

This script evaluates two hard routing strategies against gt_primary_camera:
1) center_angle: by target center azimuth
2) polygon_area: by BEV polygon overlap area with four 90-degree sectors

It saves:
- Markdown report: docs/hard_sector_route_check.md
- Debug images: output/hard_sector_route_debug/
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET_ROOT = Path(
    "prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721"
)
DEFAULT_REPORT_PATH = Path("docs/hard_sector_route_check.md")
DEFAULT_DEBUG_DIR = Path("output/hard_sector_route_debug")

SPLIT_TO_JSONL = {
    "Train": "train_samples.jsonl",
    "Valid": "valid_samples.jsonl",
    "Test": "test_samples.jsonl",
}

CAMERA_ID_TO_NAME = {
    0: "Back",
    1: "Front",
    2: "Left",
    3: "Right",
    4: "None",
}
CAMERA_IDS = [0, 1, 2, 3, 4]
CAMERA_IDS_NO_NONE = [0, 1, 2, 3]

EPS = 1e-9


@dataclass
class ModeStats:
    total: int = 0
    correct: int = 0
    per_gt_total: Dict[int, int] = field(default_factory=lambda: {cam: 0 for cam in CAMERA_IDS})
    per_gt_correct: Dict[int, int] = field(default_factory=lambda: {cam: 0 for cam in CAMERA_IDS})
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((5, 5), dtype=np.int64))
    boundary_total: int = 0
    boundary_correct: int = 0
    non_boundary_total: int = 0
    non_boundary_correct: int = 0

    def update(self, gt_camera: int, pred_camera: int, boundary_case: bool) -> None:
        if gt_camera not in CAMERA_ID_TO_NAME:
            return
        if pred_camera not in CAMERA_ID_TO_NAME:
            pred_camera = 4

        self.total += 1
        self.per_gt_total[gt_camera] += 1
        self.confusion[gt_camera, pred_camera] += 1

        is_correct = gt_camera == pred_camera
        if is_correct:
            self.correct += 1
            self.per_gt_correct[gt_camera] += 1

        if boundary_case:
            self.boundary_total += 1
            if is_correct:
                self.boundary_correct += 1
        else:
            self.non_boundary_total += 1
            if is_correct:
                self.non_boundary_correct += 1


@dataclass
class EvalObject:
    split: str
    sample_id: str
    scene_id: str
    tower_id: str
    frame_id: str
    ann_index: int
    gt_primary_camera: int
    center_angle_camera: int
    polygon_area_camera: int
    sector_areas: Dict[int, float]
    sector_main_ratio: float
    polygon_crosses_multi_sector: bool
    boundary_case: bool
    resolution: int
    range_max: float
    radar_bev_path: Path
    polygon_pixels: List[List[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check hard-sector camera route quality")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Prepared dataset root with train/valid/test_samples.jsonl",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report output path",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=DEFAULT_DEBUG_DIR,
        help="Debug image output directory",
    )
    parser.add_argument(
        "--max-vis-per-group",
        type=int,
        default=200,
        help="Maximum images to save per focus group",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260425,
        help="Random seed for deterministic sampling order",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on total frame samples across all splits (0 = no cap)",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def polygon_area(poly_xy: Sequence[Sequence[float]]) -> float:
    if len(poly_xy) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly_xy)):
        x1, y1 = float(poly_xy[i][0]), float(poly_xy[i][1])
        x2, y2 = float(poly_xy[(i + 1) % len(poly_xy)][0]), float(poly_xy[(i + 1) % len(poly_xy)][1])
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def order_polygon_clockwise(points: Sequence[Sequence[float]]) -> List[List[float]]:
    pts = [[float(p[0]), float(p[1])] for p in points]
    if len(pts) < 3:
        return pts
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return pts


def project_pixels_to_metric_xy(
    points_uv: Sequence[Sequence[float]],
    resolution: int,
    range_max: float,
) -> List[List[float]]:
    cx = resolution / 2.0
    cy = resolution / 2.0
    scale = (resolution / 2.0) / max(range_max, 1e-6)
    return [[(cy - float(v)) / scale, (float(u) - cx) / scale] for u, v in points_uv]


def project_metric_xy_to_pixels(
    points_xy: Sequence[Sequence[float]],
    resolution: int,
    range_max: float,
) -> List[List[float]]:
    cx = resolution / 2.0
    cy = resolution / 2.0
    scale = (resolution / 2.0) / max(range_max, 1e-6)
    return [[cx + float(y) * scale, cy - float(x) * scale] for x, y in points_xy]


def rbox_xywhr_to_metric_polygon(rbox_xywhr: Sequence[float]) -> List[List[float]]:
    if not isinstance(rbox_xywhr, (list, tuple)) or len(rbox_xywhr) != 5:
        return []
    cx, cy, w, h, angle = [float(v) for v in rbox_xywhr]
    half_w = max(w, 1e-6) * 0.5
    half_h = max(h, 1e-6) * 0.5
    corners = np.array(
        [[half_w, half_h], [half_w, -half_h], [-half_w, -half_h], [-half_w, half_h]],
        dtype=np.float64,
    )
    ca = math.cos(angle)
    sa = math.sin(angle)
    rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    rotated = (rot @ corners.T).T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return order_polygon_clockwise(rotated.tolist())


def _line_intersection(p1: Sequence[float], p2: Sequence[float], v1: float, v2: float) -> List[float]:
    if abs(v1 - v2) < EPS:
        return [float(p1[0]), float(p1[1])]
    t = v1 / (v1 - v2)
    return [
        float(p1[0]) + t * (float(p2[0]) - float(p1[0])),
        float(p1[1]) + t * (float(p2[1]) - float(p1[1])),
    ]


def clip_polygon_halfplane(
    poly_xy: Sequence[Sequence[float]],
    a: float,
    b: float,
    c: float = 0.0,
    keep_ge: bool = True,
) -> List[List[float]]:
    if len(poly_xy) < 3:
        return []

    def value(pt: Sequence[float]) -> float:
        return a * float(pt[0]) + b * float(pt[1]) + c

    def inside(v: float) -> bool:
        return v >= -EPS if keep_ge else v <= EPS

    output: List[List[float]] = []
    prev = [float(poly_xy[-1][0]), float(poly_xy[-1][1])]
    prev_v = value(prev)
    prev_in = inside(prev_v)

    for curr_raw in poly_xy:
        curr = [float(curr_raw[0]), float(curr_raw[1])]
        curr_v = value(curr)
        curr_in = inside(curr_v)

        if curr_in:
            if not prev_in:
                output.append(_line_intersection(prev, curr, prev_v, curr_v))
            output.append(curr)
        elif prev_in:
            output.append(_line_intersection(prev, curr, prev_v, curr_v))

        prev = curr
        prev_v = curr_v
        prev_in = curr_in

    if len(output) < 3:
        return []
    return order_polygon_clockwise(output)


def sector_overlap_areas(poly_xy: Sequence[Sequence[float]]) -> Dict[int, float]:
    poly = order_polygon_clockwise(poly_xy)
    if len(poly) < 3:
        return {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    # camera_id -> list of half-planes (a, b, c, keep_ge)
    # Back:  y - x >= 0  and x + y <= 0
    # Front: x - y >= 0  and x + y >= 0
    # Left:  x - y >= 0  and x + y <= 0
    # Right: y - x >= 0  and x + y >= 0
    sector_planes = {
        0: [(-1.0, 1.0, 0.0, True), (1.0, 1.0, 0.0, False)],
        1: [(1.0, -1.0, 0.0, True), (1.0, 1.0, 0.0, True)],
        2: [(1.0, -1.0, 0.0, True), (1.0, 1.0, 0.0, False)],
        3: [(-1.0, 1.0, 0.0, True), (1.0, 1.0, 0.0, True)],
    }

    areas: Dict[int, float] = {}
    for cam_id in [0, 1, 2, 3]:
        clipped = poly
        for a, b, c, keep_ge in sector_planes[cam_id]:
            clipped = clip_polygon_halfplane(clipped, a, b, c, keep_ge=keep_ge)
            if len(clipped) < 3:
                break
        areas[cam_id] = polygon_area(clipped)
    return areas


def angle_to_camera_id(angle_rad: float) -> int:
    if -math.pi / 4 <= angle_rad < math.pi / 4:
        return 1  # Front
    if math.pi / 4 <= angle_rad < 3 * math.pi / 4:
        return 3  # Right
    if angle_rad >= 3 * math.pi / 4 or angle_rad < -3 * math.pi / 4:
        return 0  # Back
    return 2  # Left


def choose_polygon_camera(areas: Dict[int, float], fallback_cam: int) -> int:
    best_cam = fallback_cam
    best_area = -1.0
    for cam_id in [0, 1, 2, 3]:
        area = float(areas.get(cam_id, 0.0))
        if area > best_area:
            best_area = area
            best_cam = cam_id
    return best_cam


def extract_metric_polygon(
    annotation: Dict[str, object],
    resolution: int,
    range_max: float,
) -> Tuple[List[List[float]], List[List[float]]]:
    poly_uv = annotation.get("poly", [])
    if isinstance(poly_uv, list) and len(poly_uv) >= 3:
        try:
            poly_uv_norm = [[float(p[0]), float(p[1])] for p in poly_uv if isinstance(p, (list, tuple)) and len(p) >= 2]
        except Exception:
            poly_uv_norm = []
        if len(poly_uv_norm) >= 3:
            metric_poly = project_pixels_to_metric_xy(poly_uv_norm, resolution, range_max)
            metric_poly = order_polygon_clockwise(metric_poly)
            if polygon_area(metric_poly) > EPS:
                return metric_poly, poly_uv_norm

    rbox = annotation.get("rbox_xywhr")
    metric_poly = rbox_xywhr_to_metric_polygon(rbox if isinstance(rbox, (list, tuple)) else [])
    if len(metric_poly) >= 3 and polygon_area(metric_poly) > EPS:
        pixel_poly = project_metric_xy_to_pixels(metric_poly, resolution, range_max)
        return metric_poly, pixel_poly

    return [], []


def choose_center_xy(annotation: Dict[str, object], metric_poly: Sequence[Sequence[float]]) -> Tuple[float, float]:
    rbox = annotation.get("rbox_xywhr")
    if isinstance(rbox, (list, tuple)) and len(rbox) == 5:
        try:
            return float(rbox[0]), float(rbox[1])
        except Exception:
            pass

    if metric_poly:
        cx = sum(float(p[0]) for p in metric_poly) / len(metric_poly)
        cy = sum(float(p[1]) for p in metric_poly) / len(metric_poly)
        return cx, cy
    return 0.0, 0.0


def safe_open_image(path: Path, fallback_wh: Tuple[int, int]) -> Image.Image:
    if path.is_file():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", fallback_wh, (18, 20, 24))


def draw_polygon(draw: ImageDraw.ImageDraw, poly: Sequence[Sequence[float]], color: Tuple[int, int, int], width: int = 2) -> None:
    if len(poly) < 2:
        return
    pts = [(float(p[0]), float(p[1])) for p in poly]
    pts.append(pts[0])
    draw.line(pts, fill=color, width=width)


def draw_sector_lines(draw: ImageDraw.ImageDraw, w: int, h: int, color: Tuple[int, int, int]) -> None:
    draw.line([(0, 0), (w - 1, h - 1)], fill=color, width=1)
    draw.line([(0, h - 1), (w - 1, 0)], fill=color, width=1)


def format_pct(numer: int, denom: int) -> str:
    if denom <= 0:
        return "n/a"
    return f"{(100.0 * numer / denom):.2f}%"


def fmt_ratio(value: float) -> str:
    return f"{value:.4f}"


def build_ratio_distribution(ratios: Sequence[float]) -> Dict[str, object]:
    if not ratios:
        return {
            "count": 0,
            "quantiles": {},
            "hist": {},
        }

    arr = np.asarray(list(ratios), dtype=np.float64)
    quantiles = {
        "min": float(np.percentile(arr, 0)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.percentile(arr, 100)),
    }

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.000001]
    labels = [
        "[0.0,0.2)",
        "[0.2,0.4)",
        "[0.4,0.6)",
        "[0.6,0.8)",
        "[0.8,0.9)",
        "[0.9,1.0]",
    ]
    counts, _ = np.histogram(arr, bins=bins)
    hist = {labels[i]: int(counts[i]) for i in range(len(labels))}

    return {
        "count": int(arr.size),
        "quantiles": quantiles,
        "hist": hist,
    }


def render_debug_image(item: EvalObject, out_path: Path, canvas_wh: int = 1024) -> None:
    ensure_parent(out_path)
    font = ImageFont.load_default()

    raw_img = safe_open_image(item.radar_bev_path, (item.resolution, item.resolution))
    vis_img = raw_img.resize((canvas_wh, canvas_wh), Image.BILINEAR)
    draw = ImageDraw.Draw(vis_img)

    scale = canvas_wh / max(1.0, float(item.resolution))
    poly = [[float(p[0]) * scale, float(p[1]) * scale] for p in item.polygon_pixels]

    draw_sector_lines(draw, canvas_wh, canvas_wh, color=(180, 180, 180))
    draw_polygon(draw, poly, color=(0, 255, 120), width=2)

    if poly:
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 230, 0))

    text_lines = [
        f"sample={item.sample_id} ann_idx={item.ann_index}",
        f"split={item.split} scene={item.scene_id} tower={item.tower_id} frame={item.frame_id}",
        f"gt_primary_camera={item.gt_primary_camera} ({CAMERA_ID_TO_NAME.get(item.gt_primary_camera, 'Unknown')})",
        f"center_angle_camera={item.center_angle_camera} ({CAMERA_ID_TO_NAME[item.center_angle_camera]})",
        f"polygon_area_camera={item.polygon_area_camera} ({CAMERA_ID_TO_NAME[item.polygon_area_camera]})",
        f"sector_main_ratio={item.sector_main_ratio:.4f} boundary_case={item.boundary_case}",
        (
            "areas(back/front/left/right)="
            f"{item.sector_areas.get(0, 0.0):.3f}/"
            f"{item.sector_areas.get(1, 0.0):.3f}/"
            f"{item.sector_areas.get(2, 0.0):.3f}/"
            f"{item.sector_areas.get(3, 0.0):.3f}"
        ),
    ]

    y = 8
    for line in text_lines:
        draw.rectangle([8, y, 8 + max(420, len(line) * 6), y + 16], fill=(0, 0, 0))
        draw.text((10, y + 2), line, fill=(235, 235, 235), font=font)
        y += 18

    vis_img.save(out_path)


def accuracy_of(stats: ModeStats) -> str:
    return format_pct(stats.correct, stats.total)


def write_markdown_report(
    report_path: Path,
    dataset_root: Path,
    processed_sample_count: int,
    split_sample_counts: Dict[str, int],
    total_objects: int,
    skipped_objects: int,
    split_stats_center: Dict[str, ModeStats],
    split_stats_polygon: Dict[str, ModeStats],
    overall_center: ModeStats,
    overall_polygon: ModeStats,
    ratio_dist: Dict[str, object],
    center_wrong_poly_correct_count: int,
    polygon_wrong_count: int,
    boundary_case_count: int,
) -> None:
    lines: List[str] = []
    gt_primary_none_count = int(overall_center.per_gt_total[4])
    lines.append("# Hard Sector Camera Route Check")
    lines.append("")
    lines.append(f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- dataset_root: `{dataset_root}`")
    lines.append(f"- processed_samples: `{processed_sample_count}`")
    lines.append(f"- total_objects: `{total_objects}`")
    lines.append(f"- skipped_objects: `{skipped_objects}`")
    lines.append(f"- gt_primary_camera=4 count: `{gt_primary_none_count}`")
    lines.append("")

    lines.append("## Split sample counts")
    lines.append("")
    lines.append("| split | sample_count |")
    lines.append("|---|---:|")
    for split in ["Train", "Valid", "Test"]:
        lines.append(f"| {split} | {split_sample_counts.get(split, 0)} |")
    lines.append("")

    def append_mode_section(title: str, overall: ModeStats, split_map: Dict[str, ModeStats]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- overall accuracy: `{overall.correct}/{overall.total}` ({accuracy_of(overall)})")
        lines.append(
            f"- boundary_case accuracy: `{overall.boundary_correct}/{overall.boundary_total}` "
            f"({format_pct(overall.boundary_correct, overall.boundary_total)})"
        )
        lines.append(
            f"- non_boundary_case accuracy: `{overall.non_boundary_correct}/{overall.non_boundary_total}` "
            f"({format_pct(overall.non_boundary_correct, overall.non_boundary_total)})"
        )
        lines.append("")

        lines.append("### Per-split accuracy")
        lines.append("")
        lines.append("| split | correct/total | accuracy |")
        lines.append("|---|---:|---:|")
        for split in ["Train", "Valid", "Test"]:
            st = split_map[split]
            lines.append(f"| {split} | {st.correct}/{st.total} | {accuracy_of(st)} |")
        lines.append("")

        lines.append("### Per-camera accuracy (by GT primary camera)")
        lines.append("")
        lines.append("| gt_camera | correct/total | accuracy |")
        lines.append("|---|---:|---:|")
        for cam in CAMERA_IDS:
            c = overall.per_gt_correct[cam]
            t = overall.per_gt_total[cam]
            lines.append(f"| {cam} ({CAMERA_ID_TO_NAME[cam]}) | {c}/{t} | {format_pct(c, t)} |")
        lines.append("")

        lines.append("### Confusion Matrix (rows=GT, cols=Pred)")
        lines.append("")
        header = "| GT \\\\ Pred | " + " | ".join(
            [f"{cam}({CAMERA_ID_TO_NAME[cam]})" for cam in CAMERA_IDS]
        ) + " |"
        sep = "|---|" + "---:|" * len(CAMERA_IDS)
        lines.append(header)
        lines.append(sep)
        for gt_cam in CAMERA_IDS:
            row_vals = [str(int(overall.confusion[gt_cam, pred_cam])) for pred_cam in CAMERA_IDS]
            lines.append(f"| {gt_cam} ({CAMERA_ID_TO_NAME[gt_cam]}) | " + " | ".join(row_vals) + " |")
        lines.append("")

    append_mode_section("center_angle mode", overall_center, split_stats_center)
    append_mode_section("polygon_area mode", overall_polygon, split_stats_polygon)

    lines.append("## sector_main_ratio distribution")
    lines.append("")
    lines.append(f"- count: `{ratio_dist.get('count', 0)}`")
    quantiles = ratio_dist.get("quantiles", {}) if isinstance(ratio_dist, dict) else {}
    if quantiles:
        lines.append(
            "- quantiles: "
            + ", ".join([f"{k}={float(v):.4f}" for k, v in quantiles.items()])
        )
    lines.append("")
    lines.append("| bucket | count |")
    lines.append("|---|---:|")
    hist = ratio_dist.get("hist", {}) if isinstance(ratio_dist, dict) else {}
    for bucket in ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,0.9)", "[0.9,1.0]"]:
        lines.append(f"| {bucket} | {int(hist.get(bucket, 0))} |")
    lines.append("")

    lines.append("## Focus debug sample counts")
    lines.append("")
    lines.append(f"- center_angle wrong but polygon_area correct: `{center_wrong_poly_correct_count}`")
    lines.append(f"- polygon_area wrong: `{polygon_wrong_count}`")
    lines.append(f"- boundary_case: `{boundary_case_count}`")
    lines.append("")

    ensure_parent(report_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_dataset(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    report_path = args.report_path.resolve()
    debug_dir = args.debug_dir.resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)

    split_records: Dict[str, List[Dict[str, object]]] = {}
    for split, rel_path in SPLIT_TO_JSONL.items():
        path = dataset_root / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing split jsonl: {path}")
        split_records[split] = load_jsonl(path)

    rng = random.Random(args.seed)
    split_sample_counts = {split: len(records) for split, records in split_records.items()}

    all_samples: List[Tuple[str, Dict[str, object]]] = []
    for split in ["Train", "Valid", "Test"]:
        for rec in split_records[split]:
            all_samples.append((split, rec))

    if args.max_samples and args.max_samples > 0 and args.max_samples < len(all_samples):
        all_samples = rng.sample(all_samples, args.max_samples)

    # Keep deterministic order for stable reports.
    all_samples.sort(
        key=lambda pair: (
            pair[0],
            str(pair[1].get("scene_id", "")),
            str(pair[1].get("tower_id", "")),
            str(pair[1].get("frame_id", "")),
        )
    )

    split_stats_center = {split: ModeStats() for split in ["Train", "Valid", "Test"]}
    split_stats_polygon = {split: ModeStats() for split in ["Train", "Valid", "Test"]}
    overall_center = ModeStats()
    overall_polygon = ModeStats()

    focus_center_wrong_poly_correct: List[EvalObject] = []
    focus_polygon_wrong: List[EvalObject] = []
    focus_boundary: List[EvalObject] = []
    sector_main_ratios: List[float] = []

    total_objects = 0
    skipped_objects = 0

    for split, sample in all_samples:
        ann_path = Path(str(sample.get("annotation_json_path", "")))
        radar_bev_path = Path(str(sample.get("radar_bev_path", "")))
        if not ann_path.is_file():
            continue

        payload = load_json(ann_path)
        resolution = int(payload.get("resolution", 1536))
        range_max = float(payload.get("range_max", 2000.0))
        annotations = payload.get("annotations", [])
        if not isinstance(annotations, list):
            continue

        for ann_idx, ann in enumerate(annotations):
            if not isinstance(ann, dict):
                skipped_objects += 1
                continue
            gt_cam = int(ann.get("gt_primary_camera", 4))
            if gt_cam not in CAMERA_ID_TO_NAME:
                skipped_objects += 1
                continue

            total_objects += 1

            metric_poly, pixel_poly = extract_metric_polygon(ann, resolution, range_max)
            if len(metric_poly) < 3:
                skipped_objects += 1
                continue

            cx, cy = choose_center_xy(ann, metric_poly)
            center_angle = math.atan2(cy, cx)
            center_cam = angle_to_camera_id(center_angle)

            areas = sector_overlap_areas(metric_poly)
            area_sum = float(sum(areas.values()))
            main_area = max(float(v) for v in areas.values()) if areas else 0.0
            main_ratio = main_area / max(area_sum, 1e-6)
            crosses_multi = sum(1 for v in areas.values() if v > 1e-6) > 1
            boundary_case = (main_ratio < 0.8) or crosses_multi

            poly_cam = choose_polygon_camera(areas, fallback_cam=center_cam)
            sector_main_ratios.append(main_ratio)

            split_stats_center[split].update(gt_cam, center_cam, boundary_case)
            split_stats_polygon[split].update(gt_cam, poly_cam, boundary_case)
            overall_center.update(gt_cam, center_cam, boundary_case)
            overall_polygon.update(gt_cam, poly_cam, boundary_case)

            eval_obj = EvalObject(
                split=split,
                sample_id=str(sample.get("sample_id", "")),
                scene_id=str(sample.get("scene_id", "")),
                tower_id=str(sample.get("tower_id", "")),
                frame_id=str(sample.get("frame_id", "")),
                ann_index=ann_idx,
                gt_primary_camera=gt_cam,
                center_angle_camera=center_cam,
                polygon_area_camera=poly_cam,
                sector_areas=areas,
                sector_main_ratio=main_ratio,
                polygon_crosses_multi_sector=crosses_multi,
                boundary_case=boundary_case,
                resolution=resolution,
                range_max=range_max,
                radar_bev_path=radar_bev_path,
                polygon_pixels=pixel_poly,
            )

            if center_cam != gt_cam and poly_cam == gt_cam:
                focus_center_wrong_poly_correct.append(eval_obj)
            if poly_cam != gt_cam:
                focus_polygon_wrong.append(eval_obj)
            if boundary_case:
                focus_boundary.append(eval_obj)

    def save_focus_group(items: List[EvalObject], group_name: str, max_count: int) -> int:
        out_dir = debug_dir / group_name
        out_dir.mkdir(parents=True, exist_ok=True)
        if not items:
            return 0
        chosen = items if len(items) <= max_count else rng.sample(items, max_count)
        chosen.sort(key=lambda x: (x.split, x.scene_id, x.tower_id, x.frame_id, x.ann_index))
        for idx, item in enumerate(chosen, start=1):
            filename = (
                f"{idx:04d}_{item.split}_{item.scene_id}_{item.tower_id}_{item.frame_id}"
                f"_ann{item.ann_index}_gt{item.gt_primary_camera}"
                f"_c{item.center_angle_camera}_p{item.polygon_area_camera}"
                f"_r{item.sector_main_ratio:.3f}.png"
            )
            render_debug_image(item, out_dir / filename)
        return len(chosen)

    center_wrong_poly_correct_saved = save_focus_group(
        focus_center_wrong_poly_correct,
        "center_wrong_polygon_correct",
        max(0, int(args.max_vis_per_group)),
    )
    polygon_wrong_saved = save_focus_group(
        focus_polygon_wrong,
        "polygon_wrong",
        max(0, int(args.max_vis_per_group)),
    )
    boundary_saved = save_focus_group(
        focus_boundary,
        "boundary_case",
        max(0, int(args.max_vis_per_group)),
    )

    ratio_dist = build_ratio_distribution(sector_main_ratios)
    write_markdown_report(
        report_path=report_path,
        dataset_root=dataset_root,
        processed_sample_count=len(all_samples),
        split_sample_counts=split_sample_counts,
        total_objects=total_objects,
        skipped_objects=skipped_objects,
        split_stats_center=split_stats_center,
        split_stats_polygon=split_stats_polygon,
        overall_center=overall_center,
        overall_polygon=overall_polygon,
        ratio_dist=ratio_dist,
        center_wrong_poly_correct_count=len(focus_center_wrong_poly_correct),
        polygon_wrong_count=len(focus_polygon_wrong),
        boundary_case_count=len(focus_boundary),
    )

    summary = {
        "dataset_root": str(dataset_root),
        "processed_samples": len(all_samples),
        "total_objects": total_objects,
        "skipped_objects": skipped_objects,
        "center_overall_accuracy": f"{overall_center.correct}/{overall_center.total} ({accuracy_of(overall_center)})",
        "polygon_overall_accuracy": f"{overall_polygon.correct}/{overall_polygon.total} ({accuracy_of(overall_polygon)})",
        "report_path": str(report_path),
        "debug_dir": str(debug_dir),
        "saved_debug": {
            "center_wrong_polygon_correct": center_wrong_poly_correct_saved,
            "polygon_wrong": polygon_wrong_saved,
            "boundary_case": boundary_saved,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    evaluate_dataset(args)


if __name__ == "__main__":
    main()
