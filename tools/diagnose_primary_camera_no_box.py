#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


CAMERA_CANONICAL_ORDER: Tuple[Tuple[str, str], ...] = (
    ("Back", "CamBack"),
    ("Front", "CamFront"),
    ("Left", "CamLeft"),
    ("Right", "CamRight"),
)
CANONICAL_CAMERA_KEYS: Tuple[str, ...] = tuple(item[0] for item in CAMERA_CANONICAL_ORDER)
CAMERA_NAMES: Tuple[str, ...] = tuple(item[1] for item in CAMERA_CANONICAL_ORDER)
CAMERA_ID_TO_NAME: Dict[int, str] = {idx: key for idx, key in enumerate(CANONICAL_CAMERA_KEYS)}
NONE_CAMERA_ID = 4

OPV2V_CAM_MAP = {
    "Back": "camera3",
    "Front": "camera1",
    "Left": "camera2",
    "Right": "camera0",
}
CAMERA_CV_FROM_LOCAL = [
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

EPS = 1e-9
MIN_VALID_SIDE = 1.0
IMG_W = 1024.0
IMG_H = 512.0

SPLIT_TO_JSONL = {
    "Train": "train_samples.jsonl",
    "Valid": "valid_samples.jsonl",
    "Test": "test_samples.jsonl",
}

FAIL_REASONS = [
    "raw_primary_missing",
    "raw_primary_invalid_nan",
    "raw_primary_clipped_empty",
    "raw_primary_too_small",
    "projection_box_outside_image",
    "projection_invalid_depth",
    "unknown",
]


@dataclass
class CamBoxStatus:
    valid: bool
    reason: str
    box_xyxy: Optional[List[float]]
    width: float
    height: float
    area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose primary-camera-no-box targets after None-primary fix"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721"),
    )
    parser.add_argument(
        "--issue-csv",
        type=Path,
        default=Path("output/route_loss_mask_after_none_fix/primary_with_no_box_details.csv"),
    )
    parser.add_argument(
        "--issue-jsonl",
        type=Path,
        default=Path("output/route_loss_mask_after_none_fix/primary_with_no_box_details.jsonl"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/primary_camera_no_box_diagnosis.md"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/primary_camera_no_box_debug"),
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def finite_float(value) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def clip_box_xyxy(box_xyxy: Sequence[float], width: float, height: float) -> Optional[List[float]]:
    vals = [finite_float(v) for v in box_xyxy]
    if any(v is None for v in vals):
        return None
    x1, y1, x2, y2 = [float(v) for v in vals]
    x1 = min(max(x1, 0.0), float(width - 1.0))
    y1 = min(max(y1, 0.0), float(height - 1.0))
    x2 = min(max(x2, 0.0), float(width - 1.0))
    y2 = min(max(y2, 0.0), float(height - 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def analyze_cam_entry(cam_entry: object, width: float = IMG_W, height: float = IMG_H) -> CamBoxStatus:
    if not isinstance(cam_entry, dict):
        return CamBoxStatus(False, "raw_primary_missing", None, 0.0, 0.0, 0.0)

    box_2d = cam_entry.get("box_2d")
    if not isinstance(box_2d, dict):
        return CamBoxStatus(False, "raw_primary_missing", None, 0.0, 0.0, 0.0)

    xmin = finite_float(box_2d.get("xmin"))
    ymin = finite_float(box_2d.get("ymin"))
    xmax = finite_float(box_2d.get("xmax"))
    ymax = finite_float(box_2d.get("ymax"))
    if None in (xmin, ymin, xmax, ymax):
        return CamBoxStatus(False, "raw_primary_invalid_nan", None, 0.0, 0.0, 0.0)

    clipped = clip_box_xyxy([xmin, ymin, xmax, ymax], width, height)
    if clipped is None:
        return CamBoxStatus(False, "raw_primary_clipped_empty", None, 0.0, 0.0, 0.0)

    w = float(clipped[2] - clipped[0])
    h = float(clipped[3] - clipped[1])
    area = float(max(0.0, w) * max(0.0, h))
    if w < MIN_VALID_SIDE or h < MIN_VALID_SIDE:
        return CamBoxStatus(False, "raw_primary_too_small", clipped, w, h, area)
    return CamBoxStatus(True, "valid", clipped, w, h, area)


def parse_matrix(payload: object, rows: int, cols: int) -> Optional[List[List[float]]]:
    if not isinstance(payload, (list, tuple)) or len(payload) != rows:
        return None
    out: List[List[float]] = []
    for r in payload:
        if not isinstance(r, (list, tuple)) or len(r) != cols:
            return None
        row: List[float] = []
        for v in r:
            fv = finite_float(v)
            if fv is None:
                return None
            row.append(float(fv))
        out.append(row)
    return out


def mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    r = len(a)
    c = len(b[0])
    k = len(b)
    out = [[0.0 for _ in range(c)] for _ in range(r)]
    for i in range(r):
        for j in range(c):
            s = 0.0
            for t in range(k):
                s += float(a[i][t]) * float(b[t][j])
            out[i][j] = s
    return out


def mat_vec_mul(a: List[List[float]], v: List[float]) -> List[float]:
    out = []
    for row in a:
        s = 0.0
        for i, rv in enumerate(row):
            s += float(rv) * float(v[i])
        out.append(s)
    return out


def infer_image_size_from_intrinsic(k: List[List[float]]) -> Tuple[float, float]:
    if len(k) != 3 or len(k[0]) != 3 or len(k[1]) != 3 or len(k[2]) != 3:
        return IMG_W, IMG_H
    w = float(max(2.0, float(k[0][2]) * 2.0))
    h = float(max(2.0, float(k[1][2]) * 2.0))
    return w, h


def build_cuboid_points(raw_obj: Dict[str, object], ann: Dict[str, object]) -> Tuple[Optional[List[List[float]]], str]:
    radar_proj = raw_obj.get("radar_proj") if isinstance(raw_obj, dict) else {}
    if not isinstance(radar_proj, dict):
        radar_proj = {}
    bev = raw_obj.get("bev_rot_only_yaw") if isinstance(raw_obj, dict) else {}
    if not isinstance(bev, dict):
        bev = {}

    center = radar_proj.get("center") if isinstance(radar_proj.get("center"), dict) else None
    if center is None:
        center = bev.get("center") if isinstance(bev.get("center"), dict) else None
    if center is None:
        return None, "missing_center_xyz"

    cx = finite_float(center.get("x"))
    cy = finite_float(center.get("y"))
    cz = finite_float(center.get("z"))
    if None in (cx, cy, cz):
        return None, "invalid_center_xyz"

    extent = radar_proj.get("extent") if isinstance(radar_proj.get("extent"), dict) else None
    if extent is None:
        extent = bev.get("extent") if isinstance(bev.get("extent"), dict) else None

    ex = ey = None
    if isinstance(extent, dict):
        ex = finite_float(extent.get("x"))
        ey = finite_float(extent.get("y"))

    if ex is None or ey is None:
        rbox = ann.get("rbox_xywhr")
        if isinstance(rbox, (list, tuple)) and len(rbox) == 5:
            rw = finite_float(rbox[2])
            rh = finite_float(rbox[3])
            if rw is not None and rh is not None:
                ex = max(rw, 1e-6) * 0.5
                ey = max(rh, 1e-6) * 0.5
    if ex is None or ey is None:
        return None, "missing_extent_xy"
    ex = max(float(ex), 1e-6)
    ey = max(float(ey), 1e-6)

    height = None
    bbox_m = raw_obj.get("bbox_m") if isinstance(raw_obj, dict) else None
    if isinstance(bbox_m, dict):
        height = finite_float(bbox_m.get("H"))
    if height is None and isinstance(extent, dict):
        ez = finite_float(extent.get("z"))
        if ez is not None:
            height = abs(float(ez)) * 2.0
    if height is None:
        return None, "missing_height"
    height = max(float(height), 0.1)

    yaw = finite_float(bev.get("yaw")) if isinstance(bev, dict) else None
    if yaw is None:
        rbox = ann.get("rbox_xywhr")
        if isinstance(rbox, (list, tuple)) and len(rbox) == 5:
            yaw = finite_float(rbox[4])
    if yaw is None:
        return None, "missing_yaw"

    bottom_z = float(cz) - 0.5 * float(height)
    top_z = bottom_z + float(height)
    ca = math.cos(float(yaw))
    sa = math.sin(float(yaw))

    local_xy = [
        [ex, ey],
        [ex, -ey],
        [-ex, -ey],
        [-ex, ey],
    ]
    rotated: List[List[float]] = []
    for dx, dy in local_xy:
        x = float(cx) + ca * float(dx) - sa * float(dy)
        y = float(cy) + sa * float(dx) + ca * float(dy)
        rotated.append([x, y])

    points_xyz: List[List[float]] = []
    for x, y in rotated:
        points_xyz.append([x, y, float(bottom_z)])
    for x, y in rotated:
        points_xyz.append([x, y, float(top_z)])
    return points_xyz, "ok"


def project_points(
    points_xyz: List[List[float]],
    t_camcv_ego: List[List[float]],
    intrinsic_k: List[List[float]],
) -> Tuple[List[List[float]], List[float]]:
    uv: List[List[float]] = []
    z_values: List[float] = []
    for point in points_xyz:
        vec = [float(point[0]), float(point[1]), float(point[2]), 1.0]
        cam_h = mat_vec_mul(t_camcv_ego, vec)
        cam_x, cam_y, cam_z = float(cam_h[0]), float(cam_h[1]), float(cam_h[2])
        z_values.append(cam_z)
        safe_z = cam_z if abs(cam_z) >= 1e-6 else 1e-6
        u = intrinsic_k[0][0] * cam_x / safe_z + intrinsic_k[0][2]
        v = intrinsic_k[1][1] * cam_y / safe_z + intrinsic_k[1][2]
        uv.append([u, v])
    return uv, z_values


def project_cuboid_box(
    points_xyz: List[List[float]],
    opv2v_payload: Dict[str, object],
    canonical_cam: str,
) -> Tuple[Optional[List[float]], str]:
    opv2v_cam_name = OPV2V_CAM_MAP.get(canonical_cam)
    if opv2v_cam_name is None:
        return None, "missing_opv2v_cam_mapping"
    cam_meta = opv2v_payload.get(opv2v_cam_name)
    if not isinstance(cam_meta, dict):
        return None, f"missing_opv2v_camera_meta:{opv2v_cam_name}"

    intrinsic = parse_matrix(cam_meta.get("intrinsic"), rows=3, cols=3)
    extrinsic_local = parse_matrix(cam_meta.get("extrinsic"), rows=4, cols=4)
    if intrinsic is None or extrinsic_local is None:
        return None, "invalid_camera_calibration_matrix"

    t_camcv_ego = mat_mul(CAMERA_CV_FROM_LOCAL, extrinsic_local)
    uv, z_cam = project_points(points_xyz, t_camcv_ego, intrinsic)
    valid_uv = [uv[idx] for idx, z in enumerate(z_cam) if z > 1e-4]
    if len(valid_uv) < 1:
        return None, "projection_all_points_behind_camera"

    xs = [float(p[0]) for p in valid_uv]
    ys = [float(p[1]) for p in valid_uv]
    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)

    img_w, img_h = infer_image_size_from_intrinsic(intrinsic)
    clipped = clip_box_xyxy([xmin, ymin, xmax, ymax], img_w, img_h)
    if clipped is None:
        return None, "projection_box_outside_image"
    if (clipped[2] - clipped[0]) <= 0.0 or (clipped[3] - clipped[1]) <= 0.0:
        return None, "projection_box_zero_area"
    return clipped, "ok"


def projection_reason_bucket(status: str) -> str:
    if status == "projection_box_outside_image":
        return "projection_box_outside_image"
    if status == "projection_all_points_behind_camera":
        return "projection_invalid_depth"
    return "unknown"


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
    points_uv: Sequence[Sequence[float]], resolution: int, range_max: float
) -> List[List[float]]:
    cx = resolution / 2.0
    cy = resolution / 2.0
    scale = (resolution / 2.0) / max(range_max, 1e-6)
    return [[(cy - float(v)) / scale, (float(u) - cx) / scale] for u, v in points_uv]


def rbox_xywhr_to_metric_polygon(rbox_xywhr: Sequence[float]) -> List[List[float]]:
    if not isinstance(rbox_xywhr, (list, tuple)) or len(rbox_xywhr) != 5:
        return []
    cx, cy, w, h, angle = [float(v) for v in rbox_xywhr]
    half_w = max(w, 1e-6) * 0.5
    half_h = max(h, 1e-6) * 0.5
    corners = [
        [half_w, half_h],
        [half_w, -half_h],
        [-half_w, -half_h],
        [-half_w, half_h],
    ]
    ca = math.cos(angle)
    sa = math.sin(angle)
    out = []
    for x, y in corners:
        rx = ca * x - sa * y + cx
        ry = sa * x + ca * y + cy
        out.append([rx, ry])
    return order_polygon_clockwise(out)


def _line_intersection(p1: Sequence[float], p2: Sequence[float], v1: float, v2: float) -> List[float]:
    if abs(v1 - v2) < EPS:
        return [float(p1[0]), float(p1[1])]
    t = v1 / (v1 - v2)
    return [
        float(p1[0]) + t * (float(p2[0]) - float(p1[0])),
        float(p1[1]) + t * (float(p2[1]) - float(p1[1])),
    ]


def clip_polygon_halfplane(
    poly_xy: Sequence[Sequence[float]], a: float, b: float, c: float = 0.0, keep_ge: bool = True
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


def choose_polygon_camera(areas: Dict[int, float], fallback_cam: int) -> int:
    best_cam = fallback_cam
    best_area = -1.0
    for cam_id in [0, 1, 2, 3]:
        area = float(areas.get(cam_id, 0.0))
        if area > best_area:
            best_area = area
            best_cam = cam_id
    return best_cam


def angle_to_camera_id(angle_rad: float) -> int:
    if -math.pi / 4 <= angle_rad < math.pi / 4:
        return 1
    if math.pi / 4 <= angle_rad < 3 * math.pi / 4:
        return 3
    if angle_rad >= 3 * math.pi / 4 or angle_rad < -3 * math.pi / 4:
        return 0
    return 2


def extract_metric_polygon(annotation: Dict[str, object], resolution: int, range_max: float) -> List[List[float]]:
    poly_uv = annotation.get("poly", [])
    if isinstance(poly_uv, list) and len(poly_uv) >= 3:
        points = []
        for p in poly_uv:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                u = finite_float(p[0])
                v = finite_float(p[1])
                if u is not None and v is not None:
                    points.append([u, v])
        if len(points) >= 3:
            metric = project_pixels_to_metric_xy(points, resolution, range_max)
            metric = order_polygon_clockwise(metric)
            if polygon_area(metric) > EPS:
                return metric

    rbox = annotation.get("rbox_xywhr")
    metric = rbox_xywhr_to_metric_polygon(rbox if isinstance(rbox, (list, tuple)) else [])
    if len(metric) >= 3 and polygon_area(metric) > EPS:
        return metric
    return []


def choose_center_xy(annotation: Dict[str, object], metric_poly: Sequence[Sequence[float]]) -> Tuple[float, float]:
    rbox = annotation.get("rbox_xywhr")
    if isinstance(rbox, (list, tuple)) and len(rbox) == 5:
        cx = finite_float(rbox[0])
        cy = finite_float(rbox[1])
        if cx is not None and cy is not None:
            return float(cx), float(cy)
    if metric_poly:
        cx = sum(float(p[0]) for p in metric_poly) / len(metric_poly)
        cy = sum(float(p[1]) for p in metric_poly) / len(metric_poly)
        return cx, cy
    return 0.0, 0.0


def get_tower_objects(payload: object, tower_id: str) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    towers = payload.get("towers")
    if not isinstance(towers, dict):
        return {}
    tower_entry = towers.get(tower_id)
    if not isinstance(tower_entry, dict):
        return {}
    objects = tower_entry.get("objects")
    if not isinstance(objects, dict):
        return {}
    return objects


def build_sample_map(dataset_root: Path) -> Dict[str, Dict[str, object]]:
    sample_map: Dict[str, Dict[str, object]] = {}
    for split, rel in SPLIT_TO_JSONL.items():
        path = dataset_root / rel
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = rec.get("sample_id")
                if isinstance(sid, str):
                    sample_map[sid] = rec
    return sample_map


def load_issue_rows(issue_jsonl: Path, issue_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if issue_jsonl.is_file():
        with issue_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    elif issue_csv.is_file():
        with issue_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(dict(row))
    return rows


def camera_stats_from_object(raw_obj: Optional[Dict[str, object]]) -> Tuple[List[CamBoxStatus], int]:
    statuses: List[CamBoxStatus] = []
    valid_areas = []
    cams = raw_obj.get("cams") if isinstance(raw_obj, dict) else None
    if not isinstance(cams, dict):
        for _ in range(4):
            statuses.append(CamBoxStatus(False, "raw_primary_missing", None, 0.0, 0.0, 0.0))
        return statuses, NONE_CAMERA_ID
    for cam_name in CAMERA_NAMES:
        st = analyze_cam_entry(cams.get(cam_name))
        statuses.append(st)
        valid_areas.append(st.area if st.valid else -1.0)
    best_cam = NONE_CAMERA_ID
    best_area = -1.0
    for cam_id in range(4):
        if valid_areas[cam_id] > best_area:
            best_area = valid_areas[cam_id]
            best_cam = cam_id
    if best_area <= 0.0:
        best_cam = NONE_CAMERA_ID
    return statuses, best_cam


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root
    issue_csv: Path = args.issue_csv
    issue_jsonl: Path = args.issue_jsonl
    report_path: Path = args.report_path
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_map = build_sample_map(dataset_root)
    issue_rows = load_issue_rows(issue_jsonl=issue_jsonl, issue_csv=issue_csv)
    if not issue_rows:
        raise RuntimeError("No issue rows found.")

    ann_cache: Dict[Path, Dict[str, object]] = {}
    gt_only_cache: Dict[Path, Dict[str, object]] = {}
    gt_full_cache: Dict[Path, Dict[str, object]] = {}
    opv2v_cache: Dict[Path, Dict[str, object]] = {}

    details: List[Dict[str, object]] = []

    stats_any_valid_only = 0
    stats_primary_valid_only = 0
    stats_non_primary_valid_only = 0

    stats_any_valid_full = 0
    stats_primary_valid_full = 0
    stats_non_primary_valid_full = 0

    other_cam_dist: Counter = Counter()
    boundary_case_dist: Counter = Counter()
    center_vs_rawbest_mismatch = 0
    center_vs_rawbest_match = 0

    projection_success_count = 0
    projection_failed_reason: Counter = Counter()
    final_reason_dist: Counter = Counter()

    gt_only_missing_obj_count = 0
    gt_only_obj_exists_count = 0

    for row in issue_rows:
        sample_id = str(row.get("sample_id", ""))
        ann_index = int(row.get("ann_index", 0))

        sample = sample_map.get(sample_id)
        if not isinstance(sample, dict):
            continue

        ann_path = Path(str(sample.get("annotation_json_path", "")))
        gt_only_path = Path(str(sample.get("gt_filter_only_yaw_path", "")))
        gt_full_path = Path(str(sample.get("gt_filter_path", "")))
        opv2v_path = Path(str(sample.get("opv2v_yaml_path", "")))
        tower_id = str(sample.get("tower_id", ""))

        if ann_path not in ann_cache:
            ann_cache[ann_path] = load_json(ann_path)
        if gt_only_path not in gt_only_cache:
            gt_only_cache[gt_only_path] = load_json(gt_only_path)
        if gt_full_path not in gt_full_cache:
            gt_full_cache[gt_full_path] = load_json(gt_full_path)
        if opv2v_path not in opv2v_cache:
            opv2v_cache[opv2v_path] = load_yaml(opv2v_path)

        ann_payload = ann_cache[ann_path]
        ann_list = ann_payload.get("annotations", [])
        if not isinstance(ann_list, list) or ann_index < 0 or ann_index >= len(ann_list):
            continue
        ann = ann_list[ann_index]
        if not isinstance(ann, dict):
            continue

        instance_name = str(ann.get("instance_name", ""))
        gt_primary = int(ann.get("gt_primary_camera", row.get("gt_primary_camera", NONE_CAMERA_ID)))

        objects_only = get_tower_objects(gt_only_cache[gt_only_path], tower_id)
        objects_full = get_tower_objects(gt_full_cache[gt_full_path], tower_id)
        raw_obj_only = objects_only.get(instance_name) if isinstance(objects_only, dict) else None
        raw_obj_full = objects_full.get(instance_name) if isinstance(objects_full, dict) else None
        if isinstance(raw_obj_only, dict):
            gt_only_obj_exists_count += 1
        else:
            gt_only_missing_obj_count += 1

        cam_status_only, raw_best_cam_only = camera_stats_from_object(raw_obj_only if isinstance(raw_obj_only, dict) else None)
        cam_status_full, raw_best_cam_full = camera_stats_from_object(raw_obj_full if isinstance(raw_obj_full, dict) else None)

        any_valid_only = any(st.valid for st in cam_status_only)
        any_valid_full = any(st.valid for st in cam_status_full)
        primary_valid_only = bool(0 <= gt_primary < 4 and cam_status_only[gt_primary].valid)
        primary_valid_full = bool(0 <= gt_primary < 4 and cam_status_full[gt_primary].valid)
        non_primary_valid_only = any(
            cam_status_only[i].valid for i in range(4) if i != gt_primary
        ) if 0 <= gt_primary < 4 else False
        non_primary_valid_full = any(
            cam_status_full[i].valid for i in range(4) if i != gt_primary
        ) if 0 <= gt_primary < 4 else False

        stats_any_valid_only += int(any_valid_only)
        stats_primary_valid_only += int(primary_valid_only)
        stats_non_primary_valid_only += int(non_primary_valid_only)

        stats_any_valid_full += int(any_valid_full)
        stats_primary_valid_full += int(primary_valid_full)
        stats_non_primary_valid_full += int(non_primary_valid_full)

        resolution = int(ann_payload.get("resolution", 1536))
        range_max = float(ann_payload.get("range_max", 2000.0))
        metric_poly = extract_metric_polygon(ann, resolution, range_max)
        center_x, center_y = choose_center_xy(ann, metric_poly)
        center_angle = math.atan2(center_y, center_x)
        hard_sector_center_cam = angle_to_camera_id(center_angle)
        areas = sector_overlap_areas(metric_poly) if metric_poly else {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        area_sum = float(sum(areas.values()))
        main_area = max(float(v) for v in areas.values()) if areas else 0.0
        main_ratio = main_area / max(area_sum, 1e-6)
        crosses_multi = sum(1 for v in areas.values() if v > 1e-6) > 1
        boundary_case = bool((main_ratio < 0.8) or crosses_multi)

        # Requested section (2): primary missing but other camera has box.
        if (not primary_valid_only) and non_primary_valid_only:
            other_valid_ids = [i for i in range(4) if i != gt_primary and cam_status_only[i].valid]
            for cam_id in other_valid_ids:
                other_cam_dist[cam_id] += 1
            boundary_case_dist["boundary" if boundary_case else "non_boundary"] += 1
            if raw_best_cam_only != NONE_CAMERA_ID:
                if hard_sector_center_cam != raw_best_cam_only:
                    center_vs_rawbest_mismatch += 1
                else:
                    center_vs_rawbest_match += 1

        # Primary reason in gt_filter_only_yaw source.
        if 0 <= gt_primary < 4:
            primary_reason = cam_status_only[gt_primary].reason
        else:
            primary_reason = "unknown"

        # Projection try on primary camera, using gt_filter_only_yaw object.
        proj_success = False
        proj_fail_bucket = "unknown"
        if not isinstance(raw_obj_only, dict):
            proj_fail_bucket = "raw_primary_missing"
        else:
            points_xyz, points_status = build_cuboid_points(raw_obj_only, ann)
            if points_xyz is None:
                proj_fail_bucket = "unknown"
            else:
                canonical_cam = CAMERA_ID_TO_NAME.get(gt_primary, "Back")
                proj_box, proj_status = project_cuboid_box(
                    points_xyz=points_xyz,
                    opv2v_payload=opv2v_cache[opv2v_path] if isinstance(opv2v_cache[opv2v_path], dict) else {},
                    canonical_cam=canonical_cam,
                )
                if proj_box is not None:
                    proj_success = True
                else:
                    proj_fail_bucket = projection_reason_bucket(proj_status)

        if proj_success:
            projection_success_count += 1
        else:
            projection_failed_reason[proj_fail_bucket] += 1

        if primary_reason in FAIL_REASONS:
            final_reason = primary_reason
        else:
            final_reason = "unknown"
        if final_reason == "raw_primary_missing" and proj_success:
            final_reason = "unknown"
        if final_reason in ("raw_primary_invalid_nan", "raw_primary_clipped_empty", "raw_primary_too_small"):
            if not proj_success and proj_fail_bucket in ("projection_box_outside_image", "projection_invalid_depth"):
                final_reason = proj_fail_bucket
        final_reason_dist[final_reason] += 1

        details.append(
            {
                "sample_id": sample_id,
                "ann_index": ann_index,
                "instance_name": instance_name,
                "gt_primary_camera": gt_primary,
                "hard_sector_center_camera": hard_sector_center_cam,
                "boundary_case": boundary_case,
                "raw_best_camera_gt_filter_only_yaw": raw_best_cam_only,
                "raw_best_camera_gt_filter": raw_best_cam_full,
                "any_camera_has_valid_box": bool(any_valid_only),
                "primary_camera_has_valid_box": bool(primary_valid_only),
                "non_primary_camera_has_valid_box": bool(non_primary_valid_only),
                "any_camera_has_valid_box_gt_filter": bool(any_valid_full),
                "primary_camera_has_valid_box_gt_filter": bool(primary_valid_full),
                "non_primary_camera_has_valid_box_gt_filter": bool(non_primary_valid_full),
                "primary_failure_reason": primary_reason,
                "projection_success": bool(proj_success),
                "projection_failed_reason": proj_fail_bucket if not proj_success else "",
                "final_failure_reason": final_reason,
                "cam_valid_flags_gt_filter_only_yaw": [int(st.valid) for st in cam_status_only],
                "cam_reason_gt_filter_only_yaw": [st.reason for st in cam_status_only],
                "cam_valid_flags_gt_filter": [int(st.valid) for st in cam_status_full],
                "cam_reason_gt_filter": [st.reason for st in cam_status_full],
            }
        )

    total = len(details)
    sec2_total = sum(other_cam_dist.values())
    sec2_samples = sum(
        1
        for d in details
        if (not d["primary_camera_has_valid_box"]) and d["non_primary_camera_has_valid_box"]
    )

    # Ensure all requested reason buckets exist.
    for key in FAIL_REASONS:
        _ = final_reason_dist[key]

    summary = {
        "total_issue_count": total,
        "source_for_main_stats": "gt_filter_only_yaw",
        "any_camera_has_valid_box": int(stats_any_valid_only),
        "primary_camera_has_valid_box": int(stats_primary_valid_only),
        "non_primary_camera_has_valid_box": int(stats_non_primary_valid_only),
        "sec2_primary_missing_but_other_has_box_samples": int(sec2_samples),
        "sec2_other_camera_id_dist": {str(k): int(v) for k, v in sorted(other_cam_dist.items())},
        "sec2_boundary_case_dist": {k: int(v) for k, v in boundary_case_dist.items()},
        "sec2_hard_center_vs_raw_best_mismatch": int(center_vs_rawbest_mismatch),
        "sec2_hard_center_vs_raw_best_match": int(center_vs_rawbest_match),
        "projection_success_count": int(projection_success_count),
        "projection_failed_reason": {k: int(v) for k, v in sorted(projection_failed_reason.items())},
        "final_failure_reason_dist": {k: int(v) for k, v in sorted(final_reason_dist.items())},
        "cross_check_gt_filter": {
            "any_camera_has_valid_box": int(stats_any_valid_full),
            "primary_camera_has_valid_box": int(stats_primary_valid_full),
            "non_primary_camera_has_valid_box": int(stats_non_primary_valid_full),
        },
        "gt_filter_only_yaw_object_exists_count": int(gt_only_obj_exists_count),
        "gt_filter_only_yaw_object_missing_count": int(gt_only_missing_obj_count),
    }

    # Write debug artifacts.
    summary_path = output_dir / "summary.json"
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    details_jsonl = output_dir / "diagnosis_details.jsonl"
    with details_jsonl.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    details_csv = output_dir / "diagnosis_details.csv"
    fieldnames = [
        "sample_id",
        "ann_index",
        "instance_name",
        "gt_primary_camera",
        "hard_sector_center_camera",
        "boundary_case",
        "raw_best_camera_gt_filter_only_yaw",
        "raw_best_camera_gt_filter",
        "any_camera_has_valid_box",
        "primary_camera_has_valid_box",
        "non_primary_camera_has_valid_box",
        "any_camera_has_valid_box_gt_filter",
        "primary_camera_has_valid_box_gt_filter",
        "non_primary_camera_has_valid_box_gt_filter",
        "primary_failure_reason",
        "projection_success",
        "projection_failed_reason",
        "final_failure_reason",
        "cam_valid_flags_gt_filter_only_yaw",
        "cam_reason_gt_filter_only_yaw",
        "cam_valid_flags_gt_filter",
        "cam_reason_gt_filter",
    ]
    with details_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in details:
            writer.writerow(row)

    # Render markdown report.
    lines: List[str] = []
    lines.append("# Primary Camera No-Box Diagnosis")
    lines.append("")
    lines.append(f"- total_issue_count: **{total}**")
    lines.append("- main_stats_source: `gt_filter_only_yaw` (与 None-fix 脚本一致)")
    lines.append("- cross_check_source: `gt_filter` (用于判断是否存在可恢复 raw box)")
    lines.append("")
    lines.append("## 1. 四路 raw box 可用性（gt_filter_only_yaw）")
    lines.append("")
    lines.append(f"- any_camera_has_valid_box: **{stats_any_valid_only} / {total}**")
    lines.append(f"- primary_camera_has_valid_box: **{stats_primary_valid_only} / {total}**")
    lines.append(f"- non_primary_camera_has_valid_box: **{stats_non_primary_valid_only} / {total}**")
    lines.append("")
    lines.append("## 2. primary 无 box、其它相机有 box（gt_filter_only_yaw）")
    lines.append("")
    lines.append(f"- 样本数: **{sec2_samples}**")
    lines.append("")
    lines.append("| other_camera_id | count |")
    lines.append("| --- | ---: |")
    if other_cam_dist:
        for cam_id, count in sorted(other_cam_dist.items()):
            lines.append(f"| {cam_id}({CAMERA_ID_TO_NAME.get(cam_id, 'Unknown')}) | {count} |")
    else:
        lines.append("| (empty) | 0 |")
    lines.append("")
    lines.append("| boundary_case | count |")
    lines.append("| --- | ---: |")
    if boundary_case_dist:
        for k in ["boundary", "non_boundary"]:
            lines.append(f"| {k} | {int(boundary_case_dist.get(k, 0))} |")
    else:
        lines.append("| (empty) | 0 |")
    lines.append("")
    lines.append(
        f"- hard_sector_center 与 raw_best_camera 不一致: **{center_vs_rawbest_mismatch}**"
    )
    lines.append(
        f"- hard_sector_center 与 raw_best_camera 一致: **{center_vs_rawbest_match}**"
    )
    lines.append("")
    lines.append("## 3. primary 相机 3D cuboid projection 尝试")
    lines.append("")
    lines.append(f"- projection_success_count: **{projection_success_count}**")
    lines.append("- projection_failed_reason 分布：")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("| --- | ---: |")
    if projection_failed_reason:
        for reason, count in sorted(projection_failed_reason.items()):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| (empty) | 0 |")
    lines.append("")
    lines.append("## 4. 失败原因分布（按请求口径）")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("| --- | ---: |")
    for reason in FAIL_REASONS:
        lines.append(f"| {reason} | {int(final_reason_dist.get(reason, 0))} |")
    lines.append("")
    lines.append("## 5. 交叉核查（gt_filter）")
    lines.append("")
    lines.append(
        f"- any_camera_has_valid_box(gt_filter): **{stats_any_valid_full} / {total}**"
    )
    lines.append(
        f"- primary_camera_has_valid_box(gt_filter): **{stats_primary_valid_full} / {total}**"
    )
    lines.append(
        f"- non_primary_camera_has_valid_box(gt_filter): **{stats_non_primary_valid_full} / {total}**"
    )
    lines.append(
        f"- gt_filter_only_yaw object_missing: **{gt_only_missing_obj_count} / {total}**"
    )
    lines.append("")
    lines.append("结论：`gt_filter_only_yaw` 与 `gt_filter` 的 object 可见性存在显著差异，171 个问题样本主要由此触发。")
    lines.append("")
    lines.append("## 6. 是否应 fallback 到 raw_best_camera（建议）")
    lines.append("")
    if sec2_samples > (0.5 * total):
        lines.append("- 若仅看当前问题集，`primary 无 box 但其它相机有 box` 占多数。")
        lines.append("- 建议在数据构建阶段增加 `raw_best_camera` fallback（仅在 primary 无有效 box 时触发）。")
    else:
        lines.append("- 当前问题集里，`primary 无 box 但其它相机有 box` **不是多数**。")
        lines.append("- 暂不建议把主策略改为 `raw_best_camera` fallback。")
        lines.append("- 更推荐修复数据构建/修复脚本的 raw 源选择：在 `gt_filter_only_yaw` 缺对象时，回退到 `gt_filter`（或保留原有非空监督），避免把有效 2D box 覆盖为全 0。")
    lines.append("")
    lines.append("## 7. 明细文件")
    lines.append("")
    lines.append(f"- summary: `{summary_path}`")
    lines.append(f"- details csv: `{details_csv}`")
    lines.append(f"- details jsonl: `{details_jsonl}`")
    lines.append("")

    ensure_parent(report_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "report_path": str(report_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "total_issue_count": total,
            "any_camera_has_valid_box": stats_any_valid_only,
            "primary_camera_has_valid_box": stats_primary_valid_only,
            "non_primary_camera_has_valid_box": stats_non_primary_valid_only,
            "projection_success_count": projection_success_count,
            "sec2_samples": sec2_samples,
            "cross_check_primary_camera_has_valid_box_gt_filter": stats_primary_valid_full,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

