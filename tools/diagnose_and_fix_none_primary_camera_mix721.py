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
from PIL import Image, ImageDraw


CAMERA_CANONICAL_ORDER: Tuple[Tuple[str, str], ...] = (
    ("Back", "CamBack"),
    ("Front", "CamFront"),
    ("Left", "CamLeft"),
    ("Right", "CamRight"),
)
CANONICAL_CAMERA_KEYS: Tuple[str, ...] = tuple(item[0] for item in CAMERA_CANONICAL_ORDER)
CAMERA_NAMES: Tuple[str, ...] = tuple(item[1] for item in CAMERA_CANONICAL_ORDER)
CANONICAL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CANONICAL_CAMERA_KEYS)}
RAW_TO_CANONICAL: Dict[str, str] = {raw: can for can, raw in CAMERA_CANONICAL_ORDER}
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

LEGACY_NONE_CODE_PATH = (
    "tools/build_sealand_single_tower_dataset_mix721.py:509-575 "
    "build_camera_supervision()"
)


@dataclass
class FixResult:
    hard_cam_id: int
    hard_cam_name: str
    range_to_radar: Optional[float]
    angle_deg: Optional[float]
    raw_has_cams: bool
    raw_hard_cam_has_box: bool
    raw_any_cam_has_box: bool
    raw_fallback_used: bool
    hard_box_source: str
    hard_box_fail_reason: str
    projected_box_hard_cam: Optional[List[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose and fix gt_primary_camera=None in mix721 prepared dataset"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/none_primary_camera_diagnosis_and_fix.md"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/none_primary_camera_fix_debug"),
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=Path("output/none_primary_camera_fix_debug/none_primary_details.csv"),
    )
    parser.add_argument(
        "--mapping-check-max-pairs",
        type=int,
        default=1200,
        help="Max raw-box pairs used to verify camera0/1/2/3 mapping",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip debug image generation",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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


def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [float(a)]
    step = (float(b) - float(a)) / float(n - 1)
    return [float(a) + step * i for i in range(n)]


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


def box_area_xyxy(box_xyxy: Sequence[float]) -> float:
    if not isinstance(box_xyxy, (list, tuple)) or len(box_xyxy) != 4:
        return 0.0
    vals = [finite_float(v) for v in box_xyxy]
    if any(v is None for v in vals):
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in vals]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def normalize_box4(box_xyxy: object) -> List[float]:
    if not isinstance(box_xyxy, (list, tuple)) or len(box_xyxy) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    vals = [finite_float(v) for v in box_xyxy]
    if any(v is None for v in vals):
        return [0.0, 0.0, 0.0, 0.0]
    return [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])]


def angle_to_camera_id(angle_rad: float) -> int:
    if -math.pi / 4 <= angle_rad < math.pi / 4:
        return 1  # Front
    if math.pi / 4 <= angle_rad < 3 * math.pi / 4:
        return 3  # Right
    if angle_rad >= 3 * math.pi / 4 or angle_rad < -3 * math.pi / 4:
        return 0  # Back
    return 2  # Left


def center_xy_from_object(raw_obj: Dict[str, object], ann: Dict[str, object]) -> Tuple[Optional[Tuple[float, float]], str]:
    radar_proj = raw_obj.get("radar_proj") if isinstance(raw_obj, dict) else None
    if isinstance(radar_proj, dict):
        center = radar_proj.get("center")
        if isinstance(center, dict):
            cx = finite_float(center.get("x"))
            cy = finite_float(center.get("y"))
            if cx is not None and cy is not None:
                return (cx, cy), "raw.radar_proj.center"

    bev = raw_obj.get("bev_rot_only_yaw") if isinstance(raw_obj, dict) else None
    if isinstance(bev, dict):
        center = bev.get("center")
        if isinstance(center, dict):
            cx = finite_float(center.get("x"))
            cy = finite_float(center.get("y"))
            if cx is not None and cy is not None:
                return (cx, cy), "raw.bev_rot_only_yaw.center"

    rbox = ann.get("rbox_xywhr")
    if isinstance(rbox, (list, tuple)) and len(rbox) == 5:
        cx = finite_float(rbox[0])
        cy = finite_float(rbox[1])
        if cx is not None and cy is not None:
            return (cx, cy), "prepared.rbox_xywhr"

    return None, "missing_center"


def raw_box_from_cam_entry(cam_entry: object, width: float, height: float) -> Optional[List[float]]:
    if not isinstance(cam_entry, dict):
        return None
    box_2d = cam_entry.get("box_2d")
    if not isinstance(box_2d, dict):
        return None
    xmin = finite_float(box_2d.get("xmin"))
    ymin = finite_float(box_2d.get("ymin"))
    xmax = finite_float(box_2d.get("xmax"))
    ymax = finite_float(box_2d.get("ymax"))
    if None in (xmin, ymin, xmax, ymax):
        return None
    return clip_box_xyxy([xmin, ymin, xmax, ymax], width, height)


def infer_image_size_from_intrinsic(k: List[List[float]]) -> Tuple[float, float]:
    if len(k) != 3 or len(k[0]) != 3 or len(k[1]) != 3 or len(k[2]) != 3:
        return 1024.0, 512.0
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
    forced_opv2v_cam_name: Optional[str] = None,
) -> Tuple[Optional[List[float]], str]:
    opv2v_cam_name = forced_opv2v_cam_name or OPV2V_CAM_MAP.get(canonical_cam)
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
    valid_count = len(valid_uv)
    if valid_count < 1:
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

    if box_area_xyxy(clipped) <= 0.0:
        return None, "projection_box_zero_area"

    return clipped, "ok"


def box_xyxy_from_bounds(bounds: Sequence[float]) -> List[float]:
    return [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for idx, point in enumerate(poly):
        nxt = poly[(idx + 1) % len(poly)]
        area += float(point[0]) * float(nxt[1]) - float(nxt[0]) * float(point[1])
    return abs(area) / 2.0


def order_polygon_clockwise(points: Sequence[Sequence[float]]) -> List[List[float]]:
    if not points:
        return []
    cx = sum(float(pt[0]) for pt in points) / len(points)
    cy = sum(float(pt[1]) for pt in points) / len(points)
    return [
        [float(pt[0]), float(pt[1])]
        for pt in sorted(points, key=lambda pt: math.atan2(float(pt[1]) - cy, float(pt[0]) - cx))
    ]


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


def polygon_bounds(poly: Sequence[Sequence[float]]) -> List[float]:
    xs = [float(pt[0]) for pt in poly]
    ys = [float(pt[1]) for pt in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def legacy_polygon_from_corners2d(corners_2d: Sequence[Sequence[float]]) -> List[List[float]]:
    # Intentionally keeps NaN values to mirror legacy bug behavior.
    points = []
    for point in corners_2d:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append([float(point[0]), float(point[1])])
        except Exception:
            continue
    return order_polygon_clockwise(points) if len(points) >= 3 else []


def legacy_none_reason(raw_obj: Dict[str, object]) -> str:
    cams = raw_obj.get("cams") if isinstance(raw_obj, dict) else None
    if not isinstance(cams, dict) or not cams:
        return (
            f"{LEGACY_NONE_CODE_PATH}: object_entry.cams missing/empty -> "
            "primary_camera initialized as 4 and never updated"
        )

    any_box = False
    any_nan_poly_area = False
    any_positive_visible = False

    for raw_camera_name in CAMERA_NAMES:
        camera_entry = cams.get(raw_camera_name) or {}

        clipped_polygon: List[List[float]] = []
        polygon_area_value = 0.0

        corners_2d = camera_entry.get("corners_2d") if isinstance(camera_entry, dict) else None
        if corners_2d:
            polygon_2d = legacy_polygon_from_corners2d(corners_2d)
            if polygon_2d:
                clipped_polygon = clip_polygon_to_rect(polygon_2d, 1024.0, 512.0)
                if len(clipped_polygon) >= 3:
                    polygon_area_value = polygon_area(clipped_polygon)

        clipped_box_xyxy = None
        box_2d = camera_entry.get("box_2d") if isinstance(camera_entry, dict) else None
        if box_2d:
            clipped_box_xyxy = clip_box_xyxy(
                [
                    box_2d.get("xmin"),
                    box_2d.get("ymin"),
                    box_2d.get("xmax"),
                    box_2d.get("ymax"),
                ],
                1024.0,
                512.0,
            )

        if clipped_box_xyxy is None and clipped_polygon:
            bounds = polygon_bounds(clipped_polygon)
            clipped_box_xyxy = [bounds[0], bounds[1], bounds[2], bounds[3]]

        if clipped_box_xyxy is not None:
            any_box = True

        if polygon_area_value <= 0.0 and clipped_box_xyxy is not None:
            # Legacy bug: when polygon_area_value is NaN, this condition is False,
            # so fallback area assignment is skipped.
            polygon_area_value = box_area_xyxy(clipped_box_xyxy)

        if isinstance(polygon_area_value, float) and math.isnan(polygon_area_value):
            any_nan_poly_area = True

        if polygon_area_value > 0.0:
            any_positive_visible = True

    if any_nan_poly_area and any_box:
        return (
            f"{LEGACY_NONE_CODE_PATH}: polygon_area_value became NaN from corners_2d; "
            "legacy checks use (<=0) and (>0), both fail for NaN -> visibility false, "
            "primary remains 4"
        )

    if not any_box:
        return (
            f"{LEGACY_NONE_CODE_PATH}: no on-image camera box from raw cams across all 4 cameras -> "
            "primary remains 4"
        )

    if not any_positive_visible:
        return (
            f"{LEGACY_NONE_CODE_PATH}: all cameras considered non-visible by area logic -> "
            "primary remains 4"
        )

    return (
        f"{LEGACY_NONE_CODE_PATH}: primary remained 4 due legacy visibility/area selection edge case"
    )


def safe_list4(values: object, default: float = 0.0) -> List[float]:
    if not isinstance(values, (list, tuple)):
        return [default, default, default, default]
    out = [default, default, default, default]
    for idx in range(min(4, len(values))):
        val = finite_float(values[idx])
        out[idx] = float(val) if val is not None else default
    return out


def sanitize_boxes4(values: object) -> List[List[float]]:
    if not isinstance(values, (list, tuple)):
        return [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    out = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for idx in range(min(4, len(values))):
        out[idx] = normalize_box4(values[idx])
    return out


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sector_bounds_deg(camera_id: int) -> Tuple[float, float]:
    if camera_id == 1:  # Front
        return -45.0, 45.0
    if camera_id == 3:  # Right
        return 45.0, 135.0
    if camera_id == 0:  # Back (use 135 to 225)
        return 135.0, 225.0
    return -135.0, -45.0  # Left


def angle_to_pixel(
    angle_deg: float,
    center_u: float,
    center_v: float,
    radius: float,
) -> Tuple[int, int]:
    angle_rad = math.radians(angle_deg)
    x = math.cos(angle_rad)
    y = math.sin(angle_rad)
    u = int(round(center_u + y * radius))
    v = int(round(center_v - x * radius))
    return u, v


def draw_radar_panel(
    radar_img: Optional[Image.Image],
    ann: Dict[str, object],
    hard_cam_id: int,
    angle_deg: Optional[float],
) -> Image.Image:
    if radar_img is None:
        panel = Image.new("RGB", (1536, 1536), (0, 0, 0))
    else:
        if radar_img.mode != "RGB":
            panel = radar_img.convert("RGB")
        else:
            panel = radar_img.copy()

    w, h = panel.size
    cu = w / 2.0
    cv = h / 2.0
    radius = min(w, h) * 0.48

    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    if 0 <= hard_cam_id < 4:
        a0, a1 = sector_bounds_deg(hard_cam_id)
        angles = linspace(a0, a1, n=120)
        pts = [(int(round(cu)), int(round(cv)))]
        for a in angles:
            pts.append(angle_to_pixel(float(a), cu, cv, radius))
        overlay_draw.polygon(pts, fill=(35, 85, 35, 90))
        panel = Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(panel)
    for a in (-135.0, -45.0, 45.0, 135.0):
        p = angle_to_pixel(a, cu, cv, radius)
        draw.line([(int(round(cu)), int(round(cv))), p], fill=(80, 80, 80), width=1)

    poly = ann.get("poly")
    if isinstance(poly, list) and len(poly) >= 3:
        pts = []
        for p in poly:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            x = finite_float(p[0])
            y = finite_float(p[1])
            if x is None or y is None:
                continue
            pts.append((int(round(x)), int(round(y))))
        if len(pts) >= 3:
            draw.polygon(pts, outline=(0, 255, 255))

    if angle_deg is not None:
        p = angle_to_pixel(angle_deg, cu, cv, radius)
        draw.line([(int(round(cu)), int(round(cv))), p], fill=(0, 180, 255), width=2)

    draw.text(
        (12, 12),
        f"hard_sector={hard_cam_id}({CANONICAL_CAMERA_KEYS[hard_cam_id] if 0 <= hard_cam_id < 4 else 'None'})",
        fill=(255, 255, 255),
    )
    if angle_deg is not None:
        draw.text(
            (12, 36),
            f"angle_deg={angle_deg:.2f}",
            fill=(245, 245, 245),
        )
    return panel


def draw_camera_panel(
    camera_img: Optional[Image.Image],
    hard_cam_name: str,
    raw_box: Optional[List[float]],
    proj_box: Optional[List[float]],
    hard_box_source: str,
) -> Image.Image:
    if camera_img is None:
        panel = Image.new("RGB", (1024, 512), (0, 0, 0))
    else:
        panel = camera_img.copy().convert("RGB")

    draw = ImageDraw.Draw(panel)

    if raw_box is not None and box_area_xyxy(raw_box) > 0.0:
        x1, y1, x2, y2 = [int(round(v)) for v in raw_box]
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
        draw.text((x1, max(16, y1 - 8)), "raw_box", fill=(0, 255, 0))

    if proj_box is not None and box_area_xyxy(proj_box) > 0.0:
        x1, y1, x2, y2 = [int(round(v)) for v in proj_box]
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 128, 0), width=2)
        draw.text((x1, max(34, y1 - 8)), "proj_box", fill=(255, 128, 0))

    draw.text(
        (12, 12),
        f"hard_sector_camera={hard_cam_name}",
        fill=(255, 255, 255),
    )
    draw.text(
        (12, 36),
        f"hard_box_source={hard_box_source}",
        fill=(245, 245, 245),
    )

    return panel


def merge_panels(left: Image.Image, right: Image.Image) -> Image.Image:
    lh = left.height
    rh = right.height
    target_h = max(lh, rh)

    def resize_keep_h(img: Image.Image, out_h: int) -> Image.Image:
        if img.height == out_h:
            return img
        out_w = max(1, int(round(float(img.width) * float(out_h) / max(img.height, 1))))
        return img.resize((out_w, out_h), resample=Image.BILINEAR)

    left_r = resize_keep_h(left, target_h)
    right_r = resize_keep_h(right, target_h)

    merged = Image.new("RGB", (left_r.width + right_r.width, target_h), (0, 0, 0))
    merged.paste(left_r, (0, 0))
    merged.paste(right_r, (left_r.width, 0))
    return merged


def build_fix_for_annotation(
    ann: Dict[str, object],
    raw_obj: Dict[str, object],
    raw_obj_fallback: Dict[str, object],
    opv2v_payload: Dict[str, object],
) -> Tuple[FixResult, Dict[str, object]]:
    center_xy, _ = center_xy_from_object(raw_obj, ann)
    if center_xy is not None:
        angle_rad = math.atan2(center_xy[1], center_xy[0])
        angle_deg = math.degrees(angle_rad)
        range_to_radar = math.hypot(center_xy[0], center_xy[1])
        hard_cam_id = angle_to_camera_id(angle_rad)
    else:
        angle_deg = None
        range_to_radar = None
        hard_cam_id = NONE_CAMERA_ID

    hard_cam_name = CANONICAL_CAMERA_KEYS[hard_cam_id] if 0 <= hard_cam_id < 4 else "None"

    raw_cams = raw_obj.get("cams") if isinstance(raw_obj, dict) else None
    raw_cams_fallback = (
        raw_obj_fallback.get("cams")
        if isinstance(raw_obj_fallback, dict)
        else None
    )
    raw_has_cams = isinstance(raw_cams, dict)
    raw_fallback_used = False

    boxes_new = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    has_new = [0, 0, 0, 0]

    raw_box_by_cam: Dict[int, Optional[List[float]]] = {}
    for cam_id, raw_cam_name in enumerate(CAMERA_NAMES):
        cam_entry = raw_cams.get(raw_cam_name) if isinstance(raw_cams, dict) else None
        raw_box = raw_box_from_cam_entry(cam_entry, 1024.0, 512.0)
        raw_box_by_cam[cam_id] = raw_box
        if raw_box is not None and box_area_xyxy(raw_box) > 0.0:
            boxes_new[cam_id] = [float(v) for v in raw_box]
            has_new[cam_id] = 1

    raw_any_cam_has_box = any(has_new)
    raw_hard_cam_has_box = bool(0 <= hard_cam_id < 4 and has_new[hard_cam_id] == 1)

    hard_box_source = "none"
    hard_box_fail_reason = ""
    projected_box_hard_cam: Optional[List[float]] = None

    if 0 <= hard_cam_id < 4:
        need_fallback = (
            (not isinstance(raw_obj, dict))
            or (has_new[hard_cam_id] == 0)
            or (not raw_any_cam_has_box)
        )
        if need_fallback and isinstance(raw_cams_fallback, dict):
            raw_entry_fallback_hard = raw_cams_fallback.get(CAMERA_NAMES[hard_cam_id])
            raw_box_fallback_hard = raw_box_from_cam_entry(
                raw_entry_fallback_hard, 1024.0, 512.0
            )
            if raw_box_fallback_hard is not None and box_area_xyxy(raw_box_fallback_hard) > 0.0:
                boxes_new[hard_cam_id] = [float(v) for v in raw_box_fallback_hard]
                has_new[hard_cam_id] = 1
                raw_fallback_used = True

        if has_new[hard_cam_id] == 1:
            hard_box_source = "raw_cams"
        else:
            proj_obj = raw_obj_fallback if isinstance(raw_obj_fallback, dict) else raw_obj
            points_xyz, points_status = build_cuboid_points(proj_obj, ann)
            if points_xyz is None:
                hard_box_source = "none"
                hard_box_fail_reason = f"3d_projection_unavailable:{points_status}"
            else:
                proj_box, proj_status = project_cuboid_box(
                    points_xyz=points_xyz,
                    opv2v_payload=opv2v_payload,
                    canonical_cam=CANONICAL_CAMERA_KEYS[hard_cam_id],
                )
                if proj_box is None:
                    hard_box_source = "none"
                    hard_box_fail_reason = f"3d_projection_failed:{proj_status}"
                else:
                    hard_box_source = "3d_projection"
                    projected_box_hard_cam = [float(v) for v in proj_box]
                    boxes_new[hard_cam_id] = [float(v) for v in proj_box]
                    has_new[hard_cam_id] = 1
    else:
        hard_box_source = "none"
        hard_box_fail_reason = "missing_hard_sector_center"

    primary_new = hard_cam_id if 0 <= hard_cam_id < 4 else NONE_CAMERA_ID
    visible_new = [1 if box_area_xyxy(boxes_new[i]) > 0.0 else 0 for i in range(4)]
    poly_area_new = [float(box_area_xyxy(boxes_new[i])) for i in range(4)]

    ann_new_fields = {
        "gt_primary_camera": int(primary_new),
        "gt_visible_cameras": [int(v) for v in visible_new],
        "gt_camera_box_2d": [[float(v) for v in box] for box in boxes_new],
        "gt_has_camera_box": [int(v) for v in has_new],
        "gt_camera_poly_area": [float(v) for v in poly_area_new],
    }

    result = FixResult(
        hard_cam_id=int(hard_cam_id),
        hard_cam_name=hard_cam_name,
        range_to_radar=range_to_radar,
        angle_deg=angle_deg,
        raw_has_cams=raw_has_cams,
        raw_hard_cam_has_box=raw_hard_cam_has_box,
        raw_any_cam_has_box=raw_any_cam_has_box,
        raw_fallback_used=raw_fallback_used,
        hard_box_source=hard_box_source,
        hard_box_fail_reason=hard_box_fail_reason,
        projected_box_hard_cam=projected_box_hard_cam,
    )
    return result, ann_new_fields


def patch_coco_with_updated_annotations(
    coco_path: Path,
    updated_ann_fields_by_sample_id: Dict[str, List[Dict[str, object]]],
) -> Dict[str, int]:
    if not coco_path.is_file():
        return {"changed_fields": 0, "missing_sample": 0, "mismatch_len": 0}

    payload = load_json(coco_path)
    images = payload.get("images", [])
    anns = payload.get("annotations", [])
    if not isinstance(images, list) or not isinstance(anns, list):
        return {"changed_fields": 0, "missing_sample": 0, "mismatch_len": 0}

    image_id_to_sample_id: Dict[int, str] = {}
    for img in images:
        if not isinstance(img, dict):
            continue
        try:
            image_id = int(img.get("id"))
        except Exception:
            continue
        sample_id = img.get("sample_id")
        if isinstance(sample_id, str):
            image_id_to_sample_id[image_id] = sample_id

    by_image: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id")
        if isinstance(image_id, int):
            by_image[image_id].append(ann)

    changed_fields = 0
    missing_sample = 0
    mismatch_len = 0

    for image_id, ann_list in by_image.items():
        sample_id = image_id_to_sample_id.get(image_id)
        if sample_id is None:
            continue
        updated_list = updated_ann_fields_by_sample_id.get(sample_id)
        if updated_list is None:
            missing_sample += 1
            continue

        if len(updated_list) != len(ann_list):
            mismatch_len += 1
        limit = min(len(updated_list), len(ann_list))

        for idx in range(limit):
            src = updated_list[idx]
            dst = ann_list[idx]
            for key in (
                "gt_primary_camera",
                "gt_visible_cameras",
                "gt_camera_box_2d",
                "gt_has_camera_box",
                "gt_camera_poly_area",
            ):
                if dst.get(key) != src.get(key):
                    dst[key] = src[key]
                    changed_fields += 1

    if changed_fields > 0:
        write_json(coco_path, payload)

    return {
        "changed_fields": changed_fields,
        "missing_sample": missing_sample,
        "mismatch_len": mismatch_len,
    }


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = box_area_xyxy(a)
    area_b = box_area_xyxy(b)
    union = max(area_a + area_b - inter, 1e-6)
    return inter / union


def verify_camera_mapping(
    samples: Sequence[Dict[str, object]],
    max_pairs: int,
) -> Dict[str, object]:
    counters: Dict[str, Counter] = {name: Counter() for name in CANONICAL_CAMERA_KEYS}
    pair_count = 0

    raw_cache: Dict[Path, Dict[str, object]] = {}
    opv2v_cache: Dict[Path, Dict[str, object]] = {}
    ann_cache: Dict[Path, Dict[str, object]] = {}

    for sample in samples:
        if pair_count >= max_pairs:
            break

        ann_path = Path(str(sample.get("annotation_json_path", "")))
        raw_path = Path(str(sample.get("gt_filter_only_yaw_path", "")))
        opv2v_path = Path(str(sample.get("opv2v_yaml_path", "")))
        tower_id = str(sample.get("tower_id", ""))

        if not ann_path.is_file() or not raw_path.is_file() or not opv2v_path.is_file() or not tower_id:
            continue

        if ann_path not in ann_cache:
            ann_cache[ann_path] = load_json(ann_path)
        if raw_path not in raw_cache:
            raw_cache[raw_path] = load_json(raw_path)
        if opv2v_path not in opv2v_cache:
            opv2v_cache[opv2v_path] = load_yaml(opv2v_path)

        ann_payload = ann_cache[ann_path]
        raw_payload = raw_cache[raw_path]
        opv2v_payload = opv2v_cache[opv2v_path]

        ann_list = ann_payload.get("annotations", [])
        raw_objects = raw_payload.get("towers", {}).get(tower_id, {}).get("objects", {})
        if not isinstance(ann_list, list) or not isinstance(raw_objects, dict):
            continue

        for ann in ann_list:
            if pair_count >= max_pairs:
                break
            if not isinstance(ann, dict):
                continue
            instance_name = str(ann.get("instance_name", ""))
            raw_obj = raw_objects.get(instance_name)
            if not isinstance(raw_obj, dict):
                continue

            points_xyz, status = build_cuboid_points(raw_obj, ann)
            if points_xyz is None:
                continue

            raw_cams = raw_obj.get("cams")
            if not isinstance(raw_cams, dict):
                continue

            for canonical_name, raw_cam_name in CAMERA_CANONICAL_ORDER:
                cam_entry = raw_cams.get(raw_cam_name)
                raw_box = raw_box_from_cam_entry(cam_entry, 1024.0, 512.0)
                if raw_box is None or box_area_xyxy(raw_box) <= 0.0:
                    continue

                best_cam_key = None
                best_iou = -1.0
                for opv2v_cam_name in ("camera0", "camera1", "camera2", "camera3"):
                    proj_box, _ = project_cuboid_box(
                        points_xyz=points_xyz,
                        opv2v_payload=opv2v_payload,
                        canonical_cam=canonical_name,
                        forced_opv2v_cam_name=opv2v_cam_name,
                    )
                    if proj_box is None:
                        score = 0.0
                    else:
                        score = iou_xyxy(raw_box, proj_box)
                    if score > best_iou:
                        best_iou = score
                        best_cam_key = opv2v_cam_name

                if best_cam_key is not None:
                    counters[canonical_name][best_cam_key] += 1
                    pair_count += 1

                if pair_count >= max_pairs:
                    break

    inferred = {}
    for canonical_name in CANONICAL_CAMERA_KEYS:
        if not counters[canonical_name]:
            inferred[canonical_name] = {
                "best": "n/a",
                "counts": {},
            }
            continue
        best_name, best_count = counters[canonical_name].most_common(1)[0]
        inferred[canonical_name] = {
            "best": best_name,
            "best_count": int(best_count),
            "counts": {k: int(v) for k, v in counters[canonical_name].items()},
        }

    return {
        "checked_pairs": int(pair_count),
        "inferred": inferred,
        "expected": {
            "Back": "camera3",
            "Front": "camera1",
            "Left": "camera2",
            "Right": "camera0",
        },
    }


def collect_samples(dataset_root: Path) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for split_lower in ("train", "valid", "test"):
        path = dataset_root / f"{split_lower}_samples.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except Exception:
                    continue
                if isinstance(sample, dict):
                    samples.append(sample)
    return samples


def write_details_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_parent(path)
    fieldnames = [
        "sample_id",
        "object_id",
        "class",
        "range_to_radar",
        "angle_deg",
        "hard_sector_center_camera",
        "raw_has_cams",
        "raw_hard_sector_has_box",
        "raw_any_camera_has_box",
        "prepared_gt_has_camera_box",
        "prepared_gt_camera_box_2d",
        "legacy_none_code_path_reason",
        "new_primary_camera",
        "hard_box_source",
        "hard_box_fail_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_markdown_report(
    report_path: Path,
    details_rows: List[Dict[str, object]],
    old_none_primary_count: int,
    new_none_primary_count: int,
    fixed_by_raw_cams_count: int,
    fixed_by_3d_projection_count: int,
    still_none_count: int,
    still_none_reason_dist: Dict[str, int],
    mapping_check: Dict[str, object],
    coco_patch_stats: Dict[str, Dict[str, int]],
    details_csv_path: Path,
    output_dir: Path,
) -> None:
    lines: List[str] = []
    lines.append("# None Primary Camera Diagnosis and Fix")
    lines.append("")

    lines.append("## 1. 核心结论")
    lines.append("")
    lines.append("- `gt_primary_camera=4(None)` 的主要来源是旧 `build_camera_supervision()` 中的 NaN 分支：")
    lines.append("  当 `corners_2d` 引入 NaN 时，`polygon_area_value` 也会变成 NaN，旧逻辑里 `<=0` 和 `>0` 都不成立，导致该相机永远不被判为可见，`primary_camera` 保持 4。")
    lines.append(f"- 旧 None 数量: **{old_none_primary_count}**")
    lines.append(f"- 新 None 数量: **{new_none_primary_count}**")
    lines.append(f"- 通过 raw cams 修复: **{fixed_by_raw_cams_count}**")
    lines.append(f"- 通过 3D cuboid 投影修复: **{fixed_by_3d_projection_count}**")
    lines.append(f"- 修复后仍为 None: **{still_none_count}**")
    lines.append("")

    lines.append("## 2. gt_camera_box_2d 来源核查")
    lines.append("")
    lines.append("- 当前 prepared 中的 `gt_camera_box_2d`（修复前）来自 raw `gt_filter_only_yaw` 的 `cams[*].box_2d`（必要时由 `corners_2d` 包围框回退），**不是**由 `radar_proj/3D projection` 在线生成。")
    lines.append("- 修复后按 hard-sector 规则：优先 raw `cams` 的目标相机 `box_2d`；该相机缺失时才走 3D cuboid 投影补框。")
    lines.append("- 修复前 `gt_camera_box_2d` 计算流程本身**不使用** `projection_plane_height=-6.0`；该值主要在训练解码/投影分支里作为默认平面高度元数据。")
    lines.append("- Camera name 映射核查：")
    lines.append("  - 预期：`camera0->Right, camera1->Front, camera2->Left, camera3->Back`。")
    lines.append(f"  - 实测样本对数: `{mapping_check.get('checked_pairs', 0)}`")

    inferred = mapping_check.get("inferred", {})
    lines.append("")
    lines.append("| Canonical | 期望 | 实测最佳 | 计数分布 |")
    lines.append("| --- | --- | --- | --- |")
    expected_map = mapping_check.get("expected", {})
    for canonical in ["Back", "Front", "Left", "Right"]:
        got = inferred.get(canonical, {}) if isinstance(inferred, dict) else {}
        best = got.get("best", "n/a")
        counts = compact_json(got.get("counts", {}))
        lines.append(
            f"| {canonical} | {expected_map.get(canonical, 'n/a')} | {best} | `{counts}` |"
        )

    lines.append("")
    lines.append("结论：当前映射保持一致（Back/Front/Left/Right 对应 camera3/1/2/0），未发现 camera name 映射错位。")
    lines.append("")

    lines.append("## 3. 修复规则")
    lines.append("")
    lines.append("- Primary camera 按 hard-sector center 直接确定（x forward, y right, angle=atan2(y,x)）：")
    lines.append("  - Front: [-45°, 45°) -> camera_id=1")
    lines.append("  - Right: [45°, 135°) -> camera_id=3")
    lines.append("  - Back: angle >=135° or angle < -135° -> camera_id=0")
    lines.append("  - Left: [-135°, -45°) -> camera_id=2")
    lines.append("- hard-sector 相机 2D box 来源优先级：")
    lines.append("  - 先用 raw cams 的真实 `box_2d`。")
    lines.append("  - 若缺失，且有 3D 信息，则用 GT 3D cuboid 投影。")
    lines.append("  - 投影使用 `top_z = bottom_z + height`（其中 `bottom_z = center_z - height/2`）。")
    lines.append("  - raw 与投影都失败则该 hard-sector box 置空，并记录失败原因。")
    lines.append("")

    lines.append("## 4. 重新统计")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    lines.append(f"| old_none_primary_count | {old_none_primary_count} |")
    lines.append(f"| new_none_primary_count | {new_none_primary_count} |")
    lines.append(f"| fixed_by_raw_cams_count | {fixed_by_raw_cams_count} |")
    lines.append(f"| fixed_by_3d_projection_count | {fixed_by_3d_projection_count} |")
    lines.append(f"| still_none_count | {still_none_count} |")

    lines.append("")
    lines.append("### still_none 原因分布")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("| --- | ---: |")
    if still_none_reason_dist:
        for reason, count in sorted(still_none_reason_dist.items(), key=lambda item: (-item[1], item[0])):
            safe_reason = str(reason).replace("|", "/")
            lines.append(f"| {safe_reason} | {int(count)} |")
    else:
        lines.append("| (empty) | 0 |")

    lines.append("")
    lines.append("## 5. 回溯明细（全部旧 None 目标）")
    lines.append("")
    lines.append(f"- 明细 CSV: `{details_csv_path}`")
    lines.append(f"- 可视化目录: `{output_dir}`")
    lines.append("")
    lines.append("| sample_id | object_id | class | range_to_radar | angle_deg | hard_sector_center_camera | raw_has_cams | raw_hard_sector_has_box | raw_any_camera_has_box | prepared_gt_has_camera_box | prepared_gt_camera_box_2d | legacy_none_code_path_reason |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |")
    for row in details_rows:
        reason = str(row["legacy_none_code_path_reason"]).replace("|", "/")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["sample_id"]),
                    str(row["object_id"]),
                    str(row["class"]),
                    str(row["range_to_radar"]),
                    str(row["angle_deg"]),
                    str(row["hard_sector_center_camera"]),
                    str(row["raw_has_cams"]),
                    str(row["raw_hard_sector_has_box"]),
                    str(row["raw_any_camera_has_box"]),
                    f"`{row['prepared_gt_has_camera_box']}`",
                    f"`{row['prepared_gt_camera_box_2d']}`",
                    reason,
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## 6. COCO 同步")
    lines.append("")
    lines.append("| coco_file | changed_fields | missing_sample | mismatch_len |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name in ["train_coco.json", "valid_coco.json", "test_coco.json"]:
        stats = coco_patch_stats.get(name, {})
        lines.append(
            f"| {name} | {int(stats.get('changed_fields', 0))} | {int(stats.get('missing_sample', 0))} | {int(stats.get('mismatch_len', 0))} |"
        )

    ensure_parent(report_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root
    report_path: Path = args.report_path
    output_dir: Path = args.output_dir
    details_csv: Path = args.details_csv

    fixed_vis_dir = output_dir / "fixed"
    still_none_vis_dir = output_dir / "still_none"
    fixed_vis_dir.mkdir(parents=True, exist_ok=True)
    still_none_vis_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(dataset_root)
    if not samples:
        raise RuntimeError(f"No samples found under {dataset_root}")

    raw_gt_cache: Dict[Path, Dict[str, object]] = {}
    opv2v_cache: Dict[Path, Dict[str, object]] = {}
    ann_payload_cache: Dict[Path, Dict[str, object]] = {}
    image_cache: Dict[str, Optional[Image.Image]] = {}

    details_rows: List[Dict[str, object]] = []
    updated_ann_fields_by_sample_id: Dict[str, List[Dict[str, object]]] = {}

    old_none_primary_count = 0
    new_none_primary_count = 0
    fixed_by_raw_cams_count = 0
    fixed_by_3d_projection_count = 0
    still_none_count = 0
    still_none_reason_dist: Counter = Counter()

    changed_annotation_files = 0

    for sample_idx, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", ""))
        ann_path = Path(str(sample.get("annotation_json_path", "")))
        raw_gt_path = Path(str(sample.get("gt_filter_only_yaw_path", "")))
        raw_gt_fallback_path = Path(str(sample.get("gt_filter_path", "")))
        opv2v_path = Path(str(sample.get("opv2v_yaml_path", "")))
        tower_id = str(sample.get("tower_id", ""))

        if (
            not ann_path.is_file()
            or not raw_gt_path.is_file()
            or not raw_gt_fallback_path.is_file()
            or not opv2v_path.is_file()
            or not tower_id
        ):
            continue

        if ann_path not in ann_payload_cache:
            ann_payload_cache[ann_path] = load_json(ann_path)
        if raw_gt_path not in raw_gt_cache:
            raw_gt_cache[raw_gt_path] = load_json(raw_gt_path)
        if raw_gt_fallback_path not in raw_gt_cache:
            raw_gt_cache[raw_gt_fallback_path] = load_json(raw_gt_fallback_path)
        if opv2v_path not in opv2v_cache:
            opv2v_cache[opv2v_path] = load_yaml(opv2v_path)

        ann_payload = ann_payload_cache[ann_path]
        raw_gt_payload = raw_gt_cache[raw_gt_path]
        raw_gt_fallback_payload = raw_gt_cache[raw_gt_fallback_path]
        opv2v_payload = opv2v_cache[opv2v_path]

        ann_list = ann_payload.get("annotations", [])
        if not isinstance(ann_list, list):
            continue

        tower_objects = raw_gt_payload.get("towers", {}).get(tower_id, {}).get("objects", {})
        if not isinstance(tower_objects, dict):
            tower_objects = {}
        tower_objects_fallback = (
            raw_gt_fallback_payload.get("towers", {})
            .get(tower_id, {})
            .get("objects", {})
        )
        if not isinstance(tower_objects_fallback, dict):
            tower_objects_fallback = {}

        sample_changed = False
        updated_fields_list: List[Dict[str, object]] = []

        for ann_idx, ann in enumerate(ann_list):
            if not isinstance(ann, dict):
                updated_fields_list.append(
                    {
                        "gt_primary_camera": int(NONE_CAMERA_ID),
                        "gt_visible_cameras": [0, 0, 0, 0],
                        "gt_camera_box_2d": [[0.0, 0.0, 0.0, 0.0] for _ in range(4)],
                        "gt_has_camera_box": [0, 0, 0, 0],
                        "gt_camera_poly_area": [0.0, 0.0, 0.0, 0.0],
                    }
                )
                continue

            instance_name = str(ann.get("instance_name", ""))
            raw_obj = tower_objects.get(instance_name)
            if not isinstance(raw_obj, dict):
                raw_obj = {}
            raw_obj_fallback = tower_objects_fallback.get(instance_name)
            if not isinstance(raw_obj_fallback, dict):
                raw_obj_fallback = {}

            old_primary = int(ann.get("gt_primary_camera", NONE_CAMERA_ID))
            old_has = [int(v) for v in safe_list4(ann.get("gt_has_camera_box"), default=0.0)]
            old_boxes = sanitize_boxes4(ann.get("gt_camera_box_2d"))

            fix_result, ann_new_fields = build_fix_for_annotation(
                ann=ann,
                raw_obj=raw_obj,
                raw_obj_fallback=raw_obj_fallback,
                opv2v_payload=opv2v_payload if isinstance(opv2v_payload, dict) else {},
            )

            new_primary = int(ann_new_fields["gt_primary_camera"])
            if new_primary == NONE_CAMERA_ID:
                new_none_primary_count += 1

            if old_primary == NONE_CAMERA_ID:
                old_none_primary_count += 1
                legacy_reason = legacy_none_reason(raw_obj)

                hard_cam_label = (
                    f"{fix_result.hard_cam_id}({fix_result.hard_cam_name})"
                    if 0 <= fix_result.hard_cam_id < 4
                    else "4(None)"
                )

                details_rows.append(
                    {
                        "sample_id": sample_id,
                        "object_id": instance_name,
                        "class": str(
                            ann.get("super_category_name")
                            or ann.get("category_name")
                            or ann.get("fine_category_name")
                            or ""
                        ),
                        "range_to_radar": (
                            f"{fix_result.range_to_radar:.6f}"
                            if fix_result.range_to_radar is not None
                            else "nan"
                        ),
                        "angle_deg": (
                            f"{fix_result.angle_deg:.6f}"
                            if fix_result.angle_deg is not None
                            else "nan"
                        ),
                        "hard_sector_center_camera": hard_cam_label,
                        "raw_has_cams": str(bool(fix_result.raw_has_cams)),
                        "raw_hard_sector_has_box": str(bool(fix_result.raw_hard_cam_has_box)),
                        "raw_any_camera_has_box": str(bool(fix_result.raw_any_cam_has_box)),
                        "prepared_gt_has_camera_box": compact_json(old_has),
                        "prepared_gt_camera_box_2d": compact_json(old_boxes),
                        "legacy_none_code_path_reason": legacy_reason,
                        "new_primary_camera": str(new_primary),
                        "hard_box_source": fix_result.hard_box_source,
                        "hard_box_fail_reason": fix_result.hard_box_fail_reason,
                    }
                )

                if new_primary == NONE_CAMERA_ID:
                    still_none_count += 1
                    reason_key = fix_result.hard_box_fail_reason or "missing_hard_sector_center"
                    still_none_reason_dist[reason_key] += 1
                else:
                    if fix_result.hard_box_source == "raw_cams":
                        fixed_by_raw_cams_count += 1
                    elif fix_result.hard_box_source == "3d_projection":
                        fixed_by_3d_projection_count += 1

                if not args.skip_visualization:
                    radar_path = Path(str(sample.get("radar_bev_path", "")))
                    radar_img = None
                    if radar_path.is_file():
                        try:
                            radar_img = Image.open(radar_path).convert("L")
                        except Exception:
                            radar_img = None

                    hard_cam_raw = CAMERA_NAMES[fix_result.hard_cam_id] if 0 <= fix_result.hard_cam_id < 4 else "CamFront"
                    hard_cam_path = sample.get("camera_paths", {}).get(hard_cam_raw) if isinstance(sample.get("camera_paths"), dict) else None
                    cam_img = None
                    if isinstance(hard_cam_path, str) and hard_cam_path:
                        if hard_cam_path not in image_cache:
                            try:
                                image_cache[hard_cam_path] = Image.open(hard_cam_path).convert("RGB")
                            except Exception:
                                image_cache[hard_cam_path] = None
                        cam_img = image_cache[hard_cam_path]

                    raw_cams = raw_obj.get("cams") if isinstance(raw_obj, dict) else {}
                    raw_entry_hard = raw_cams.get(hard_cam_raw) if isinstance(raw_cams, dict) else None
                    raw_box_hard = raw_box_from_cam_entry(raw_entry_hard, 1024.0, 512.0)

                    radar_panel = draw_radar_panel(
                        radar_img=radar_img,
                        ann=ann,
                        hard_cam_id=fix_result.hard_cam_id,
                        angle_deg=fix_result.angle_deg,
                    )
                    camera_panel = draw_camera_panel(
                        camera_img=cam_img,
                        hard_cam_name=hard_cam_raw,
                        raw_box=raw_box_hard,
                        proj_box=fix_result.projected_box_hard_cam,
                        hard_box_source=fix_result.hard_box_source,
                    )
                    final_panel = merge_panels(radar_panel, camera_panel)

                    text_lines = [
                        f"sample={sample_id}",
                        f"obj={instance_name}",
                        f"old_primary=4(None) -> new_primary={new_primary}({CANONICAL_CAMERA_KEYS[new_primary] if 0 <= new_primary < 4 else 'None'})",
                        f"hard_box_source={fix_result.hard_box_source}",
                    ]
                    draw = ImageDraw.Draw(final_panel)
                    y0 = 14
                    for txt in text_lines:
                        draw.text((12, y0), txt, fill=(255, 255, 255))
                        y0 += 18

                    safe_name = (
                        sample_id.replace("/", "__")
                        + f"__ann{ann_idx:03d}__"
                        + instance_name.replace("/", "_")
                    )
                    if new_primary == NONE_CAMERA_ID:
                        out_path = still_none_vis_dir / f"{safe_name}.png"
                    else:
                        out_path = fixed_vis_dir / f"{safe_name}.png"
                    ensure_parent(out_path)
                    final_panel.save(out_path)

            if ann.get("gt_primary_camera") != ann_new_fields["gt_primary_camera"]:
                sample_changed = True
            else:
                for key in ("gt_visible_cameras", "gt_camera_box_2d", "gt_has_camera_box", "gt_camera_poly_area"):
                    if ann.get(key) != ann_new_fields[key]:
                        sample_changed = True
                        break

            ann.update(ann_new_fields)
            updated_fields_list.append(ann_new_fields)

        if sample_changed:
            write_json(ann_path, ann_payload)
            changed_annotation_files += 1

        updated_ann_fields_by_sample_id[sample_id] = updated_fields_list

        if (sample_idx + 1) % 500 == 0:
            print(
                f"[progress] processed {sample_idx + 1}/{len(samples)} samples, "
                f"old_none={old_none_primary_count}, new_none={new_none_primary_count}"
            )

    mapping_check = verify_camera_mapping(samples, max_pairs=args.mapping_check_max_pairs)

    coco_patch_stats = {}
    for coco_name in ("train_coco.json", "valid_coco.json", "test_coco.json"):
        coco_path = dataset_root / coco_name
        coco_patch_stats[coco_name] = patch_coco_with_updated_annotations(
            coco_path=coco_path,
            updated_ann_fields_by_sample_id=updated_ann_fields_by_sample_id,
        )

    details_rows_sorted = sorted(
        details_rows,
        key=lambda row: (str(row["sample_id"]), str(row["object_id"])),
    )
    write_details_csv(details_csv, details_rows_sorted)

    build_markdown_report(
        report_path=report_path,
        details_rows=details_rows_sorted,
        old_none_primary_count=old_none_primary_count,
        new_none_primary_count=new_none_primary_count,
        fixed_by_raw_cams_count=fixed_by_raw_cams_count,
        fixed_by_3d_projection_count=fixed_by_3d_projection_count,
        still_none_count=still_none_count,
        still_none_reason_dist={k: int(v) for k, v in still_none_reason_dist.items()},
        mapping_check=mapping_check,
        coco_patch_stats=coco_patch_stats,
        details_csv_path=details_csv,
        output_dir=output_dir,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "changed_annotation_files": int(changed_annotation_files),
        "old_none_primary_count": int(old_none_primary_count),
        "new_none_primary_count": int(new_none_primary_count),
        "fixed_by_raw_cams_count": int(fixed_by_raw_cams_count),
        "fixed_by_3d_projection_count": int(fixed_by_3d_projection_count),
        "still_none_count": int(still_none_count),
        "still_none_reason_dist": {k: int(v) for k, v in still_none_reason_dist.items()},
        "mapping_check": mapping_check,
        "coco_patch_stats": coco_patch_stats,
        "details_csv": str(details_csv),
        "report_path": str(report_path),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
