#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


CAMERA_CANONICAL_ORDER: Tuple[Tuple[str, str], ...] = (
    ("Back", "CamBack"),
    ("Front", "CamFront"),
    ("Left", "CamLeft"),
    ("Right", "CamRight"),
)
CANONICAL_CAMERA_KEYS: Tuple[str, ...] = tuple(item[0] for item in CAMERA_CANONICAL_ORDER)
CAMERA_NAMES: Tuple[str, ...] = tuple(item[1] for item in CAMERA_CANONICAL_ORDER)
CANONICAL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CANONICAL_CAMERA_KEYS)}
OPV2V_CAM_MAP = {
    "Back": "camera3",
    "Front": "camera1",
    "Left": "camera2",
    "Right": "camera0",
}
CAMERA_CV_FROM_LOCAL = np.asarray(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

RGB_COLOR_BY_LABEL = {
    "raw_box": (0, 255, 0),
    "current_projection_box": (255, 0, 0),  # red in RGB
    "cuboid_8pt_box": (0, 0, 255),  # blue in RGB
    "gt_camera_box_2d": (0, 255, 255),  # cyan in RGB
}

DISTANCE_BINS: Tuple[Tuple[str, float, float], ...] = (
    ("[0,250)", 0.0, 250.0),
    ("[250,500)", 250.0, 500.0),
    ("[500,750)", 500.0, 750.0),
    ("[750,1000)", 750.0, 1000.0),
    ("[1000,1250)", 1000.0, 1250.0),
    ("[1250,1500)", 1250.0, 1500.0),
    ("[1500,1750)", 1500.0, 1750.0),
    ("[1750,2000)", 1750.0, 2000.0),
    ("[2000,inf)", 2000.0, float("inf")),
)
LONG_EDGE_BINS: Tuple[Tuple[str, float, float], ...] = (
    ("[0,16)", 0.0, 16.0),
    ("[16,32)", 16.0, 32.0),
    ("[32,64)", 32.0, 64.0),
    ("[64,128)", 64.0, 128.0),
    ("[128,256)", 128.0, 256.0),
    ("[256,inf)", 256.0, float("inf")),
)


@dataclass
class ProjectionResult:
    box_xyxy: Optional[List[float]]
    status: str
    valid_depth_count: int
    points_total: int


@dataclass
class CurrentCuboidInfo:
    points_xyz: Optional[np.ndarray]
    status: str
    center_xyz: Optional[Tuple[float, float, float]]
    extent_xy: Optional[Tuple[float, float]]
    height_m: Optional[float]
    yaw_rad: Optional[float]
    source_center: str
    source_extent: str
    source_height: str
    source_yaw: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose RouteROI current projection box vs cuboid-8pt projection box"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721"),
    )
    parser.add_argument(
        "--samples-jsonl",
        nargs="+",
        default=[],
        help="Optional jsonl list. If empty, use train/valid/test from --dataset-root.",
    )
    parser.add_argument(
        "--height-prior-path",
        type=Path,
        default=Path("configs/priors/sealand_height_prior_super4.json"),
    )
    parser.add_argument("--height-prior-stat", type=str, default="p75")
    parser.add_argument("--projection-plane-height", type=float, default=-6.0)
    parser.add_argument("--roi-expand-ratio", type=float, default=1.15)
    parser.add_argument("--topk-vis", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/route_roi_3d_projection_box_debug"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/route_roi_3d_projection_box_diagnosis.md"),
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


def write_json(path: Path, payload: object) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def box_area_xyxy(box_xyxy: Optional[Sequence[float]]) -> float:
    if box_xyxy is None or len(box_xyxy) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_width_height(box_xyxy: Optional[Sequence[float]]) -> Tuple[float, float]:
    if box_xyxy is None or len(box_xyxy) != 4:
        return 0.0, 0.0
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


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


def expand_box_xyxy(box_xyxy: Sequence[float], ratio: float, width: float, height: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * max(0.0, x2 - x1) * max(float(ratio), 1e-6)
    half_h = 0.5 * max(0.0, y2 - y1) * max(float(ratio), 1e-6)
    expanded = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
    clipped = clip_box_xyxy(expanded, width, height)
    if clipped is None:
        return [0.0, 0.0, 0.0, 0.0]
    return clipped


def iou_xyxy(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if a is None or b is None:
        return 0.0
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


def angle_diff_mod_pi_rad(a: float, b: float) -> float:
    # Orientation axis is symmetric under 180 degrees.
    d = (float(a) - float(b)) % math.pi
    if d > math.pi / 2.0:
        d = math.pi - d
    return abs(d)


def angle_to_bin(value: float, bins: Sequence[Tuple[str, float, float]]) -> str:
    for name, low, high in bins:
        if low <= value < high:
            return name
    return bins[-1][0] if bins else "all"


def parse_matrix(payload: object, rows: int, cols: int) -> Optional[np.ndarray]:
    if not isinstance(payload, (list, tuple)) or len(payload) != rows:
        return None
    out = np.zeros((rows, cols), dtype=np.float64)
    for r, row in enumerate(payload):
        if not isinstance(row, (list, tuple)) or len(row) != cols:
            return None
        for c, value in enumerate(row):
            fv = finite_float(value)
            if fv is None:
                return None
            out[r, c] = float(fv)
    return out


def infer_image_size_from_intrinsic(k: np.ndarray) -> Tuple[float, float]:
    if k.shape != (3, 3):
        return 1024.0, 512.0
    width = float(max(2.0, k[0, 2] * 2.0))
    height = float(max(2.0, k[1, 2] * 2.0))
    return width, height


def get_camera_matrices(
    opv2v_payload: Dict[str, object], canonical_cam: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float, str]:
    opv2v_cam_name = OPV2V_CAM_MAP.get(canonical_cam)
    if opv2v_cam_name is None:
        return None, None, 0.0, 0.0, "missing_opv2v_cam_mapping"
    cam_meta = opv2v_payload.get(opv2v_cam_name)
    if not isinstance(cam_meta, dict):
        return None, None, 0.0, 0.0, f"missing_opv2v_camera_meta:{opv2v_cam_name}"
    intrinsic = parse_matrix(cam_meta.get("intrinsic"), rows=3, cols=3)
    extrinsic_local = parse_matrix(cam_meta.get("extrinsic"), rows=4, cols=4)
    if intrinsic is None or extrinsic_local is None:
        return None, None, 0.0, 0.0, "invalid_camera_calibration_matrix"
    t_camcv_ego = CAMERA_CV_FROM_LOCAL @ extrinsic_local
    img_w, img_h = infer_image_size_from_intrinsic(intrinsic)
    return intrinsic, t_camcv_ego, img_w, img_h, "ok"


def project_points(
    points_xyz: np.ndarray,
    t_camcv_ego: np.ndarray,
    intrinsic_k: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    homo = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (t_camcv_ego @ homo.T).T[:, :3]
    z_cam = cam[:, 2]
    safe_z = np.where(np.abs(z_cam) < 1e-6, 1e-6, z_cam)
    u = intrinsic_k[0, 0] * cam[:, 0] / safe_z + intrinsic_k[0, 2]
    v = intrinsic_k[1, 1] * cam[:, 1] / safe_z + intrinsic_k[1, 2]
    uv = np.stack([u, v], axis=1)
    return uv, z_cam


def project_points_to_box(
    points_xyz: np.ndarray,
    intrinsic_k: np.ndarray,
    t_camcv_ego: np.ndarray,
    img_w: float,
    img_h: float,
    expand_ratio: float = 1.0,
) -> ProjectionResult:
    if points_xyz.shape[0] <= 0:
        return ProjectionResult(None, "projection_no_points", 0, 0)
    uv, z_cam = project_points(points_xyz, t_camcv_ego, intrinsic_k)
    valid = z_cam > 1e-4
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 1:
        return ProjectionResult(None, "projection_all_points_behind_camera", valid_count, int(points_xyz.shape[0]))
    uv_valid = uv[valid]
    xmin = float(np.min(uv_valid[:, 0]))
    ymin = float(np.min(uv_valid[:, 1]))
    xmax = float(np.max(uv_valid[:, 0]))
    ymax = float(np.max(uv_valid[:, 1]))
    clipped = clip_box_xyxy([xmin, ymin, xmax, ymax], img_w, img_h)
    if clipped is None:
        return ProjectionResult(None, "projection_box_outside_image", valid_count, int(points_xyz.shape[0]))
    if box_area_xyxy(clipped) <= 0.0:
        return ProjectionResult(None, "projection_box_zero_area", valid_count, int(points_xyz.shape[0]))
    if expand_ratio > 1.0 + 1e-8:
        clipped = expand_box_xyxy(clipped, ratio=expand_ratio, width=img_w, height=img_h)
        if box_area_xyxy(clipped) <= 0.0:
            return ProjectionResult(None, "projection_box_zero_area_after_expand", valid_count, int(points_xyz.shape[0]))
    return ProjectionResult([float(v) for v in clipped], "ok", valid_count, int(points_xyz.shape[0]))


def raw_box_from_cam_entry(cam_entry: object, img_w: float, img_h: float) -> Optional[List[float]]:
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
    return clip_box_xyxy([xmin, ymin, xmax, ymax], img_w, img_h)


def safe_box4(value: object) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    vals = [finite_float(v) for v in value]
    if any(v is None for v in vals):
        return None
    box = [float(v) for v in vals]
    if box_area_xyxy(box) <= 0.0:
        return None
    return box


def load_records(paths: Sequence[Path]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    out.append(payload)
    return out


def load_height_prior(path: Path, stat: str) -> Dict[str, float]:
    default_prior = {
        "CargoShip": 46.55,
        "CruiseShip": 69.77,
        "FishingVessel": 34.33,
        "RecreationalBoat": 9.18,
    }
    if not path.is_file():
        return dict(default_prior)
    try:
        payload = load_json(path)
    except Exception:
        return dict(default_prior)
    if not isinstance(payload, dict):
        return dict(default_prior)

    candidate = None
    if isinstance(payload.get("stats"), dict):
        candidate = payload["stats"].get(stat)
    if candidate is None:
        candidate = payload.get(stat)
    if candidate is None:
        by_class = {}
        for key, value in payload.items():
            if isinstance(value, dict) and stat in value:
                by_class[str(key)] = value[stat]
        if by_class:
            candidate = by_class
    if candidate is None and all(not isinstance(v, (dict, list, tuple)) for v in payload.values()):
        candidate = payload

    out = dict(default_prior)
    if isinstance(candidate, dict):
        for key, value in candidate.items():
            fv = finite_float(value)
            if fv is None:
                continue
            out[str(key)] = max(float(fv), 0.5)
    return out


def rbox_to_bottom_corners(rbox_xywhr: Sequence[float]) -> np.ndarray:
    cx, cy, w, h, angle = [float(v) for v in rbox_xywhr]
    half_w = max(float(w), 1e-6) * 0.5
    half_h = max(float(h), 1e-6) * 0.5
    local = np.asarray(
        [[half_w, half_h], [half_w, -half_h], [-half_w, -half_h], [-half_w, half_h]],
        dtype=np.float64,
    )
    ca = math.cos(float(angle))
    sa = math.sin(float(angle))
    rot = np.asarray([[ca, -sa], [sa, ca]], dtype=np.float64)
    rotated = (rot @ local.T).T
    rotated[:, 0] += float(cx)
    rotated[:, 1] += float(cy)
    return rotated


def build_current_cuboid_points(raw_obj: Dict[str, object], ann: Dict[str, object]) -> CurrentCuboidInfo:
    radar_proj = raw_obj.get("radar_proj") if isinstance(raw_obj, dict) else None
    if not isinstance(radar_proj, dict):
        radar_proj = {}
    bev = raw_obj.get("bev_rot_only_yaw") if isinstance(raw_obj, dict) else None
    if not isinstance(bev, dict):
        bev = {}

    center = radar_proj.get("center") if isinstance(radar_proj.get("center"), dict) else None
    source_center = "radar_proj.center"
    if center is None:
        center = bev.get("center") if isinstance(bev.get("center"), dict) else None
        source_center = "bev_rot_only_yaw.center"
    if center is None:
        return CurrentCuboidInfo(None, "missing_center_xyz", None, None, None, None, source_center, "", "", "")

    cx = finite_float(center.get("x"))
    cy = finite_float(center.get("y"))
    cz = finite_float(center.get("z"))
    if None in (cx, cy, cz):
        return CurrentCuboidInfo(None, "invalid_center_xyz", None, None, None, None, source_center, "", "", "")

    extent = radar_proj.get("extent") if isinstance(radar_proj.get("extent"), dict) else None
    source_extent = "radar_proj.extent"
    if extent is None:
        extent = bev.get("extent") if isinstance(bev.get("extent"), dict) else None
        source_extent = "bev_rot_only_yaw.extent"

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
                ex = max(float(rw), 1e-6) * 0.5
                ey = max(float(rh), 1e-6) * 0.5
                source_extent = "ann.rbox_xywhr"
    if ex is None or ey is None:
        return CurrentCuboidInfo(None, "missing_extent_xy", None, None, None, None, source_center, source_extent, "", "")
    ex = max(float(ex), 1e-6)
    ey = max(float(ey), 1e-6)

    height = None
    source_height = ""
    bbox_m = raw_obj.get("bbox_m") if isinstance(raw_obj, dict) else None
    if isinstance(bbox_m, dict):
        height = finite_float(bbox_m.get("H"))
        if height is not None:
            source_height = "bbox_m.H"
    if height is None and isinstance(extent, dict):
        ez = finite_float(extent.get("z"))
        if ez is not None:
            height = abs(float(ez)) * 2.0
            source_height = "extent.z*2"
    if height is None:
        return CurrentCuboidInfo(None, "missing_height", None, None, None, None, source_center, source_extent, source_height, "")
    height = max(float(height), 0.1)

    yaw = finite_float(bev.get("yaw")) if isinstance(bev, dict) else None
    source_yaw = "bev_rot_only_yaw.yaw"
    if yaw is None:
        rbox = ann.get("rbox_xywhr")
        if isinstance(rbox, (list, tuple)) and len(rbox) == 5:
            yaw = finite_float(rbox[4])
            source_yaw = "ann.rbox_xywhr.yaw"
    if yaw is None:
        return CurrentCuboidInfo(
            None,
            "missing_yaw",
            (float(cx), float(cy), float(cz)),
            (float(ex), float(ey)),
            float(height),
            None,
            source_center,
            source_extent,
            source_height,
            source_yaw,
        )

    bottom_z = float(cz) - 0.5 * float(height)
    top_z = bottom_z + float(height)
    ca = math.cos(float(yaw))
    sa = math.sin(float(yaw))
    local_xy = np.asarray(
        [[ex, ey], [ex, -ey], [-ex, -ey], [-ex, ey]],
        dtype=np.float64,
    )
    rotated = np.zeros_like(local_xy)
    rotated[:, 0] = float(cx) + ca * local_xy[:, 0] - sa * local_xy[:, 1]
    rotated[:, 1] = float(cy) + sa * local_xy[:, 0] + ca * local_xy[:, 1]

    bottom = np.concatenate([rotated, np.full((4, 1), bottom_z, dtype=np.float64)], axis=1)
    top = np.concatenate([rotated, np.full((4, 1), top_z, dtype=np.float64)], axis=1)
    points_xyz = np.concatenate([bottom, top], axis=0)
    return CurrentCuboidInfo(
        points_xyz=points_xyz,
        status="ok",
        center_xyz=(float(cx), float(cy), float(cz)),
        extent_xy=(float(ex), float(ey)),
        height_m=float(height),
        yaw_rad=float(yaw),
        source_center=source_center,
        source_extent=source_extent,
        source_height=source_height,
        source_yaw=source_yaw,
    )


def get_cuboid_8pt_points(raw_obj: Dict[str, object]) -> Tuple[Optional[np.ndarray], str]:
    radar_proj = raw_obj.get("radar_proj") if isinstance(raw_obj, dict) else None
    if not isinstance(radar_proj, dict):
        return None, "missing_radar_proj"
    corners_3d = radar_proj.get("corners_3d")
    if not isinstance(corners_3d, list) or len(corners_3d) < 8:
        return None, "missing_radar_proj.corners_3d"
    points: List[List[float]] = []
    for item in corners_3d[:8]:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            return None, "invalid_radar_proj.corners_3d"
        x = finite_float(item[0])
        y = finite_float(item[1])
        z = finite_float(item[2])
        if None in (x, y, z):
            return None, "invalid_radar_proj.corners_3d"
        points.append([float(x), float(y), float(z)])
    return np.asarray(points, dtype=np.float64), "ok"


def metric_rect_from_points(points_xy: np.ndarray) -> Tuple[float, float, float]:
    # Returns (long_edge, short_edge, yaw_of_long_edge_rad)
    if points_xy.shape[0] < 4:
        return 0.0, 0.0, 0.0
    pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    (_, _), (w, h), angle_deg = rect
    w = float(abs(w))
    h = float(abs(h))
    if w >= h:
        long_edge = w
        short_edge = h
        yaw_deg = float(angle_deg)
    else:
        long_edge = h
        short_edge = w
        yaw_deg = float(angle_deg) + 90.0
    yaw_rad = math.radians(yaw_deg)
    # normalize to [0, pi)
    yaw_rad = yaw_rad % math.pi
    return float(long_edge), float(short_edge), float(yaw_rad)


def normalize_rbox_long_yaw(rbox_xywhr: Sequence[float]) -> float:
    _, _, w, h, angle = [float(v) for v in rbox_xywhr]
    yaw = float(angle)
    if h > w:
        yaw += math.pi * 0.5
    return yaw % math.pi


def draw_box_with_label(
    image_bgr: np.ndarray,
    box_xyxy: Optional[Sequence[float]],
    label: str,
    color_rgb: Tuple[int, int, int],
) -> None:
    if box_xyxy is None or box_area_xyxy(box_xyxy) <= 0.0:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color_bgr, 2, lineType=cv2.LINE_AA)
    text_y = max(14, y1 - 6)
    cv2.putText(
        image_bgr,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color_bgr,
        1,
        lineType=cv2.LINE_AA,
    )


def draw_legend(image_bgr: np.ndarray, labels: Sequence[str]) -> None:
    x = 12
    y = 18
    for label in labels:
        color_rgb = RGB_COLOR_BY_LABEL[label]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        cv2.rectangle(image_bgr, (x, y - 10), (x + 16, y + 4), color_bgr, thickness=-1)
        cv2.putText(
            image_bgr,
            label,
            (x + 22, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            label,
            (x + 22, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color_bgr,
            1,
            lineType=cv2.LINE_AA,
        )
        y += 22


def draw_radar_rbox(image_bgr: np.ndarray, rbox_xywhr: Sequence[float]) -> None:
    corners = rbox_to_bottom_corners(rbox_xywhr)
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image_bgr, [pts], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)


def aggregate_metric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count_valid": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count_valid": float(arr.shape[0]),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def write_bucket_stats_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_parent(path)
    if not rows:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_camera_for_visualization(
    raw_boxes_by_cam: Dict[int, Optional[List[float]]],
    gt_primary: int,
) -> Optional[int]:
    if 0 <= gt_primary < 4:
        raw_box = raw_boxes_by_cam.get(gt_primary)
        if raw_box is not None and box_area_xyxy(raw_box) > 0.0:
            return gt_primary
    best_cam = None
    best_area = -1.0
    for cam_id in range(4):
        raw_box = raw_boxes_by_cam.get(cam_id)
        area = box_area_xyxy(raw_box)
        if area > best_area:
            best_area = area
            best_cam = cam_id
    if best_area <= 0.0:
        return None
    return best_cam


def split_sample_paths(dataset_root: Path, custom_paths: Sequence[str]) -> List[Path]:
    if custom_paths:
        return [Path(p) for p in custom_paths]
    return [
        dataset_root / "train_samples.jsonl",
        dataset_root / "valid_samples.jsonl",
        dataset_root / "test_samples.jsonl",
    ]


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    vis_dir = output_dir / "top50_large_ship_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = split_sample_paths(args.dataset_root, args.samples_jsonl)
    records = load_records(sample_paths)
    if not records:
        raise RuntimeError("No sample records loaded.")

    height_prior = load_height_prior(args.height_prior_path, args.height_prior_stat)
    height_prior_default = float(median(list(height_prior.values()))) if height_prior else 20.0

    ann_cache: Dict[Path, Dict[str, object]] = {}
    raw_yaw_cache: Dict[Path, Dict[str, object]] = {}
    raw_full_cache: Dict[Path, Dict[str, object]] = {}
    opv2v_cache: Dict[Path, Dict[str, object]] = {}

    all_rows: List[Dict[str, object]] = []
    object_level_rows: List[Dict[str, object]] = []
    vis_candidates: List[Dict[str, object]] = []

    source_check = {
        "fixed_visualization_raw_box_source": "raw_obj.cams[hard_cam].box_2d (gt_filter_only_yaw)",
        "fixed_visualization_current_projection_source": (
            "build_current_cuboid_points(center/extent/yaw/height) -> project_points_to_box(no roi_expand_ratio)"
        ),
        "fixed_visualization_has_cuboid_8pt_box": False,
        "fixed_visualization_has_gt_camera_box_2d": False,
    }

    task2_counts = {
        "objects_checked": 0,
        "current_projection_success": 0,
        "current_projection_points_total_is_8": 0,
        "current_projection_points_total_not_8": 0,
    }

    t_start = time.time()
    print(f"[info] loaded {len(records)} samples, start diagnosing...", flush=True)

    for rec_idx, rec in enumerate(records):
        sample_id = str(rec.get("sample_id", ""))
        ann_path = Path(str(rec.get("annotation_json_path", "")))
        raw_yaw_path = Path(str(rec.get("gt_filter_only_yaw_path", "")))
        raw_full_path = Path(str(rec.get("gt_filter_path", "")))
        opv2v_path = Path(str(rec.get("opv2v_yaml_path", "")))
        tower_id = str(rec.get("tower_id", ""))
        if not ann_path.is_file() or not raw_yaw_path.is_file() or not raw_full_path.is_file() or not opv2v_path.is_file():
            continue

        if ann_path not in ann_cache:
            ann_cache[ann_path] = load_json(ann_path)
        if raw_yaw_path not in raw_yaw_cache:
            raw_yaw_cache[raw_yaw_path] = load_json(raw_yaw_path)
        if raw_full_path not in raw_full_cache:
            raw_full_cache[raw_full_path] = load_json(raw_full_path)
        if opv2v_path not in opv2v_cache:
            opv2v_cache[opv2v_path] = load_yaml(opv2v_path)

        ann_payload = ann_cache[ann_path]
        raw_yaw_payload = raw_yaw_cache[raw_yaw_path]
        raw_full_payload = raw_full_cache[raw_full_path]
        opv2v_payload = opv2v_cache[opv2v_path]
        if not isinstance(ann_payload, dict) or not isinstance(opv2v_payload, dict):
            continue

        ann_list = ann_payload.get("annotations", [])
        if not isinstance(ann_list, list):
            continue

        raw_yaw_objs = raw_yaw_payload.get("towers", {}).get(tower_id, {}).get("objects", {})
        raw_full_objs = raw_full_payload.get("towers", {}).get(tower_id, {}).get("objects", {})
        if not isinstance(raw_yaw_objs, dict):
            raw_yaw_objs = {}
        if not isinstance(raw_full_objs, dict):
            raw_full_objs = {}

        camera_cache_by_cam: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float, str]] = {}
        for cam_id, canonical_cam in enumerate(CANONICAL_CAMERA_KEYS):
            camera_cache_by_cam[cam_id] = get_camera_matrices(
                opv2v_payload=opv2v_payload,
                canonical_cam=canonical_cam,
            )

        for ann_idx, ann in enumerate(ann_list):
            if not isinstance(ann, dict):
                continue
            instance_name = str(ann.get("instance_name", ""))
            class_name = str(ann.get("super_category_name") or ann.get("category_name") or "")
            gt_boxes = ann.get("gt_camera_box_2d", [])
            if not isinstance(gt_boxes, list) or len(gt_boxes) < 4:
                gt_boxes = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
            gt_primary = int(ann.get("gt_primary_camera", 4))
            rbox = ann.get("rbox_xywhr")
            if not isinstance(rbox, (list, tuple)) or len(rbox) != 5:
                continue

            long_edge_px = float(max(float(rbox[2]), float(rbox[3])))

            raw_obj_yaw = raw_yaw_objs.get(instance_name)
            raw_obj_full = raw_full_objs.get(instance_name)
            if not isinstance(raw_obj_yaw, dict):
                raw_obj_yaw = {}
            if not isinstance(raw_obj_full, dict):
                raw_obj_full = raw_obj_yaw

            center_src = raw_obj_full.get("radar_proj", {}).get("center")
            if not isinstance(center_src, dict):
                center_src = raw_obj_full.get("bev_rot_only_yaw", {}).get("center")
            if isinstance(center_src, dict):
                cx = finite_float(center_src.get("x"))
                cy = finite_float(center_src.get("y"))
                distance_m = float(math.hypot(float(cx), float(cy))) if None not in (cx, cy) else float("nan")
            else:
                distance_m = float("nan")

            current_info = build_current_cuboid_points(raw_obj_full, ann)
            cuboid_8pt_points, cuboid_8pt_status = get_cuboid_8pt_points(raw_obj_full)

            bbox_m = raw_obj_full.get("bbox_m") if isinstance(raw_obj_full, dict) else None
            bbox_l = finite_float(bbox_m.get("L")) if isinstance(bbox_m, dict) else None
            bbox_w = finite_float(bbox_m.get("W")) if isinstance(bbox_m, dict) else None

            corners_long = corners_short = corners_yaw = None
            if cuboid_8pt_points is not None:
                corners_long, corners_short, corners_yaw = metric_rect_from_points(cuboid_8pt_points[:, :2])

            current_len_m = float(current_info.extent_xy[0] * 2.0) if current_info.extent_xy is not None else float("nan")
            current_wid_m = float(current_info.extent_xy[1] * 2.0) if current_info.extent_xy is not None else float("nan")
            current_yaw = float(current_info.yaw_rad) if current_info.yaw_rad is not None else float("nan")
            rbox_long_yaw = normalize_rbox_long_yaw(rbox)

            lw_swap_suspected = 0
            yaw_off90_suspected = 0
            extent_vs_bbox_swap_suspected = 0
            current_vs_corners_yaw_deg = float("nan")

            if corners_long is not None and corners_short is not None and math.isfinite(current_len_m) and math.isfinite(current_wid_m):
                direct_err = abs(current_len_m - corners_long) + abs(current_wid_m - corners_short)
                swap_err = abs(current_len_m - corners_short) + abs(current_wid_m - corners_long)
                if swap_err + 1e-6 < direct_err:
                    lw_swap_suspected = 1
            if corners_yaw is not None and math.isfinite(current_yaw):
                yaw_diff = angle_diff_mod_pi_rad(current_yaw, corners_yaw)
                current_vs_corners_yaw_deg = math.degrees(yaw_diff)
                if abs(current_vs_corners_yaw_deg - 90.0) < current_vs_corners_yaw_deg and abs(current_vs_corners_yaw_deg - 90.0) <= 15.0:
                    yaw_off90_suspected = 1
            if bbox_l is not None and bbox_w is not None and math.isfinite(current_len_m) and math.isfinite(current_wid_m):
                direct_err = abs(current_len_m - float(bbox_l)) + abs(current_wid_m - float(bbox_w))
                swap_err = abs(current_len_m - float(bbox_w)) + abs(current_wid_m - float(bbox_l))
                if swap_err + 1e-6 < direct_err:
                    extent_vs_bbox_swap_suspected = 1

            raw_boxes_by_cam: Dict[int, Optional[List[float]]] = {}
            for cam_id, raw_cam_name in enumerate(CAMERA_NAMES):
                cam_entry = raw_obj_yaw.get("cams", {}).get(raw_cam_name) if isinstance(raw_obj_yaw.get("cams"), dict) else None
                raw_boxes_by_cam[cam_id] = raw_box_from_cam_entry(cam_entry, img_w=1024.0, img_h=512.0)

            # Task 2 counters at object granularity.
            task2_counts["objects_checked"] += 1
            if current_info.points_xyz is not None:
                if int(current_info.points_xyz.shape[0]) == 8:
                    task2_counts["current_projection_points_total_is_8"] += 1
                else:
                    task2_counts["current_projection_points_total_not_8"] += 1

            for cam_id, canonical_cam in enumerate(CANONICAL_CAMERA_KEYS):
                raw_box = raw_boxes_by_cam.get(cam_id)
                if raw_box is None or box_area_xyxy(raw_box) <= 0.0:
                    continue

                intrinsic, t_camcv_ego, img_w, img_h, calib_status = camera_cache_by_cam[cam_id]
                if intrinsic is None or t_camcv_ego is None:
                    continue

                gt_box = safe_box4(gt_boxes[cam_id] if cam_id < len(gt_boxes) else None)

                current_proj = ProjectionResult(None, "projection_unavailable", 0, 0)
                if current_info.points_xyz is not None:
                    current_proj = project_points_to_box(
                        points_xyz=current_info.points_xyz,
                        intrinsic_k=intrinsic,
                        t_camcv_ego=t_camcv_ego,
                        img_w=img_w,
                        img_h=img_h,
                        expand_ratio=1.0,  # current implementation in fix scripts has no roi_expand_ratio
                    )
                cuboid_8pt_proj = ProjectionResult(None, f"cuboid_8pt_unavailable:{cuboid_8pt_status}", 0, 0)
                if cuboid_8pt_points is not None:
                    cuboid_8pt_proj = project_points_to_box(
                        points_xyz=cuboid_8pt_points,
                        intrinsic_k=intrinsic,
                        t_camcv_ego=t_camcv_ego,
                        img_w=img_w,
                        img_h=img_h,
                        expand_ratio=float(args.roi_expand_ratio),
                    )

                bottom_only_proj = ProjectionResult(None, "projection_unavailable", 0, 0)
                top_only_proj = ProjectionResult(None, "projection_unavailable", 0, 0)
                if current_info.points_xyz is not None and current_info.points_xyz.shape[0] == 8:
                    bottom_only_proj = project_points_to_box(
                        points_xyz=current_info.points_xyz[:4],
                        intrinsic_k=intrinsic,
                        t_camcv_ego=t_camcv_ego,
                        img_w=img_w,
                        img_h=img_h,
                        expand_ratio=1.0,
                    )
                    top_only_proj = project_points_to_box(
                        points_xyz=current_info.points_xyz[4:],
                        intrinsic_k=intrinsic,
                        t_camcv_ego=t_camcv_ego,
                        img_w=img_w,
                        img_h=img_h,
                        expand_ratio=1.0,
                    )

                if current_proj.box_xyxy is not None:
                    task2_counts["current_projection_success"] += 1

                raw_w, raw_h = box_width_height(raw_box)
                current_w, current_h = box_width_height(current_proj.box_xyxy)
                cuboid_w, cuboid_h = box_width_height(cuboid_8pt_proj.box_xyxy)
                gt_w, gt_h = box_width_height(gt_box)

                row = {
                    "sample_id": sample_id,
                    "split": str(rec.get("split", "")),
                    "scene_id": str(rec.get("scene_id", "")),
                    "tower_id": str(rec.get("tower_id", "")),
                    "frame_id": str(rec.get("frame_id", "")),
                    "ann_index": int(ann_idx),
                    "instance_name": instance_name,
                    "class_name": class_name,
                    "camera_id": int(cam_id),
                    "camera_name": canonical_cam,
                    "distance_m": float(distance_m) if math.isfinite(distance_m) else float("nan"),
                    "rbox_long_edge_px": float(long_edge_px),
                    "raw_box_x1": float(raw_box[0]),
                    "raw_box_y1": float(raw_box[1]),
                    "raw_box_x2": float(raw_box[2]),
                    "raw_box_y2": float(raw_box[3]),
                    "raw_box_width": float(raw_w),
                    "raw_box_height": float(raw_h),
                    "current_projection_status": current_proj.status,
                    "current_projection_box_x1": float(current_proj.box_xyxy[0]) if current_proj.box_xyxy else float("nan"),
                    "current_projection_box_y1": float(current_proj.box_xyxy[1]) if current_proj.box_xyxy else float("nan"),
                    "current_projection_box_x2": float(current_proj.box_xyxy[2]) if current_proj.box_xyxy else float("nan"),
                    "current_projection_box_y2": float(current_proj.box_xyxy[3]) if current_proj.box_xyxy else float("nan"),
                    "current_projection_box_width": float(current_w),
                    "current_projection_box_height": float(current_h),
                    "cuboid_8pt_status": cuboid_8pt_proj.status,
                    "cuboid_8pt_box_x1": float(cuboid_8pt_proj.box_xyxy[0]) if cuboid_8pt_proj.box_xyxy else float("nan"),
                    "cuboid_8pt_box_y1": float(cuboid_8pt_proj.box_xyxy[1]) if cuboid_8pt_proj.box_xyxy else float("nan"),
                    "cuboid_8pt_box_x2": float(cuboid_8pt_proj.box_xyxy[2]) if cuboid_8pt_proj.box_xyxy else float("nan"),
                    "cuboid_8pt_box_y2": float(cuboid_8pt_proj.box_xyxy[3]) if cuboid_8pt_proj.box_xyxy else float("nan"),
                    "cuboid_8pt_box_width": float(cuboid_w),
                    "cuboid_8pt_box_height": float(cuboid_h),
                    "gt_camera_box_2d_x1": float(gt_box[0]) if gt_box else float("nan"),
                    "gt_camera_box_2d_y1": float(gt_box[1]) if gt_box else float("nan"),
                    "gt_camera_box_2d_x2": float(gt_box[2]) if gt_box else float("nan"),
                    "gt_camera_box_2d_y2": float(gt_box[3]) if gt_box else float("nan"),
                    "gt_camera_box_2d_width": float(gt_w),
                    "gt_camera_box_2d_height": float(gt_h),
                    "raw_to_gt_iou": float(iou_xyxy(raw_box, gt_box)),
                    "current_to_raw_iou": float(iou_xyxy(current_proj.box_xyxy, raw_box)),
                    "cuboid8_to_raw_iou": float(iou_xyxy(cuboid_8pt_proj.box_xyxy, raw_box)),
                    "current_projection_width_over_raw_width": (
                        float(current_w / raw_w) if raw_w > 1e-6 and current_w > 0.0 else float("nan")
                    ),
                    "cuboid_8pt_width_over_raw_width": (
                        float(cuboid_w / raw_w) if raw_w > 1e-6 and cuboid_w > 0.0 else float("nan")
                    ),
                    "current_projection_height_over_raw_height": (
                        float(current_h / raw_h) if raw_h > 1e-6 and current_h > 0.0 else float("nan")
                    ),
                    "cuboid_8pt_height_over_raw_height": (
                        float(cuboid_h / raw_h) if raw_h > 1e-6 and cuboid_h > 0.0 else float("nan")
                    ),
                    "bbox_m_L": float(bbox_l) if bbox_l is not None else float("nan"),
                    "bbox_m_W": float(bbox_w) if bbox_w is not None else float("nan"),
                    "current_length_m": float(current_len_m) if math.isfinite(current_len_m) else float("nan"),
                    "current_width_m": float(current_wid_m) if math.isfinite(current_wid_m) else float("nan"),
                    "corners_long_m": float(corners_long) if corners_long is not None else float("nan"),
                    "corners_short_m": float(corners_short) if corners_short is not None else float("nan"),
                    "current_vs_corners_yaw_deg": float(current_vs_corners_yaw_deg),
                    "current_vs_rbox_long_yaw_deg": (
                        float(math.degrees(angle_diff_mod_pi_rad(current_yaw, rbox_long_yaw)))
                        if math.isfinite(current_yaw)
                        else float("nan")
                    ),
                    "lw_swap_suspected": int(lw_swap_suspected),
                    "yaw_off90_suspected": int(yaw_off90_suspected),
                    "extent_vs_bbox_swap_suspected": int(extent_vs_bbox_swap_suspected),
                    "current_points_status": current_info.status,
                    "current_points_total": int(current_info.points_xyz.shape[0]) if current_info.points_xyz is not None else 0,
                    "current_valid_depth_count": int(current_proj.valid_depth_count),
                    "bottom_only_status": bottom_only_proj.status,
                    "top_only_status": top_only_proj.status,
                    "current_equals_bottom_only_iou": float(iou_xyxy(current_proj.box_xyxy, bottom_only_proj.box_xyxy)),
                    "current_equals_top_only_iou": float(iou_xyxy(current_proj.box_xyxy, top_only_proj.box_xyxy)),
                    "current_source_center": current_info.source_center,
                    "current_source_extent": current_info.source_extent,
                    "current_source_height": current_info.source_height,
                    "current_source_yaw": current_info.source_yaw,
                    "cuboid_8pt_points_status": cuboid_8pt_status,
                    "calibration_status": calib_status,
                    "camera_path": str(rec.get("camera_paths", {}).get(CAMERA_NAMES[cam_id], "")) if isinstance(rec.get("camera_paths"), dict) else "",
                    "radar_bev_path": str(rec.get("radar_bev_path", "")),
                    "gt_primary_camera": int(gt_primary),
                    "selected_for_visualization": 0,
                }
                all_rows.append(row)

            # one-row object-level summary for L/W/yaw diagnosis
            object_level_rows.append(
                {
                    "sample_id": sample_id,
                    "ann_index": int(ann_idx),
                    "instance_name": instance_name,
                    "class_name": class_name,
                    "distance_m": float(distance_m) if math.isfinite(distance_m) else float("nan"),
                    "rbox_long_edge_px": float(long_edge_px),
                    "bbox_m_L": float(bbox_l) if bbox_l is not None else float("nan"),
                    "bbox_m_W": float(bbox_w) if bbox_w is not None else float("nan"),
                    "current_length_m": float(current_len_m) if math.isfinite(current_len_m) else float("nan"),
                    "current_width_m": float(current_wid_m) if math.isfinite(current_wid_m) else float("nan"),
                    "corners_long_m": float(corners_long) if corners_long is not None else float("nan"),
                    "corners_short_m": float(corners_short) if corners_short is not None else float("nan"),
                    "current_yaw_rad": float(current_yaw) if math.isfinite(current_yaw) else float("nan"),
                    "corners_yaw_rad": float(corners_yaw) if corners_yaw is not None else float("nan"),
                    "rbox_long_yaw_rad": float(rbox_long_yaw),
                    "lw_swap_suspected": int(lw_swap_suspected),
                    "yaw_off90_suspected": int(yaw_off90_suspected),
                    "extent_vs_bbox_swap_suspected": int(extent_vs_bbox_swap_suspected),
                    "current_points_status": current_info.status,
                    "current_points_total": int(current_info.points_xyz.shape[0]) if current_info.points_xyz is not None else 0,
                    "current_source_center": current_info.source_center,
                    "current_source_extent": current_info.source_extent,
                    "current_source_height": current_info.source_height,
                    "current_source_yaw": current_info.source_yaw,
                    "cuboid_8pt_points_status": cuboid_8pt_status,
                }
            )

            # visualization candidate by object: select one camera with raw box.
            vis_cam_id = select_camera_for_visualization(raw_boxes_by_cam, gt_primary=gt_primary)
            if vis_cam_id is not None:
                vis_candidates.append(
                    {
                        "sample_id": sample_id,
                        "record": rec,
                        "ann": ann,
                        "ann_index": int(ann_idx),
                        "instance_name": instance_name,
                        "class_name": class_name,
                        "long_edge_px": float(long_edge_px),
                        "distance_m": float(distance_m) if math.isfinite(distance_m) else float("nan"),
                        "camera_id": int(vis_cam_id),
                        "camera_name": CANONICAL_CAMERA_KEYS[vis_cam_id],
                        "raw_obj_yaw": raw_obj_yaw,
                        "raw_obj_full": raw_obj_full,
                        "opv2v_payload": opv2v_payload,
                        "current_info": current_info,
                        "cuboid_8pt_points": cuboid_8pt_points,
                        "cuboid_8pt_status": cuboid_8pt_status,
                    }
                )

        if (rec_idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            print(
                f"[progress] {rec_idx + 1}/{len(records)} samples "
                f"rows={len(all_rows)} obj_rows={len(object_level_rows)} "
                f"elapsed_sec={elapsed:.1f}",
                flush=True,
            )

    if not all_rows:
        raise RuntimeError("No raw_box rows were collected. Please check dataset paths.")

    # Top-K large ship visualization.
    vis_candidates = sorted(
        vis_candidates,
        key=lambda row: (-float(row["long_edge_px"]), str(row["sample_id"]), int(row["ann_index"])),
    )
    topk = vis_candidates[: max(0, int(args.topk_vis))]
    vis_manifest_rows: List[Dict[str, object]] = []

    for idx, item in enumerate(topk):
        rec = item["record"]
        ann = item["ann"]
        ann_idx = int(item["ann_index"])
        instance_name = str(item["instance_name"])
        sample_id = str(item["sample_id"])
        camera_id = int(item["camera_id"])
        canonical_cam = CANONICAL_CAMERA_KEYS[camera_id]
        raw_cam_name = CAMERA_NAMES[camera_id]
        opv2v_payload = item["opv2v_payload"]
        raw_obj_yaw = item["raw_obj_yaw"]
        current_info: CurrentCuboidInfo = item["current_info"]
        cuboid_8pt_points = item["cuboid_8pt_points"]
        cuboid_8pt_status = str(item["cuboid_8pt_status"])

        intrinsic, t_camcv_ego, img_w, img_h, calib_status = get_camera_matrices(opv2v_payload, canonical_cam)
        if intrinsic is None or t_camcv_ego is None:
            continue
        raw_entry = raw_obj_yaw.get("cams", {}).get(raw_cam_name) if isinstance(raw_obj_yaw.get("cams"), dict) else None
        raw_box = raw_box_from_cam_entry(raw_entry, img_w=1024.0, img_h=512.0)
        gt_boxes = ann.get("gt_camera_box_2d", [[0.0, 0.0, 0.0, 0.0] for _ in range(4)])
        gt_box = safe_box4(gt_boxes[camera_id] if camera_id < len(gt_boxes) else None)

        current_proj = ProjectionResult(None, "projection_unavailable", 0, 0)
        if current_info.points_xyz is not None:
            current_proj = project_points_to_box(
                points_xyz=current_info.points_xyz,
                intrinsic_k=intrinsic,
                t_camcv_ego=t_camcv_ego,
                img_w=img_w,
                img_h=img_h,
                expand_ratio=1.0,
            )
        cuboid_8pt_proj = ProjectionResult(None, f"cuboid_8pt_unavailable:{cuboid_8pt_status}", 0, 0)
        if cuboid_8pt_points is not None:
            cuboid_8pt_proj = project_points_to_box(
                points_xyz=cuboid_8pt_points,
                intrinsic_k=intrinsic,
                t_camcv_ego=t_camcv_ego,
                img_w=img_w,
                img_h=img_h,
                expand_ratio=float(args.roi_expand_ratio),
            )

        cam_path = None
        if isinstance(rec.get("camera_paths"), dict):
            cam_path = rec.get("camera_paths", {}).get(raw_cam_name)
        rgb_img = cv2.imread(str(cam_path), cv2.IMREAD_COLOR) if isinstance(cam_path, str) and Path(cam_path).is_file() else None
        if rgb_img is None:
            rgb_img = np.zeros((512, 1024, 3), dtype=np.uint8)
        draw_box_with_label(rgb_img, raw_box, "raw_box", RGB_COLOR_BY_LABEL["raw_box"])
        draw_box_with_label(rgb_img, current_proj.box_xyxy, "current_projection_box", RGB_COLOR_BY_LABEL["current_projection_box"])
        draw_box_with_label(rgb_img, cuboid_8pt_proj.box_xyxy, "cuboid_8pt_box", RGB_COLOR_BY_LABEL["cuboid_8pt_box"])
        draw_box_with_label(rgb_img, gt_box, "gt_camera_box_2d", RGB_COLOR_BY_LABEL["gt_camera_box_2d"])
        draw_legend(
            rgb_img,
            labels=["raw_box", "current_projection_box", "cuboid_8pt_box", "gt_camera_box_2d"],
        )

        info_lines = [
            f"sample={sample_id}",
            f"ann={ann_idx} obj={instance_name} class={item['class_name']}",
            f"camera={canonical_cam} long_edge_px={float(item['long_edge_px']):.2f}",
            f"current_status={current_proj.status} cuboid_8pt_status={cuboid_8pt_proj.status}",
            f"calib_status={calib_status}",
        ]
        y = 20
        for text in info_lines:
            cv2.putText(rgb_img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(rgb_img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, lineType=cv2.LINE_AA)
            y += 22

        radar_path = Path(str(rec.get("radar_bev_path", "")))
        radar_img = cv2.imread(str(radar_path), cv2.IMREAD_COLOR) if radar_path.is_file() else None
        if radar_img is None:
            radar_img = np.zeros((1536, 1536, 3), dtype=np.uint8)
        draw_radar_rbox(radar_img, ann.get("rbox_xywhr", [0, 0, 0, 0, 0]))
        cv2.putText(
            radar_img,
            "radar_bev + rbox_xywhr",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        # Resize radar panel to match RGB height.
        radar_resized = cv2.resize(radar_img, (int(round(radar_img.shape[1] * (rgb_img.shape[0] / max(radar_img.shape[0], 1)))), rgb_img.shape[0]))
        merged = np.concatenate([rgb_img, radar_resized], axis=1)

        safe_sample = sample_id.replace("/", "__")
        safe_obj = instance_name.replace("/", "_")
        out_name = (
            f"{idx + 1:02d}__{safe_sample}__ann{ann_idx:03d}__{safe_obj}"
            "__raw_box__current_projection_box__cuboid_8pt_box__gt_camera_box_2d.png"
        )
        out_path = vis_dir / out_name
        cv2.imwrite(str(out_path), merged)

        vis_manifest_rows.append(
            {
                "rank": idx + 1,
                "sample_id": sample_id,
                "ann_index": ann_idx,
                "instance_name": instance_name,
                "class_name": item["class_name"],
                "camera_name": canonical_cam,
                "long_edge_px": float(item["long_edge_px"]),
                "distance_m": float(item["distance_m"]) if math.isfinite(float(item["distance_m"])) else float("nan"),
                "visualization_file": str(out_path),
                "raw_box_status": "ok" if raw_box is not None else "missing_raw_box",
                "current_projection_status": current_proj.status,
                "cuboid_8pt_status": cuboid_8pt_proj.status,
                "gt_camera_box_2d_status": "ok" if gt_box is not None else "missing_gt_camera_box_2d",
            }
        )

        # tag selected rows
        for row in all_rows:
            if (
                row["sample_id"] == sample_id
                and int(row["ann_index"]) == ann_idx
                and int(row["camera_id"]) == camera_id
            ):
                row["selected_for_visualization"] = 1

    # Save per-pair stats.
    pairs_csv = output_dir / "all_raw_box_projection_pairs.csv"
    with pairs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    object_csv = output_dir / "object_level_lw_yaw_diagnosis.csv"
    with object_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(object_level_rows[0].keys()))
        writer.writeheader()
        for row in object_level_rows:
            writer.writerow(row)

    vis_manifest_csv = output_dir / "top50_visualization_manifest.csv"
    if vis_manifest_rows:
        with vis_manifest_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(vis_manifest_rows[0].keys()))
            writer.writeheader()
            for row in vis_manifest_rows:
                writer.writerow(row)

    # Bucket stats for ratios.
    metric_keys = [
        "current_projection_width_over_raw_width",
        "cuboid_8pt_width_over_raw_width",
        "current_projection_height_over_raw_height",
        "cuboid_8pt_height_over_raw_height",
    ]
    group_specs = [
        ("stats_by_category.csv", ["class_name"]),
        ("stats_by_distance_bin.csv", ["distance_bin"]),
        ("stats_by_camera.csv", ["camera_name"]),
        ("stats_by_long_edge_bin.csv", ["long_edge_bin"]),
        ("stats_by_all_buckets.csv", ["class_name", "distance_bin", "camera_name", "long_edge_bin"]),
    ]

    for row in all_rows:
        distance_m = float(row["distance_m"]) if math.isfinite(float(row["distance_m"])) else float("inf")
        row["distance_bin"] = angle_to_bin(distance_m, DISTANCE_BINS)
        row["long_edge_bin"] = angle_to_bin(float(row["rbox_long_edge_px"]), LONG_EDGE_BINS)

    bucket_outputs: Dict[str, List[Dict[str, object]]] = {}
    for out_name, group_keys in group_specs:
        buckets: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
        for row in all_rows:
            key = tuple(row[k] for k in group_keys)
            buckets[key].append(row)

        out_rows: List[Dict[str, object]] = []
        for key, rows in sorted(buckets.items(), key=lambda item: item[0]):
            item: Dict[str, object] = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
            item["count_total"] = int(len(rows))
            for mk in metric_keys:
                vals = [float(r[mk]) for r in rows if math.isfinite(float(r[mk]))]
                stats = aggregate_metric(vals)
                item[f"{mk}_count_valid"] = int(stats["count_valid"])
                item[f"{mk}_mean"] = float(stats["mean"])
                item[f"{mk}_median"] = float(stats["median"])
                item[f"{mk}_p10"] = float(stats["p10"])
                item[f"{mk}_p90"] = float(stats["p90"])
            out_rows.append(item)
        bucket_outputs[out_name] = out_rows
        write_bucket_stats_csv(output_dir / out_name, out_rows)

    # Global summary metrics.
    def finite_values(key: str) -> List[float]:
        return [float(row[key]) for row in all_rows if math.isfinite(float(row[key]))]

    global_summary = {
        "num_samples_loaded": int(len(records)),
        "num_raw_box_pairs": int(len(all_rows)),
        "num_object_rows": int(len(object_level_rows)),
        "num_topk_visualizations": int(len(vis_manifest_rows)),
        "projection_plane_height": float(args.projection_plane_height),
        "roi_expand_ratio": float(args.roi_expand_ratio),
        "task1_fixed_source_audit": source_check,
        "task2_corner_usage_check": task2_counts,
        "task4_global_ratio_stats": {},
        "task5_lw_yaw_flags": {},
        "artifacts": {
            "all_raw_box_projection_pairs_csv": str(pairs_csv),
            "object_level_lw_yaw_csv": str(object_csv),
            "top50_visualization_manifest_csv": str(vis_manifest_csv),
            "visualization_dir": str(vis_dir),
        },
    }
    for mk in metric_keys:
        vals = finite_values(mk)
        global_summary["task4_global_ratio_stats"][mk] = aggregate_metric(vals)

    lw_swap_count = int(sum(int(row.get("lw_swap_suspected", 0)) for row in object_level_rows))
    yaw_off90_count = int(sum(int(row.get("yaw_off90_suspected", 0)) for row in object_level_rows))
    extent_bbox_swap_count = int(sum(int(row.get("extent_vs_bbox_swap_suspected", 0)) for row in object_level_rows))
    global_summary["task5_lw_yaw_flags"] = {
        "lw_swap_suspected_count": lw_swap_count,
        "yaw_off90_suspected_count": yaw_off90_count,
        "extent_vs_bbox_swap_suspected_count": extent_bbox_swap_count,
        "num_objects": int(len(object_level_rows)),
    }

    summary_json = output_dir / "summary.json"
    write_json(summary_json, global_summary)

    # Markdown report.
    lines: List[str] = []
    lines.append("# RouteROI 3D Projection Box Diagnosis")
    lines.append("")
    lines.append(f"- dataset_root: `{args.dataset_root}`")
    lines.append(f"- num_samples_loaded: **{len(records)}**")
    lines.append(f"- num_raw_box_pairs: **{len(all_rows)}**")
    lines.append(f"- num_object_rows: **{len(object_level_rows)}**")
    lines.append(f"- num_topk_visualizations: **{len(vis_manifest_rows)}**")
    lines.append(f"- projection_plane_height: **{float(args.projection_plane_height):.4f}**")
    lines.append(f"- roi_expand_ratio: **{float(args.roi_expand_ratio):.4f}**")
    lines.append("")

    lines.append("## Task 1: Fixed Visualization Box Source Audit")
    lines.append("")
    lines.append("| label | source |")
    lines.append("| --- | --- |")
    lines.append("| raw_box | raw `cams[*].box_2d` from `gt_filter_only_yaw` |")
    lines.append("| current_projection_box | current fix-chain cuboid build (`center+extent+yaw+height`) then projection (no roi expand) |")
    lines.append("| cuboid_8pt_box | projected from raw `radar_proj.corners_3d` (8 points) + `roi_expand_ratio` |")
    lines.append("| gt_camera_box_2d | prepared annotation `gt_camera_box_2d[camera_id]` |")
    lines.append("")
    lines.append("说明：现有 `output/none_primary_camera_fix_debug/fixed/` 历史图仅绘制了 `raw_box` 与 `current_projection_box`。")
    lines.append("本次新增 top50 诊断图在图例、文件名、manifest 中显式标注了四类框。")
    lines.append("")

    lines.append("## Task 2: 8-Corner Usage Check")
    lines.append("")
    lines.append(f"- objects_checked: **{task2_counts['objects_checked']}**")
    lines.append(f"- current_projection_success: **{task2_counts['current_projection_success']}**")
    lines.append(
        f"- current_projection_points_total_is_8: **{task2_counts['current_projection_points_total_is_8']}**"
    )
    lines.append(
        f"- current_projection_points_total_not_8: **{task2_counts['current_projection_points_total_not_8']}**"
    )
    lines.append("- 代码核查：`current_projection_box` 由 bottom 4 点 + top 4 点共 8 点参与 min/max 包围盒计算。")
    lines.append("")

    lines.append("## Task 3: Top 50 Large-Ship Visualization")
    lines.append("")
    lines.append(f"- visualization_dir: `{vis_dir}`")
    lines.append(f"- visualization_manifest: `{vis_manifest_csv}`")
    lines.append("- 每张图左侧为 RGB 叠框，右侧为雷达 BEV + 对应 rbox。")
    lines.append("")

    lines.append("## Task 4: Width/Height Ratio Statistics")
    lines.append("")
    lines.append("| metric | valid_count | mean | median | p10 | p90 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for mk in metric_keys:
        stats = global_summary["task4_global_ratio_stats"][mk]
        lines.append(
            f"| {mk} | {int(stats['count_valid'])} | {stats['mean']:.6f} | {stats['median']:.6f} | {stats['p10']:.6f} | {stats['p90']:.6f} |"
        )
    lines.append("")
    lines.append("分桶统计文件：")
    lines.append(f"- `{output_dir / 'stats_by_category.csv'}`")
    lines.append(f"- `{output_dir / 'stats_by_distance_bin.csv'}`")
    lines.append(f"- `{output_dir / 'stats_by_camera.csv'}`")
    lines.append(f"- `{output_dir / 'stats_by_long_edge_bin.csv'}`")
    lines.append(f"- `{output_dir / 'stats_by_all_buckets.csv'}`")
    lines.append("")

    lines.append("## Task 5: L/W/Yaw Diagnosis")
    lines.append("")
    lines.append(f"- lw_swap_suspected_count: **{lw_swap_count}**")
    lines.append(f"- yaw_off90_suspected_count: **{yaw_off90_count}**")
    lines.append(f"- extent_vs_bbox_swap_suspected_count: **{extent_bbox_swap_count}**")
    lines.append(f"- object_level_csv: `{object_csv}`")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- summary_json: `{summary_json}`")
    lines.append(f"- all_pairs_csv: `{pairs_csv}`")
    lines.append(f"- report_generated_by: `tools/diagnose_route_roi_3d_projection_box.py`")
    lines.append("")

    ensure_parent(args.report_path)
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "report_path": str(args.report_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "summary_json": str(summary_json.resolve()),
                "pairs_csv": str(pairs_csv.resolve()),
                "vis_dir": str(vis_dir.resolve()),
                "num_raw_box_pairs": len(all_rows),
                "num_topk_visualizations": len(vis_manifest_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[info] done in {time.time() - t_start:.1f} sec", flush=True)


if __name__ == "__main__":
    main()
