#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

SPLIT_TO_JSONL = {
    "Train": "train_samples.jsonl",
    "Valid": "valid_samples.jsonl",
    "Test": "test_samples.jsonl",
}

IMG_W = 1024.0
IMG_H = 512.0
MIN_VALID_SIDE = 1.0


@dataclass
class CamBoxStatus:
    valid: bool
    reason: str
    box_xyxy: Optional[List[float]]
    width: float
    height: float
    area: float


@dataclass
class PrimaryFixResult:
    source: str
    fallback_triggered: bool
    fallback_trigger_reason: str
    projection_status: str
    primary_reason_yaw: str
    primary_reason_full: str
    new_primary_has_box: int
    final_invalid_reason: str
    overwrite_guarded: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix mix721 primary camera no-box using gt_filter fallback (without changing route rule)"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/primary_camera_no_box_fix.md"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/primary_camera_no_box_fix_debug"),
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


def safe_list4(payload: object, default: float = 0.0) -> List[float]:
    if not isinstance(payload, (list, tuple)):
        return [float(default)] * 4
    out = [float(default)] * 4
    for idx in range(min(4, len(payload))):
        value = finite_float(payload[idx])
        out[idx] = float(value) if value is not None else float(default)
    return out


def sanitize_boxes4(payload: object) -> List[List[float]]:
    if not isinstance(payload, (list, tuple)):
        return [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    out: List[List[float]] = []
    for idx in range(4):
        if idx < len(payload):
            out.append(normalize_box4(payload[idx]))
        else:
            out.append([0.0, 0.0, 0.0, 0.0])
    return out


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


def camera_statuses_from_object(raw_obj: object) -> List[CamBoxStatus]:
    statuses: List[CamBoxStatus] = []
    if not isinstance(raw_obj, dict):
        for _ in range(4):
            statuses.append(CamBoxStatus(False, "raw_primary_missing", None, 0.0, 0.0, 0.0))
        return statuses
    cams = raw_obj.get("cams")
    if not isinstance(cams, dict):
        for _ in range(4):
            statuses.append(CamBoxStatus(False, "raw_primary_missing", None, 0.0, 0.0, 0.0))
        return statuses
    for cam_name in CAMERA_NAMES:
        statuses.append(analyze_cam_entry(cams.get(cam_name)))
    return statuses


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
    if box_area_xyxy(clipped) <= 0.0:
        return None, "projection_box_zero_area"
    return clipped, "ok"


def get_tower_objects(payload: object, tower_id: str) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    towers = payload.get("towers")
    if not isinstance(towers, dict):
        return {}
    tower = towers.get(tower_id)
    if not isinstance(tower, dict):
        return {}
    objects = tower.get("objects")
    if not isinstance(objects, dict):
        return {}
    return objects


def projection_bucket(status: str) -> str:
    if status == "projection_box_outside_image":
        return "projection_box_outside_image"
    if status == "projection_all_points_behind_camera":
        return "projection_invalid_depth"
    return "unknown"


def fix_primary_camera_box(
    ann: Dict[str, object],
    old_has: List[int],
    old_boxes: List[List[float]],
    old_visible: List[int],
    old_poly_area: List[float],
    raw_obj_yaw: object,
    raw_obj_full: object,
    opv2v_payload: Dict[str, object],
) -> Tuple[Dict[str, object], PrimaryFixResult]:
    gt_primary = int(ann.get("gt_primary_camera", NONE_CAMERA_ID))

    new_has = [int(v) for v in old_has]
    new_boxes = [[float(v) for v in box] for box in old_boxes]
    new_visible = [int(v) for v in old_visible]
    new_poly_area = [float(v) for v in old_poly_area]

    fallback_triggered = False
    fallback_trigger_reason = ""
    projection_status = ""
    primary_reason_yaw = "unknown"
    primary_reason_full = "unknown"
    source = "unchanged"
    overwrite_guarded = False

    if gt_primary < 0 or gt_primary >= 4:
        fields = {
            "gt_primary_camera": gt_primary,
            "gt_has_camera_box": new_has,
            "gt_camera_box_2d": new_boxes,
            "gt_visible_cameras": new_visible,
            "gt_camera_poly_area": new_poly_area,
        }
        return fields, PrimaryFixResult(
            source=source,
            fallback_triggered=fallback_triggered,
            fallback_trigger_reason=fallback_trigger_reason,
            projection_status=projection_status,
            primary_reason_yaw=primary_reason_yaw,
            primary_reason_full=primary_reason_full,
            new_primary_has_box=int(new_has[gt_primary]) if 0 <= gt_primary < 4 else 0,
            final_invalid_reason="unknown",
            overwrite_guarded=overwrite_guarded,
        )

    old_primary_valid = bool(old_has[gt_primary] == 1 and box_area_xyxy(old_boxes[gt_primary]) > 0.0)
    if old_primary_valid:
        source = "existing_kept"
        fields = {
            "gt_primary_camera": gt_primary,
            "gt_has_camera_box": new_has,
            "gt_camera_box_2d": new_boxes,
            "gt_visible_cameras": new_visible,
            "gt_camera_poly_area": new_poly_area,
        }
        return fields, PrimaryFixResult(
            source=source,
            fallback_triggered=fallback_triggered,
            fallback_trigger_reason=fallback_trigger_reason,
            projection_status=projection_status,
            primary_reason_yaw="existing_non_empty",
            primary_reason_full=primary_reason_full,
            new_primary_has_box=1,
            final_invalid_reason="",
            overwrite_guarded=overwrite_guarded,
        )

    statuses_yaw = camera_statuses_from_object(raw_obj_yaw)
    primary_reason_yaw = statuses_yaw[gt_primary].reason
    primary_valid_yaw = statuses_yaw[gt_primary].valid
    all_empty_yaw = not any(st.valid for st in statuses_yaw)
    obj_missing_yaw = not isinstance(raw_obj_yaw, dict)

    if primary_valid_yaw and statuses_yaw[gt_primary].box_xyxy is not None:
        box = statuses_yaw[gt_primary].box_xyxy
        new_boxes[gt_primary] = [float(v) for v in box]
        new_has[gt_primary] = 1
        new_visible[gt_primary] = 1
        new_poly_area[gt_primary] = float(box_area_xyxy(box))
        source = "gt_filter_only_yaw_raw_primary"
    else:
        if obj_missing_yaw:
            fallback_trigger_reason = "object_missing_in_gt_filter_only_yaw"
        elif all_empty_yaw:
            fallback_trigger_reason = "all_cams_empty_in_gt_filter_only_yaw"
        elif not primary_valid_yaw:
            fallback_trigger_reason = f"primary_invalid_in_gt_filter_only_yaw:{primary_reason_yaw}"
        else:
            fallback_trigger_reason = "unknown"
        fallback_triggered = True

        statuses_full = camera_statuses_from_object(raw_obj_full)
        primary_reason_full = statuses_full[gt_primary].reason
        primary_valid_full = statuses_full[gt_primary].valid

        if primary_valid_full and statuses_full[gt_primary].box_xyxy is not None:
            box = statuses_full[gt_primary].box_xyxy
            new_boxes[gt_primary] = [float(v) for v in box]
            new_has[gt_primary] = 1
            new_visible[gt_primary] = 1
            new_poly_area[gt_primary] = float(box_area_xyxy(box))
            source = "gt_filter_fallback_raw_primary"
        else:
            proj_obj = raw_obj_full if isinstance(raw_obj_full, dict) else raw_obj_yaw
            if isinstance(proj_obj, dict):
                points_xyz, points_status = build_cuboid_points(proj_obj, ann)
                if points_xyz is None:
                    projection_status = f"projection_unavailable:{points_status}"
                else:
                    proj_box, proj_status = project_cuboid_box(
                        points_xyz=points_xyz,
                        opv2v_payload=opv2v_payload if isinstance(opv2v_payload, dict) else {},
                        canonical_cam=CAMERA_ID_TO_NAME[gt_primary],
                    )
                    projection_status = proj_status
                    if proj_box is not None:
                        new_boxes[gt_primary] = [float(v) for v in proj_box]
                        new_has[gt_primary] = 1
                        new_visible[gt_primary] = 1
                        new_poly_area[gt_primary] = float(box_area_xyxy(proj_box))
                        source = "3d_projection"
                    else:
                        source = "still_no_primary_box"
            else:
                projection_status = "projection_unavailable:missing_object_in_both_sources"
                source = "still_no_primary_box"

    # Keep primary box semantic consistent.
    if new_has[gt_primary] <= 0:
        new_has[gt_primary] = 0
        new_boxes[gt_primary] = [0.0, 0.0, 0.0, 0.0]
        new_visible[gt_primary] = 0
        new_poly_area[gt_primary] = 0.0

    # Overwrite guard: prevent replacing non-empty existing supervision with all-zero result.
    if any(v > 0 for v in old_has) and not any(v > 0 for v in new_has):
        overwrite_guarded = True
        new_has = [int(v) for v in old_has]
        new_boxes = [[float(v) for v in box] for box in old_boxes]
        new_visible = [int(v) for v in old_visible]
        new_poly_area = [float(v) for v in old_poly_area]
        source = "overwrite_guard_keep_old_non_empty"

    final_invalid_reason = ""
    if new_has[gt_primary] == 0:
        if source == "still_no_primary_box":
            if primary_reason_full and primary_reason_full != "unknown":
                final_invalid_reason = primary_reason_full
            elif primary_reason_yaw and primary_reason_yaw != "unknown":
                final_invalid_reason = primary_reason_yaw
            else:
                final_invalid_reason = projection_bucket(projection_status)
        elif source.startswith("overwrite_guard"):
            final_invalid_reason = "overwrite_guard_keep_old_non_empty"
        else:
            final_invalid_reason = primary_reason_yaw or "unknown"

    fields = {
        "gt_primary_camera": int(gt_primary),
        "gt_has_camera_box": [int(v) for v in new_has],
        "gt_camera_box_2d": [[float(v) for v in box] for box in new_boxes],
        "gt_visible_cameras": [int(v) for v in new_visible],
        "gt_camera_poly_area": [float(v) for v in new_poly_area],
    }
    return fields, PrimaryFixResult(
        source=source,
        fallback_triggered=fallback_triggered,
        fallback_trigger_reason=fallback_trigger_reason,
        projection_status=projection_status,
        primary_reason_yaw=primary_reason_yaw,
        primary_reason_full=primary_reason_full,
        new_primary_has_box=int(fields["gt_has_camera_box"][gt_primary]),
        final_invalid_reason=final_invalid_reason,
        overwrite_guarded=overwrite_guarded,
    )


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
        sample_id = image_id_to_sample_id.get(image_id, "")
        if sample_id not in updated_ann_fields_by_sample_id:
            continue
        updated_fields_list = updated_ann_fields_by_sample_id[sample_id]
        ann_list.sort(key=lambda a: int(a.get("id", 0)))
        if len(ann_list) != len(updated_fields_list):
            mismatch_len += 1
            continue

        for ann_idx, ann in enumerate(ann_list):
            new_fields = updated_fields_list[ann_idx]
            for key in (
                "gt_primary_camera",
                "gt_visible_cameras",
                "gt_camera_box_2d",
                "gt_has_camera_box",
                "gt_camera_poly_area",
            ):
                if ann.get(key) != new_fields[key]:
                    ann[key] = new_fields[key]
                    changed_fields += 1

    write_json(coco_path, payload)
    return {
        "changed_fields": int(changed_fields),
        "missing_sample": int(missing_sample),
        "mismatch_len": int(mismatch_len),
    }


def collect_samples(dataset_root: Path) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for split in ["train", "valid", "test"]:
        path = dataset_root / f"{split}_samples.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    samples.append(payload)
    return samples


def summarize_counts_from_annotations(dataset_root: Path) -> Dict[str, int]:
    count_primary_none = 0
    count_primary_no_box = 0
    total = 0
    for split in ["train", "valid", "test"]:
        path = dataset_root / f"{split}_samples.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                ann_path = Path(str(sample.get("annotation_json_path", "")))
                if not ann_path.is_file():
                    continue
                payload = load_json(ann_path)
                ann_list = payload.get("annotations", [])
                if not isinstance(ann_list, list):
                    continue
                for ann in ann_list:
                    if not isinstance(ann, dict):
                        continue
                    total += 1
                    p = int(ann.get("gt_primary_camera", NONE_CAMERA_ID))
                    has = [int(v) for v in safe_list4(ann.get("gt_has_camera_box"), default=0.0)]
                    if p == NONE_CAMERA_ID:
                        count_primary_none += 1
                    if 0 <= p < 4 and has[p] == 0:
                        count_primary_no_box += 1
    return {
        "total_objects": int(total),
        "gt_primary_camera_none_count": int(count_primary_none),
        "gt_primary_non_none_but_no_primary_box_count": int(count_primary_no_box),
    }


def write_report(
    report_path: Path,
    dataset_root: Path,
    summary: Dict[str, object],
    coco_patch_stats: Dict[str, Dict[str, int]],
    details_csv_path: Path,
    details_jsonl_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Primary Camera No-Box Fix")
    lines.append("")
    lines.append(f"- dataset_root: `{dataset_root}`")
    lines.append(f"- total_objects: **{summary['total_objects']}**")
    lines.append(f"- gt_primary_camera=4 count: **{summary['gt_primary_camera_none_count']}**")
    lines.append(
        "- gt_primary_camera != 4 && gt_has_camera_box[primary] == 0 count: "
        f"**{summary['gt_primary_non_none_but_no_primary_box_count']}**"
    )
    lines.append("")
    lines.append("## Fix Stats")
    lines.append("")
    lines.append(f"- fixed_by_gt_filter_fallback_count: **{summary['fixed_by_gt_filter_fallback_count']}**")
    lines.append(f"- fixed_by_3d_projection_count: **{summary['fixed_by_3d_projection_count']}**")
    lines.append(f"- still_no_primary_box_count: **{summary['still_no_primary_box_count']}**")
    lines.append(f"- overwrite_guard_protected_count: **{summary['overwrite_guard_protected_count']}**")
    lines.append("")
    lines.append("### still_no_primary_box reason 分布")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("| --- | ---: |")
    reason_dist = summary.get("still_no_primary_box_reason_dist", {})
    if isinstance(reason_dist, dict) and reason_dist:
        for reason, count in sorted(reason_dist.items(), key=lambda x: (-int(x[1]), x[0])):
            lines.append(f"| {reason} | {int(count)} |")
    else:
        lines.append("| (empty) | 0 |")
    lines.append("")
    lines.append("## COCO 同步")
    lines.append("")
    lines.append("| coco_file | changed_fields | missing_sample | mismatch_len |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name in ["train_coco.json", "valid_coco.json", "test_coco.json"]:
        stats = coco_patch_stats.get(name, {})
        lines.append(
            f"| {name} | {int(stats.get('changed_fields', 0))} | "
            f"{int(stats.get('missing_sample', 0))} | {int(stats.get('mismatch_len', 0))} |"
        )
    lines.append("")
    lines.append("## Debug 明细")
    lines.append("")
    lines.append(f"- details csv: `{details_csv_path}`")
    lines.append(f"- details jsonl: `{details_jsonl_path}`")
    lines.append("")
    ensure_parent(report_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root
    report_path: Path = args.report_path
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(dataset_root)
    if not samples:
        raise RuntimeError(f"No samples found under {dataset_root}")

    ann_cache: Dict[Path, Dict[str, object]] = {}
    raw_yaw_cache: Dict[Path, Dict[str, object]] = {}
    raw_full_cache: Dict[Path, Dict[str, object]] = {}
    opv2v_cache: Dict[Path, Dict[str, object]] = {}

    updated_ann_fields_by_sample_id: Dict[str, List[Dict[str, object]]] = {}
    details_rows: List[Dict[str, object]] = []

    fixed_by_gt_filter_fallback_count = 0
    fixed_by_3d_projection_count = 0
    overwrite_guard_protected_count = 0
    still_no_primary_box_reason_dist: Counter = Counter()

    changed_annotation_files = 0

    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        tower_id = str(sample.get("tower_id", ""))
        ann_path = Path(str(sample.get("annotation_json_path", "")))
        raw_yaw_path = Path(str(sample.get("gt_filter_only_yaw_path", "")))
        raw_full_path = Path(str(sample.get("gt_filter_path", "")))
        opv2v_path = Path(str(sample.get("opv2v_yaml_path", "")))

        if not ann_path.is_file() or not raw_yaw_path.is_file() or not raw_full_path.is_file() or not opv2v_path.is_file():
            continue

        if ann_path not in ann_cache:
            ann_cache[ann_path] = load_json(ann_path)
        if raw_yaw_path not in raw_yaw_cache:
            raw_yaw_cache[raw_yaw_path] = load_json(raw_yaw_path)
        if raw_full_path not in raw_full_cache:
            raw_full_cache[raw_full_path] = load_json(raw_full_path)
        ann_payload = ann_cache[ann_path]
        ann_list = ann_payload.get("annotations", [])
        if not isinstance(ann_list, list):
            continue

        tower_objects_yaw = get_tower_objects(raw_yaw_cache[raw_yaw_path], tower_id)
        tower_objects_full = get_tower_objects(raw_full_cache[raw_full_path], tower_id)

        sample_changed = False
        updated_fields_list: List[Dict[str, object]] = []

        for ann_idx, ann in enumerate(ann_list):
            if not isinstance(ann, dict):
                updated_fields_list.append(
                    {
                        "gt_primary_camera": NONE_CAMERA_ID,
                        "gt_visible_cameras": [0, 0, 0, 0],
                        "gt_camera_box_2d": [[0.0, 0.0, 0.0, 0.0] for _ in range(4)],
                        "gt_has_camera_box": [0, 0, 0, 0],
                        "gt_camera_poly_area": [0.0, 0.0, 0.0, 0.0],
                    }
                )
                continue

            instance_name = str(ann.get("instance_name", ""))
            raw_obj_yaw = tower_objects_yaw.get(instance_name) if isinstance(tower_objects_yaw, dict) else None
            raw_obj_full = tower_objects_full.get(instance_name) if isinstance(tower_objects_full, dict) else None

            old_primary = int(ann.get("gt_primary_camera", NONE_CAMERA_ID))
            old_has = [int(v) for v in safe_list4(ann.get("gt_has_camera_box"), default=0.0)]
            old_boxes = sanitize_boxes4(ann.get("gt_camera_box_2d"))
            old_visible = [int(v) for v in safe_list4(ann.get("gt_visible_cameras"), default=0.0)]
            old_poly_area = [float(v) for v in safe_list4(ann.get("gt_camera_poly_area"), default=0.0)]

            old_no_primary_box = bool(0 <= old_primary < 4 and old_has[old_primary] == 0)

            # Lazy-load camera calibration only when current annotation may need projection.
            opv2v_payload: Dict[str, object] = {}
            if old_no_primary_box:
                if opv2v_path not in opv2v_cache:
                    opv2v_cache[opv2v_path] = load_yaml(opv2v_path)
                cached = opv2v_cache.get(opv2v_path)
                if isinstance(cached, dict):
                    opv2v_payload = cached

            new_fields, fix_result = fix_primary_camera_box(
                ann=ann,
                old_has=old_has,
                old_boxes=old_boxes,
                old_visible=old_visible,
                old_poly_area=old_poly_area,
                raw_obj_yaw=raw_obj_yaw,
                raw_obj_full=raw_obj_full,
                opv2v_payload=opv2v_payload,
            )

            new_primary = int(new_fields["gt_primary_camera"])
            new_has = [int(v) for v in new_fields["gt_has_camera_box"]]
            new_no_primary_box = bool(0 <= new_primary < 4 and new_has[new_primary] == 0)

            if old_no_primary_box and not new_no_primary_box:
                if fix_result.source == "gt_filter_fallback_raw_primary":
                    fixed_by_gt_filter_fallback_count += 1
                elif fix_result.source == "3d_projection":
                    fixed_by_3d_projection_count += 1

            if new_no_primary_box:
                reason_key = fix_result.final_invalid_reason or "unknown"
                still_no_primary_box_reason_dist[reason_key] += 1

            if fix_result.overwrite_guarded:
                overwrite_guard_protected_count += 1

            details_rows.append(
                {
                    "sample_id": sample_id,
                    "ann_index": ann_idx,
                    "instance_name": instance_name,
                    "old_primary": old_primary,
                    "new_primary": new_primary,
                    "old_primary_has_box": int(old_has[old_primary]) if 0 <= old_primary < 4 else 0,
                    "new_primary_has_box": int(new_has[new_primary]) if 0 <= new_primary < 4 else 0,
                    "old_no_primary_box": int(old_no_primary_box),
                    "new_no_primary_box": int(new_no_primary_box),
                    "fix_source": fix_result.source,
                    "fallback_triggered": int(fix_result.fallback_triggered),
                    "fallback_trigger_reason": fix_result.fallback_trigger_reason,
                    "primary_reason_yaw": fix_result.primary_reason_yaw,
                    "primary_reason_full": fix_result.primary_reason_full,
                    "projection_status": fix_result.projection_status,
                    "final_invalid_reason": fix_result.final_invalid_reason,
                    "overwrite_guarded": int(fix_result.overwrite_guarded),
                }
            )

            for key in ("gt_primary_camera", "gt_visible_cameras", "gt_camera_box_2d", "gt_has_camera_box", "gt_camera_poly_area"):
                if ann.get(key) != new_fields[key]:
                    ann[key] = new_fields[key]
                    sample_changed = True

            updated_fields_list.append(new_fields)

        if sample_changed:
            write_json(ann_path, ann_payload)
            changed_annotation_files += 1
        updated_ann_fields_by_sample_id[sample_id] = updated_fields_list

    # Sync COCO.
    coco_patch_stats: Dict[str, Dict[str, int]] = {}
    for coco_name in ("train_coco.json", "valid_coco.json", "test_coco.json"):
        coco_path = dataset_root / coco_name
        coco_patch_stats[coco_name] = patch_coco_with_updated_annotations(
            coco_path=coco_path,
            updated_ann_fields_by_sample_id=updated_ann_fields_by_sample_id,
        )

    counts = summarize_counts_from_annotations(dataset_root)
    summary = {
        **counts,
        "samples_count": int(len(samples)),
        "changed_annotation_files": int(changed_annotation_files),
        "fixed_by_gt_filter_fallback_count": int(fixed_by_gt_filter_fallback_count),
        "fixed_by_3d_projection_count": int(fixed_by_3d_projection_count),
        "still_no_primary_box_count": int(counts["gt_primary_non_none_but_no_primary_box_count"]),
        "still_no_primary_box_reason_dist": {k: int(v) for k, v in sorted(still_no_primary_box_reason_dist.items())},
        "overwrite_guard_protected_count": int(overwrite_guard_protected_count),
    }

    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)

    details_jsonl = output_dir / "fix_details.jsonl"
    with details_jsonl.open("w", encoding="utf-8") as handle:
        for row in details_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    details_csv = output_dir / "fix_details.csv"
    fieldnames = [
        "sample_id",
        "ann_index",
        "instance_name",
        "old_primary",
        "new_primary",
        "old_primary_has_box",
        "new_primary_has_box",
        "old_no_primary_box",
        "new_no_primary_box",
        "fix_source",
        "fallback_triggered",
        "fallback_trigger_reason",
        "primary_reason_yaw",
        "primary_reason_full",
        "projection_status",
        "final_invalid_reason",
        "overwrite_guarded",
    ]
    with details_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in details_rows:
            writer.writerow(row)

    write_report(
        report_path=report_path,
        dataset_root=dataset_root,
        summary=summary,
        coco_patch_stats=coco_patch_stats,
        details_csv_path=details_csv,
        details_jsonl_path=details_jsonl,
    )

    print(
        json.dumps(
            {
                "report_path": str(report_path.resolve()),
                "summary_path": str(summary_path.resolve()),
                "details_csv": str(details_csv.resolve()),
                "details_jsonl": str(details_jsonl.resolve()),
                "gt_primary_camera_none_count": summary["gt_primary_camera_none_count"],
                "gt_primary_non_none_but_no_primary_box_count": summary[
                    "gt_primary_non_none_but_no_primary_box_count"
                ],
                "fixed_by_gt_filter_fallback_count": summary["fixed_by_gt_filter_fallback_count"],
                "fixed_by_3d_projection_count": summary["fixed_by_3d_projection_count"],
                "still_no_primary_box_count": summary["still_no_primary_box_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
