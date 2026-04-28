#!/usr/bin/env python3
"""Check radar/ego z-axis direction and RouteROI top/bottom projection convention.

This script does NOT modify training code. It only reads raw data/calibration,
produces debug artifacts, and writes a markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

CAM_ORDER = ["CamBack", "CamFront", "CamLeft", "CamRight"]
OPV2V_CAM_MAP = {
    "CamBack": "camera3",
    "CamFront": "camera1",
    "CamLeft": "camera2",
    "CamRight": "camera0",
}
CAMERA_CV_FROM_LOCAL = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check z-axis direction from raw data")
    parser.add_argument(
        "--samples-jsonl",
        nargs="+",
        default=[
            "prepared/sealand_single_tower_2km_super4_1536_route_roi_min8/train_samples.jsonl",
            "prepared/sealand_single_tower_2km_super4_1536_route_roi_min8/valid_samples.jsonl",
            "prepared/sealand_single_tower_2km_super4_1536_route_roi_min8/test_samples.jsonl",
        ],
        help="Input sample index jsonl files",
    )
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--pcd-samples", type=int, default=12)
    parser.add_argument("--projection-samples", type=int, default=12)
    parser.add_argument("--box-frames", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/z_axis_direction_debug"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/z_axis_direction_check.md"),
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_records(paths: Sequence[Path]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.6f}"


def percentile_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.percentile(values, 0.0)),
        "p1": float(np.percentile(values, 1.0)),
        "median": float(np.percentile(values, 50.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.percentile(values, 100.0)),
    }


def read_pcd_xyz(pcd_path: Path) -> Tuple[List[str], np.ndarray]:
    fields: Optional[List[str]] = None
    x_idx = y_idx = z_idx = -1
    max_idx = -1
    points: List[Tuple[float, float, float]] = []
    in_data = False

    with pcd_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if not in_data:
                if stripped.startswith("FIELDS"):
                    fields = stripped.split()[1:]
                elif stripped.startswith("DATA"):
                    if fields is None:
                        raise ValueError(f"Missing FIELDS in {pcd_path}")
                    if "ascii" not in stripped.lower():
                        raise ValueError(f"Only ASCII PCD is supported: {pcd_path}")
                    if "x" not in fields or "y" not in fields or "z" not in fields:
                        raise ValueError(f"x/y/z fields missing in {pcd_path}: {fields}")
                    x_idx = fields.index("x")
                    y_idx = fields.index("y")
                    z_idx = fields.index("z")
                    max_idx = max(x_idx, y_idx, z_idx)
                    in_data = True
                continue

            tokens = stripped.split()
            if len(tokens) <= max_idx:
                continue
            try:
                points.append(
                    (
                        float(tokens[x_idx]),
                        float(tokens[y_idx]),
                        float(tokens[z_idx]),
                    )
                )
            except ValueError:
                continue

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = np.zeros((0, 3), dtype=np.float64)
    return fields or [], pts


def quat_to_rot(q: Dict[str, float]) -> np.ndarray:
    w = float(q.get("w", 1.0))
    x = float(q.get("x", 0.0))
    y = float(q.get("y", 0.0))
    z = float(q.get("z", 0.0))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def obb_inside_mask(
    points_xyz: np.ndarray,
    center_xyz: np.ndarray,
    extent_xyz: np.ndarray,
    rotation_local_to_global: np.ndarray,
) -> np.ndarray:
    # world = R @ local + center -> local(row) = (world - center) @ R
    local = (points_xyz - center_xyz.reshape(1, 3)) @ rotation_local_to_global
    return (
        (np.abs(local[:, 0]) <= extent_xyz[0] + 1e-6)
        & (np.abs(local[:, 1]) <= extent_xyz[1] + 1e-6)
        & (np.abs(local[:, 2]) <= extent_xyz[2] + 1e-6)
    )


def area_of_box2d(box: Dict[str, object]) -> float:
    try:
        w = max(0.0, float(box["xmax"]) - float(box["xmin"]))
        h = max(0.0, float(box["ymax"]) - float(box["ymin"]))
        return w * h
    except Exception:
        return 0.0


def choose_primary_camera(obj_rgb: Dict[str, object]) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    cams = obj_rgb.get("cams", {}) if isinstance(obj_rgb, dict) else {}
    best_cam: Optional[str] = None
    best_box: Optional[Dict[str, float]] = None
    best_area = -1.0
    for cam_name in CAM_ORDER:
        cam_entry = cams.get(cam_name) if isinstance(cams, dict) else None
        if not isinstance(cam_entry, dict):
            continue
        box = cam_entry.get("box_2d")
        if not isinstance(box, dict):
            continue
        area = area_of_box2d(box)
        if area > best_area:
            best_area = area
            best_cam = cam_name
            best_box = {
                "xmin": float(box["xmin"]),
                "ymin": float(box["ymin"]),
                "xmax": float(box["xmax"]),
                "ymax": float(box["ymax"]),
            }
    return best_cam, best_box


def project_points(
    points_xyz: np.ndarray,
    t_camcv_ego: np.ndarray,
    intrinsic_k: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    homo = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (t_camcv_ego @ homo.T).T[:, :3]
    z_cam = cam[:, 2]
    uv = np.full((points_xyz.shape[0], 2), np.nan, dtype=np.float64)
    valid = z_cam > 1e-6
    if np.any(valid):
        proj = (intrinsic_k @ cam[valid].T).T
        uv[valid, 0] = proj[:, 0] / proj[:, 2]
        uv[valid, 1] = proj[:, 1] / proj[:, 2]
    return uv, z_cam


def build_corners_ab(
    center_xyz: np.ndarray,
    extent_xy: np.ndarray,
    height: float,
    rotation_local_to_global: np.ndarray,
) -> Dict[str, np.ndarray]:
    ex, ey = float(extent_xy[0]), float(extent_xy[1])
    # 4 corners on XY plane in object local frame.
    local_xy = np.array(
        [[ex, ey], [ex, -ey], [-ex, -ey], [-ex, ey]],
        dtype=np.float64,
    )

    def local_to_world(local_pts: np.ndarray) -> np.ndarray:
        return (rotation_local_to_global @ local_pts.T).T + center_xyz.reshape(1, 3)

    # A hypothesis: z-up -> top_z = bottom_z + height
    local_bottom_a = np.concatenate([local_xy, np.full((4, 1), -height / 2.0)], axis=1)
    local_top_a = np.concatenate([local_xy, np.full((4, 1), +height / 2.0)], axis=1)

    # B hypothesis: z-down -> top_z = bottom_z - height
    local_bottom_b = np.concatenate([local_xy, np.full((4, 1), +height / 2.0)], axis=1)
    local_top_b = np.concatenate([local_xy, np.full((4, 1), -height / 2.0)], axis=1)

    return {
        "bottom_a": local_to_world(local_bottom_a),
        "top_a": local_to_world(local_top_a),
        "bottom_b": local_to_world(local_bottom_b),
        "top_b": local_to_world(local_top_b),
    }


def draw_polyline(image: np.ndarray, points_uv: np.ndarray, color: Tuple[int, int, int], closed: bool = True) -> None:
    finite = np.isfinite(points_uv[:, 0]) & np.isfinite(points_uv[:, 1])
    if np.count_nonzero(finite) < 2:
        return
    pts = points_uv[finite].astype(np.int32)
    cv2.polylines(image, [pts.reshape(-1, 1, 2)], closed, color, 2, lineType=cv2.LINE_AA)


def draw_cuboid(image: np.ndarray, bottom_uv: np.ndarray, top_uv: np.ndarray, color: Tuple[int, int, int]) -> None:
    draw_polyline(image, bottom_uv, color, closed=True)
    draw_polyline(image, top_uv, color, closed=True)
    for idx in range(4):
        p0 = bottom_uv[idx]
        p1 = top_uv[idx]
        if np.isfinite(p0[0]) and np.isfinite(p0[1]) and np.isfinite(p1[0]) and np.isfinite(p1[1]):
            cv2.line(
                image,
                (int(round(p0[0])), int(round(p0[1]))),
                (int(round(p1[0])), int(round(p1[1]))),
                color,
                2,
                lineType=cv2.LINE_AA,
            )


def sanitize_name(text: str) -> str:
    safe = text.replace("/", "__").replace(" ", "_")
    for ch in [":", "\\", "*", "?", '"', "<", ">", "|"]:
        safe = safe.replace(ch, "_")
    return safe


@dataclass
class ProjectionResult:
    sample_id: str
    split: str
    scene_id: str
    tower_id: str
    frame_id: str
    object_name: str
    camera_name: str
    gt_box_xmin: float
    gt_box_ymin: float
    gt_box_xmax: float
    gt_box_ymax: float
    mean_v_bottom_a: float
    mean_v_top_a: float
    mean_v_bottom_b: float
    mean_v_top_b: float
    a_top_less_bottom: bool
    b_top_less_bottom: bool
    vis_path: str


def collect_pcd_stats(
    records: Sequence[Dict[str, object]],
    rng: random.Random,
    sample_count: int,
) -> List[Dict[str, object]]:
    if not records:
        return []
    if sample_count >= len(records):
        selected = list(records)
    else:
        selected = rng.sample(list(records), sample_count)

    rows: List[Dict[str, object]] = []
    for rec in selected:
        pcd_path = Path(rec["radar_pcd_path"])  # type: ignore[index]
        gt_filter_path = Path(rec["gt_filter_path"])  # type: ignore[index]
        tower_id = str(rec["tower_id"])

        fields, pts = read_pcd_xyz(pcd_path)
        if pts.shape[0] == 0:
            continue
        z_stats = percentile_stats(pts[:, 2])

        gt = load_json(gt_filter_path)
        tower = gt.get("towers", {}).get(tower_id, {})
        objects = tower.get("objects", {}) if isinstance(tower, dict) else {}

        best_obj = None
        best_inside = 0
        best_inside_stats = None
        for obj_name, obj in objects.items():
            if not isinstance(obj, dict):
                continue
            radar_proj = obj.get("radar_proj")
            if not isinstance(radar_proj, dict):
                continue
            center = radar_proj.get("center", {})
            extent = radar_proj.get("extent", {})
            quat = radar_proj.get("rotation_quat", {})
            if not isinstance(center, dict) or not isinstance(extent, dict) or not isinstance(quat, dict):
                continue
            try:
                center_xyz = np.array(
                    [float(center["x"]), float(center["y"]), float(center["z"])], dtype=np.float64
                )
                extent_xyz = np.array(
                    [float(extent["x"]), float(extent["y"]), float(extent["z"])], dtype=np.float64
                )
            except Exception:
                continue
            if np.any(extent_xyz <= 0.0):
                continue
            rot = quat_to_rot(quat)
            inside = obb_inside_mask(pts, center_xyz, extent_xyz, rot)
            inside_count = int(np.count_nonzero(inside))
            if inside_count <= best_inside:
                continue
            best_inside = inside_count
            best_obj = obj_name
            if inside_count > 0:
                z_inside = pts[inside, 2]
                best_inside_stats = percentile_stats(z_inside)
            else:
                best_inside_stats = None

        row = {
            "sample_id": rec["sample_id"],
            "tower_id": tower_id,
            "frame_id": rec["frame_id"],
            "radar_pcd_path": str(pcd_path),
            "pcd_fields": ",".join(fields),
            "num_points": int(pts.shape[0]),
            "z_min": z_stats["min"],
            "z_p1": z_stats["p1"],
            "z_median": z_stats["median"],
            "z_p99": z_stats["p99"],
            "z_max": z_stats["max"],
            "inside_object_name": best_obj or "",
            "inside_point_count": best_inside,
            "inside_z_min": best_inside_stats["min"] if best_inside_stats else float("nan"),
            "inside_z_p1": best_inside_stats["p1"] if best_inside_stats else float("nan"),
            "inside_z_median": best_inside_stats["median"] if best_inside_stats else float("nan"),
            "inside_z_p99": best_inside_stats["p99"] if best_inside_stats else float("nan"),
            "inside_z_max": best_inside_stats["max"] if best_inside_stats else float("nan"),
        }
        rows.append(row)

    return rows


def collect_box_semantics(
    records: Sequence[Dict[str, object]],
    frame_limit: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    field_sample: Dict[str, object] = {}
    if records:
        rec0 = records[0]
        tower = str(rec0["tower_id"])
        for key in ["gt_filter_path", "gt_filter_only_yaw_path", "gt_sensor_path"]:
            path = Path(rec0[key])  # type: ignore[index]
            data = load_json(path)
            top_keys = list(data.keys())
            tower_data = data.get("towers", {}).get(tower, {}) if isinstance(data, dict) else {}
            objects = tower_data.get("objects", {}) if isinstance(tower_data, dict) else {}
            obj_name = next(iter(objects.keys())) if objects else ""
            obj_keys: List[str] = []
            sample_values: Dict[str, object] = {}
            if obj_name:
                obj = objects[obj_name]
                if isinstance(obj, dict):
                    obj_keys = list(obj.keys())
                    if "bbox_m" in obj:
                        sample_values["bbox_m"] = obj["bbox_m"]
                    if "radar_proj" in obj:
                        rp = obj["radar_proj"]
                        if isinstance(rp, dict):
                            sample_values["radar_proj.center"] = rp.get("center")
                            sample_values["radar_proj.extent"] = rp.get("extent")
                            corners = rp.get("corners_3d")
                            if isinstance(corners, list) and corners:
                                sample_values["radar_proj.corners_3d[0]"] = corners[0]
                    if "bev_rot_only_yaw" in obj:
                        sample_values["bev_rot_only_yaw"] = obj["bev_rot_only_yaw"]
            field_sample[key] = {
                "path": str(path),
                "top_keys": top_keys,
                "tower_keys": list(tower_data.keys()) if isinstance(tower_data, dict) else [],
                "sample_object": obj_name,
                "object_keys": obj_keys,
                "sample_values": sample_values,
            }

    for rec in records[:frame_limit]:
        gt = load_json(Path(rec["gt_filter_path"]))
        tower = str(rec["tower_id"])
        objects = gt.get("towers", {}).get(tower, {}).get("objects", {})
        if not isinstance(objects, dict):
            continue
        for obj_name, obj in objects.items():
            if not isinstance(obj, dict):
                continue
            bbox_m = obj.get("bbox_m")
            radar_proj = obj.get("radar_proj")
            if not isinstance(bbox_m, dict) or not isinstance(radar_proj, dict):
                continue
            center = radar_proj.get("center")
            corners = radar_proj.get("corners_3d")
            if not isinstance(center, dict) or not isinstance(corners, list) or not corners:
                continue
            try:
                center_z = float(center["z"])
                height = float(bbox_m["H"])
                z_vals = np.array([float(pt[2]) for pt in corners], dtype=np.float64)
            except Exception:
                continue
            z_min = float(np.min(z_vals))
            z_max = float(np.max(z_vals))
            z_mid = 0.5 * (z_min + z_max)

            rows.append(
                {
                    "sample_id": rec["sample_id"],
                    "tower_id": rec["tower_id"],
                    "frame_id": rec["frame_id"],
                    "object_name": obj_name,
                    "center_z": center_z,
                    "height": height,
                    "possible_bottom_z": center_z - 0.5 * height,
                    "possible_top_z": center_z + 0.5 * height,
                    "corners_z_min": z_min,
                    "corners_z_max": z_max,
                    "corners_z_mid": z_mid,
                    "corners_height": z_max - z_min,
                    "err_center_vs_mid": abs(center_z - z_mid),
                    "err_center_vs_min": abs(center_z - z_min),
                    "err_center_vs_max": abs(center_z - z_max),
                }
            )

    return rows, field_sample


def run_projection_ab_test(
    records: Sequence[Dict[str, object]],
    rng: random.Random,
    target_count: int,
    vis_dir: Path,
) -> List[ProjectionResult]:
    out: List[ProjectionResult] = []
    vis_dir.mkdir(parents=True, exist_ok=True)

    shuffled = list(records)
    rng.shuffle(shuffled)

    for rec in shuffled:
        if len(out) >= target_count:
            break

        tower = str(rec["tower_id"])
        gt_filter = load_json(Path(rec["gt_filter_path"]))
        gt_rgb = load_json(Path(rec["gt_filter_only_yaw_path"]))
        opv2v = load_yaml(Path(rec["opv2v_yaml_path"]))

        objects_metric = gt_filter.get("towers", {}).get(tower, {}).get("objects", {})
        objects_rgb = gt_rgb.get("towers", {}).get(tower, {}).get("objects", {})
        if not isinstance(objects_metric, dict) or not isinstance(objects_rgb, dict):
            continue

        object_names = list(set(objects_metric.keys()) & set(objects_rgb.keys()))
        rng.shuffle(object_names)

        chosen_row: Optional[ProjectionResult] = None
        for obj_name in object_names:
            obj_metric = objects_metric.get(obj_name)
            obj_rgb = objects_rgb.get(obj_name)
            if not isinstance(obj_metric, dict) or not isinstance(obj_rgb, dict):
                continue

            cam_name, gt_box = choose_primary_camera(obj_rgb)
            if cam_name is None or gt_box is None:
                continue

            radar_proj = obj_metric.get("radar_proj")
            bbox_m = obj_metric.get("bbox_m")
            if not isinstance(radar_proj, dict) or not isinstance(bbox_m, dict):
                continue

            center = radar_proj.get("center")
            extent = radar_proj.get("extent")
            quat = radar_proj.get("rotation_quat")
            if not isinstance(center, dict) or not isinstance(extent, dict) or not isinstance(quat, dict):
                continue

            try:
                center_xyz = np.array(
                    [float(center["x"]), float(center["y"]), float(center["z"])],
                    dtype=np.float64,
                )
                extent_xy = np.array([float(extent["x"]), float(extent["y"])], dtype=np.float64)
                height = float(bbox_m.get("H", 0.0))
                if height <= 0.0:
                    height = float(extent.get("z", 0.0)) * 2.0
            except Exception:
                continue
            if height <= 0.0 or np.any(extent_xy <= 0.0):
                continue

            cam_meta_name = OPV2V_CAM_MAP[cam_name]
            cam_meta = opv2v.get(cam_meta_name)
            if not isinstance(cam_meta, dict):
                continue

            try:
                intrinsic = np.array(cam_meta["intrinsic"], dtype=np.float64)
                extrinsic_local = np.array(cam_meta["extrinsic"], dtype=np.float64)
            except Exception:
                continue
            if intrinsic.shape != (3, 3) or extrinsic_local.shape != (4, 4):
                continue

            t_camcv_ego = CAMERA_CV_FROM_LOCAL @ extrinsic_local

            rot = quat_to_rot(quat)
            corners = build_corners_ab(center_xyz, extent_xy, height, rot)

            bottom_a_uv, _ = project_points(corners["bottom_a"], t_camcv_ego, intrinsic)
            top_a_uv, _ = project_points(corners["top_a"], t_camcv_ego, intrinsic)
            bottom_b_uv, _ = project_points(corners["bottom_b"], t_camcv_ego, intrinsic)
            top_b_uv, _ = project_points(corners["top_b"], t_camcv_ego, intrinsic)

            mean_v_bottom_a = float(np.nanmean(bottom_a_uv[:, 1]))
            mean_v_top_a = float(np.nanmean(top_a_uv[:, 1]))
            mean_v_bottom_b = float(np.nanmean(bottom_b_uv[:, 1]))
            mean_v_top_b = float(np.nanmean(top_b_uv[:, 1]))

            if (
                math.isnan(mean_v_bottom_a)
                or math.isnan(mean_v_top_a)
                or math.isnan(mean_v_bottom_b)
                or math.isnan(mean_v_top_b)
            ):
                continue

            cam_img_path = Path(rec["camera_paths"][cam_name])  # type: ignore[index]
            if not cam_img_path.is_file():
                continue
            image = cv2.imread(str(cam_img_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            # GT 2D box (green)
            cv2.rectangle(
                image,
                (int(round(gt_box["xmin"])), int(round(gt_box["ymin"]))),
                (int(round(gt_box["xmax"])), int(round(gt_box["ymax"]))),
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )

            # A (blue), B (red)
            draw_cuboid(image, bottom_a_uv, top_a_uv, (255, 0, 0))
            draw_cuboid(image, bottom_b_uv, top_b_uv, (0, 0, 255))

            text_lines = [
                f"sample={rec['sample_id']}",
                f"obj={obj_name} cam={cam_name}",
                f"A blue: mean_v_bottom={mean_v_bottom_a:.2f} mean_v_top={mean_v_top_a:.2f} top<bottom={mean_v_top_a < mean_v_bottom_a}",
                f"B red:  mean_v_bottom={mean_v_bottom_b:.2f} mean_v_top={mean_v_top_b:.2f} top<bottom={mean_v_top_b < mean_v_bottom_b}",
            ]
            y = 24
            for text in text_lines:
                cv2.putText(
                    image,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (15, 15, 15),
                    1,
                    lineType=cv2.LINE_AA,
                )
                y += 22

            vis_name = sanitize_name(f"{rec['sample_id']}__{obj_name}__{cam_name}.png")
            vis_path = vis_dir / vis_name
            ensure_parent(vis_path)
            cv2.imwrite(str(vis_path), image)

            chosen_row = ProjectionResult(
                sample_id=str(rec["sample_id"]),
                split=str(rec["split"]),
                scene_id=str(rec["scene_id"]),
                tower_id=str(rec["tower_id"]),
                frame_id=str(rec["frame_id"]),
                object_name=str(obj_name),
                camera_name=cam_name,
                gt_box_xmin=float(gt_box["xmin"]),
                gt_box_ymin=float(gt_box["ymin"]),
                gt_box_xmax=float(gt_box["xmax"]),
                gt_box_ymax=float(gt_box["ymax"]),
                mean_v_bottom_a=mean_v_bottom_a,
                mean_v_top_a=mean_v_top_a,
                mean_v_bottom_b=mean_v_bottom_b,
                mean_v_top_b=mean_v_top_b,
                a_top_less_bottom=bool(mean_v_top_a < mean_v_bottom_a),
                b_top_less_bottom=bool(mean_v_top_b < mean_v_bottom_b),
                vis_path=str(vis_path),
            )
            break

        if chosen_row is not None:
            out.append(chosen_row)

    return out


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_parent(path)
    if not rows:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def projection_rows_to_dicts(rows: Sequence[ProjectionResult]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "sample_id": row.sample_id,
                "split": row.split,
                "scene_id": row.scene_id,
                "tower_id": row.tower_id,
                "frame_id": row.frame_id,
                "object_name": row.object_name,
                "camera_name": row.camera_name,
                "gt_box_xmin": row.gt_box_xmin,
                "gt_box_ymin": row.gt_box_ymin,
                "gt_box_xmax": row.gt_box_xmax,
                "gt_box_ymax": row.gt_box_ymax,
                "mean_v_bottom_A": row.mean_v_bottom_a,
                "mean_v_top_A": row.mean_v_top_a,
                "mean_v_bottom_B": row.mean_v_bottom_b,
                "mean_v_top_B": row.mean_v_top_b,
                "A_top_less_bottom": int(row.a_top_less_bottom),
                "B_top_less_bottom": int(row.b_top_less_bottom),
                "vis_path": row.vis_path,
            }
        )
    return out


def build_report(
    report_path: Path,
    output_dir: Path,
    field_sample: Dict[str, object],
    pcd_rows: Sequence[Dict[str, object]],
    box_rows: Sequence[Dict[str, object]],
    projection_rows: Sequence[ProjectionResult],
) -> None:
    ensure_parent(report_path)

    center_match = [row["err_center_vs_mid"] for row in box_rows]
    min_match = [np.minimum(row["err_center_vs_min"], row["err_center_vs_max"]) for row in box_rows]
    height_err = [abs(row["height"] - row["corners_height"]) for row in box_rows]

    center_match_med = float(np.median(center_match)) if center_match else float("nan")
    min_match_med = float(np.median(min_match)) if min_match else float("nan")
    height_err_med = float(np.median(height_err)) if height_err else float("nan")

    center_z_semantics = "center_z"
    if not (center_match_med < min_match_med):
        center_z_semantics = "uncertain"

    a_true = sum(1 for row in projection_rows if row.a_top_less_bottom)
    b_true = sum(1 for row in projection_rows if row.b_top_less_bottom)

    # Decision by majority on top<bottom condition.
    if a_true > b_true:
        z_direction = "z increases upward"
        recommended_formula = "top_z = bottom_z + height"
    elif b_true > a_true:
        z_direction = "z increases downward"
        recommended_formula = "top_z = bottom_z - height"
    else:
        z_direction = "inconclusive"
        recommended_formula = "inconclusive"

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Z Axis Direction Check\n\n")
        handle.write("## Scope\n")
        handle.write("- Data source: raw `radar_pcd`, `gt_filter`, `gt_filter_only_yaw`, `gt_sensor`, `opv2v_yaml`.\n")
        handle.write(f"- Output directory: `{output_dir}`.\n")
        handle.write(f"- Projection validation samples: {len(projection_rows)}.\n\n")

        handle.write("## 1) Raw radar PCD z statistics\n")
        handle.write("- Parsed `FIELDS` from each sampled PCD and extracted `x/y/z`.\n")
        handle.write("- Per-sample stats and in-box stats saved to `pcd_z_stats.csv`.\n\n")
        handle.write("| sample_id | num_points | z_min | z_p1 | z_median | z_p99 | z_max | inside_object | inside_points | inside_z_median |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|\n")
        for row in pcd_rows:
            handle.write(
                "| {sample_id} | {num_points} | {z_min} | {z_p1} | {z_median} | {z_p99} | {z_max} | {inside_object_name} | {inside_point_count} | {inside_z_median} |\n".format(
                    sample_id=row["sample_id"],
                    num_points=row["num_points"],
                    z_min=format_float(float(row["z_min"])),
                    z_p1=format_float(float(row["z_p1"])),
                    z_median=format_float(float(row["z_median"])),
                    z_p99=format_float(float(row["z_p99"])),
                    z_max=format_float(float(row["z_max"])),
                    inside_object_name=row["inside_object_name"] or "-",
                    inside_point_count=row["inside_point_count"],
                    inside_z_median=format_float(float(row["inside_z_median"])),
                )
            )
        handle.write("\n")

        handle.write("## 2) Raw 3D box annotation field check\n")
        handle.write("- Field snapshots from one sample file for each source:\n")
        for key, info in field_sample.items():
            handle.write(f"- `{key}`: path=`{info['path']}`\n")
            handle.write(f"  top_keys={info['top_keys']}\n")
            handle.write(f"  tower_keys={info['tower_keys']}\n")
            handle.write(f"  sample_object={info['sample_object']}\n")
            handle.write(f"  object_keys={info['object_keys']}\n")
            handle.write(f"  sample_values={info['sample_values']}\n")
        handle.write("\n")

        handle.write("- Per-target z semantic rows saved to `box_z_semantics.csv` (includes `center_z`, `height`, `possible_bottom_z=center_z-height/2`, `possible_top_z=center_z+height/2`).\n")
        handle.write(f"- Median |center_z - (z_min+z_max)/2| = {center_match_med:.6f}\n")
        handle.write(f"- Median min(|center_z-z_min|, |center_z-z_max|) = {min_match_med:.6f}\n")
        handle.write(f"- Median |height - (z_max-z_min)| = {height_err_med:.6f}\n")
        handle.write(f"- Inference: radar_proj.center.z behaves as **{center_z_semantics}**.\n\n")

        handle.write("## 3) Projection A/B test on primary camera\n")
        handle.write("- A hypothesis: `top_z = bottom_z + height` (blue).\n")
        handle.write("- B hypothesis: `top_z = bottom_z - height` (red).\n")
        handle.write("- Visualizations are under `vis/` and include GT 2D box (green), A (blue), B (red), and mean-v text.\n\n")
        handle.write("| # | sample_id | object | camera | mean_v_bottom_A | mean_v_top_A | A(top<bottom) | mean_v_bottom_B | mean_v_top_B | B(top<bottom) | vis |\n")
        handle.write("|---:|---|---|---|---:|---:|---|---:|---:|---|---|\n")
        for idx, row in enumerate(projection_rows, start=1):
            rel_vis = Path(row.vis_path).as_posix()
            handle.write(
                f"| {idx} | {row.sample_id} | {row.object_name} | {row.camera_name} | "
                f"{row.mean_v_bottom_a:.3f} | {row.mean_v_top_a:.3f} | {row.a_top_less_bottom} | "
                f"{row.mean_v_bottom_b:.3f} | {row.mean_v_top_b:.3f} | {row.b_top_less_bottom} | `{rel_vis}` |\n"
            )
        handle.write("\n")
        handle.write(f"- A(top<bottom)=True count: {a_true}/{len(projection_rows)}\n")
        handle.write(f"- B(top<bottom)=True count: {b_true}/{len(projection_rows)}\n\n")

        handle.write("## 4) Camera extrinsics z-flip check\n")
        m = CAMERA_CV_FROM_LOCAL[:3, :3]
        det_m = float(np.linalg.det(m))
        handle.write(
            "- Loader uses a fixed conversion `camera_cv_from_local = [[0,1,0],[0,0,-1],[1,0,0]]` before per-camera extrinsic.\n"
        )
        handle.write(f"- det(camera_cv_from_local[:3,:3]) = {det_m:.6f}.\n")
        handle.write(
            "- The `-1` term (second row, third column) flips local z when mapping to CV y (image-down axis). "
            "This is a fixed axis-convention conversion, not a sample-dependent random flip.\n\n"
        )

        handle.write("## Final Answers\n")
        handle.write(
            f"- 当前 radar/ego 坐标系 z 增大方向：**{z_direction}** （由 A/B 投影统计主导判断）。\n"
        )
        handle.write(f"- RouteROI 3D cuboid 建议使用：**`{recommended_formula}`**。\n")
        handle.write(
            "- 是否存在 camera_extrinsics 造成的额外 z 翻转：**存在固定坐标系转换中的符号映射（local z -> CV y 含负号）**，"
            "但它是统一定义，不会在样本间额外随机翻转。\n"
        )
        handle.write(
            f"- 投影验证统计样本数：**{len(projection_rows)}**（见上表，>=10）。\n"
        )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    sample_paths = [Path(p) for p in args.samples_jsonl]
    records = load_records(sample_paths)
    if not records:
        raise RuntimeError("No sample records loaded. Check --samples-jsonl paths.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"

    pcd_rows = collect_pcd_stats(records, rng, args.pcd_samples)
    box_rows, field_sample = collect_box_semantics(records, args.box_frames)
    projection_results = run_projection_ab_test(records, rng, args.projection_samples, vis_dir)

    if len(projection_results) < 10:
        raise RuntimeError(
            f"Projection AB valid samples only {len(projection_results)} (<10). "
            "Please increase --projection-samples or inspect calibration/data quality."
        )

    pcd_csv = output_dir / "pcd_z_stats.csv"
    box_csv = output_dir / "box_z_semantics.csv"
    proj_csv = output_dir / "projection_ab_stats.csv"

    write_csv(pcd_csv, pcd_rows)
    write_csv(box_csv, box_rows)
    write_csv(proj_csv, projection_rows_to_dicts(projection_results))

    summary = {
        "num_records": len(records),
        "num_pcd_rows": len(pcd_rows),
        "num_box_rows": len(box_rows),
        "num_projection_rows": len(projection_results),
        "pcd_csv": str(pcd_csv),
        "box_csv": str(box_csv),
        "projection_csv": str(proj_csv),
        "vis_dir": str(vis_dir),
        "report_path": str(args.report_path),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    build_report(
        report_path=args.report_path,
        output_dir=output_dir,
        field_sample=field_sample,
        pcd_rows=pcd_rows,
        box_rows=box_rows,
        projection_rows=projection_results,
    )

    print("[OK] Z-axis check completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
