#!/usr/bin/env python3
"""Compare old fixed-plane RouteROI projection vs new p75-height cuboid projection.

Outputs:
- metrics summary JSON/MD
- per-box CSV
- visualization overlays
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

CAM_NAMES = ["CamBack", "CamFront", "CamLeft", "CamRight"]
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
DEFAULT_HEIGHT_PRIOR = {
    "CargoShip": 46.55,
    "CruiseShip": 69.77,
    "FishingVessel": 34.33,
    "RecreationalBoat": 9.18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check RouteROI 3D projection quality against GT camera boxes"
    )
    parser.add_argument(
        "--samples-jsonl",
        nargs="+",
        default=[
            "prepared/sealand_single_tower_2km_super4_1536_route_roi_min8/train_samples.jsonl",
            "prepared/sealand_single_tower_2km_super4_1536_route_roi_min8/valid_samples.jsonl",
            "prepared/sealand_single_tower_2km_super4_1536_route_roi_min8/test_samples.jsonl",
        ],
    )
    parser.add_argument(
        "--height-prior-path",
        default="configs/priors/sealand_height_prior_super4.json",
        help="JSON file for class height prior",
    )
    parser.add_argument("--height-prior-stat", default="p75")
    parser.add_argument("--projection-plane-height", type=float, default=-6.0)
    parser.add_argument("--roi-expand-ratio", type=float, default=1.15)
    parser.add_argument("--max-records", type=int, default=2000)
    parser.add_argument("--max-vis", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/route_roi_3d_projection_debug"),
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def box_area_xyxy(box: Sequence[float]) -> float:
    if box is None or len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_records(paths: Sequence[Path]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def load_height_prior(path: Path, stat: str) -> Dict[str, float]:
    prior = dict(DEFAULT_HEIGHT_PRIOR)
    if not path.is_file():
        return prior

    try:
        payload = load_json(path)
    except Exception:
        return prior
    if not isinstance(payload, dict):
        return prior

    candidate = None
    if "stats" in payload and isinstance(payload["stats"], dict):
        candidate = payload["stats"].get(stat)
    if candidate is None:
        candidate = payload.get(stat)
    if candidate is None:
        per_class = {}
        for key, value in payload.items():
            if isinstance(value, dict) and stat in value:
                per_class[key] = value[stat]
        if per_class:
            candidate = per_class
    if candidate is None and all(not isinstance(v, (dict, list, tuple)) for v in payload.values()):
        candidate = payload

    if isinstance(candidate, dict):
        for key, value in candidate.items():
            try:
                prior[str(key)] = max(float(value), 0.5)
            except Exception:
                continue
    return prior


def rbox_to_bottom_corners(rbox_xywhr: Sequence[float]) -> np.ndarray:
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
    return rotated


def project_points(points_xyz: np.ndarray, t_camcv_ego: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    homo = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (t_camcv_ego @ homo.T).T[:, :3]
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = k[0, 0] * cam[:, 0] / safe_z + k[0, 2]
    v = k[1, 1] * cam[:, 1] / safe_z + k[1, 2]
    uv = np.stack([u, v], axis=1)
    return uv, z


def clip_box(box: Sequence[float], img_w: float, img_h: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), img_w - 1.0)
    y1 = min(max(y1, 0.0), img_h - 1.0)
    x2 = min(max(x2, 0.0), img_w - 1.0)
    y2 = min(max(y2, 0.0), img_h - 1.0)
    return [x1, y1, x2, y2]


def expand_box(box: Sequence[float], ratio: float, img_w: float, img_h: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * max(x2 - x1, 0.0) * ratio
    half_h = 0.5 * max(y2 - y1, 0.0) * ratio
    nx1 = cx - half_w
    ny1 = cy - half_h
    nx2 = cx + half_w
    ny2 = cy + half_h
    return clip_box([nx1, ny1, nx2, ny2], img_w, img_h)


def project_old_fixed_plane(
    bottom_xy: np.ndarray,
    k: np.ndarray,
    t_camcv_ego: np.ndarray,
    bottom_z: float,
    img_w: float,
    img_h: float,
) -> Tuple[List[float], bool]:
    points = np.concatenate([bottom_xy, np.full((4, 1), float(bottom_z), dtype=np.float64)], axis=1)
    uv, z = project_points(points, t_camcv_ego, k)

    xmin = float(np.min(uv[:, 0]))
    ymin = float(np.min(uv[:, 1]))
    xmax = float(np.max(uv[:, 0]))
    ymax = float(np.max(uv[:, 1]))
    box = clip_box([xmin, ymin, xmax, ymax], img_w, img_h)

    valid_depth = bool(np.all(z > 1e-4))
    visible = valid_depth and (box[2] > box[0]) and (box[3] > box[1])
    if not visible:
        box = [0.0, 0.0, 0.0, 0.0]
    return box, visible


def project_new_height_cuboid(
    bottom_xy: np.ndarray,
    k: np.ndarray,
    t_camcv_ego: np.ndarray,
    bottom_z: float,
    est_height: float,
    img_w: float,
    img_h: float,
    expand_ratio: float,
) -> Tuple[List[float], bool]:
    bottom = np.concatenate([bottom_xy, np.full((4, 1), float(bottom_z), dtype=np.float64)], axis=1)
    top = np.concatenate([bottom_xy, np.full((4, 1), float(bottom_z + est_height), dtype=np.float64)], axis=1)
    points = np.concatenate([bottom, top], axis=0)

    uv, z = project_points(points, t_camcv_ego, k)
    valid = z > 1e-4
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 1:
        return [0.0, 0.0, 0.0, 0.0], False

    uv_valid = uv[valid]
    xmin = float(np.min(uv_valid[:, 0]))
    ymin = float(np.min(uv_valid[:, 1]))
    xmax = float(np.max(uv_valid[:, 0]))
    ymax = float(np.max(uv_valid[:, 1]))

    clipped = clip_box([xmin, ymin, xmax, ymax], img_w, img_h)
    inter_w = max(0.0, clipped[2] - clipped[0])
    inter_h = max(0.0, clipped[3] - clipped[1])
    inter_area = inter_w * inter_h
    has_intersection = (inter_w > 0.0) and (inter_h > 0.0)

    visible = (valid_count >= 1) and has_intersection and (inter_area > 4.0)
    if not visible:
        return [0.0, 0.0, 0.0, 0.0], False

    expanded = expand_box(clipped, ratio=max(float(expand_ratio), 1e-6), img_w=img_w, img_h=img_h)
    return expanded, True


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
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(area_a + area_b - inter, 1e-6)
    return inter / union


def contains_gt(pred: Sequence[float], gt: Sequence[float]) -> bool:
    px1, py1, px2, py2 = [float(v) for v in pred]
    gx1, gy1, gx2, gy2 = [float(v) for v in gt]
    return (px1 <= gx1) and (py1 <= gy1) and (px2 >= gx2) and (py2 >= gy2)


def center_error(pred: Sequence[float], gt: Sequence[float]) -> float:
    px1, py1, px2, py2 = [float(v) for v in pred]
    gx1, gy1, gx2, gy2 = [float(v) for v in gt]
    pcx = 0.5 * (px1 + px2)
    pcy = 0.5 * (py1 + py2)
    gcx = 0.5 * (gx1 + gx2)
    gcy = 0.5 * (gy1 + gy2)
    return math.hypot(pcx - gcx, pcy - gcy)


def draw_box(image: np.ndarray, box: Sequence[float], color: Tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def summarize_metric(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.median(arr))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    output_dir = args.output_dir
    vis_dir = output_dir / "vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    prior = load_height_prior(Path(args.height_prior_path), args.height_prior_stat)
    prior_default = float(np.median(list(prior.values()))) if prior else 20.0

    sample_paths = [Path(p) for p in args.samples_jsonl]
    records = load_records(sample_paths)
    if not records:
        raise RuntimeError("No sample records loaded.")

    if args.max_records > 0 and len(records) > args.max_records:
        records = rng.sample(records, args.max_records)

    rows: List[Dict[str, object]] = []
    vis_count = 0

    for rec in records:
        ann_path = Path(rec["annotation_json_path"])
        opv2v_path = Path(rec["opv2v_yaml_path"])
        if not ann_path.is_file() or not opv2v_path.is_file():
            continue

        ann_payload = load_json(ann_path)
        opv2v = load_yaml(opv2v_path)
        if not isinstance(ann_payload, dict) or not isinstance(opv2v, dict):
            continue

        cam_images: Dict[str, Optional[np.ndarray]] = {}
        anns = ann_payload.get("annotations", [])
        if not isinstance(anns, list):
            continue

        for ann_idx, ann in enumerate(anns):
            if not isinstance(ann, dict):
                continue
            rbox = ann.get("rbox_xywhr")
            if not isinstance(rbox, (list, tuple)) or len(rbox) != 5:
                continue
            bottom_xy = rbox_to_bottom_corners(rbox)

            class_name = str(ann.get("super_category_name") or ann.get("category_name") or "")
            est_height = float(prior.get(class_name, prior_default))

            gt_boxes = ann.get("gt_camera_box_2d", [])
            if not isinstance(gt_boxes, list) or len(gt_boxes) < 4:
                continue

            for cam_id, cam_name in enumerate(CAM_NAMES):
                gt_box = gt_boxes[cam_id] if cam_id < len(gt_boxes) else None
                if not isinstance(gt_box, (list, tuple)) or len(gt_box) != 4:
                    continue
                gt_box = [float(v) for v in gt_box]
                if box_area_xyxy(gt_box) <= 0.0:
                    continue

                opv2v_cam = OPV2V_CAM_MAP[cam_name]
                cam_meta = opv2v.get(opv2v_cam)
                if not isinstance(cam_meta, dict):
                    continue
                try:
                    intrinsic = np.asarray(cam_meta["intrinsic"], dtype=np.float64)
                    extrinsic_local = np.asarray(cam_meta["extrinsic"], dtype=np.float64)
                except Exception:
                    continue
                if intrinsic.shape != (3, 3) or extrinsic_local.shape != (4, 4):
                    continue

                t_camcv_ego = CAMERA_CV_FROM_LOCAL @ extrinsic_local
                img_w = float(max(2.0, intrinsic[0, 2] * 2.0))
                img_h = float(max(2.0, intrinsic[1, 2] * 2.0))
                diag = math.hypot(img_w, img_h)

                old_box, old_visible = project_old_fixed_plane(
                    bottom_xy=bottom_xy,
                    k=intrinsic,
                    t_camcv_ego=t_camcv_ego,
                    bottom_z=args.projection_plane_height,
                    img_w=img_w,
                    img_h=img_h,
                )
                new_box, new_visible = project_new_height_cuboid(
                    bottom_xy=bottom_xy,
                    k=intrinsic,
                    t_camcv_ego=t_camcv_ego,
                    bottom_z=args.projection_plane_height,
                    est_height=est_height,
                    img_w=img_w,
                    img_h=img_h,
                    expand_ratio=args.roi_expand_ratio,
                )

                old_iou = iou_xyxy(old_box, gt_box) if old_visible else 0.0
                new_iou = iou_xyxy(new_box, gt_box) if new_visible else 0.0

                old_contains = 1 if (old_visible and contains_gt(old_box, gt_box)) else 0
                new_contains = 1 if (new_visible and contains_gt(new_box, gt_box)) else 0

                old_center_err = center_error(old_box, gt_box) if old_visible else diag
                new_center_err = center_error(new_box, gt_box) if new_visible else diag

                row = {
                    "sample_id": rec["sample_id"],
                    "scene_id": rec["scene_id"],
                    "tower_id": rec["tower_id"],
                    "frame_id": rec["frame_id"],
                    "annotation_index": ann_idx,
                    "instance_name": ann.get("instance_name", ""),
                    "class_name": class_name,
                    "camera_id": cam_id,
                    "camera_name": cam_name,
                    "gt_x1": gt_box[0],
                    "gt_y1": gt_box[1],
                    "gt_x2": gt_box[2],
                    "gt_y2": gt_box[3],
                    "old_visible": int(old_visible),
                    "new_visible": int(new_visible),
                    "old_iou": old_iou,
                    "new_iou": new_iou,
                    "old_contains_gt": old_contains,
                    "new_contains_gt": new_contains,
                    "old_center_error": old_center_err,
                    "new_center_error": new_center_err,
                    "old_x1": old_box[0],
                    "old_y1": old_box[1],
                    "old_x2": old_box[2],
                    "old_y2": old_box[3],
                    "new_x1": new_box[0],
                    "new_y1": new_box[1],
                    "new_x2": new_box[2],
                    "new_y2": new_box[3],
                    "estimated_height": est_height,
                }
                rows.append(row)

                if vis_count < args.max_vis:
                    if cam_name not in cam_images:
                        cam_path = Path(rec["camera_paths"][cam_name])
                        img = cv2.imread(str(cam_path), cv2.IMREAD_COLOR) if cam_path.is_file() else None
                        cam_images[cam_name] = img
                    base_img = cam_images.get(cam_name)
                    if base_img is not None:
                        vis = base_img.copy()
                        draw_box(vis, gt_box, (0, 255, 0), 2)  # GT green
                        if old_visible:
                            draw_box(vis, old_box, (0, 0, 255), 2)  # old red
                        if new_visible:
                            draw_box(vis, new_box, (255, 0, 0), 2)  # new blue

                        lines = [
                            f"sample={rec['sample_id']} cam={cam_name}",
                            f"obj={ann.get('instance_name', '')} class={class_name}",
                            f"old iou={old_iou:.3f} vis={old_visible}",
                            f"new iou={new_iou:.3f} vis={new_visible} h={est_height:.2f}",
                        ]
                        y0 = 24
                        for txt in lines:
                            cv2.putText(
                                vis,
                                txt,
                                (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (255, 255, 255),
                                2,
                                lineType=cv2.LINE_AA,
                            )
                            cv2.putText(
                                vis,
                                txt,
                                (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (20, 20, 20),
                                1,
                                lineType=cv2.LINE_AA,
                            )
                            y0 += 22

                        vis_name = (
                            f"{str(rec['sample_id']).replace('/', '__')}"
                            f"__ann{ann_idx:03d}__{str(ann.get('instance_name','obj')).replace('/', '_')}"
                            f"__{cam_name}.png"
                        )
                        cv2.imwrite(str(vis_dir / vis_name), vis)
                        vis_count += 1

    if not rows:
        raise RuntimeError("No valid GT camera boxes found for evaluation.")

    old_iou_vals = [float(r["old_iou"]) for r in rows]
    new_iou_vals = [float(r["new_iou"]) for r in rows]
    old_center_vals = [float(r["old_center_error"]) for r in rows]
    new_center_vals = [float(r["new_center_error"]) for r in rows]

    old_mean_iou, old_median_iou = summarize_metric(old_iou_vals)
    new_mean_iou, new_median_iou = summarize_metric(new_iou_vals)

    old_recall_iou_01 = float(np.mean(np.asarray(old_iou_vals) >= 0.1))
    new_recall_iou_01 = float(np.mean(np.asarray(new_iou_vals) >= 0.1))

    old_contains_rate = float(np.mean([float(r["old_contains_gt"]) for r in rows]))
    new_contains_rate = float(np.mean([float(r["new_contains_gt"]) for r in rows]))

    old_center_mean = float(np.mean(old_center_vals))
    new_center_mean = float(np.mean(new_center_vals))

    summary = {
        "num_records_loaded": len(records),
        "num_box_pairs": len(rows),
        "height_prior_path": str(Path(args.height_prior_path)),
        "height_prior_stat": args.height_prior_stat,
        "projection_plane_height": args.projection_plane_height,
        "roi_expand_ratio": args.roi_expand_ratio,
        "old_mean_iou": old_mean_iou,
        "new_mean_iou": new_mean_iou,
        "old_median_iou": old_median_iou,
        "new_median_iou": new_median_iou,
        "old_recall_iou_0.1": old_recall_iou_01,
        "new_recall_iou_0.1": new_recall_iou_01,
        "old_contains_gt_rate": old_contains_rate,
        "new_contains_gt_rate": new_contains_rate,
        "old_center_error": old_center_mean,
        "new_center_error": new_center_mean,
        "delta_mean_iou": new_mean_iou - old_mean_iou,
        "delta_median_iou": new_median_iou - old_median_iou,
        "delta_recall_iou_0.1": new_recall_iou_01 - old_recall_iou_01,
        "delta_contains_gt_rate": new_contains_rate - old_contains_rate,
        "delta_center_error": new_center_mean - old_center_mean,
    }
    summary["new_significantly_better"] = bool(
        (summary["delta_contains_gt_rate"] >= 0.02)
        and (summary["delta_recall_iou_0.1"] >= 0.02)
        and (summary["delta_mean_iou"] >= 0.0)
    )

    summary_path = output_dir / "projection_quality_summary.json"
    ensure_parent(summary_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "per_box_metrics.csv"
    ensure_parent(csv_path)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = output_dir / "projection_quality_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# RouteROI 3D Projection Quality\n\n")
        f.write(f"- num_box_pairs: {summary['num_box_pairs']}\n")
        f.write(f"- old_mean_iou: {summary['old_mean_iou']:.6f}\n")
        f.write(f"- new_mean_iou: {summary['new_mean_iou']:.6f}\n")
        f.write(f"- old_median_iou: {summary['old_median_iou']:.6f}\n")
        f.write(f"- new_median_iou: {summary['new_median_iou']:.6f}\n")
        f.write(f"- old_recall_iou_0.1: {summary['old_recall_iou_0.1']:.6f}\n")
        f.write(f"- new_recall_iou_0.1: {summary['new_recall_iou_0.1']:.6f}\n")
        f.write(f"- old_contains_gt_rate: {summary['old_contains_gt_rate']:.6f}\n")
        f.write(f"- new_contains_gt_rate: {summary['new_contains_gt_rate']:.6f}\n")
        f.write(f"- old_center_error: {summary['old_center_error']:.6f}\n")
        f.write(f"- new_center_error: {summary['new_center_error']:.6f}\n")
        f.write(f"- new_significantly_better: {summary['new_significantly_better']}\n")

    print("[OK] Projection quality check finished")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
