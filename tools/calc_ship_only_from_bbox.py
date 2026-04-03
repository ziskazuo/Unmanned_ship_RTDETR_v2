#!/usr/bin/env python3
"""Compute class-agnostic (ship-only) metrics from saved bbox.json predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Force CPU so this postprocess doesn't contend with GPU jobs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FLAGS_use_gpu", "false")

from ppdet.metrics.map_utils import DetectionMAP
from ppdet.utils.rbox_min_size import clamp_segmentation_min_edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno-all", required=True, help="ALL split COCO annotation json")
    parser.add_argument("--pred-all", required=True, help="ALL split bbox.json predictions")
    parser.add_argument("--anno-s01", required=True)
    parser.add_argument("--pred-s01", required=True)
    parser.add_argument("--anno-s02", required=True)
    parser.add_argument("--pred-s02", required=True)
    parser.add_argument("--anno-s03", required=True)
    parser.add_argument("--pred-s03", required=True)
    parser.add_argument("--output-dir", required=True, help="Output directory for summary json/csv")
    parser.add_argument(
        "--min-gt-rbox-edge",
        type=float,
        default=2.0,
        help="Clamp GT rotated box min edge length in pixels (<=0 disables).")
    return parser.parse_args()


def flatten_poly(segmentation) -> List[float]:
    if segmentation is None:
        return []
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        return [float(x) for x in segmentation[0]]
    if isinstance(segmentation, list):
        return [float(x) for x in segmentation]
    return []


def load_split(anno_file: str, pred_file: str, min_gt_rbox_edge: float):
    with open(anno_file, "r", encoding="utf-8") as f:
        anno = json.load(f)
    with open(pred_file, "r", encoding="utf-8") as f:
        preds = json.load(f)

    gt_by_img: Dict[int, List[List[float]]] = defaultdict(list)
    for ann in anno.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        seg = ann.get("segmentation")
        if min_gt_rbox_edge > 0:
            seg = clamp_segmentation_min_edge(seg, min_edge=min_gt_rbox_edge)
        poly = flatten_poly(seg)
        if len(poly) != 8:
            continue
        gt_by_img[int(ann["image_id"])].append(poly)

    pred_by_img: Dict[int, List[Tuple[List[float], float]]] = defaultdict(list)
    for pred in preds:
        bbox = pred.get("bbox", [])
        if len(bbox) != 8:
            continue
        pred_by_img[int(pred["image_id"])].append(
            ([float(x) for x in bbox], float(pred.get("score", 0.0)))
        )

    image_ids = set(int(im["id"]) for im in anno.get("images", []))
    return gt_by_img, pred_by_img, image_ids


def eval_threshold(
    gt_by_img: Dict[int, List[List[float]]],
    pred_by_img: Dict[int, List[Tuple[List[float], float]]],
    image_ids: Iterable[int],
    overlap_thresh: float,
    map_type: str,
) -> float:
    metric = DetectionMAP(
        class_num=1,
        overlap_thresh=overlap_thresh,
        map_type=map_type,
        is_bbox_normalized=False,
        evaluate_difficult=False,
        catid2name={1: "ship"},
        classwise=False,
    )

    for image_id in image_ids:
        gt_items = gt_by_img.get(image_id, [])
        pred_items = pred_by_img.get(image_id, [])
        gt_boxes = np.array(gt_items, dtype="float32") if gt_items else np.zeros((0, 8), dtype="float32")
        gt_labels = (
            np.zeros((len(gt_items),), dtype="int32") if gt_items else np.zeros((0,), dtype="int32")
        )
        pred_boxes = [box for box, _ in pred_items]
        pred_scores = [score for _, score in pred_items]
        pred_labels = [0] * len(pred_items)
        metric.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)

    metric.accumulate()
    return float(metric.get_map())


def eval_split(
    gt_by_img: Dict[int, List[List[float]]],
    pred_by_img: Dict[int, List[Tuple[List[float], float]]],
    image_ids: Iterable[int],
):
    image_ids = list(image_ids)
    ious = [round(0.50 + i * 0.05, 2) for i in range(10)]
    ap_by_iou = {
        f"{iou:.2f}": eval_threshold(gt_by_img, pred_by_img, image_ids, overlap_thresh=iou, map_type="integral")
        for iou in ious
    }
    ap50_11point = eval_threshold(
        gt_by_img, pred_by_img, image_ids, overlap_thresh=0.50, map_type="11point"
    )
    return {
        "num_images": len(image_ids),
        "AP50": ap_by_iou["0.50"],
        "AP75": ap_by_iou["0.75"],
        "mAP50_95": float(sum(ap_by_iou.values()) / len(ap_by_iou)),
        "mAP50_11point": ap50_11point,
        "AP_by_IoU": ap_by_iou,
    }


def main() -> None:
    args = parse_args()

    splits = {
        "ALL": (args.anno_all, args.pred_all),
        "S01": (args.anno_s01, args.pred_s01),
        "S02": (args.anno_s02, args.pred_s02),
        "S03": (args.anno_s03, args.pred_s03),
    }

    summary = {}
    for split_name, (anno_file, pred_file) in splits.items():
        gt_by_img, pred_by_img, image_ids = load_split(
            anno_file, pred_file, min_gt_rbox_edge=args.min_gt_rbox_edge)
        summary[split_name] = eval_split(gt_by_img, pred_by_img, image_ids)
        print(
            f"{split_name}: AP50={summary[split_name]['AP50']:.6f}, "
            f"mAP50_95={summary[split_name]['mAP50_95']:.6f}, "
            f"mAP50_11point={summary[split_name]['mAP50_11point']:.6f}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "ship_only_summary.json"
    csv_path = out_dir / "ship_only_summary.csv"
    summary["_meta"] = {"min_gt_rbox_edge": args.min_gt_rbox_edge}
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "num_images", "AP50", "AP75", "mAP50_95", "mAP50_11point"],
        )
        writer.writeheader()
        for split in ["ALL", "S01", "S02", "S03"]:
            row = summary[split]
            writer.writerow(
                {
                    "split": split,
                    "num_images": row["num_images"],
                    "AP50": row["AP50"],
                    "AP75": row["AP75"],
                    "mAP50_95": row["mAP50_95"],
                    "mAP50_11point": row["mAP50_11point"],
                }
            )

    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
