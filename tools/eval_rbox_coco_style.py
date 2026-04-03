#!/usr/bin/env python3
"""Evaluate rotated bbox predictions with COCO-style IoU sweep.

This script is intentionally independent from PaddleDetection's built-in
RBoxMetric because the current implementation only reports a single-IoU mAP
with VOC-style `11point` / `integral` options. Here we expose:

- AP50 (integral)
- AP75 (integral)
- mAP50:95 (integral, step=0.05)

The prediction json is expected to be the `bbox.json` written by
`tools/eval.py --save_prediction_only` for an RBOX dataset, where each record is:

{
  "image_id": int,
  "category_id": int,
  "bbox": [x1,y1,x2,y2,x3,y3,x4,y4],
  "score": float
}
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# This evaluator only parses json and computes metrics; force CPU to avoid
# contending with training/eval jobs that may already occupy the GPUs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FLAGS_use_gpu", "false")

from ppdet.data.source.category import get_categories
from ppdet.metrics.map_utils import DetectionMAP
from ppdet.utils.rbox_min_size import clamp_segmentation_min_edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno-file", required=True, help="COCO-style annotation json")
    parser.add_argument("--pred-json", required=True, help="bbox.json prediction file")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to dump metrics json")
    parser.add_argument(
        "--classwise",
        action="store_true",
        help="Include per-class AP in the underlying metric accumulators")
    parser.add_argument(
        "--min-gt-rbox-edge",
        type=float,
        default=2.0,
        help="Clamp GT rotated box min edge length in pixels (<=0 disables).")
    return parser.parse_args()


def _flatten_poly(segmentation) -> List[float]:
    if segmentation is None:
        return []
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        return [float(x) for x in segmentation[0]]
    if isinstance(segmentation, list):
        return [float(x) for x in segmentation]
    return []


def load_gt(
    anno_file: str,
    min_gt_rbox_edge: float,
) -> Tuple[Dict[int, List[Tuple[List[float], int]]], Dict[int, int], Dict[int, str]]:
    with open(anno_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    clsid2catid, catid2name = get_categories("RBOX", anno_file)
    catid2clsid = {catid: clsid for clsid, catid in clsid2catid.items()}

    gt_by_image: Dict[int, List[Tuple[List[float], int]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        seg = ann.get("segmentation")
        if min_gt_rbox_edge > 0:
            seg = clamp_segmentation_min_edge(seg, min_edge=min_gt_rbox_edge)
        poly = _flatten_poly(seg)
        if len(poly) != 8:
            continue
        catid = int(ann["category_id"])
        if catid not in catid2clsid:
            continue
        gt_by_image[int(ann["image_id"])].append((poly, catid2clsid[catid]))

    return gt_by_image, catid2clsid, catid2name


def load_pred(pred_json: str, catid2clsid: Dict[int, int]):
    with open(pred_json, "r", encoding="utf-8") as f:
        preds = json.load(f)

    pred_by_image: Dict[int, List[Tuple[List[float], float, int]]] = defaultdict(list)
    for pred in preds:
        bbox = pred.get("bbox", [])
        if len(bbox) != 8:
            continue
        catid = int(pred["category_id"])
        if catid not in catid2clsid:
            continue
        pred_by_image[int(pred["image_id"])].append(
            ([float(x) for x in bbox], float(pred["score"]), catid2clsid[catid])
        )
    return pred_by_image


def evaluate_threshold(
    gt_by_image: Dict[int, List[Tuple[List[float], int]]],
    pred_by_image: Dict[int, List[Tuple[List[float], float, int]]],
    catid2name: Dict[int, str],
    overlap_thresh: float,
    classwise: bool,
) -> float:
    metric = DetectionMAP(
        class_num=len(catid2name),
        overlap_thresh=overlap_thresh,
        map_type="integral",
        is_bbox_normalized=False,
        evaluate_difficult=False,
        catid2name=catid2name,
        classwise=classwise,
    )

    image_ids = sorted(set(gt_by_image.keys()) | set(pred_by_image.keys()))
    for image_id in image_ids:
        gt_items = gt_by_image.get(image_id, [])
        pred_items = pred_by_image.get(image_id, [])

        gt_boxes = np.array([poly for poly, _ in gt_items], dtype="float32")
        gt_labels = np.array([clsid for _, clsid in gt_items], dtype="int32")
        pred_boxes = [poly for poly, _, _ in pred_items]
        pred_scores = [score for _, score, _ in pred_items]
        pred_labels = [clsid for _, _, clsid in pred_items]

        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 8), dtype="float32")
            gt_labels = np.zeros((0,), dtype="int32")

        metric.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)

    metric.accumulate()
    return float(metric.get_map())


def main():
    args = parse_args()
    gt_by_image, catid2clsid, catid2name = load_gt(
        args.anno_file, min_gt_rbox_edge=args.min_gt_rbox_edge)
    pred_by_image = load_pred(args.pred_json, catid2clsid)

    thresholds = [round(0.50 + i * 0.05, 2) for i in range(10)]
    ap_by_thresh = {}
    for thr in thresholds:
        ap_by_thresh[f"{thr:.2f}"] = evaluate_threshold(
            gt_by_image, pred_by_image, catid2name, thr, args.classwise
        )

    summary = {
        "pred_json": str(Path(args.pred_json).resolve()),
        "anno_file": str(Path(args.anno_file).resolve()),
        "metric_type": "rotated_bbox_integral",
        "min_gt_rbox_edge": args.min_gt_rbox_edge,
        "AP50": ap_by_thresh["0.50"],
        "AP75": ap_by_thresh["0.75"],
        "mAP50_95": float(sum(ap_by_thresh.values()) / len(ap_by_thresh)),
        "AP_by_IoU": ap_by_thresh,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
