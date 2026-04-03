#!/usr/bin/env python3
"""Post-process full-test predictions into ALL/S01/S02/S03 metrics.

Outputs:
1) Integral rotated metrics: AP50 / AP75 / mAP50:95
2) VOC-style mAP@0.50 (11point)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ppdet.data.source.category import get_categories
from ppdet.metrics.map_utils import DetectionMAP
from ppdet.utils.rbox_min_size import clamp_segmentation_min_edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno-file", required=True, help="COCO annotation json")
    parser.add_argument("--pred-json", required=True, help="bbox.json from eval.py")
    parser.add_argument("--output-dir", required=True, help="Directory to write summaries")
    parser.add_argument("--scene-key", default="scene_id", help="Image field for split scene key")
    parser.add_argument("--scenes", default="", help="Comma separated scene names, e.g. S01,S02,S03")
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


def load_annotations(anno_file: str, min_gt_rbox_edge: float):
    with open(anno_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    clsid2catid, catid2name = get_categories("RBOX", anno_file)
    catid2clsid = {catid: clsid for clsid, catid in clsid2catid.items()}

    images = data.get("images", [])
    anns = data.get("annotations", [])

    gt_by_image: Dict[int, List[Tuple[List[float], int]]] = defaultdict(list)
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        seg = ann.get("segmentation")
        if min_gt_rbox_edge > 0:
            seg = clamp_segmentation_min_edge(seg, min_edge=min_gt_rbox_edge)
        poly = flatten_poly(seg)
        if len(poly) != 8:
            continue
        catid = int(ann["category_id"])
        if catid not in catid2clsid:
            continue
        gt_by_image[int(ann["image_id"])].append((poly, catid2clsid[catid]))

    return images, gt_by_image, catid2name, catid2clsid


def load_predictions(pred_json: str, catid2clsid: Dict[int, int]):
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


def evaluate_at_thresh(
    gt_by_image: Dict[int, List[Tuple[List[float], int]]],
    pred_by_image: Dict[int, List[Tuple[List[float], float, int]]],
    catid2name: Dict[int, str],
    overlap_thresh: float,
    map_type: str,
):
    metric = DetectionMAP(
        class_num=len(catid2name),
        overlap_thresh=overlap_thresh,
        map_type=map_type,
        is_bbox_normalized=False,
        evaluate_difficult=False,
        catid2name=catid2name,
        classwise=False,
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


def evaluate_split(
    image_ids: Iterable[int],
    gt_full: Dict[int, List[Tuple[List[float], int]]],
    pred_full: Dict[int, List[Tuple[List[float], float, int]]],
    catid2name: Dict[int, str],
):
    image_ids = set(image_ids)
    gt_by_image = {k: v for k, v in gt_full.items() if k in image_ids}
    pred_by_image = {k: v for k, v in pred_full.items() if k in image_ids}

    thresholds = [round(0.50 + i * 0.05, 2) for i in range(10)]
    ap_by_iou = {}
    for thr in thresholds:
        ap_by_iou[f"{thr:.2f}"] = evaluate_at_thresh(
            gt_by_image, pred_by_image, catid2name, thr, map_type="integral"
        )

    ap50_11point = evaluate_at_thresh(
        gt_by_image, pred_by_image, catid2name, 0.50, map_type="11point"
    )

    return {
        "num_images": len(image_ids),
        "AP50": ap_by_iou["0.50"],
        "AP75": ap_by_iou["0.75"],
        "mAP50_95": float(sum(ap_by_iou.values()) / len(ap_by_iou)),
        "mAP50_11point": ap50_11point,
        "AP_by_IoU": ap_by_iou,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images, gt_by_image, catid2name, catid2clsid = load_annotations(
        args.anno_file, min_gt_rbox_edge=args.min_gt_rbox_edge)
    pred_by_image = load_predictions(args.pred_json, catid2clsid)

    image_ids_all = [int(img["id"]) for img in images]
    split_ids: Dict[str, List[int]] = defaultdict(list)
    for img in images:
        split_ids[str(img.get(args.scene_key, "UNKNOWN"))].append(int(img["id"]))

    requested_scenes = [x.strip() for x in args.scenes.split(",") if x.strip()]
    if requested_scenes:
        scene_order = [name for name in requested_scenes if name in split_ids]
    else:
        scene_order = sorted(k for k in split_ids.keys() if k and k != "UNKNOWN")

    all_metrics = evaluate_split(image_ids_all, gt_by_image, pred_by_image, catid2name)

    metrics_all = {
        "pred_json": str(Path(args.pred_json).resolve()),
        "anno_file": str(Path(args.anno_file).resolve()),
        "metric_type": "rotated_bbox_integral",
        "min_gt_rbox_edge": args.min_gt_rbox_edge,
        "AP50": all_metrics["AP50"],
        "AP75": all_metrics["AP75"],
        "mAP50_95": all_metrics["mAP50_95"],
        "AP_by_IoU": all_metrics["AP_by_IoU"],
    }
    (output_dir / "metrics_all_integral.json").write_text(
        json.dumps(metrics_all, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    subset_metrics = {
        "ALL": {
            "num_images": all_metrics["num_images"],
            "AP50": all_metrics["AP50"],
            "AP75": all_metrics["AP75"],
            "mAP50_95": all_metrics["mAP50_95"],
            "mAP50_11point": all_metrics["mAP50_11point"],
        }
    }
    for scene in scene_order:
        scene_metrics = evaluate_split(split_ids[scene], gt_by_image, pred_by_image, catid2name)
        subset_metrics[scene] = {
            "num_images": scene_metrics["num_images"],
            "AP50": scene_metrics["AP50"],
            "AP75": scene_metrics["AP75"],
            "mAP50_95": scene_metrics["mAP50_95"],
            "mAP50_11point": scene_metrics["mAP50_11point"],
        }

    (output_dir / "subset_metrics.json").write_text(
        json.dumps(subset_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary_all_splits.json").write_text(
        json.dumps(subset_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    map11_summary = {
        k: {"mAP50_11point": v["mAP50_11point"], "num_images": v["num_images"]}
        for k, v in subset_metrics.items()
    }
    (output_dir / "map11point_summary.json").write_text(
        json.dumps(map11_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(subset_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
