#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def box_area_xyxy(box: List[float]) -> float:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def normalize_annotation(annotation: Dict[str, object]) -> bool:
    boxes = annotation.get(
        "gt_camera_box_2d", [[0.0, 0.0, 0.0, 0.0] for _ in range(4)])
    has_box = annotation.get("gt_has_camera_box", [0, 0, 0, 0])
    poly_area = annotation.get("gt_camera_poly_area", [0.0, 0.0, 0.0, 0.0])

    normalized_has_box: List[int] = []
    normalized_areas: List[float] = []
    visible_cameras: List[int] = []
    primary_camera = 4
    best_area = -1.0

    for i in range(4):
        hb = 1 if i < len(has_box) and float(has_box[i]) > 0.5 else 0
        area = float(poly_area[i]) if i < len(poly_area) else 0.0
        if area <= 0.0 and hb and i < len(boxes):
            area = box_area_xyxy(boxes[i])
        area = max(0.0, area)
        visible = 1 if area > 0.0 else 0

        normalized_has_box.append(hb)
        normalized_areas.append(area)
        visible_cameras.append(visible)

        if visible and area > best_area:
            best_area = area
            primary_camera = i

    changed = False
    if annotation.get("gt_has_camera_box") != normalized_has_box:
        annotation["gt_has_camera_box"] = normalized_has_box
        changed = True
    if annotation.get("gt_camera_poly_area") != normalized_areas:
        annotation["gt_camera_poly_area"] = normalized_areas
        changed = True
    if annotation.get("gt_visible_cameras") != visible_cameras:
        annotation["gt_visible_cameras"] = visible_cameras
        changed = True
    if int(annotation.get("gt_primary_camera", 4)) != primary_camera:
        annotation["gt_primary_camera"] = primary_camera
        changed = True
    return changed


def patch_coco(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    changed = 0
    for ann in data.get("annotations", []):
        if normalize_annotation(ann):
            changed += 1
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix RouteROI camera supervision fields in prepared COCO files.")
    parser.add_argument("paths", nargs="+", help="COCO json files to patch")
    args = parser.parse_args()

    summary = {}
    for raw_path in args.paths:
        path = Path(raw_path)
        summary[str(path)] = patch_coco(path)
    print(summary)


if __name__ == "__main__":
    main()
