#!/usr/bin/env python3
"""Export per-subclass RGB crops from gt_filter_only_yaw annotations."""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


FINE_TO_SUPER = {
    "Containership": "CargoShip",
    "libertyship": "FishingVessel",
    "smallcargo": "CargoShip",
    "suppliervessel": "CargoShip",
    "queenmarry": "CruiseShip",
    "ramonasteam": "CruiseShip",
    "fishingboat": "FishingVessel",
    "Yacht": "RecreationalBoat",
    "Sailboat": "RecreationalBoat",
    "HouseBoat": "CruiseShip",
    "Motorboat": "RecreationalBoat",
    "Boataaa": "FishingVessel",
    "CoastGuard": "RecreationalBoat",
}

CAMERA_ORDER = ("CamBack", "CamFront", "CamLeft", "CamRight")
RGB_SIZE = (1024, 512)  # W, H


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 5 large and clear crops for each fine-grained class."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/subclass_gt_crops_20260401"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Valid", "Test"],
    )
    parser.add_argument(
        "--per-subclass",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min-box-side",
        type=float,
        default=24.0,
        help="Minimum width/height in pixels before ranking.",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=3.0,
        help="Require box to stay away from image border by this margin.",
    )
    parser.add_argument(
        "--crop-pad-ratio",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear output root before export.",
    )
    return parser.parse_args()


def normalize_category_name(object_name: str) -> str:
    name = object_name
    if name.startswith("USV_"):
        name = name[4:]
    if "_C_" in name:
        name = name.split("_C_", 1)[0]
    while name and name[-1].isdigit():
        name = name[:-1]
    return name or object_name


def parse_frame_id(gt_filename: str) -> str:
    # gt_000123_tosensor_filter.json -> 000123
    stem = gt_filename.replace("gt_", "").replace("_tosensor_filter.json", "")
    return stem


def clip_box_xyxy(box_2d: Dict[str, float], width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    x1 = float(box_2d.get("xmin", 0.0))
    x2 = float(box_2d.get("xmax", 0.0))
    y1 = float(box_2d.get("ymin", 0.0))
    y2 = float(box_2d.get("ymax", 0.0))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(x1, float(width - 1)))
    x2 = max(0.0, min(x2, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    y2 = max(0.0, min(y2, float(height - 1)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def collect_candidates(args: argparse.Namespace) -> Dict[str, List[Dict[str, object]]]:
    width, height = RGB_SIZE
    candidates: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for split in args.splits:
        split_dir = args.dataset_root / split
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            gt_dir = scene_dir / "gt_filter_only_yaw"
            if not gt_dir.is_dir():
                continue
            gt_files = sorted(gt_dir.glob("gt_*_tosensor_filter.json"))
            for gt_path in gt_files:
                frame_id = parse_frame_id(gt_path.name)
                with gt_path.open("r", encoding="utf-8") as f:
                    content = json.load(f)
                towers = content.get("towers", {})
                for tower_name, tower_value in towers.items():
                    objects = tower_value.get("objects", {})
                    for object_name, object_entry in objects.items():
                        fine = normalize_category_name(object_name)
                        super_class = FINE_TO_SUPER.get(fine)
                        if super_class is None:
                            continue

                        cams = object_entry.get("cams", {})
                        best = None
                        for cam_name in CAMERA_ORDER:
                            cam_entry = cams.get(cam_name, {})
                            if not cam_entry:
                                continue
                            visible = cam_entry.get("visible", True)
                            if visible is False:
                                continue
                            box_2d = cam_entry.get("box_2d")
                            if not box_2d:
                                continue
                            clipped = clip_box_xyxy(box_2d, width, height)
                            if clipped is None:
                                continue
                            x1, y1, x2, y2 = clipped
                            w = x2 - x1
                            h = y2 - y1
                            if w < args.min_box_side or h < args.min_box_side:
                                continue
                            if (
                                x1 < args.edge_margin
                                or y1 < args.edge_margin
                                or x2 > (width - 1 - args.edge_margin)
                                or y2 > (height - 1 - args.edge_margin)
                            ):
                                continue

                            area = w * h
                            short_side = min(w, h)
                            score = area + short_side * 30.0
                            rgb_path = (
                                scene_dir
                                / tower_name
                                / "cams"
                                / cam_name
                                / "rgb"
                                / f"{frame_id}.png"
                            )
                            if not rgb_path.is_file():
                                continue

                            item = {
                                "fine": fine,
                                "super": super_class,
                                "split": split,
                                "scene": scene_dir.name,
                                "tower": tower_name,
                                "frame_id": frame_id,
                                "camera": cam_name,
                                "object_name": object_name,
                                "rgb_path": str(rgb_path),
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "w": w,
                                "h": h,
                                "area": area,
                                "short_side": short_side,
                                "score": score,
                            }
                            if best is None or item["score"] > best["score"]:
                                best = item

                        if best is not None:
                            candidates[fine].append(best)
    return candidates


def select_top_k(items: List[Dict[str, object]], k: int) -> List[Dict[str, object]]:
    items = sorted(items, key=lambda x: (float(x["score"]), float(x["area"])), reverse=True)
    picked: List[Dict[str, object]] = []
    seen_frame = set()

    for item in items:
        key = (item["split"], item["scene"], item["tower"], item["frame_id"])
        if key in seen_frame:
            continue
        picked.append(item)
        seen_frame.add(key)
        if len(picked) >= k:
            return picked

    if len(picked) < k:
        used = {
            (x["split"], x["scene"], x["tower"], x["frame_id"], x["camera"], x["object_name"])
            for x in picked
        }
        for item in items:
            key = (item["split"], item["scene"], item["tower"], item["frame_id"], item["camera"], item["object_name"])
            if key in used:
                continue
            picked.append(item)
            used.add(key)
            if len(picked) >= k:
                break
    return picked


def export_crops(args: argparse.Namespace, selected: Dict[str, List[Dict[str, object]]]) -> None:
    if args.output_root.exists() and not args.keep_existing:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    width, height = RGB_SIZE

    for fine, items in selected.items():
        super_class = FINE_TO_SUPER[fine]
        out_dir = args.output_root / super_class / fine
        out_dir.mkdir(parents=True, exist_ok=True)

        meta_path = out_dir / "meta.csv"
        with meta_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "idx",
                "super_class",
                "fine_class",
                "split",
                "scene",
                "tower",
                "frame_id",
                "camera",
                "object_name",
                "rgb_path",
                "x1",
                "y1",
                "x2",
                "y2",
                "w",
                "h",
                "area",
                "crop_path",
            ])

            for idx, item in enumerate(items, start=1):
                image = Image.open(item["rgb_path"]).convert("RGB")
                x1 = float(item["x1"])
                y1 = float(item["y1"])
                x2 = float(item["x2"])
                y2 = float(item["y2"])
                w = x2 - x1
                h = y2 - y1

                pad_x = max(4.0, w * float(args.crop_pad_ratio))
                pad_y = max(4.0, h * float(args.crop_pad_ratio))
                cx1 = max(0.0, x1 - pad_x)
                cy1 = max(0.0, y1 - pad_y)
                cx2 = min(float(width), x2 + pad_x)
                cy2 = min(float(height), y2 + pad_y)

                crop = image.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
                out_name = (
                    f"{idx:02d}_{item['split']}_{item['scene']}_{item['tower']}_"
                    f"{item['frame_id']}_{item['camera']}.png"
                )
                out_path = out_dir / out_name
                crop.save(out_path)

                row = [
                    idx,
                    super_class,
                    fine,
                    item["split"],
                    item["scene"],
                    item["tower"],
                    item["frame_id"],
                    item["camera"],
                    item["object_name"],
                    item["rgb_path"],
                    f"{x1:.2f}",
                    f"{y1:.2f}",
                    f"{x2:.2f}",
                    f"{y2:.2f}",
                    f"{w:.2f}",
                    f"{h:.2f}",
                    f"{float(item['area']):.2f}",
                    str(out_path),
                ]
                writer.writerow(row)
                summary_rows.append(row)

    summary_path = args.output_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx",
            "super_class",
            "fine_class",
            "split",
            "scene",
            "tower",
            "frame_id",
            "camera",
            "object_name",
            "rgb_path",
            "x1",
            "y1",
            "x2",
            "y2",
            "w",
            "h",
            "area",
            "crop_path",
        ])
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    candidates = collect_candidates(args)
    selected = {}
    for fine in sorted(FINE_TO_SUPER.keys()):
        picked = select_top_k(candidates.get(fine, []), args.per_subclass)
        selected[fine] = picked
    export_crops(args, selected)

    print(f"Output root: {args.output_root}")
    for fine in sorted(FINE_TO_SUPER.keys()):
        print(f"{fine}: {len(selected[fine])} samples")


if __name__ == "__main__":
    main()
