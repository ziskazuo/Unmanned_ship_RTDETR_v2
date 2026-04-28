#!/usr/bin/env python3
"""Visualize representative label samples from the rebuilt sealand dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


Color = Tuple[int, int, int]

COLORS: Dict[str, Color] = {
    "CargoShip": (255, 128, 0),
    "CruiseShip": (0, 200, 255),
    "FishingVessel": (80, 255, 80),
    "RecreationalBoat": (255, 80, 180),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def boundary_distance(bbox: Sequence[float], image_size: int) -> float:
    x, y, w, h = bbox
    return min(x, y, image_size - (x + w), image_size - (y + h))


def select_samples(
    root: Path,
    split: str,
    normal_count: int,
    edge_count: int,
    image_size: int,
    rng: random.Random,
) -> Tuple[List[dict], List[dict]]:
    coco = load_json(root / f"{split.lower()}_coco.json")
    images = coco["images"]
    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    edge: List[Tuple[float, dict]] = []
    normal: List[dict] = []
    for image in images:
        anns = anns_by_image.get(image["id"], [])
        if not anns:
            continue
        min_dist = min(boundary_distance(ann["bbox"], image_size) for ann in anns)
        if min_dist <= 5:
            edge.append((min_dist, image))
        else:
            normal.append(image)

    edge.sort(key=lambda item: (item[0], item[1]["sample_id"]))
    rng.shuffle(normal)
    selected_normal = sorted(normal[:normal_count], key=lambda item: item["sample_id"])
    selected_edge = [image for _, image in edge[:edge_count]]
    return selected_normal, selected_edge


def draw_polygon(draw: ImageDraw.ImageDraw, poly: Sequence[Sequence[float]], color: Color, width: int = 2) -> None:
    if not poly:
        return
    points = [(float(x), float(y)) for x, y in poly]
    draw.line(points + [points[0]], fill=color, width=width)


def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, color: Color, font: ImageFont.ImageFont) -> None:
    x, y = xy
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)


def render_image(root: Path, image_meta: dict, output_path: Path, show_raw: bool = False) -> dict:
    radar_path = root / image_meta["radar_im_file"]
    ann_path = (
        root
        / image_meta["split"]
        / "annotations"
        / image_meta["scene_id"]
        / image_meta["tower_id"]
        / f"{image_meta['frame_id']}.json"
    )
    ann = load_json(ann_path)
    image = Image.open(radar_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    edge_like = False
    for obj in ann.get("annotations", []):
        cat = obj.get("category_name", "unknown")
        color = COLORS.get(cat, (255, 255, 255))
        poly = obj.get("poly") or []
        raw_poly = obj.get("raw_poly") or []
        bbox = obj.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox
        if min(x, y, 1000 - (x + w), 1000 - (y + h)) <= 5:
            edge_like = True
        draw_polygon(draw, poly, color, width=2)
        if show_raw and raw_poly:
            draw_polygon(draw, raw_poly, (255, 255, 0), width=1)
        top_left = min(poly, key=lambda p: (p[1], p[0])) if poly else (x, y)
        draw_label(draw, (top_left[0] + 2, max(0, top_left[1] - 12)), cat, color, font)

    title = f"{image_meta['split']}/{image_meta['scene_id']}/{image_meta['tower_id']}/{image_meta['frame_id']}"
    draw_label(draw, (8, 8), title, (255, 255, 255), font)
    if show_raw:
        draw_label(draw, (8, 24), "yellow=raw_poly, class color=final poly", (255, 255, 0), font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "sample_id": image_meta["sample_id"],
        "split": image_meta["split"],
        "scene_id": image_meta["scene_id"],
        "tower_id": image_meta["tower_id"],
        "frame_id": image_meta["frame_id"],
        "output_file": str(output_path),
        "edge_like": edge_like,
        "show_raw": show_raw,
    }


def build_contact_sheet(image_paths: Sequence[Path], output_path: Path, cols: int = 3, thumb_size: int = 420) -> None:
    if not image_paths:
        return
    rows = math.ceil(len(image_paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (24, 24, 24))
    for idx, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        col = idx % cols
        row = idx // cols
        x = col * thumb_size + (thumb_size - image.width) // 2
        y = row * thumb_size + (thumb_size - image.height) // 2
        sheet.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_index(rows: Iterable[dict], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "split",
                "scene_id",
                "tower_id",
                "frame_id",
                "output_file",
                "edge_like",
                "show_raw",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared/sealand_single_tower_4km_super4",
    )
    parser.add_argument(
        "--output-root",
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/output/label_check_4km_super4_20260323",
    )
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--normal-per-split", type=int, default=4)
    parser.add_argument("--edge-per-split", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    rng = random.Random(args.seed)

    final_paths: List[Path] = []
    compare_paths: List[Path] = []
    index_rows: List[dict] = []

    for split in ["Train", "Valid", "Test"]:
        normal, edge = select_samples(
            root,
            split,
            normal_count=args.normal_per_split,
            edge_count=args.edge_per_split,
            image_size=args.image_size,
            rng=rng,
        )
        for sample in normal:
            out = output_root / "final" / split / f"{sample['scene_id']}_{sample['tower_id']}_{sample['frame_id']}.png"
            index_rows.append(render_image(root, sample, out, show_raw=False))
            final_paths.append(out)
        for sample in edge:
            out = output_root / "edge_compare" / split / f"{sample['scene_id']}_{sample['tower_id']}_{sample['frame_id']}.png"
            index_rows.append(render_image(root, sample, out, show_raw=True))
            compare_paths.append(out)

    build_contact_sheet(final_paths, output_root / "contact_sheet_final.png")
    build_contact_sheet(compare_paths, output_root / "contact_sheet_edge_compare.png")
    write_index(index_rows, output_root / "samples.csv")


if __name__ == "__main__":
    main()
