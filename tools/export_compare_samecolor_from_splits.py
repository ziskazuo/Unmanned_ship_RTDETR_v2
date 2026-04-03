#!/usr/bin/env python3
"""Export GT/Pred/Compare visualization samples with shared class colors."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_DIR = (
    REPO_ROOT / "output" / "testfull_eval_best_onlineval_r1536_20260401" / "splits_runtimefix_20260401"
)
DEFAULT_DATASET_ROOT = REPO_ROOT / "prepared" / "sealand_single_tower_4km_super4_1536_route_roi"
DEFAULT_OUTDIR = REPO_ROOT / "output" / "compare40_samecolor_20260401"


CLASS_COLORS = {
    1: (255, 0, 0),      # CargoShip
    2: (0, 0, 255),      # CruiseShip
    3: (0, 255, 0),      # FishingVessel
    4: (255, 165, 0),    # RecreationalBoat
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260401)
    parser.add_argument("--draw-titles", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_color(category_id: int) -> Tuple[int, int, int]:
    return CLASS_COLORS.get(int(category_id), (255, 255, 0))


def text_size(draw: ImageDraw.ImageDraw, text: str, font=None) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    if hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)
    if font is None:
        font = ImageFont.load_default()
    return font.getsize(text)


def draw_polygon_with_label(
    draw: ImageDraw.ImageDraw,
    polygon_xy: List[Tuple[float, float]],
    label: str,
    color: Tuple[int, int, int],
):
    if len(polygon_xy) < 2:
        return
    draw.line(polygon_xy + [polygon_xy[0]], width=2, fill=color)
    xmin = min(p[0] for p in polygon_xy)
    ymin = min(p[1] for p in polygon_xy)
    font = None
    tw, th = text_size(draw, label, font=font)
    top = max(ymin - th, 0)
    draw.rectangle([(xmin + 1, top), (xmin + tw + 2, top + th)], fill=color)
    draw.text((xmin + 1, top), label, fill=(255, 255, 255), font=font)


def ann_to_polygon(ann: Dict) -> List[Tuple[float, float]]:
    seg = ann.get("segmentation")
    if isinstance(seg, list) and seg and isinstance(seg[0], list) and len(seg[0]) >= 8:
        pts = seg[0]
        return [(float(pts[i]), float(pts[i + 1])) for i in range(0, len(pts), 2)]
    bbox = ann.get("bbox", [])
    if len(bbox) == 4:
        x, y, w, h = [float(v) for v in bbox]
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if len(bbox) == 8:
        pts = [float(v) for v in bbox]
        return [(pts[i], pts[i + 1]) for i in range(0, 8, 2)]
    return []


def pred_to_polygon(pred: Dict) -> List[Tuple[float, float]]:
    bbox = pred.get("bbox", [])
    if len(bbox) == 8:
        pts = [float(v) for v in bbox]
        return [(pts[i], pts[i + 1]) for i in range(0, 8, 2)]
    if len(bbox) == 4:
        x, y, w, h = [float(v) for v in bbox]
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return []


def add_title(image: Image.Image, title: str) -> Image.Image:
    font = ImageFont.load_default()
    title_h = 20
    canvas = Image.new("RGB", (image.width, image.height + title_h), (0, 0, 0))
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    tw, th = text_size(draw, title, font=font)
    draw.text(((image.width - tw) // 2, max((title_h - th) // 2, 0)), title, fill=(255, 255, 255), font=font)
    return canvas


def make_compare(pred: Image.Image, gt: Image.Image, draw_titles: bool) -> Image.Image:
    if draw_titles:
        pred = add_title(pred, "Prediction")
        gt = add_title(gt, "GT")
    sep = 8
    out = Image.new("RGB", (pred.width + gt.width + sep, max(pred.height, gt.height)), (0, 0, 0))
    out.paste(pred, (0, 0))
    out.paste(gt, (pred.width + sep, 0))
    return out


def read_split_pair(splits_dir: Path):
    pairs = []
    for coco_path in sorted(splits_dir.glob("*_test_coco.json")):
        prefix = coco_path.name.replace("_test_coco.json", "")
        pred_path = splits_dir / f"{prefix}_bbox.json"
        if pred_path.exists():
            pairs.append((prefix, coco_path, pred_path))
    return pairs


def main():
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    pred_dir = output_dir / "pred"
    gt_dir = output_dir / "GT"
    compare_dir = output_dir / "compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    split_pairs = read_split_pair(splits_dir)
    if not split_pairs:
        raise FileNotFoundError(f"No split pairs found under {splits_dir}")

    img_db: Dict[int, Dict] = {}
    gt_by_img: Dict[int, List[Dict]] = defaultdict(list)
    pred_by_img: Dict[int, List[Dict]] = defaultdict(list)
    catid2name: Dict[int, str] = {}
    gid_to_split: Dict[int, str] = {}

    global_id = 0
    for split_name, coco_path, pred_path in split_pairs:
        coco = load_json(coco_path)
        preds = load_json(pred_path)

        local_to_global = {}
        for cat in coco.get("categories", []):
            catid2name[int(cat["id"])] = cat.get("name", str(cat["id"]))

        for img in coco.get("images", []):
            gid = global_id
            global_id += 1
            local_to_global[int(img["id"])] = gid
            img_db[gid] = img
            gid_to_split[gid] = split_name

        for ann in coco.get("annotations", []):
            gid = local_to_global[int(ann["image_id"])]
            gt_by_img[gid].append(ann)

        for pred in preds:
            if float(pred.get("score", 0.0)) < float(args.score_thr):
                continue
            local_img_id = int(pred["image_id"])
            if local_img_id not in local_to_global:
                continue
            gid = local_to_global[local_img_id]
            pred_by_img[gid].append(pred)

    all_gids = sorted(img_db.keys())
    if len(all_gids) == 0:
        raise RuntimeError("No images found from split json files.")

    rng = random.Random(args.seed)
    if args.num_samples >= len(all_gids):
        selected = all_gids
    else:
        selected = sorted(rng.sample(all_gids, args.num_samples))

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "global_image_id", "split", "scene_id", "tower_id", "frame_name"])

        for idx, gid in enumerate(selected, start=1):
            info = img_db[gid]
            radar_rel = info.get("radar_im_file") or info.get("im_file")
            if not radar_rel:
                continue
            radar_path = Path(radar_rel)
            if not radar_path.is_absolute():
                radar_path = dataset_root / radar_path
            if not radar_path.exists():
                continue

            base = Image.open(radar_path).convert("RGB")
            pred_img = base.copy()
            gt_img = base.copy()
            draw_pred = ImageDraw.Draw(pred_img)
            draw_gt = ImageDraw.Draw(gt_img)

            for ann in gt_by_img.get(gid, []):
                catid = int(ann["category_id"])
                label = catid2name.get(catid, str(catid))
                poly = ann_to_polygon(ann)
                draw_polygon_with_label(draw_gt, poly, label, get_color(catid))

            for pred in pred_by_img.get(gid, []):
                catid = int(pred["category_id"])
                score = float(pred.get("score", 0.0))
                label = f"{catid2name.get(catid, str(catid))}:{score:.2f}"
                poly = pred_to_polygon(pred)
                draw_polygon_with_label(draw_pred, poly, label, get_color(catid))

            frame_name = Path(info.get("file_name", f"{gid:06d}.png")).name
            stem = Path(frame_name).stem
            out_name = f"{idx:03d}_{gid_to_split[gid]}_{stem}.png"
            pred_img.save(pred_dir / out_name, format="PNG")
            gt_img.save(gt_dir / out_name, format="PNG")
            compare = make_compare(pred_img, gt_img, args.draw_titles)
            compare.save(compare_dir / out_name, format="PNG")

            writer.writerow(
                [
                    idx,
                    gid,
                    gid_to_split[gid],
                    info.get("scene_id", ""),
                    info.get("tower_id", ""),
                    frame_name,
                ]
            )

    print(f"Done. Exported {len(selected)} samples to: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
