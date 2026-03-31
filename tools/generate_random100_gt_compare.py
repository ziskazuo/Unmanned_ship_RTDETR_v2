#!/usr/bin/env python3
"""Generate GT and compare images for random-100 sealand evaluation outputs."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / 'prepared' / 'sealand_single_tower_4km_super4_960'

from PIL import Image, ImageDraw, ImageFont


PREDEFINED_COLORS = {
    0: (255, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (255, 165, 0),
    4: (160, 32, 240),
    5: (0, 255, 255),
    6: (148, 87, 235),
    7: (255, 127, 80),
    8: (135, 206, 250),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--draw-titles", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_color(catid: int):
    return PREDEFINED_COLORS.get(catid, (255, 255, 0))


def text_size(draw: ImageDraw.ImageDraw, text: str, font=None) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    if hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)
    if font is None:
        font = ImageFont.load_default()
    return font.getsize(text)


def draw_rot_or_hbb(draw: ImageDraw.ImageDraw, ann: Dict, catid2name: Dict[int, str]):
    catid = int(ann["category_id"])
    color = get_color(catid)
    font = None

    seg = ann.get("segmentation")
    if seg and isinstance(seg, list) and seg[0]:
        pts = seg[0]
        xy = [(float(pts[i]), float(pts[i + 1])) for i in range(0, len(pts), 2)]
        if len(xy) >= 2:
            draw.line(xy + [xy[0]], width=2, fill=color)
            xmin = min(p[0] for p in xy)
            ymin = min(p[1] for p in xy)
        else:
            xmin = ymin = 0.0
    else:
        xmin, ymin, w, h = ann["bbox"]
        xmax, ymax = xmin + w, ymin + h
        draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)], width=2, fill=color)

    label = catid2name.get(catid, str(catid))
    tw, th = text_size(draw, label, font=font)
    box_top = max(float(ymin) - th, 0)
    draw.rectangle([(float(xmin) + 1, box_top), (float(xmin) + tw + 1, box_top + th)], fill=color)
    draw.text((float(xmin) + 1, box_top), label, fill=(255, 255, 255), font=font)


def add_title(image: Image.Image, title: str) -> Image.Image:
    font = ImageFont.load_default()
    title_h = 18
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


def process_subset(subset_path: Path, task_dir: Path, dataset_root: Path, draw_titles: bool):
    subset = load_json(subset_path)
    catid2name = {int(cat["id"]): cat["name"] for cat in subset["categories"]}
    imgs = {int(img["id"]): img for img in subset["images"]}
    anns_by_img: Dict[int, List[Dict]] = {}
    for ann in subset["annotations"]:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    vis_dir = task_dir / "vis"
    gt_dir = task_dir / "GT"
    compare_dir = task_dir / "compare"
    gt_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    for image_id, image_info in imgs.items():
        radar_path = Path(image_info.get("radar_im_file") or image_info.get("im_file"))
        if not radar_path.is_absolute():
            radar_path = dataset_root / radar_path
        if not radar_path.exists():
            continue
        base = Image.open(radar_path).convert("RGB")
        gt_img = base.copy()
        draw = ImageDraw.Draw(gt_img)
        for ann in anns_by_img.get(image_id, []):
            draw_rot_or_hbb(draw, ann, catid2name)

        filename = f"{image_id:06d}.png"
        gt_path = gt_dir / filename
        gt_img.save(gt_path, format="PNG")

        pred_path = vis_dir / filename
        if pred_path.exists():
            pred_img = Image.open(pred_path).convert("RGB")
            compare = make_compare(pred_img, gt_img, draw_titles)
            compare.save(compare_dir / filename, format="PNG")


def main():
    args = parse_args()
    result_root = Path(args.result_root)
    dataset_root = Path(args.dataset_root)
    annotations_dir = result_root / "annotations"
    for subset_path in sorted(annotations_dir.glob("*.json")):
        name = subset_path.stem
        parts = name.split("_")
        if len(parts) < 3:
            continue
        scene = parts[0]
        tower = parts[1]
        task_dir = result_root / scene / tower
        if task_dir.exists():
            process_subset(subset_path, task_dir, dataset_root, args.draw_titles)


if __name__ == "__main__":
    main()
