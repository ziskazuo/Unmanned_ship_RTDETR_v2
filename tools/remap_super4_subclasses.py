#!/usr/bin/env python3
"""Remap selected fine-grained subclasses to new super4 classes in prepared datasets."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FINE_TO_SUPER_OVERRIDES = {
    "libertyship": "FishingVessel",
    "Boataaa": "FishingVessel",
    "HouseBoat": "CruiseShip",
}

SUPER_TO_ID = {
    "CargoShip": 1,
    "CruiseShip": 2,
    "FishingVessel": 3,
    "RecreationalBoat": 4,
}

SUPER4_NAMES = set(SUPER_TO_ID.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply subclass remapping to prepared super4 dataset json files."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "prepared/sealand_single_tower_4km_super4_1536_route_roi",
            "prepared/sealand_single_tower_4km_super4_960_route_roi",
            "prepared/sealand_single_tower_4km_super4_960",
            "prepared/sealand_single_tower_4km_super4",
        ],
        help="Prepared dataset roots to update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args()


def scene_to_split(scene_id: str) -> Optional[str]:
    if scene_id.startswith("T"):
        return "Train"
    if scene_id.startswith("V"):
        return "Valid"
    if scene_id.startswith("S"):
        return "Test"
    if scene_id.startswith("H"):
        return "StressTest"
    return None


def image_key_from_file_name(file_name: str) -> Optional[Tuple[str, str, str, str]]:
    parts = file_name.split("/")
    if len(parts) != 3:
        return None
    scene, tower, frame_png = parts
    split = scene_to_split(scene)
    if split is None:
        return None
    frame = frame_png.replace(".png", "")
    return split, scene, tower, frame


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def remap_per_frame_annotation_file(path: Path, dry_run: bool) -> Tuple[int, List[str]]:
    data = load_json(path)
    anns = data.get("annotations", [])
    changed = 0
    supers: List[str] = []
    for ann in anns:
        fine = ann.get("fine_category_name")
        old_super = ann.get("super_category_name")
        new_super = FINE_TO_SUPER_OVERRIDES.get(fine, old_super)
        if new_super is None:
            new_super = old_super
        if new_super != old_super:
            ann["super_category_name"] = new_super
            ann["category_name"] = new_super
            changed += 1
        supers.append(ann.get("super_category_name", old_super))
    if changed > 0 and (not dry_run):
        dump_json(path, data)
    return changed, supers


def process_root(root: Path, dry_run: bool) -> Dict[str, int]:
    stats = {
        "frame_files_changed": 0,
        "frame_annotations_changed": 0,
        "coco_files_changed": 0,
        "coco_annotations_changed": 0,
        "coco_images_missed": 0,
    }

    # Cache ordered super labels per image key, from per-frame annotations.
    frame_super_cache: Dict[Tuple[str, str, str, str], List[str]] = {}

    for split in ("Train", "Valid", "Test", "StressTest"):
        ann_dir = root / split / "annotations"
        if not ann_dir.is_dir():
            continue
        for frame_file in sorted(ann_dir.glob("*/*/*.json")):
            changed, supers = remap_per_frame_annotation_file(frame_file, dry_run=dry_run)
            if changed > 0:
                stats["frame_files_changed"] += 1
                stats["frame_annotations_changed"] += changed

            scene = frame_file.parts[-3]
            tower = frame_file.parts[-2]
            frame = frame_file.stem
            key = (split, scene, tower, frame)
            frame_super_cache[key] = supers

    # Update top-level coco-like files
    for json_file in sorted(root.glob("*.json")):
        try:
            data = load_json(json_file)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if "images" not in data or "annotations" not in data or "categories" not in data:
            continue

        cat_names = {c.get("name") for c in data.get("categories", []) if isinstance(c, dict)}
        if not SUPER4_NAMES.issubset(cat_names):
            # Skip non-super4 json (e.g., one-class derivatives)
            continue

        anns_by_img: Dict[int, List[dict]] = {}
        for ann in data["annotations"]:
            anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

        changed_count = 0
        missed_img = 0

        for image in data["images"]:
            image_id = int(image["id"])
            ann_list = anns_by_img.get(image_id, [])
            if not ann_list:
                continue

            key = image_key_from_file_name(str(image.get("file_name", "")))
            if key is None:
                missed_img += 1
                continue

            supers = frame_super_cache.get(key)
            if supers is None:
                # Lazy load from frame annotation if not in cache yet.
                split, scene, tower, frame = key
                frame_path = root / split / "annotations" / scene / tower / f"{frame}.json"
                if not frame_path.is_file():
                    missed_img += 1
                    continue
                _, supers = remap_per_frame_annotation_file(frame_path, dry_run=dry_run)
                frame_super_cache[key] = supers

            if len(supers) != len(ann_list):
                # Fallback: do not touch this image if ordering is inconsistent.
                missed_img += 1
                continue

            for ann, super_name in zip(ann_list, supers):
                target_id = SUPER_TO_ID.get(super_name)
                if target_id is None:
                    continue
                if int(ann.get("category_id", -1)) != target_id:
                    ann["category_id"] = target_id
                    changed_count += 1

        if changed_count > 0:
            stats["coco_files_changed"] += 1
            stats["coco_annotations_changed"] += changed_count
            if not dry_run:
                dump_json(json_file, data)

        stats["coco_images_missed"] += missed_img

    return stats


def main() -> None:
    args = parse_args()
    for root_text in args.roots:
        root = Path(root_text)
        if not root.is_dir():
            print(f"[SKIP] Missing root: {root}")
            continue
        stats = process_root(root, dry_run=args.dry_run)
        print(f"[ROOT] {root}")
        for k, v in stats.items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
