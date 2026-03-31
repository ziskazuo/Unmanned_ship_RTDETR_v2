#!/usr/bin/env python3
"""Create a fixed random-400 test subset COCO json for online validation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to the full test_coco.json file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to write the fixed subset COCO json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260327,
        help="Random seed used for fixed subset sampling.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=400,
        help="Number of images to keep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input_json.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = list(coco.get("images", []))
    if args.count <= 0 or args.count > len(images):
        raise ValueError(
            f"Requested subset size {args.count} is invalid for {len(images)} images."
        )

    rng = random.Random(args.seed)
    selected_images = rng.sample(images, args.count)
    selected_images = sorted(selected_images, key=lambda item: int(item["id"]))
    selected_ids = {int(image["id"]) for image in selected_images}

    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": selected_images,
        "annotations": [
            ann for ann in coco.get("annotations", [])
            if int(ann["image_id"]) in selected_ids
        ],
        "categories": coco.get("categories", []),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(subset, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "input_json": str(args.input_json),
                "output_json": str(args.output_json),
                "seed": args.seed,
                "count": len(selected_images),
                "annotation_count": len(subset["annotations"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
