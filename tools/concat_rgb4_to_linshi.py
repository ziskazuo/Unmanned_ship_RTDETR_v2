#!/usr/bin/env python3
"""Concatenate 4-view RGB images into one 512x256 image with original split structure."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image


CAM_ORDER = ("CamBack", "CamFront", "CamLeft", "CamRight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate CamBack/CamFront/CamLeft/CamRight into one image."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/output/linshi"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Test"],
        help="Splits to process.",
    )
    parser.add_argument(
        "--out-size",
        type=str,
        default="512x256",
        help="Output WxH, e.g. 512x256.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )
    return parser.parse_args()


def parse_size(size_text: str) -> Tuple[int, int]:
    w_text, h_text = size_text.lower().split("x")
    return int(w_text), int(h_text)


def scene_dirs(split_dir: Path) -> List[Path]:
    if split_dir.name == "Train":
        prefix = "T"
    elif split_dir.name == "Test":
        prefix = "S"
    elif split_dir.name == "Valid":
        prefix = "V"
    else:
        prefix = ""
    out = []
    for p in sorted(split_dir.iterdir()):
        if not p.is_dir():
            continue
        if prefix and not p.name.startswith(prefix):
            continue
        out.append(p)
    return out


def tower_dirs(scene_dir: Path) -> List[Path]:
    out = []
    for p in sorted(scene_dir.iterdir()):
        if p.is_dir() and p.name.startswith("CoastGuard"):
            out.append(p)
    return out


def frames_intersection(cam_rgb_dirs: Sequence[Path]) -> List[str]:
    frame_sets = []
    for d in cam_rgb_dirs:
        if not d.is_dir():
            return []
        frame_sets.append({p.name for p in d.glob("*.png")})
    inter = set.intersection(*frame_sets) if frame_sets else set()
    return sorted(inter)


def concat_one(
    frame_name: str,
    cam_paths: Dict[str, Path],
    out_path: Path,
    tile_size: Tuple[int, int],
    out_size: Tuple[int, int],
) -> bool:
    tw, th = tile_size
    out_w, out_h = out_size
    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    # layout: Back Front / Left Right
    layout = {
        "CamBack": (0, 0),
        "CamFront": (tw, 0),
        "CamLeft": (0, th),
        "CamRight": (tw, th),
    }
    try:
        for cam in CAM_ORDER:
            img = Image.open(cam_paths[cam] / frame_name).convert("RGB")
            img = img.resize((tw, th), Image.BILINEAR)
            ox, oy = layout[cam]
            canvas.paste(img, (ox, oy))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    out_w, out_h = parse_size(args.out_size)
    tile_size = (out_w // 2, out_h // 2)
    args.output_root.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    total_ok = 0
    total_fail = 0
    total_towers = 0

    for split in args.splits:
        split_dir = args.dataset_root / split
        if not split_dir.is_dir():
            print(f"[WARN] Missing split dir: {split_dir}")
            continue
        for scene_dir in scene_dirs(split_dir):
            for tower_dir in tower_dirs(scene_dir):
                cam_rgb = {
                    cam: tower_dir / "cams" / cam / "rgb"
                    for cam in CAM_ORDER
                }
                frames = frames_intersection(list(cam_rgb.values()))
                if not frames:
                    continue

                out_dir = (
                    args.output_root
                    / split
                    / scene_dir.name
                    / tower_dir.name
                    / "cams"
                    / "Cam4Concat"
                    / "rgb"
                )
                total_towers += 1

                futures = []
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    for frame_name in frames:
                        out_path = out_dir / frame_name
                        futures.append(
                            ex.submit(
                                concat_one,
                                frame_name=frame_name,
                                cam_paths=cam_rgb,
                                out_path=out_path,
                                tile_size=tile_size,
                                out_size=(out_w, out_h),
                            )
                        )
                    for fut in as_completed(futures):
                        total_tasks += 1
                        if fut.result():
                            total_ok += 1
                        else:
                            total_fail += 1

                print(
                    f"[DONE] {split}/{scene_dir.name}/{tower_dir.name}: "
                    f"{len(frames)} frames -> {out_dir}"
                )

    print(
        f"[SUMMARY] towers={total_towers}, total={total_tasks}, "
        f"ok={total_ok}, fail={total_fail}, output_root={args.output_root}"
    )


if __name__ == "__main__":
    main()
