#!/usr/bin/env python3
"""Run per-tower random-100 evaluation/visualization on the sealand test split."""

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/data1/zuokun/vene/Unmanned_ship_RTDETR/bin/python"
DEFAULT_CONFIG = "configs/rtdetr/sealand_radardetr_r50vd_50e_bs4_4km_super4_960.yml"
DEFAULT_WEIGHTS = (
    "output/sealand_radardetr_r50vd_50e_bs4_4km_super4_960/"
    "sealand_radardetr_r50vd_50e_bs4_4km_super4_960/model_final.pdparams"
)
DEFAULT_DATASET_ROOT = "prepared/sealand_single_tower_4km_super4_960"
DEFAULT_TEST_COCO = "prepared/sealand_single_tower_4km_super4_960/test_coco.json"
DEFAULT_OUTPUT_ROOT = "/data1/zuokun/Result/RTDETR/test_random100_4km_super4_960"
DEFAULT_NCCL = (
    "/data1/zuokun/vene/nccl/2.20.3-cuda11.0/"
    "nccl_2.20.3-1+cuda11.0_x86_64/lib"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--test-coco", default=DEFAULT_TEST_COCO)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--samples-per-tower", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--draw-threshold", type=float, default=0.3)
    parser.add_argument("--scenes", nargs="*", default=["S01", "S02", "S03"])
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_coco(coco_path: Path):
    with coco_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def group_images(images, scenes):
    grouped = {}
    for image in images:
        scene = image.get("scene_id")
        tower = image.get("tower_id")
        if scene not in scenes:
            continue
        grouped.setdefault((scene, tower), []).append(image)
    return grouped


def sample_groups(grouped, samples_per_tower, seed):
    sampled = {}
    for index, key in enumerate(sorted(grouped)):
        images = sorted(grouped[key], key=lambda item: (item["frame_id"], item["id"]))
        rng = random.Random(seed + index)
        sample_count = min(samples_per_tower, len(images))
        selected = rng.sample(images, sample_count)
        sampled[key] = sorted(selected, key=lambda item: (item["frame_id"], item["id"]))
    return sampled


def write_subset_cocos(coco_data, sampled_groups, output_root: Path):
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    anns_by_img = {}
    for ann in coco_data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    subset_tasks = []
    for scene, tower in sorted(sampled_groups):
        images = sampled_groups[(scene, tower)]
        annotations = []
        for image in images:
            annotations.extend(anns_by_img.get(image["id"], []))
        subset = {
            "images": images,
            "annotations": annotations,
            "categories": coco_data["categories"],
            "licenses": coco_data.get("licenses", []),
            "info": {
                "source": "sealand test subset",
                "scene": scene,
                "tower": tower,
                "sample_count": len(images),
            },
        }
        subset_name = f"{scene}_{tower}_random{len(images)}.json"
        subset_path = annotations_dir / subset_name
        with subset_path.open("w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)

        task_dir = output_root / scene / tower
        task_dir.mkdir(parents=True, exist_ok=True)
        selection_path = task_dir / "selected_samples.json"
        with selection_path.open("w", encoding="utf-8") as f:
            json.dump(images, f, ensure_ascii=False, indent=2)

        subset_tasks.append(
            {
                "scene": scene,
                "tower": tower,
                "sample_count": len(images),
                "subset_path": subset_path,
                "task_dir": task_dir,
                "selection_path": selection_path,
            }
        )
    return subset_tasks


def build_env(gpu: str):
    env = os.environ.copy()
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{DEFAULT_NCCL}:/usr/local/cuda/lib64:{current}"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def extract_map(log_text: str):
    match = re.search(r"mAP\(0\.50,\s*11point\)\s*=\s*([0-9.]+)%", log_text)
    if match:
        return float(match.group(1))
    return None


def run_command(cmd, env, log_path: Path):
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return process.returncode


def write_eval_config(task, args):
    config_path = task["task_dir"] / "eval_config.yml"
    config_text = f"""_BASE_:\n  - {args.base_config}\n\nweights: {args.weights}\n\nEvalDataset:\n  !CameraRadar_COCODataSet\n    dataset_dir: {args.dataset_root}\n    radar_image_dir: ''\n    camera_image_dir: ''\n    anno_path: {task['subset_path']}\n    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd', 'gt_poly']\n    allow_empty: true\n"""
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def run_task(task, gpu, args):
    scene = task["scene"]
    tower = task["tower"]
    task_dir = task["task_dir"]
    vis_dir = task_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    eval_log = task_dir / "eval.log"
    vis_log = task_dir / "visualize.log"
    metrics_path = task_dir / "metrics.json"
    task_config = write_eval_config(task, args)

    env = build_env(gpu)

    eval_cmd = [args.python, "tools/eval.py", "-c", str(task_config)]
    eval_return = run_command(eval_cmd, env, eval_log)

    eval_text = eval_log.read_text(encoding="utf-8", errors="ignore")
    map50 = extract_map(eval_text)

    vis_cmd = [
        args.python,
        "tools/visualize_eval.py",
        "-c",
        str(task_config),
        "--weights",
        args.weights,
        "--output_dir",
        str(vis_dir),
        "--max_images",
        str(task["sample_count"]),
        "--draw_threshold",
        str(args.draw_threshold),
    ]
    vis_return = run_command(vis_cmd, env, vis_log)

    vis_count = len(list(vis_dir.glob("*.png")))
    metrics = {
        "scene": scene,
        "tower": tower,
        "gpu": gpu,
        "sample_count": task["sample_count"],
        "mAP_50_11point": map50,
        "eval_returncode": eval_return,
        "visualize_returncode": vis_return,
        "visualization_count": vis_count,
        "subset_path": str(task["subset_path"]),
        "selection_path": str(task["selection_path"]),
        "config_path": str(task_config),
        "eval_log": str(eval_log),
        "visualize_log": str(vis_log),
        "vis_dir": str(vis_dir),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def write_summary(results, output_root: Path):
    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "tower",
                "gpu",
                "sample_count",
                "mAP_50_11point",
                "eval_returncode",
                "visualize_returncode",
                "visualization_count",
                "subset_path",
                "selection_path",
                "config_path",
                "vis_dir",
                "eval_log",
                "visualize_log",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():
    args = parse_args()
    args.base_config = str(resolve_path(args.config))
    args.weights = str(resolve_path(args.weights))
    args.dataset_root = str(resolve_path(args.dataset_root))
    test_coco = resolve_path(args.test_coco)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    coco_data = load_coco(test_coco)
    grouped = group_images(coco_data["images"], set(args.scenes))
    sampled = sample_groups(grouped, args.samples_per_tower, args.seed)
    tasks = write_subset_cocos(coco_data, sampled, output_root)

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    results = []
    pending = list(tasks)
    futures = {}
    free_gpus = list(gpus)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        while pending or futures:
            while pending and free_gpus:
                gpu = free_gpus.pop(0)
                task = pending.pop(0)
                future = executor.submit(run_task, task, gpu, args)
                futures[future] = gpu
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                gpu = futures.pop(future)
                results.append(future.result())
                free_gpus.append(gpu)

    results = sorted(results, key=lambda item: (item["scene"], item["tower"]))
    write_summary(results, output_root)
    for row in results:
        print(
            f'{row["scene"]}/{row["tower"]}: '
            f'mAP@0.50={row["mAP_50_11point"]}, '
            f'vis={row["visualization_count"]}'
        )
    print(f"summary_json={output_root / 'summary.json'}")
    print(f"summary_csv={output_root / 'summary.csv'}")


if __name__ == "__main__":
    sys.exit(main())
