#!/usr/bin/env python3
"""Sweep checkpoints with save_prediction_only eval + rotated AP50 / mAP50:95."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import threading
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/data1/zuokun/vene/Unmanned_ship_RTDETR/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--anno-file", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--start-epoch", type=int, default=None)
    parser.add_argument("--end-epoch", type=int, default=None)
    return parser.parse_args()


def discover_checkpoints(
    ckpt_dir: Path,
    start_epoch: int | None,
    end_epoch: int | None,
    include_final: bool,
) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for path in sorted(ckpt_dir.glob("*.pdparams")):
        stem = path.stem
        if stem == "model_final":
            if include_final:
                items.append((stem, path))
            continue
        if not re.fullmatch(r"\d+", stem):
            continue
        epoch = int(stem)
        if start_epoch is not None and epoch < start_epoch:
            continue
        if end_epoch is not None and epoch > end_epoch:
            continue
        items.append((stem, path))
    return items


def worker(
    gpu_id: str,
    config: str,
    anno_file: str,
    output_root: Path,
    jobs: "queue.Queue[Tuple[str, Path]]",
    summaries: List[dict],
):
    while True:
        try:
            name, ckpt_path = jobs.get_nowait()
        except queue.Empty:
            return

        run_dir = output_root / name
        pred_dir = run_dir / "pred"
        pred_dir.mkdir(parents=True, exist_ok=True)
        eval_log = run_dir / "eval.log"
        metric_json = run_dir / "metrics.json"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["LD_LIBRARY_PATH"] = (
            "/data1/zuokun/vene/nccl/2.20.3-cuda11.0/nccl_2.20.3-1+cuda11.0_x86_64/lib:"
            "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        )
        env["FLAGS_allocator_strategy"] = "auto_growth"

        eval_cmd = [
            str(PYTHON_BIN),
            "tools/eval.py",
            "-c",
            config,
            "--save_prediction_only",
            "--output_eval",
            str(pred_dir),
            "-o",
            f"weights={ckpt_path}",
        ]
        with open(eval_log, "w", encoding="utf-8") as f:
            subprocess.run(
                eval_cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )

        cpu_env = env.copy()
        cpu_env["CUDA_VISIBLE_DEVICES"] = ""
        cpu_env["FLAGS_use_gpu"] = "false"
        metric_cmd = [
            str(PYTHON_BIN),
            "tools/eval_rbox_coco_style.py",
            "--anno-file",
            anno_file,
            "--pred-json",
            str(pred_dir / "bbox.json"),
            "--output-json",
            str(metric_json),
        ]
        subprocess.run(metric_cmd, cwd=REPO_ROOT, env=cpu_env, check=True)
        with open(metric_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["checkpoint"] = name
        data["weights"] = str(ckpt_path)
        summaries.append(data)
        jobs.task_done()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(
        Path(args.ckpt_dir),
        args.start_epoch,
        args.end_epoch,
        args.include_final,
    )
    if not checkpoints:
        raise SystemExit("No checkpoints found for the requested range.")

    jobs: "queue.Queue[Tuple[str, Path]]" = queue.Queue()
    for item in checkpoints:
        jobs.put(item)

    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    summaries: List[dict] = []
    threads = [
        threading.Thread(
            target=worker,
            args=(gpu_id, args.config, args.anno_file, output_root, jobs, summaries),
            daemon=True,
        )
        for gpu_id in gpu_ids
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summaries.sort(key=lambda x: x["checkpoint"])
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if summaries:
        best_ap50 = max(summaries, key=lambda x: x["AP50"])
        best_map = max(summaries, key=lambda x: x["mAP50_95"])
        print(json.dumps({
            "num_checkpoints": len(summaries),
            "best_AP50": best_ap50,
            "best_mAP50_95": best_map,
            "summary_json": str(summary_path.resolve()),
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
