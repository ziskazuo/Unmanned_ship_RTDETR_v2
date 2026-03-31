#!/usr/bin/env python3
"""Wait for free GPUs, then launch a single rotated-box checkpoint sweep."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/data1/zuokun/vene/Unmanned_ship_RTDETR/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-pool", default="4,7")
    parser.add_argument("--min-free", type=int, default=1)
    parser.add_argument("--check-interval", type=int, default=60)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--config", required=True)
    parser.add_argument("--anno-file", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--end-epoch", type=int, default=49)
    return parser.parse_args()


def gpu_is_free(gpu_id: str) -> bool:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-i", gpu_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception:
        return False

    if proc.returncode != 0:
        return False
    text = proc.stdout
    if "No running processes found" in text:
        return True
    # Fallback: treat the GPU as occupied if any python/C process is listed.
    return not bool(re.search(r"\|\s+\d+\s+N/A\s+N/A\s+\d+\s+[CG]\s+", text))


def main():
    args = parse_args()
    gpu_pool = [x.strip() for x in args.gpu_pool.split(",") if x.strip()]
    stable_hits = 0
    last_free = []

    while True:
        free = [gpu for gpu in gpu_pool if gpu_is_free(gpu)]
        print(f"[{time.strftime('%F %T')}] free_gpus={free}", flush=True)
        if len(free) >= args.min_free:
            if free == last_free:
                stable_hits += 1
            else:
                stable_hits = 1
            last_free = free
            if stable_hits >= args.stable_checks:
                launch_gpus = ",".join(free)
                cmd = [
                    str(PYTHON_BIN),
                    "tools/sweep_rbox_coco_style.py",
                    "--config",
                    args.config,
                    "--anno-file",
                    args.anno_file,
                    "--ckpt-dir",
                    args.ckpt_dir,
                    "--output-root",
                    args.output_root,
                    "--gpus",
                    launch_gpus,
                    "--start-epoch",
                    str(args.start_epoch),
                    "--end-epoch",
                    str(args.end_epoch),
                ]
                env = os.environ.copy()
                env["LD_LIBRARY_PATH"] = (
                    "/data1/zuokun/vene/nccl/2.20.3-cuda11.0/nccl_2.20.3-1+cuda11.0_x86_64/lib:"
                    "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
                )
                env["FLAGS_allocator_strategy"] = "auto_growth"
                print(
                    f"[{time.strftime('%F %T')}] launching sweep on gpus={launch_gpus}",
                    flush=True,
                )
                subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
                return
        else:
            stable_hits = 0
            last_free = free
        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
