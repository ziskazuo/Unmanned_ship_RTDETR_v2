#!/usr/bin/env python3
"""Run class-agnostic ship-only evaluation on existing sealand random-100 test subsets."""

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/data1/zuokun/vene/Unmanned_ship_RTDETR/bin/python"
DEFAULT_ROOT = "/data1/zuokun/Result/RTDETR/test_random100_4km_super4_960"
DEFAULT_NCCL = (
    "/data1/zuokun/vene/nccl/2.20.3-cuda11.0/"
    "nccl_2.20.3-1+cuda11.0_x86_64/lib"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default=DEFAULT_PYTHON)
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--gpus', default='0,1,2,3')
    return parser.parse_args()


def build_env(gpu: str):
    env = os.environ.copy()
    current = env.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = f"{DEFAULT_NCCL}:/usr/local/cuda/lib64:{current}"
    env['CUDA_VISIBLE_DEVICES'] = gpu
    return env


def run_task(task_dir: Path, gpu: str, python_bin: str):
    eval_config = task_dir / 'eval_config.yml'
    output_json = task_dir / 'ship_only_metrics.json'
    log_path = task_dir / 'ship_only_eval.log'
    cmd = [python_bin, 'tools/eval_ship_only.py', '-c', str(eval_config), '--output_json', str(output_json)]
    with log_path.open('w', encoding='utf-8') as f:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=build_env(gpu), stdout=f, stderr=subprocess.STDOUT, text=True)
    result = {'scene': task_dir.parent.name, 'tower': task_dir.name, 'gpu': gpu, 'eval_returncode': proc.returncode, 'log': str(log_path)}
    if output_json.exists():
        with output_json.open('r', encoding='utf-8') as f:
            result.update(json.load(f))
        result['output_json'] = str(output_json)
    return result


def main():
    args = parse_args()
    root = Path(args.root)
    task_dirs = sorted([p.parent for p in root.glob('S*/CoastGuard*/eval_config.yml')])
    gpus = [gpu.strip() for gpu in args.gpus.split(',') if gpu.strip()]
    results = []
    pending = list(task_dirs)
    futures = {}
    free_gpus = list(gpus)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        while pending or futures:
            while pending and free_gpus:
                gpu = free_gpus.pop(0)
                task_dir = pending.pop(0)
                future = executor.submit(run_task, task_dir, gpu, args.python)
                futures[future] = gpu
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                gpu = futures.pop(future)
                results.append(future.result())
                free_gpus.append(gpu)

    results = sorted(results, key=lambda x: (x['scene'], x['tower']))
    summary_json = root / 'ship_only_summary.json'
    summary_csv = root / 'ship_only_summary.csv'
    with summary_json.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with summary_csv.open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['scene', 'tower', 'gpu', 'ship_mAP_50_11point', 'overlap_thresh', 'map_type', 'eval_returncode', 'output_json', 'log']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fieldnames})
    for row in results:
        print(f"{row['scene']}/{row['tower']}: ship_mAP@0.50={row.get('ship_mAP_50_11point')}")
    print(f'summary_json={summary_json}')
    print(f'summary_csv={summary_csv}')


if __name__ == '__main__':
    sys.exit(main())
