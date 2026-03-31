#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_gpu_pool(value: str):
    gpu_ids = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        gpu_ids.append(int(part))
    if not gpu_ids:
        raise argparse.ArgumentTypeError('GPU pool cannot be empty.')
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError('GPU ids must be unique.')
    return gpu_ids


def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(message: str):
    print(f'[{timestamp()}] {message}', flush=True)


def query_gpu_states():
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    states = {}
    for raw_line in result.stdout.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        index_str, mem_str, util_str = [item.strip() for item in line.split(',')]
        states[int(index_str)] = {
            'memory_used_mib': int(mem_str),
            'util_percent': int(util_str),
        }
    return states


def free_gpus_in_pool(states, pool, max_memory_mib, max_util_percent):
    free = []
    busy = []
    for gpu_id in pool:
        state = states.get(gpu_id)
        if state is None:
            busy.append((gpu_id, None))
            continue
        if state['memory_used_mib'] <= max_memory_mib and state['util_percent'] <= max_util_percent:
            free.append(gpu_id)
        else:
            busy.append((gpu_id, state))
    return free, busy


def choose_group(free_ids, min_gpus, max_gpus):
    if len(free_ids) >= max_gpus:
        return free_ids[:max_gpus]
    if len(free_ids) >= min_gpus:
        return free_ids[:min_gpus]
    return None


def build_launch_command(args, selected_gpus):
    cmd = [
        'bash',
        args.train_script,
        '--gpus',
        ','.join(str(gpu_id) for gpu_id in selected_gpus),
    ]
    if args.config:
        cmd += ['--config', args.config]
    if args.python:
        cmd += ['--python', args.python]
    if args.env_dir:
        cmd += ['--env-dir', args.env_dir]
    if args.no_amp or '_fp32_' in args.config:
        cmd.append('--no-amp')
    if args.extra_args:
        cmd.append('--')
        cmd.extend(args.extra_args)
    return cmd


def write_launch_manifest(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Wait for 3 or 4 GPUs to become free, then launch Sealand multi-GPU training.'
    )
    parser.add_argument('--gpu-pool', type=parse_gpu_pool, default=parse_gpu_pool('0,1,2,3,4,5,6,7'),
                        help='Ordered GPU ids to watch, e.g. 0,1,2,3')
    parser.add_argument('--min-gpus', type=int, default=3,
                        help='Launch when at least this many GPUs are free.')
    parser.add_argument('--max-gpus', type=int, default=4,
                        help='Prefer this many GPUs when enough are free.')
    parser.add_argument('--max-memory-mib', type=int, default=256,
                        help='Treat a GPU as free only if used memory is below this threshold.')
    parser.add_argument('--max-util-percent', type=int, default=10,
                        help='Treat a GPU as free only if utilization is below this threshold.')
    parser.add_argument('--poll-seconds', type=int, default=60,
                        help='Seconds between GPU availability checks.')
    parser.add_argument('--stable-checks', type=int, default=2,
                        help='Require the same candidate GPU group to remain free for this many consecutive checks.')
    parser.add_argument('--train-script', default='tools/train_sealand_gpu4.sh',
                        help='Launcher script to execute once GPUs are free.')
    parser.add_argument('--config', default='configs/rtdetr/sealand_radardetr_r50vd_50e_bs2_fp32_stable_4km_super4_960.yml',
                        help='Training config passed to the launcher.')
    parser.add_argument('--python', default='',
                        help='Optional python interpreter passed through to the launcher.')
    parser.add_argument('--env-dir', default='/data1/zuokun/vene/Unmanned_ship_RTDETR',
                        help='Optional env dir passed through to the launcher.')
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP for the launched training job.')
    parser.add_argument('--launch-log-dir', default='output/gpu_queue_launcher',
                        help='Directory for queue manifests and child launcher logs.')
    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
                        help='Extra args appended after -- and passed to train.py via the launcher.')
    args = parser.parse_args()

    if args.min_gpus < 1:
        parser.error('--min-gpus must be >= 1')
    if args.max_gpus < args.min_gpus:
        parser.error('--max-gpus must be >= --min-gpus')

    root_dir = Path(__file__).resolve().parents[1]
    launch_log_dir = (root_dir / args.launch_log_dir).resolve()
    launch_log_dir.mkdir(parents=True, exist_ok=True)

    cleaned_extra = list(args.extra_args)
    if cleaned_extra[:1] == ['--']:
        cleaned_extra = cleaned_extra[1:]
    args.extra_args = cleaned_extra

    log('Watching GPUs ' + ','.join(str(gpu_id) for gpu_id in args.gpu_pool)
        + f' with thresholds mem<={args.max_memory_mib}MiB util<={args.max_util_percent}%')
    log(f'Will launch when {args.min_gpus}-{args.max_gpus} GPUs are free and stable for {args.stable_checks} checks.')

    previous_candidate = None
    stable_count = 0

    while True:
        states = query_gpu_states()
        free_ids, busy = free_gpus_in_pool(states, args.gpu_pool, args.max_memory_mib, args.max_util_percent)
        candidate = choose_group(free_ids, args.min_gpus, args.max_gpus)

        busy_desc = []
        for gpu_id, state in busy:
            if state is None:
                busy_desc.append(f'{gpu_id}:missing')
            else:
                busy_desc.append(f"{gpu_id}:mem={state['memory_used_mib']}MiB util={state['util_percent']}%")

        if candidate is None:
            previous_candidate = None
            stable_count = 0
            log('No launch group yet. Free GPUs=' + ','.join(map(str, free_ids))
                + ' busy=' + '; '.join(busy_desc))
            time.sleep(args.poll_seconds)
            continue

        candidate_tuple = tuple(candidate)
        if candidate_tuple == previous_candidate:
            stable_count += 1
        else:
            previous_candidate = candidate_tuple
            stable_count = 1

        log('Candidate GPUs=' + ','.join(map(str, candidate))
            + f' stable_checks={stable_count}/{args.stable_checks}')

        if stable_count < args.stable_checks:
            time.sleep(args.poll_seconds)
            continue

        cmd = build_launch_command(args, candidate)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        manifest_path = launch_log_dir / f'launch_{stamp}.json'
        launcher_log_path = launch_log_dir / f'launcher_{stamp}.log'
        manifest = {
            'timestamp': timestamp(),
            'selected_gpus': candidate,
            'command': cmd,
            'cwd': str(root_dir),
            'launcher_log': str(launcher_log_path),
        }
        write_launch_manifest(manifest_path, manifest)
        log(f'Launching training on GPUs {candidate}.')
        log('Command: ' + ' '.join(shlex.quote(part) for part in cmd))
        log(f'Manifest: {manifest_path}')
        log(f'Launcher log: {launcher_log_path}')

        with launcher_log_path.open('a', encoding='utf-8') as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=root_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        log(f'Launched PID {process.pid}. Queue watcher exiting.')
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log('Interrupted.')
        raise SystemExit(130)
