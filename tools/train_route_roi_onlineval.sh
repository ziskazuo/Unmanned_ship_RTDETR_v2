#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/data1/zuokun/vene/Unmanned_ship_RTDETR/bin/python"
NCCL_LIB_DIR="/data1/zuokun/vene/nccl/2.20.3-cuda11.0/nccl_2.20.3-1+cuda11.0_x86_64/lib"
DEFAULT_CONFIG="configs/rtdetr/sealand_radardetr_r50vd_route_roi_p2_16e_bs2x4_fp32_test400_onlineval_4km_super4_960.yml"
DEFAULT_GPUS="0,1,2,3,4,5,6,7"

CONFIG="$DEFAULT_CONFIG"
GPUS="$DEFAULT_GPUS"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"

export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export FLAGS_allocator_strategy="auto_growth"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
GPU_COUNT="${#GPU_ARR[@]}"

if [[ "$GPU_COUNT" -le 1 ]]; then
  CUDA_VISIBLE_DEVICES="$GPUS" \
    "$PYTHON_BIN" tools/train.py \
    -c "$CONFIG" \
    --eval \
    "${EXTRA_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="$GPUS" \
    "$PYTHON_BIN" -m paddle.distributed.launch \
    --gpus "$GPUS" \
    tools/train.py \
    -c "$CONFIG" \
    --eval \
    "${EXTRA_ARGS[@]}"
fi
