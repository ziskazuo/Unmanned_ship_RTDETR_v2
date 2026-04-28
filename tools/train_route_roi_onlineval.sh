#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/data1/zuokun/vene/Unmanned_ship_RTDETR/bin/python"
DEFAULT_CONFIG="configs/rtdetr/sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721.yml"
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

source "${ROOT_DIR}/tools/paddle_runtime_env.sh"
setup_paddle_runtime_env
MIN_CUDNN="${MIN_CUDNN:-8400}"
show_runtime_cudnn_version "${PYTHON_BIN}"
RUNTIME_CUDNN="$(get_runtime_cudnn_version "${PYTHON_BIN}")"
if [[ "${RUNTIME_CUDNN}" -lt "${MIN_CUDNN}" ]]; then
  echo "ERROR: runtime cuDNN=${RUNTIME_CUDNN}, expected >= ${MIN_CUDNN}" >&2
  echo "Please fix LD_LIBRARY_PATH before launching training." >&2
  exit 2
fi

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
