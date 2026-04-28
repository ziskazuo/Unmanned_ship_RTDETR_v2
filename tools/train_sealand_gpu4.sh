#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="/data1/zuokun/vene/Unmanned_ship_RTDETR"
PYTHON_BIN="${ENV_DIR}/bin/python"
CONFIG_PATH="configs/rtdetr/sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721.yml"
GPUS="4"
USE_AMP=0
ALLOCATOR_STRATEGY="auto_growth"
DEFAULT_CONFIG_NAME="$(basename "${CONFIG_PATH}" .yml)"
LOG_ROOT="${ROOT_DIR}/output/${DEFAULT_CONFIG_NAME}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra train.py args]

Options:
  --gpus IDS           GPU ids, e.g. 4 or 4,5,6,7
  --config PATH        Training config path
  --python PATH        Python interpreter path
  --env-dir PATH       Environment directory
  --amp                Force mixed precision
  --no-amp             Disable mixed precision
  --log-file PATH      Explicit log file path
  -h, --help           Show this message

Examples:
  $(basename "$0")
  $(basename "$0") --gpus 4,5,6,7
  $(basename "$0") --gpus 4 -- --eval
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --env-dir)
      ENV_DIR="$2"
      PYTHON_BIN="${ENV_DIR}/bin/python"
      shift 2
      ;;
    --amp)
      USE_AMP=1
      shift
      ;;
    --no-amp)
      USE_AMP=0
      shift
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

CONFIG_NAME="$(basename "${CONFIG_PATH}" .yml)"
if [[ "${LOG_ROOT}" == "${ROOT_DIR}/output/${DEFAULT_CONFIG_NAME}/logs" ]]; then
  LOG_ROOT="${ROOT_DIR}/output/${CONFIG_NAME}/logs"
fi
mkdir -p "${LOG_ROOT}"
if [[ -z "${LOG_FILE}" ]]; then
  SAFE_GPUS="${GPUS//,/ _}"
  SAFE_GPUS="${SAFE_GPUS// /}"
  LOG_FILE="${LOG_ROOT}/train_gpus_${SAFE_GPUS}_${TIMESTAMP}.log"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

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
export FLAGS_allocator_strategy="${ALLOCATOR_STRATEGY}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

TRAIN_ARGS=(
  "${PYTHON_BIN}"
  "tools/train.py"
  -c "${CONFIG_PATH}"
)

if [[ "${USE_AMP}" == "1" ]]; then
  TRAIN_ARGS+=(--amp)
fi

TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

{
  echo "[$(date '+%F %T')] root=${ROOT_DIR}"
  echo "[$(date '+%F %T')] python=${PYTHON_BIN}"
  echo "[$(date '+%F %T')] config=${CONFIG_PATH}"
  echo "[$(date '+%F %T')] gpus=${GPUS}"
  echo "[$(date '+%F %T')] amp=${USE_AMP}"
  echo "[$(date '+%F %T')] log_file=${LOG_FILE}"
} | tee -a "${LOG_FILE}"

cd "${ROOT_DIR}"

if [[ "${GPUS}" == *,* ]]; then
  DIST_LOG_DIR="${LOG_ROOT}/dist_${TIMESTAMP}"
  mkdir -p "${DIST_LOG_DIR}"
  CMD=(
    "${PYTHON_BIN}"
    -m paddle.distributed.launch
    --gpus "${GPUS}"
    --log_dir "${DIST_LOG_DIR}"
    tools/train.py
    -c "${CONFIG_PATH}"
  )
  if [[ "${USE_AMP}" == "1" ]]; then
    CMD+=(--amp)
  fi
  CMD+=("${EXTRA_ARGS[@]}")
else
  CMD=("${TRAIN_ARGS[@]}")
fi

printf '[%s] command=' "$(date '+%F %T')" | tee -a "${LOG_FILE}"
printf ' %q' "${CMD[@]}" | tee -a "${LOG_FILE}"
printf '\n' | tee -a "${LOG_FILE}"

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
