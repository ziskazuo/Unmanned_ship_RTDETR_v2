#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/data1/zuokun/vene/Unmanned_ship_RTDETR/bin/python"
GPUS="0"
CONFIG_PATH=""
OUTPUT_EVAL=""
WEIGHTS=""
USE_AMP=0
SAVE_PRED_ONLY=1
CHECK_CUDNN=1
MIN_CUDNN=8400
DRY_RUN=0
OPT_ARGS=()
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") --config PATH [options] [-- extra eval.py args]

Options:
  --config PATH         Eval config path (required)
  --weights PATH        Override weights via -o weights=...
  --gpus IDS            CUDA_VISIBLE_DEVICES, e.g. 3 or 3,4 (default: 0)
  --output-eval DIR     Output dir passed to --output_eval
  --amp                 Enable AMP eval
  --no-amp              Disable AMP eval (default)
  --save-pred-only      Enable --save_prediction_only (default)
  --no-save-pred-only   Disable --save_prediction_only
  --opt KEY=VALUE       Additional -o options (repeatable)
  --no-check-cudnn      Skip printing runtime libcudnn version
  --min-cudnn N         Minimal required cuDNN version (default: 8400)
  --dry-run             Print command only, do not execute
  -h, --help            Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --weights)
      WEIGHTS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --output-eval)
      OUTPUT_EVAL="$2"
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
    --save-pred-only)
      SAVE_PRED_ONLY=1
      shift
      ;;
    --no-save-pred-only)
      SAVE_PRED_ONLY=0
      shift
      ;;
    --opt)
      OPT_ARGS+=("$2")
      shift 2
      ;;
    --no-check-cudnn)
      CHECK_CUDNN=0
      shift
      ;;
    --min-cudnn)
      MIN_CUDNN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "--config is required" >&2
  usage
  exit 1
fi

cd "${ROOT_DIR}"
source "${ROOT_DIR}/tools/paddle_runtime_env.sh"
setup_paddle_runtime_env
export CUDA_VISIBLE_DEVICES="${GPUS}"

if [[ "${CHECK_CUDNN}" == "1" ]]; then
  show_runtime_cudnn_version "${PYTHON_BIN}"
  RUNTIME_CUDNN="$(get_runtime_cudnn_version "${PYTHON_BIN}")"
  if [[ "${RUNTIME_CUDNN}" -lt "${MIN_CUDNN}" ]]; then
    echo "[eval_with_runtime_env] ERROR: runtime cuDNN=${RUNTIME_CUDNN}, expected >= ${MIN_CUDNN}" >&2
    echo "[eval_with_runtime_env] Check LD_LIBRARY_PATH and restart with the shared runtime env." >&2
    exit 2
  fi
fi

CMD=(
  "${PYTHON_BIN}"
  "tools/eval.py"
  -c "${CONFIG_PATH}"
)

if [[ "${USE_AMP}" == "1" ]]; then
  CMD+=(--amp)
fi

if [[ "${SAVE_PRED_ONLY}" == "1" ]]; then
  CMD+=(--save_prediction_only)
fi

if [[ -n "${OUTPUT_EVAL}" ]]; then
  mkdir -p "${OUTPUT_EVAL}"
  CMD+=(--output_eval "${OUTPUT_EVAL}")
fi

if [[ -n "${WEIGHTS}" ]]; then
  OPT_ARGS+=("weights=${WEIGHTS}")
fi

if [[ ${#OPT_ARGS[@]} -gt 0 ]]; then
  CMD+=(-o)
  CMD+=("${OPT_ARGS[@]}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf '[eval_with_runtime_env] command='
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"
