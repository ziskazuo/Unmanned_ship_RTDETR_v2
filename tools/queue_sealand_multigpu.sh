#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="/data1/zuokun/vene/Unmanned_ship_RTDETR"
PYTHON_BIN="${ENV_DIR}/bin/python"
QUEUE_SCRIPT="${ROOT_DIR}/tools/wait_and_launch_sealand_multigpu.py"
QUEUE_LOG_DIR="${ROOT_DIR}/output/gpu_queue/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG_FILE="${QUEUE_LOG_DIR}/queue_${TIMESTAMP}.log"

mkdir -p "${QUEUE_LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

{
  echo "[$(date '+%F %T')] python=${PYTHON_BIN}"
  echo "[$(date '+%F %T')] queue_script=${QUEUE_SCRIPT}"
  echo "[$(date '+%F %T')] queue_log=${QUEUE_LOG_FILE}"
  echo "[$(date '+%F %T')] args=$*"
} | tee -a "${QUEUE_LOG_FILE}"

PID="$({
  cd "${ROOT_DIR}" && \
  setsid bash -lc "exec \"${PYTHON_BIN}\" \"${QUEUE_SCRIPT}\" \"\$@\" >> \"${QUEUE_LOG_FILE}\" 2>&1" _ "$@" \
    < /dev/null > /dev/null 2>&1 & echo \$!
})"

echo "${PID}" | tee -a "${QUEUE_LOG_FILE}"
echo "Queue watcher started: PID=${PID}" | tee -a "${QUEUE_LOG_FILE}"
echo "Queue log: ${QUEUE_LOG_FILE}"
