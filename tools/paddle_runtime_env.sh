#!/usr/bin/env bash

# Shared runtime env setup to keep train/eval on the same CUDA/cuDNN stack.

PADDLE_NCCL_LIB_DEFAULT="/data1/zuokun/vene/nccl/2.20.3-cuda11.0/nccl_2.20.3-1+cuda11.0_x86_64/lib"
PADDLE_CUDA_LIB_DEFAULT="/usr/local/cuda/lib64"

setup_paddle_runtime_env() {
  local nccl_lib="${1:-${PADDLE_NCCL_LIB_DEFAULT}}"
  local cuda_lib="${2:-${PADDLE_CUDA_LIB_DEFAULT}}"
  local current_ld="${LD_LIBRARY_PATH:-}"

  export LD_LIBRARY_PATH="${nccl_lib}:${cuda_lib}:${current_ld}"
  export FLAGS_allocator_strategy="${FLAGS_allocator_strategy:-auto_growth}"
}

show_runtime_cudnn_version() {
  local python_bin="${1:-python3}"
  "${python_bin}" - <<'PY'
import ctypes

try:
    lib = ctypes.CDLL("libcudnn.so")
    lib.cudnnGetVersion.restype = ctypes.c_size_t
    print(f"[runtime] libcudnn={lib.cudnnGetVersion()}")
except Exception as ex:
    print(f"[runtime] libcudnn=unavailable ({ex})")
PY
}

get_runtime_cudnn_version() {
  local python_bin="${1:-python3}"
  "${python_bin}" - <<'PY'
import ctypes
import sys

try:
    lib = ctypes.CDLL("libcudnn.so")
    lib.cudnnGetVersion.restype = ctypes.c_size_t
    print(int(lib.cudnnGetVersion()))
except Exception:
    print(0)
    sys.exit(0)
PY
}
