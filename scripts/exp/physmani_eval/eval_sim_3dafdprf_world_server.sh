#!/bin/bash
# 启动策略服务器 - 运行在 conda_env_b (Python 3.10)

set -e

ENV_NAME="lerobot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
THIRD_PARTY_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"
echo $PROJECT_ROOT
# 初始化 conda
eval "$(conda shell.bash hook)"
cd "${SCRIPT_DIR}"

# 激活环境
# 重要：将 conda activate 的输出完全重定向到 /dev/null，避免其输出被误解析为 Python 参数
# 某些 conda 配置（如 CONDA_SUBDIR）可能会输出信息到标准输出
conda activate ${ENV_NAME} >/dev/null 2>&1

export PYTHONPATH="${PROJECT_ROOT}:${THIRD_PARTY_DIR}/RLBench:${THIRD_PARTY_DIR}/PyRep:${PYTHONPATH:-}"

# 解析命令行参数
GPUID="${1:-0}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8766}"
export CUDA_VISIBLE_DEVICES=$GPUID
# 调试：打印实际传递的参数（重定向到 stderr，避免干扰）
echo "激活环境 ${ENV_NAME} 并启动策略服务器..."
echo "服务器地址: ${HOST}:${PORT}"
echo "调试信息: HOST=${HOST}, PORT=${PORT}" >&2

# 运行 Python 服务器
# 使用 exec 避免额外的 shell 层，确保参数正确传递
exec python -m online_evaluation_rlbench.world_server --host "${HOST}" --port "${PORT}"
