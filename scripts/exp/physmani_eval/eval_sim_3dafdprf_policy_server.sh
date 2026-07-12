#!/bin/bash
# 启动策略服务器 - 运行在 conda_env_b (Python 3.10)

set -e

ENV_NAME="3d_diffuser_actor"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
THIRD_PARTY_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"
echo $PROJECT_ROOT
# 解析命令行参数
GPUID="${1:-0}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8765}"

echo "激活环境 ${ENV_NAME} 并启动策略服务器..."
echo "服务器地址: ${HOST}:${PORT}"

# 初始化 conda
eval "$(conda shell.bash hook)"
cd "${SCRIPT_DIR}"

# 激活环境并运行
conda activate ${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${THIRD_PARTY_DIR}/RLBench:${THIRD_PARTY_DIR}/PyRep:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=$GPUID
python -m online_evaluation_rlbench.policy_server --host "${HOST}" --port "${PORT}"
