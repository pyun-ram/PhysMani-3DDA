#!/usr/bin/env bash
set -e

# ENV_DIR="/root/autodl-tmp/lerobot"


echo "==> [1/6] Create conda env (local prefix)"
conda env create -n lerobot -f docker/lerobot_environment.yaml

echo "==> [2/6] Activate env"
conda activate lerobot

echo "==> [3/6] Install PyTorch 2.7.1 + CUDA 12.6 (pip wheel)"
pip install --no-cache-dir \
  torch==2.7.1+cu126 \
  torchvision \
  torchaudio \
  --index-url https://download.pytorch.org/whl/cu126

echo "==> [4/6] Install CUDA compiler toolchain (nvcc only)"
conda install -y -c nvidia \
  cuda-nvcc=12.6 \
  cuda-cudart-dev=12.6 \
  --no-channel-priority

echo "==> [5/6] Sanity check"
python - << 'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
EOF

nvcc --version || echo "nvcc not found!"

echo "==> [6/6] Done. Environment ready."

# conda activate ${ENV_DIR}
# diff-gaussian-rasterization-extentions
if [ ! -d "diff-gaussian-rasterization-extentions" ]; then
            git clone https://github.com/ingra14m/diff-gaussian-rasterization-extentions
    else
                echo "diff-gaussian-rasterization-extentions 目录已存在，跳过 clone"
fi
conda install -c conda-forge glm -y
cd diff-gaussian-rasterization-extentions && git checkout 2eb32ea251d3b339dab3af8b6fd78d7dec3caf8e && pip install -e . --no-build-isolation && cd ..

# ngv_robot
if [ ! -d "ngv_robot" ]; then
            git clone git@github.com:pyun-ram/ngv_robot.git
    else
                echo "ngv_robot 目录已存在，跳过 clone"
fi
cd ngv_robot && git checkout acc1e6a444fc7b13e2db42087976359ab9cfdd35 && cd ..

# RLBench
if [ ! -d "RLBench" ]; then
            git clone git@github.com:pyun-ram/RLBench.git
    else
                echo "RLBench 目录已存在，跳过 clone"
fi
cd RLBench && git checkout 50830e9cabdb968fbf0ac0d422665092c7b99f76 && pip install -e . --no-build-isolation && cd ..
cd RLBench && git checkout 50830e9cabdb968fbf0ac0d422665092c7b99f76 && pip install -r requirements.txt && cd ..

# Deformable-3D-Gaussians
if [ ! -d "Deformable-3D-Gaussians" ]; then
            git clone https://github.com/ingra14m/Deformable-3D-Gaussians.git --recursive
    else
                echo "Deformable-3D-Gaussians 目录已存在，跳过 clone"
fi
cd Deformable-3D-Gaussians 
# 补丁：在 submodules/simple-knn/simple_knn.cu 的头文件区加入 #include <cfloat>
if ! grep -q '#include <cfloat>' submodules/simple-knn/simple_knn.cu; then
    sed -i '1i#include <cfloat>' submodules/simple-knn/simple_knn.cu
    echo "已在 submodules/simple-knn/simple_knn.cu 文件头部添加 #include <cfloat>"
else
    echo "submodules/simple-knn/simple_knn.cu 已包含 #include <cfloat>，跳过修改"
fi
pip install submodules/simple-knn --no-build-isolation && cd ..


# PyRep
if [ ! -d "PyRep" ]; then
            git clone git@github.com:pyun-ram/PyRep.git
    else
                echo "PyRep 目录已存在，跳过 clone"
fi
cd PyRep && git checkout 7b7f6328a22c35262b4e93446563a0e68a31b870 && pip install -e . --no-build-isolation && cd ..

# gsplat
if [ ! -d "gsplat" ]; then
            git clone https://github.com/nerfstudio-project/gsplat
    else
                echo "gsplat 目录已存在，跳过 clone"
fi
pip install git+https://github.com/nerfstudio-project/gsplat.git

# pytorch3d
if [ ! -d "pytorch3d" ]; then
            git clone https://github.com/facebookresearch/pytorch3d.git
    else
                echo "pytorch3d 目录已存在，跳过 clone"
fi
conda install -c nvidia libcusparse-dev libcublas-dev libcusolver-dev libcurand-dev
cd pytorch3d && git checkout f5f6b78e70e0a1b70f3be9a09b5b001e9b3a7a03 && pip install -e . --no-build-isolation && cd ..

pip install fire open3d websockets msgpack lion_pytorch
cd 3d_diffuser_actor && pip install submodules/fps --no-build-isolation