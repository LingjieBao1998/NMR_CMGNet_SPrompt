#!/bin/bash
# install_dependencies.sh

echo "=== 开始安装所有依赖包 ==="

# 1. 安装PyTorch和相关库
echo "安装PyTorch 2.7.0 + CUDA 12.6..."
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# 2. 安装RDKit
echo "安装RDKit..."
pip install rdkit

# 3. 安装PyTorch Lightning
echo "安装PyTorch Lightning 1.8.6..."
pip install pytorch_lightning==1.8.6

# 4. 安装Transformers
echo "安装Transformers..."
pip install transformers

# 5. 安装Flash Attention（需要先下载whl文件）
echo "安装Flash Attention..."
# 请确保flash_attn-2.8.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl文件在当前目录
if [ -f "flash_attn-2.8.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" ]; then
    pip install flash_attn-2.8.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
else
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
    pip install flash_attn-2.8.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
    # echo "警告: flash_attn wheel文件不存在，尝试从PyPI安装..."
    # pip install flash-attn
fi

# 6. 安装其他依赖
echo "安装Pandas..."
pip install pandas

echo "安装Pystow..."
pip install pystow

echo "安装lmdb..."
pip install lmdb

echo "=== 所有依赖安装完成 ==="