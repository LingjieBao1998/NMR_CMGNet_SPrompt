FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniforge
COPY Miniforge3-Linux-x86_64.sh ./miniforge.sh
RUN bash miniforge.sh -b -p /opt/conda && \
    rm miniforge.sh

# 设置 conda 环境变量
ENV PATH=/opt/conda/bin:${PATH}
ENV CONDA_DEFAULT_ENV=base

# 创建 Python 3.10 环境
RUN conda create -n py310 python=3.10 -y && \
    conda run -n py310 pip install --upgrade pip

# 激活 py310 环境
ENV CONDA_DEFAULT_ENV=py310
ENV CONDA_PREFIX=/opt/conda/envs/py310
ENV PATH=/opt/conda/envs/py310/bin:${PATH}
ENV CONDA_PROMPT_MODIFIER="(py310) "

# 设置工作目录
WORKDIR /app

# 1. 先安装 PyTorch (CUDA 12.1)
RUN conda run -n py310 pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Flash Attention已禁用（兼容性问题），CMGNet将使用标准注意力机制

# 3. 安装其他依赖 - 添加setuptools支持pkg_resources
RUN conda run -n py310 pip install --no-cache-dir \
    setuptools==69.5.1 \
    Django==4.2.16 \
    celery==5.3.4 \
    redis==5.0.1 \
    django-redis==5.4.0 \
    gunicorn==21.2.0 \
    whitenoise==6.6.0 \
    Pillow==10.1.0 \
    transformers==4.35.2 \
    pytorch-lightning==2.1.0 \
    xgboost==2.0.3 \
    numpy==1.24.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    joblib==1.3.2 \
    cryptography==41.0.7 \
    tqdm==4.66.1

# 4. 安装 RDKit
RUN conda run -n py310 conda install -c conda-forge rdkit=2023.9.1 -y

# 基础环境镜像构建完成
# 使用方法: docker run --gpus all -it your-image-name bash

