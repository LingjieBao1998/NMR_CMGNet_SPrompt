#!/bin/bash

# Docker 启动脚本 - NMR检测系统 (docker compose 方式)
# 镜像 my-nmr-service:v1.0-web = my-nmr-service:v1.0 + dnspython/py3-validate-email/python-dotenv

set -e
cd "$(dirname "$0")"

command -v docker >/dev/null 2>&1 || { echo "Error: docker not found"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Error: docker compose (v2) not found"; exit 1; }

# 清理可能存在的同名遗留容器(避免 up 冲突)
docker rm -f nmr-core-system 2>/dev/null || true

echo "=========================================="
echo "Starting NMR Core Detection System Container"
echo "=========================================="

docker compose up -d

echo "=========================================="
echo "Container started!"
echo "=========================================="
echo "To view logs:  docker compose logs -f"
echo "To access:     http://localhost:8888"
echo "To stop:       bash docker_stop.sh"
echo "=========================================="
