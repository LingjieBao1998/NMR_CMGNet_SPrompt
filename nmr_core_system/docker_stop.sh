#!/bin/bash

# Docker 停止脚本 - NMR检测系统

cd "$(dirname "$0")"

echo "Stopping NMR Core Detection System container..."
docker compose stop
echo "Stopped. (Container remains for restart; use 'docker compose down' to remove it.)"
