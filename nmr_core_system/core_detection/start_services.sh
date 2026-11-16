#!/bin/bash

echo "=== Starting NMR Core System Services ==="

# 启动 Redis 服务
echo "Starting Redis server..."
redis-server --daemonize yes

# 检查 Redis 是否启动成功
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis server started successfully"
else
    echo "❌ Redis server failed to start"
    exit 1
fi

# 启动 Django 开发服务器
echo "Starting Django development server..."
cd /app/nmr_core_system
python manage.py runserver 0.0.0.0:8888

