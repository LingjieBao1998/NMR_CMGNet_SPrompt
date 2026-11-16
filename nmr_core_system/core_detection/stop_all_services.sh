#!/bin/bash

# 停止所有服务脚本
# 用途: 停止Django、Redis、Celery和SSH反向代理
# 作者: Claude Code
# 日期: 2026-01-08

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 项目根目录
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# PID文件目录
PID_DIR="$PROJECT_DIR/logs"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "    NMR Core Detection - 停止所有服务"
echo "=========================================="

# 1. 停止SSH反向代理
echo ""
echo "🛑 停止SSH反向代理..."
SSH_PID_FILE="$SCRIPT_DIR/ssh_tunnel.pid"
if [ -f "$SSH_PID_FILE" ]; then
    SSH_PID=$(cat "$SSH_PID_FILE")
    if ps -p "$SSH_PID" > /dev/null 2>&1; then
        kill "$SSH_PID"
        sleep 2
        if ps -p "$SSH_PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  强制停止SSH隧道${NC}"
            kill -9 "$SSH_PID"
        fi
        echo -e "${GREEN}✅ SSH隧道已停止${NC}"
    else
        echo -e "${YELLOW}⚠️  SSH隧道进程不存在${NC}"
    fi
    rm -f "$SSH_PID_FILE"
else
    echo -e "${YELLOW}⚠️  SSH隧道PID文件不存在${NC}"
fi

# 停止所有autossh进程（清理遗留进程）
pkill -f "autossh.*118.31.69.203" 2>/dev/null

# 2. 停止Django服务
echo ""
echo "🛑 停止Django服务..."
DJANGO_PIDS=$(pgrep -f "runserver 0.0.0.0:8888")
if [ -n "$DJANGO_PIDS" ]; then
    echo "找到Django进程: $DJANGO_PIDS"
    pkill -f "runserver 0.0.0.0:8888"
    sleep 2
    
    # 强制停止遗留进程
    REMAINING=$(pgrep -f "runserver 0.0.0.0:8888")
    if [ -n "$REMAINING" ]; then
        echo -e "${YELLOW}⚠️  强制停止Django进程${NC}"
        pkill -9 -f "runserver 0.0.0.0:8888"
    fi
    echo -e "${GREEN}✅ Django服务已停止${NC}"
else
    echo -e "${YELLOW}⚠️  Django服务未运行${NC}"
fi

# 3. 停止Celery Worker
echo ""
echo "🛑 停止Celery Worker服务..."
CELERY_PID_FILE="$PID_DIR/celery.pid"
if [ -f "$CELERY_PID_FILE" ]; then
    CELERY_PID=$(cat "$CELERY_PID_FILE")
    if ps -p "$CELERY_PID" > /dev/null 2>&1; then
        kill "$CELERY_PID"
        sleep 2
        if ps -p "$CELERY_PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  强制停止Celery Worker${NC}"
            kill -9 "$CELERY_PID"
        fi
        echo -e "${GREEN}✅ Celery Worker已停止${NC}"
    else
        echo -e "${YELLOW}⚠️  Celery Worker进程不存在${NC}"
    fi
    rm -f "$CELERY_PID_FILE"
fi

# 停止所有Celery进程（清理遗留进程）
pkill -f "celery.*worker" 2>/dev/null
sleep 1

# 4. 停止Redis服务
echo ""
echo "🛑 停止Redis服务..."
REDIS_PID_FILE="$PID_DIR/redis.pid"
if [ -f "$REDIS_PID_FILE" ]; then
    REDIS_PID=$(cat "$REDIS_PID_FILE")
    if ps -p "$REDIS_PID" > /dev/null 2>&1; then
        kill "$REDIS_PID"
        sleep 2
        if ps -p "$REDIS_PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  强制停止Redis${NC}"
            kill -9 "$REDIS_PID"
        fi
        echo -e "${GREEN}✅ Redis服务已停止${NC}"
    else
        echo -e "${YELLOW}⚠️  Redis进程不存在${NC}"
    fi
    rm -f "$REDIS_PID_FILE"
else
    # 如果没有PID文件，尝试查找并停止Redis
    if pgrep redis-server > /dev/null; then
        echo "找到Redis进程，停止中..."
        pkill redis-server
        sleep 2
        echo -e "${GREEN}✅ Redis服务已停止${NC}"
    else
        echo -e "${YELLOW}⚠️  Redis服务未运行${NC}"
    fi
fi

echo ""
echo "=========================================="
echo "    所有服务已停止"
echo "=========================================="
echo ""
echo "💡 提示:"
echo "  • 重新启动服务: $SCRIPT_DIR/start_all_services.sh"
echo "  • 检查进程: ps aux | grep -E '(django|celery|redis|autossh)'"
echo "=========================================="