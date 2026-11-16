#!/bin/bash

# SSH反向代理停止脚本

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/ssh_tunnel.pid"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "    停止SSH反向代理"
echo "=========================================="

if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}⚠️  未找到PID文件，可能没有运行中的SSH隧道${NC}"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    echo "正在停止SSH隧道进程 (PID: $PID)..."
    kill "$PID"
    sleep 2
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  进程未响应，强制停止...${NC}"
        kill -9 "$PID"
        sleep 1
    fi
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${RED}❌ 无法停止进程${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ SSH隧道已停止${NC}"
        rm -f "$PID_FILE"
    fi
else
    echo -e "${YELLOW}⚠️  进程 $PID 未运行，清理PID文件${NC}"
    rm -f "$PID_FILE"
fi

echo "=========================================="