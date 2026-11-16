#!/bin/bash

# 反向代理启动脚本
# 用途: 将本地Django服务反向代理到远程服务器
# 作者: Claude Code
# 日期: 2026-01-08

# 配置参数
REMOTE_HOST="root@118.31.69.203"
REMOTE_PORT="8888"
LOCAL_PORT="8888"
LOCAL_SERVICE="http://127.0.0.1:8888"

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# PID文件
PID_FILE="$SCRIPT_DIR/ssh_tunnel.pid"
LOG_FILE="$SCRIPT_DIR/ssh_tunnel.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "    NMR Core Detection - SSH反向代理启动"
echo "=========================================="
echo "远程主机: $REMOTE_HOST"
echo "远程端口: $REMOTE_PORT"
echo "本地端口: $LOCAL_PORT"
echo "本地服务: $LOCAL_SERVICE"
echo "=========================================="

# 检查autossh是否安装
if ! command -v autossh &> /dev/null; then
    echo -e "${RED}❌ autossh 未安装，正在安装...${NC}"
    sudo apt-get update && sudo apt-get install -y autossh
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ autossh 安装失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ autossh 安装成功${NC}"
fi

# 检查是否已有进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  发现已有SSH隧道进程运行 (PID: $OLD_PID)${NC}"
        read -p "是否停止旧进程并启动新进程? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "停止旧进程..."
            kill "$OLD_PID"
            sleep 2
        else
            echo "取消操作"
            exit 0
        fi
    fi
fi

# 检查本地服务是否运行
echo "检查本地Django服务..."
if ! curl -s "$LOCAL_SERVICE" > /dev/null 2>&1; then
    echo -e "${RED}❌ 本地服务 $LOCAL_SERVICE 未运行${NC}"
    echo "请先启动Django服务"
    exit 1
fi
echo -e "${GREEN}✅ 本地服务运行正常${NC}"

# 测试SSH连接
echo "测试SSH连接..."
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$REMOTE_HOST" "echo 'SSH连接测试成功'" > /dev/null 2>&1; then
    echo -e "${RED}❌ 无法连接到远程主机 $REMOTE_HOST${NC}"
    echo "请检查:"
    echo "1. 网络连接"
    echo "2. SSH密钥配置"
    echo "3. 远程主机是否在线"
    exit 1
fi
echo -e "${GREEN}✅ SSH连接测试成功${NC}"

# 创建autossh启动命令
echo "启动autossh反向代理..."

# 使用nohup启动autossh
nohup autossh \
    -M 0 \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -N \
    -R "$REMOTE_PORT:127.0.0.1:$LOCAL_PORT" \
    "$REMOTE_HOST" \
    > "$LOG_FILE" 2>&1 &

SSH_PID=$!
echo $SSH_PID > "$PID_FILE"

# 等待几秒检查进程是否成功启动
sleep 3

if ps -p "$SSH_PID" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ SSH反向代理启动成功！${NC}"
    echo "进程ID: $SSH_PID"
    echo "日志文件: $LOG_FILE"
    echo "PID文件: $PID_FILE"
    echo ""
    echo "访问方式:"
    echo "  本地: http://127.0.0.1:8000"
    echo "  远程: http://$REMOTE_HOST:$REMOTE_PORT"
    echo ""
    echo "管理命令:"
    echo "  查看日志: tail -f $LOG_FILE"
    echo "  停止服务: kill $SSH_PID"
    echo "  查看状态: ps -p $SSH_PID"
else
    echo -e "${RED}❌ SSH反向代理启动失败${NC}"
    echo "请检查日志: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi