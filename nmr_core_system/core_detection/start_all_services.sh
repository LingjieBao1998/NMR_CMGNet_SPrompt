#!/bin/bash

# 完整服务启动脚本
# 用途: 启动Django、Redis、Celery和SSH反向代理
# 作者: Claude Code
# 日期: 2026-01-08

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 项目根目录
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PID文件目录
PID_DIR="$PROJECT_DIR/logs"
mkdir -p "$PID_DIR"

echo "=========================================="
echo "    NMR Core Detection - 完整服务启动"
echo "=========================================="

# 1. 启动Redis
echo ""
echo "📦 启动Redis服务..."
if pgrep redis-server > /dev/null; then
    echo -e "${YELLOW}⚠️  Redis已在运行${NC}"
    REDIS_PID=$(pgrep redis-server)
    echo "   进程ID: $REDIS_PID"
else
    redis-server --daemonize yes --port 6379 --logfile "$PID_DIR/redis.log" --pidfile "$PID_DIR/redis.pid"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Redis启动成功${NC}"
        sleep 1
    else
        echo -e "${RED}❌ Redis启动失败${NC}"
        exit 1
    fi
fi

# 2. 启动Celery Worker
echo ""
echo "📦 启动Celery Worker服务..."
if pgrep -f "celery.*worker" > /dev/null; then
    echo -e "${YELLOW}⚠️  Celery Worker已在运行${NC}"
    CELERY_PID=$(pgrep -f "celery.*worker" | head -1)
    echo "   主进程ID: $CELERY_PID"
else
    # 清理旧的PID文件
    rm -f "$PID_DIR/celery.pid"
    
    # 启动Celery
    nohup celery -A nmr_core_system worker --loglevel=info --logfile="$PID_DIR/celery.log" --pidfile="$PID_DIR/celery.pid" --detach > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Celery Worker启动成功${NC}"
        sleep 3
    else
        echo -e "${RED}❌ Celery Worker启动失败${NC}"
        exit 1
    fi
fi

# 3. 启动Django服务
echo ""
echo "📦 启动Django服务 (8888端口)..."
if pgrep -f "runserver 0.0.0.0:8888" > /dev/null; then
    echo -e "${YELLOW}⚠️  Django服务已在运行${NC}"
    DJANGO_PID=$(pgrep -f "runserver 0.0.0.0:8888" | head -1)
    echo "   进程ID: $DJANGO_PID"
else
    # 启动Django
    nohup python manage.py runserver 0.0.0.0:8888 > "$PID_DIR/django.log" 2>&1 &
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Django服务启动成功${NC}"
        sleep 3
        
        # 检查服务是否正常响应
        if curl -s http://127.0.0.1:8888 > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Django服务响应正常${NC}"
        else
            echo -e "${YELLOW}⚠️  Django服务启动但未响应，请检查日志${NC}"
        fi
    else
        echo -e "${RED}❌ Django服务启动失败${NC}"
        exit 1
    fi
fi

# 4. 启动SSH反向代理
echo ""
echo "📦 启动SSH反向代理..."

# 配置参数
REMOTE_HOST="root@118.31.69.203"
REMOTE_PORT="8888"
LOCAL_PORT="8888"
LOCAL_SERVICE="http://127.0.0.1:8888"

SSH_PID_FILE="$SCRIPT_DIR/ssh_tunnel.pid"
SSH_LOG_FILE="$SCRIPT_DIR/ssh_tunnel.log"

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

# 检查是否已有SSH隧道进程运行
if [ -f "$SSH_PID_FILE" ]; then
    OLD_PID=$(cat "$SSH_PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  发现已有SSH隧道进程运行 (PID: $OLD_PID)${NC}"
        echo "停止旧进程..."
        kill "$OLD_PID"
        sleep 2
    fi
fi

# 检查本地Django服务是否运行
if ! curl -s "$LOCAL_SERVICE" > /dev/null 2>&1; then
    echo -e "${RED}❌ 本地服务 $LOCAL_SERVICE 未运行${NC}"
    echo "请检查Django服务启动状态"
    exit 1
fi

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
    > "$SSH_LOG_FILE" 2>&1 &

SSH_PID=$!
echo $SSH_PID > "$SSH_PID_FILE"

# 等待几秒检查进程是否成功启动
sleep 3

if ps -p "$SSH_PID" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ SSH反向代理启动成功！${NC}"
else
    echo -e "${RED}❌ SSH反向代理启动失败${NC}"
    echo "请检查日志: $SSH_LOG_FILE"
    rm -f "$SSH_PID_FILE"
    exit 1
fi

# 最终状态检查
echo ""
echo "=========================================="
echo "    服务启动完成！"
echo "=========================================="
echo -e "${GREEN}✅ 所有服务启动成功${NC}"
echo ""
echo "📊 服务状态:"
echo "  • Django:     http://127.0.0.1:8888"
echo "  • 远程访问:   http://$REMOTE_HOST:$REMOTE_PORT"
echo "  • Admin:      http://127.0.0.1:8888/admin"
echo ""
echo "📁 日志文件:"
echo "  • Django:     $PID_DIR/django.log"
echo "  • Celery:     $PID_DIR/celery.log"
echo "  • Redis:      $PID_DIR/redis.log"
echo "  • SSH:        $SSH_LOG_FILE"
echo ""
echo "🔧 管理命令:"
echo "  • 查看服务状态: $SCRIPT_DIR/check_services.sh"
echo "  • 停止所有服务: $SCRIPT_DIR/stop_all_services.sh"
echo "  • 查看Django日志: tail -f $PID_DIR/django.log"
echo "  • 查看Celery日志: tail -f $PID_DIR/celery.log"
echo "  • 查看SSH日志: tail -f $SSH_LOG_FILE"
echo "=========================================="