#!/bin/bash

# SSH隧道快速启动示例
# 这是一个简化的演示脚本，展示如何启动SSH隧道

echo "=========================================="
echo "    NMR Core Detection - SSH隧道快速启动"
echo "=========================================="
echo ""
echo "🚀 正在启动SSH反向代理..."
echo ""

# 配置
REMOTE_HOST="root@118.31.69.203"
REMOTE_PORT="8888"
LOCAL_PORT="8888"
LOG_FILE="ssh_tunnel.log"

# 启动命令
echo "执行命令:"
echo "autossh -M 0 -N -R $REMOTE_PORT:127.0.0.1:$LOCAL_PORT $REMOTE_HOST"
echo ""

# 实际启动（后台运行）
nohup autossh -M 0 \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=no \
    -N \
    -R "$REMOTE_PORT:127.0.0.1:$LOCAL_PORT" \
    "$REMOTE_HOST" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ SSH隧道已启动！"
echo ""
echo "📊 进程信息:"
echo "   进程ID: $PID"
echo "   日志文件: $LOG_FILE"
echo ""
echo "🌐 访问地址:"
echo "   本地: http://127.0.0.1:8000"
echo "   远程: http://$REMOTE_HOST:$REMOTE_PORT"
echo ""
echo "📋 管理命令:"
echo "   查看日志: tail -f $LOG_FILE"
echo "   停止服务: kill $PID"
echo "   检查状态: ps -p $PID"
echo ""
echo "=========================================="