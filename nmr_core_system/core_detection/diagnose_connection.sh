#!/bin/bash

# SSH隧道连接诊断脚本

echo "=========================================="
echo "    SSH隧道连接诊断"
echo "=========================================="
echo ""

REMOTE_HOST="118.31.69.203"
REMOTE_PORT="8888"
LOCAL_PORT="8888"

# 1. 检查本地Django服务
echo "1. 检查本地Django服务..."
if curl -s "http://127.0.0.1:$LOCAL_PORT" > /dev/null 2>&1; then
    echo "   ✅ 本地Django服务正常 (http://127.0.0.1:$LOCAL_PORT)"
    curl -s "http://127.0.0.1:$LOCAL_PORT" | grep -o "<title>.*</title>"
else
    echo "   ❌ 本地Django服务未响应"
fi
echo ""

# 2. 检查SSH隧道状态
echo "2. 检查SSH隧道状态..."
SSH_PID=$(cat ssh_tunnel.pid 2>/dev/null)
if ps -p "$SSH_PID" > /dev/null 2>&1; then
    echo "   ✅ SSH隧道进程运行中 (PID: $SSH_PID)"
    
    # 检查网络连接
    if netstat -tuln 2>/dev/null | grep ":$LOCAL_PORT " > /dev/null; then
        echo "   ✅ 本地端口$LOCAL_PORT正在监听"
    else
        echo "   ⚠️  本地端口$LOCAL_PORT未监听"
    fi
    
    # 检查SSH连接
    if ps -p "$SSH_PID" -o args= | grep -q "ssh.*$REMOTE_HOST"; then
        echo "   ✅ SSH连接到 $REMOTE_HOST 正常"
    else
        echo "   ❌ SSH连接异常"
    fi
else
    echo "   ❌ SSH隧道进程未运行"
fi
echo ""

# 3. 从服务器端测试
echo "3. 从服务器端测试远程访问..."
echo "   正在测试 http://$REMOTE_HOST:$REMOTE_PORT ..."
if timeout 10 curl -I "http://$REMOTE_HOST:$REMOTE_PORT" > /dev/null 2>&1; then
    echo "   ✅ 远程访问正常"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://$REMOTE_HOST:$REMOTE_PORT")
    echo "   HTTP状态码: $HTTP_CODE"
else
    echo "   ❌ 远程访问失败"
fi
echo ""

# 4. 检查远程服务器端口监听
echo "4. 检查远程服务器端口监听..."
REMOTE_LISTEN=$(ssh root@$REMOTE_HOST "netstat -tuln | grep :$REMOTE_PORT" 2>/dev/null)
if [ -n "$REMOTE_LISTEN" ]; then
    echo "   ✅ 远程服务器端口$REMOTE_PORT正在监听:"
    echo "$REMOTE_LISTEN" | sed 's/^/      /'
else
    echo "   ❌ 远程服务器端口$REMOTE_PORT未监听"
fi
echo ""

# 5. 检查防火墙状态
echo "5. 检查远程服务器防火墙..."
FIREWALL_STATUS=$(ssh root@$REMOTE_HOST "firewall-cmd --list-ports 2>/dev/null | grep $REMOTE_PORT || iptables -L -n | grep $REMOTE_PORT" 2>/dev/null)
if [ -n "$FIREWALL_STATUS" ]; then
    echo "   ✅ 防火墙已配置端口$REMOTE_PORT"
else
    echo "   ⚠️  防火墙配置未明确找到端口$REMOTE_PORT"
fi
echo ""

# 6. 提供解决方案建议
echo "=========================================="
echo "📋 诊断结果和解决方案"
echo "=========================================="
echo ""

if curl -s "http://127.0.0.1:$LOCAL_PORT" > /dev/null 2>&1 && timeout 10 curl -I "http://$REMOTE_HOST:$REMOTE_PORT" > /dev/null 2>&1; then
    echo "✅ SSH隧道工作正常！"
    echo ""
    echo "🌐 访问地址:"
    echo "   本地: http://127.0.0.1:$LOCAL_PORT"
    echo "   远程: http://$REMOTE_HOST:$REMOTE_PORT"
    echo ""
    echo "💡 如果您无法从浏览器访问，请尝试:"
    echo "1. 清除浏览器缓存和Cookie"
    echo "2. 尝试无痕/隐私模式"
    echo "3. 检查您的网络连接"
    echo "4. 如果在公司网络，可能有代理限制"
    echo "5. 尝试用手机(关闭WiFi)测试"
else
    echo "❌ 发现问题，请按以下步骤排查:"
    echo ""
    echo "1. 重启SSH隧道:"
    echo "   ./stop_ssh_tunnel.sh"
    echo "   ./start_ssh_tunnel.sh"
    echo ""
    echo "2. 检查远程服务器防火墙:"
    echo "   ssh root@$REMOTE_HOST 'firewall-cmd --add-port=$REMOTE_PORT/tcp --permanent && firewall-cmd --reload'"
    echo ""
    echo "3. 查看详细日志:"
    echo "   tail -50 ssh_tunnel.log"
fi

echo "=========================================="
