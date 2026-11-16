# NMR Core Detection - 服务管理指南

## 🚀 快速启动

### 启动所有服务
```bash
cd /home/lingjiebao/project/NMR_Core_Mol_Detection/nmr_core_system/core_detection
./start_all_services.sh
```

这个脚本会自动启动：
- ✅ Redis服务 (6379端口)
- ✅ Celery Worker (33个worker进程)
- ✅ Django服务 (8888端口)
- ✅ SSH反向代理 (autossh连接到118.31.69.203:8888)

### 检查服务状态
```bash
./check_services.sh
```

### 停止所有服务
```bash
./stop_all_services.sh
```

## 📊 服务详情

### Django服务
- **端口**: 8888
- **本地访问**: http://127.0.0.1:8888
- **远程访问**: http://118.31.69.203:8888
- **Admin**: http://127.0.0.1:8888/admin
- **日志**: `../logs/django.log`

### Celery Worker
- **Worker数量**: 33个进程
- **日志**: `../logs/celery.log`
- **功能**: 处理NMR核心分子检测任务

### Redis服务
- **端口**: 6379
- **日志**: `../logs/redis.log`
- **功能**: Celery任务队列

### SSH反向代理
- **工具**: autossh
- **远程服务器**: root@118.31.69.203:8888
- **本地端口**: 8888
- **PID文件**: `ssh_tunnel.pid`
- **日志**: `ssh_tunnel.log`

## 🔧 日志查看

### 实时查看Django日志
```bash
tail -f ../logs/django.log
```

### 实时查看Celery日志
```bash
tail -f ../logs/celery.log
```

### 实时查看SSH隧道日志
```bash
tail -f ssh_tunnel.log
```

### 查看最近的错误日志
```bash
grep -i error ../logs/django.log | tail -10
grep -i error ../logs/celery.log | tail -10
```

## 🎯 任务管理

### 检查任务状态
```bash
cd /home/lingjiebao/project/NMR_Core_Mol_Detection/nmr_core_system
python check_tasks.py
```

### 手动触发任务处理
通过Web界面访问: http://118.31.69.203:8888/submission/

## 🛠️ 故障排除

### Django无法启动
```bash
# 检查端口占用
netstat -tlnp | grep 8888

# 如果端口被占用，停止占用进程
sudo fuser -k 8888/tcp
```

### Celery无法启动
```bash
# 清理旧的PID文件
rm -f ../logs/celery.pid

# 重新启动
../start_celery.sh
```

### SSH隧道无法连接
```bash
# 测试SSH连接
ssh root@118.31.69.203

# 检查autossh是否安装
which autossh

# 如果未安装，安装autossh
sudo apt-get install autossh
```

### Redis连接失败
```bash
# 检查Redis状态
redis-cli ping

# 重启Redis
sudo systemctl restart redis-server
```

## 📈 性能监控

### 查看系统资源
```bash
# CPU和内存使用
top

# 磁盘使用
df -h

# 进程监控
htop
```

### Celery任务队列
```bash
# 查看活动任务
celery -A nmr_core_system inspect active

# 查看统计信息
celery -A nmr_core_system inspect stats
```

## 🔐 安全注意事项

1. **SSH密钥**: 确保SSH密钥安全，不要泄露
2. **防火墙**: 确保远程服务器防火墙允许8888端口
3. **日志管理**: 定期清理日志文件，避免磁盘空间不足
4. **进程监控**: 定期检查服务运行状态

## 📱 联系方式

如有问题，请检查：
1. 日志文件位置
2. 服务状态检查脚本输出
3. 系统资源使用情况

---
**创建日期**: 2026-01-08
**版本**: 1.0
**维护**: Claude Code