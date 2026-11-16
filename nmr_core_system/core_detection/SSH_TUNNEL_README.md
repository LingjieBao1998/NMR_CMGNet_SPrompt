# SSH反向代理使用说明

## 📋 概述
这组脚本用于建立从本地Django服务到远程服务器的SSH反向代理隧道。

## 🔧 配置参数
- **远程主机**: `root@118.31.69.203`
- **远程端口**: `8888`
- **本地端口**: `8888`
- **本地服务**: `http://127.0.0.1:8000`

## 🚀 使用方法

### 1. 启动SSH隧道
```bash
cd /home/lingjiebao/project/NMR_Core_Mol_Detection/nmr_core_system/core_detection
./start_ssh_tunnel.sh
```

**脚本功能**:
- 自动检查并安装autossh
- 检查本地Django服务是否运行
- 测试SSH连接
- 启动nohup后台进程
- 生成PID文件和日志文件

### 2. 检查隧道状态
```bash
./check_ssh_tunnel.sh
```

**显示信息**:
- 进程运行状态
- 网络连接状态
- 最近日志输出

### 3. 停止SSH隧道
```bash
./stop_ssh_tunnel.sh
```

**功能**:
- 优雅停止SSH隧道进程
- 清理PID文件

## 📁 生成的文件

### 运行时文件
- `ssh_tunnel.pid` - 进程ID文件
- `ssh_tunnel.log` - 运行日志文件

### 日志查看
```bash
# 查看实时日志
tail -f ssh_tunnel.log

# 查看最近日志
tail -20 ssh_tunnel.log
```

## 🌐 访问方式

隧道启动后，可以通过以下方式访问您的Django应用：

### 本地访问
```
http://127.0.0.1:8000
http://localhost:8000
```

### 远程访问（通过隧道）
```
http://118.31.69.203:8888
```

## 🔍 故障排除

### 问题1: SSH连接失败
**检查项**:
1. 网络连接是否正常
2. SSH密钥是否正确配置
3. 远程主机是否在线

**解决方法**:
```bash
# 测试SSH连接
ssh root@118.31.69.203

# 检查SSH密钥
ls -la ~/.ssh/id_rsa*
```

### 问题2: 本地服务未运行
**检查项**:
1. Gunicorn是否启动
2. 端口8000是否被占用

**解决方法**:
```bash
# 检查Gunicorn进程
ps aux | grep gunicorn

# 检查端口占用
netstat -tuln | grep 8000
```

### 问题3: 隧道自动断开
**特点**: autossh会自动重连

**监控重连**:
```bash
# 查看日志中的重连记录
grep "autossh" ssh_tunnel.log
```

## 🔒 安全注意事项

1. **SSH密钥**: 确保SSH密钥安全，不要泄露
2. **防火墙**: 确保远程服务器防火墙允许8888端口
3. **权限**: 脚本文件权限设置为755
4. **日志**: 定期清理日志文件，避免磁盘空间不足

## 📊 性能优化

### autossh参数说明
```bash
-M 0                          # 不使用监控端口
ServerAliveInterval 30        # 30秒发送心跳
ServerAliveCountMax 3         # 3次失败后断开
ExitOnForwardFailure yes      # 转发失败时退出
```

### 带宽优化
如需优化带宽使用，可以在SSH命令中添加压缩参数：
```bash
autossh ... -C -c blowfish ...
```

## 🛠️ 高级用法

### 自定义端口
修改脚本中的端口参数：
```bash
REMOTE_PORT="自定义远程端口"
LOCAL_PORT="自定义本地端口"
```

### 多隧道
复制脚本并修改参数，可以运行多个隧道。

## 📞 支持

如遇问题，请检查：
1. 日志文件：`ssh_tunnel.log`
2. 系统日志：`/var/log/auth.log`
3. 进程状态：`ps aux | grep ssh`

## 🔄 自动重启

如需开机自动启动，可以添加到crontab：
```bash
@reboot /path/to/start_ssh_tunnel.sh
```

或创建systemd服务（需要额外配置）。

---

**创建日期**: 2026-01-08
**版本**: 1.0
**维护**: Claude Code