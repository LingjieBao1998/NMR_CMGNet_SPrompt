# Docker 容器启动说明（当前部署方式）

> 主 README.md 中也有同样内容。必须使用新版 `docker compose`(v2)；旧版 `docker-compose` 1.29 不兼容本配置。

## 镜像

- **镜像名**: `my-nmr-service:v1.0-web`
- 构成: `my-nmr-service:v1.0`（torch cu118 + transformers + xgboost + rdkit + Django 4.2.16 + celery + redis-server）+ dnspython / py3-validate-email / python-dotenv + `setuptools<81`（pytorch_lightning 需要 pkg_resources）
- 构建（在 nmr_core_system/ 目录下，依赖变化时执行一次）:

```bash
docker build -f Dockerfile.web -t my-nmr-service:v1.0-web .
```

## 启动 / 停止 / 管理

```bash
# 启动
bash docker_start.sh          # 或 docker compose up -d

# 状态 / 日志
./manage_docker.sh status
./manage_docker.sh logs       # 或 docker compose logs -f

# 停止 / 重启
bash docker_stop.sh           # 或 docker compose stop
./manage_docker.sh restart

# 进入容器
./manage_docker.sh shell

# 移除容器
./manage_docker.sh down
```

访问地址: `http://localhost:8888`

## 配置说明（docker-compose.yml）

| 项 | 值 |
|---|---|
| 端口 | `8888:8888` |
| GPU | `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=1`（宿主 GPU1 = RTX 4090，容器内为 cuda:0；镜像 torch cu118 不支持 RTX 5090） |
| 代码挂载 | `.:/app/nmr_core_system`（工作目录；db.sqlite3 / media / logs / result 随项目持久化） |
| 结果挂载 | `../result:/app/result`（ensemble 预测的 result.json 落点） |
| 模型挂载(只读) | `../models:/app/models`、`../checkpoints:/app/checkpoints` |
| 环境变量 | `.env.docker`（邮件配置、SITE_URL） |
| 重启策略 | `restart: unless-stopped` |

容器启动命令自动完成: 清理残留 pid → `migrate` → 容器内 redis-server → celery worker（GPU）→ `runserver 0.0.0.0:8888`。

## 验证启动

```bash
docker ps | grep nmr-core-system
docker logs -f nmr-core-system
curl -s -o /dev/null -w '%{http_code}' http://localhost:8888/   # 302(跳登录页)为正常
docker compose exec nmr-core-system redis-cli ping              # PONG
docker compose exec nmr-core-system celery -A nmr_core_system inspect ping   # pong
docker compose exec nmr-core-system python -c "import torch;print(torch.cuda.get_device_name(0))"  # RTX 4090
```

## 故障排除

| 问题 | 处理 |
|---|---|
| 容器起不来 | `docker logs nmr-core-system` 看 traceback |
| 容器内 GPU 报 `NVML: Unknown Error` / `No CUDA GPUs` | `docker compose up -d --force-recreate` 重建（宿主机 GPU 正常时重建必好） |
| 端口 8888 被占 | `ss -tlnp \| grep 8888`，宿主 nginx/redis 已在 Docker 化时停用 |
| 模型加载失败 | `docker compose exec nmr-core-system ls /app/models /app/checkpoints` 确认挂载 |
| db 损坏 | 用 `db.sqlite3.backup_*` 恢复后 `docker compose restart` |

## 备份和恢复

```bash
cp db.sqlite3 db.sqlite3.backup_$(date +%Y%m%d)
# 恢复
cp db.sqlite3.backup_YYYYMMDD db.sqlite3
docker compose restart
```
