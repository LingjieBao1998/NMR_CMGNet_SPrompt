# NMR Core Detection System

A Django-based web application for NMR molecular core detection with real-time task processing using Redis and Celery.

## Features

- **4 Main Sections**:
  - **Home**: Dashboard with statistics and recent tasks
  - **Submit**: Task submission interface for NMR analysis
  - **Queue**: Real-time task monitoring and management
  - **Contact**: User contact and support form

- **Task Management**:
  - Asynchronous task processing with Celery
  - Real-time progress tracking
  - Task status monitoring (Pending, Processing, Completed, Failed)
  - Detailed task logs and error reporting

- **User System** (email-based accounts):
  - Registration with email verification (48h link validity)
  - Click the verification link to activate and log in automatically
  - Login with email address (username is a non-unique display name)
  - Per-user task isolation — each account only sees its own tasks

- **User Interface**:
  - Responsive Bootstrap 5 design
  - Real-time updates using JavaScript
  - AJAX-powered task status monitoring
  - Clean, modern interface

## Docker 启动（当前部署方式）

系统当前以 Docker 方式运行（镜像 `my-nmr-service:v1.0-web` = `my-nmr-service:v1.0` + dnspython/py3-validate-email/python-dotenv + setuptools<81）。容器内自带 Redis 和 Celery，GPU 使用宿主 GPU1（RTX 4090，经 `runtime: nvidia`）。

```bash
# 构建镜像（首次或依赖变化时，需在 nmr_core_system/ 目录下）
docker build -f Dockerfile.web -t my-nmr-service:v1.0-web .

# 启动（任选其一）
bash docker_start.sh            # 推荐
docker compose up -d

# 查看状态 / 日志
./manage_docker.sh status
./manage_docker.sh logs         # 或 docker compose logs -f

# 停止 / 重启
bash docker_stop.sh             # 或 docker compose stop
./manage_docker.sh restart

# 进入容器
./manage_docker.sh shell
```

- **访问地址**: `http://localhost:8888`（公网: `http://118.31.69.203:8888`）
- **数据持久化**（宿主机挂载）: 项目目录整体挂载到 `/app/nmr_core_system`（db.sqlite3 / media / logs / result 均随项目持久化）；预测结果 result.json 额外挂载 `../result:/app/result`；模型只读挂载 `/app/models`、`/app/checkpoints`
- **排障**: 容器内 GPU 异常时 `docker compose up -d --force-recreate` 重建；验证 GPU: `docker compose exec nmr-core-system python -c "import torch;print(torch.cuda.get_device_name(0))"`
- 旧版 `DOCKER_README.md` 中的 `docker-compose`(1.29) 命令已废弃，请使用 `docker compose`(v2)

## Installation（原生方式，备用）

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and Start Redis**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   sudo service redis-server start
   
   # Or use the included script
   chmod +x start_redis.sh
   ./start_redis.sh
   ```

3. **Run Database Migrations**:
   ```bash
   python manage.py migrate
   ```

4. **Create Superuser** (if not already created):
   ```bash
   python manage.py createsuperuser
   ```

5. **Start the Application**:
   ```bash
   # Start everything (Redis, Celery, Django)
   chmod +x start_all.sh
   ./start_all.sh
   
   # Or start services individually
   chmod +x start_celery.sh
   ./start_celery.sh        # Start Celery worker
   python manage.py runserver # Start Django server
   ```

## Usage

1. **Access the Application**:
   - Main page: `http://localhost:8000`
   - Admin interface: `http://localhost:8000/admin`

2. **Submit a Task**:
   - Navigate to "Submit" page
   - Fill in task details (name, type, input data)
   - Click "Submit Task"
   - Task will be processed asynchronously

3. **Monitor Progress**:
   - Go to "Queue" page to see all tasks
   - Tasks update automatically every 5 seconds
   - Click on task name for detailed information

4. **View Results**:
   - Completed tasks show detailed results
   - Download generated files if available
   - View processing logs

## Project Structure

```
nmr_core_system/
├── core_detection/           # Main Django app
│   ├── models.py            # Database models (NMRTask, TaskLog, etc.)
│   ├── views.py             # View functions for all pages
│   ├── tasks.py             # Celery async tasks
│   ├── urls.py              # URL routing
│   ├── admin.py             # Django admin configuration
│   └── templates/           # HTML templates
│       ├── base.html        # Base template
│       ├── home.html        # Home page
│       ├── submit.html      # Task submission page
│       ├── queue.html       # Task queue page
│       ├── contact.html     # Contact page
│       └── task_detail.html # Task detail page
├── static/                  # Static files
│   ├── css/
│   │   └── style.css        # Custom styles
│   └── js/
│       └── main.js          # JavaScript functionality
├── nmr_core_system/         # Django project settings
│   ├── settings.py          # Project configuration
│   ├── urls.py              # Main URL configuration
│   └── celery.py            # Celery configuration
├── media/                   # User uploaded files
├── logs/                    # Application logs
├── manage.py                # Django management script
├── start_all.sh             # Start all services
├── stop_all.sh              # Stop all services
├── start_celery.sh          # Start Celery only
└── start_redis.sh           # Start Redis only
```

## Configuration

### Settings (settings.py)

- **Database**: SQLite (default)
- **Redis**: localhost:6379/0 (Celery broker)
- **Static Files**: Located in `static/` directory
- **Media Files**: Located in `media/` directory

### Celery Configuration

- **Broker**: Redis
- **Result Backend**: Redis
- **Task Serialization**: JSON
- **Timezone**: Asia/Shanghai

## Email Configuration (邮箱授权与域名)

系统邮件(注册验证、任务通知)通过 **网易 163 SMTP** 发送,当前发件账号为 `15857612673@163.com`。

### 1. 邮箱授权码的获取与配置

163 邮箱**不允许用登录密码做 SMTP 认证**,必须使用"客户端授权码":

1. 登录网页版邮箱 [mail.163.com](https://mail.163.com)
2. 进入 **设置 → POP3/SMTP/IMAP**
3. 开启 **SMTP 服务**(需手机短信验证)
4. 新增/查看 **客户端授权码**(一串字母,只完整显示一次,务必保存)

授权码等邮箱配置存放在项目根目录的 **`.env`** 文件(不提交到 git):

```bash
# .env
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.163.com
EMAIL_PORT=465
EMAIL_USE_SSL=True
EMAIL_USE_TLS=False
EMAIL_HOST_USER=15857612673@163.com
EMAIL_HOST_PASSWORD=<你的客户端授权码>      # 不是邮箱登录密码!
DEFAULT_FROM_EMAIL=15857612673@163.com    # 必须与 EMAIL_HOST_USER 完全一致(163不允许代发)
EMAIL_TIMEOUT=30
EMAIL_SUBJECT_PREFIX=[NMR检测系统]
```

`settings.py` 中的同名项是默认值,`.env` 里的配置优先生效。更换发件邮箱时只改 `.env` 即可。

**常见错误对照**:

| 报错 | 原因 |
|------|------|
| `535 Error: authentication failed` | 填了登录密码而非授权码 |
| `501/553 发件人拒绝` | `DEFAULT_FROM_EMAIL` 与登录账号不一致 |
| 邮件进垃圾箱 | 163 自发信正常现象,提醒用户查收垃圾邮件文件夹 |

### 2. 域名与站点地址 (SITE_URL)

`SITE_URL` 决定**注册验证邮件里链接指向的地址**,必须填写用户能实际访问到的公网地址。系统通过 SSH 反向隧道暴露到公网:

```bash
# .env
SITE_URL=http://118.31.69.203:8888/
```

当前部署形态:

```
用户浏览器 ──> http://118.31.69.203:8888 (公网服务器)
                    │ SSH 反向隧道 (autossh)
                    ▼
              http://127.0.0.1:8888 (本机 Django)
```

隧道管理(在项目机器上执行):

```bash
# 启动(需先确保 Django 已在 8888 端口运行)
bash core_detection/start_ssh_tunnel.sh

# 手动方式
autossh -M 0 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -N -R 8888:127.0.0.1:8888 root@118.31.69.203
```

**注意**: 服务器 IP 或端口变化后,必须同步更新 `.env` 中的 `SITE_URL`,否则用户收到的验证链接打不开。

### 3. 邮箱真实存在性校验

注册时系统会在前后端两层校验邮箱是否真实存在(`core_detection/email_verify.py`):

- 查询邮箱域名的 **MX 记录**,域名不存在直接拒绝
- 连接收件方邮件服务器做 **SMTP RCPT 探测**,用户不存在返回"邮箱不存在"
- QQ/Gmail 等不响应探测的服务器,不阻断注册,由验证邮件兜底
- 依赖 `dnspython`、`py3-validate-email`(`pip install -r requirements.txt` 已含)

AJAX 校验端点: `GET /api/check-email/?email=xxx@example.com`

### 4. 注册流程全景

```
用户填写(显示名/邮箱/密码)
  → 前端 AJAX 实时校验(格式/已注册/邮箱存在)
  → 后端再次校验(双保险)
  → 发送验证邮件(48小时有效)
  → 用户点击邮件链接 → 账号激活 + 自动登录 → 跳转主页
  → 之后用邮箱(或用户名)正常登录
```

管理员仍可在 `/admin/` 查看/禁用用户(审批环节已移除,邮箱验证即激活)。

## API Endpoints

- `GET /api/task-status/<task_id>/` - Get task status
- `GET /api/queue-stats/` - Get queue statistics
- `GET /api/recent-tasks/` - Get recent tasks

## Admin Access

- **URL**: `http://localhost:8888/admin`（公网: `http://118.31.69.203:8888/admin`）
- **Username**: `admin`
- **初始密码**: `NMRadmin2026!`（首次登录后请及时在 `/admin/` 修改密码）
- `admin` 账号仅用用户名登录(不接邮箱)

## Development

### Running Tests

```bash
python manage.py test core_detection
```

### Creating New Tasks

Add new tasks to `core_detection/tasks.py`:

```python
@shared_task
def your_new_task(param1, param2):
    # Your task logic here
    return result
```

### Adding New Pages

1. Create view function in `core_detection/views.py`
2. Add URL pattern in `core_detection/urls.py`
3. Create template in `core_detection/templates/`

## Troubleshooting

### Celery Issues

- Check Celery logs: `logs/celery.log`
- Ensure Redis is running: `redis-cli ping`
- Restart Celery: `./stop_all.sh && ./start_all.sh`

### Database Issues

- Reset database: `rm db.sqlite3 && python manage.py migrate`
- Create superuser: `python manage.py createsuperuser`

### Static Files

- Collect static files: `python manage.py collectstatic`
- Clear browser cache if styles don't load

## Technologies Used

- **Backend**: Django 4.2+, Python 3.8+
- **Task Queue**: Celery 5.3+
- **Cache/Message Broker**: Redis 5.0+
- **Frontend**: Bootstrap 5, jQuery
- **Database**: SQLite (default)
- **Static Files**: WhiteNoise

## License

This project is for academic and research purposes.

## Support

For issues or questions, use the contact form in the application or check the logs in the `logs/` directory.