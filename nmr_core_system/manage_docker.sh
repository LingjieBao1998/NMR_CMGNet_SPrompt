#!/bin/bash

# Docker容器管理脚本 - NMR检测系统 (docker compose v2+)
# 注意: 必须用新版 `docker compose`(v5.0.1)，旧版 docker-compose 1.29 会忽略 deploy 段导致无 GPU

cd "$(dirname "$0")"

command -v docker >/dev/null 2>&1 || { echo "Error: docker not found"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Error: docker compose (v2) not found"; exit 1; }

case "$1" in
    start)
        echo "Starting NMR Core Detection System..."
        # 清理可能存在的同名遗留容器(避免 up 冲突)
        docker rm -f nmr-core-system 2>/dev/null || true
        docker compose up -d
        echo "System started successfully!"
        echo "Access: http://localhost:8888"
        ;;

    stop)
        echo "Stopping NMR Core Detection System..."
        docker compose stop
        echo "System stopped successfully!"
        ;;

    restart)
        echo "Restarting NMR Core Detection System..."
        docker compose restart
        echo "System restarted successfully!"
        ;;

    logs)
        echo "Showing logs (press Ctrl+C to exit)..."
        docker compose logs -f
        ;;

    status)
        echo "Checking system status..."
        docker compose ps
        ;;

    shell)
        echo "Opening shell in container..."
        docker compose exec nmr-core-system bash
        ;;

    down)
        echo "Stopping and removing container..."
        docker compose down
        echo "Container removed."
        ;;

    clean)
        echo "Cleaning up container and volumes..."
        read -p "This will remove the container and volumes. Are you sure? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker compose down -v
            echo "Cleanup completed!"
        else
            echo "Cleanup cancelled."
        fi
        ;;

    *)
        echo "NMR Core Detection System - Docker Management"
        echo "Usage: $0 {start|stop|restart|logs|status|shell|down|clean}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the system"
        echo "  stop    - Stop the system"
        echo "  restart - Restart the system"
        echo "  logs    - View logs"
        echo "  status  - Check system status"
        echo "  shell   - Open shell in container"
        echo "  down    - Stop and remove container"
        echo "  clean   - Remove container and volumes"
        exit 1
        ;;
esac
