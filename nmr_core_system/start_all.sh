#!/bin/bash

# Complete startup script for NMR Core Detection System with CUDA 1

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting NMR Core Detection System with CUDA 1..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Stop any existing processes first
echo "Stopping any existing processes..."
if [ -f logs/celery.pid ]; then
    celery_pid=$(cat logs/celery.pid)
    if ps -p $celery_pid > /dev/null 2>&1; then
        kill $celery_pid 2>/dev/null || true
    fi
    rm -f logs/celery.pid
fi

if [ -f logs/redis.pid ]; then
    redis_pid=$(cat logs/redis.pid)
    if ps -p $redis_pid > /dev/null 2>&1; then
        kill $redis_pid 2>/dev/null || true
    fi
    rm -f logs/redis.pid
fi

# Kill any existing Django processes
pkill -f "manage.py runserver" 2>/dev/null || true

sleep 2

# Start Redis
echo "Starting Redis server..."
redis-server --daemonize yes --port 6379 --logfile logs/redis.log --pidfile logs/redis.pid

# Wait for Redis to start
sleep 2

# Start Celery worker with CUDA 1
echo "Starting Celery worker with CUDA 1..."
CUDA_VISIBLE_DEVICES=0 celery -A nmr_core_system worker --loglevel=info --logfile=logs/celery.log --pidfile=logs/celery.pid --detach

# Start Django development server
echo "Starting Django development server on port 8888..."
# nohup python manage.py runserver 0.0.0.0:8888 > logs/django.log 2>&1 &

echo "NMR Core Detection System is now running with CUDA 1!"
echo "Django: http://localhost:8888"
echo "Admin: http://localhost:8888/admin"
echo "Redis logs: logs/redis.log"
echo "Celery logs: logs/celery.log"
echo "CUDA Device: 0"

python manage.py runserver 0.0.0.0:8888 #> logs/django.log 2>&1 &