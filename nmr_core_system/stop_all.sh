#!/bin/bash

# Stop script for NMR Core Detection System

# Ensure we're in the correct directory
cd "$(dirname "$0")"

echo "Stopping NMR Core Detection System..."

# Stop Celery worker
if [ -f logs/celery.pid ]; then
    echo "Stopping Celery worker..."
    kill $(cat logs/celery.pid) 2>/dev/null
    rm logs/celery.pid
    echo "Celery worker stopped."
fi

# Stop Redis server
if [ -f logs/redis.pid ]; then
    echo "Stopping Redis server..."
    kill $(cat logs/redis.pid) 2>/dev/null
    rm logs/redis.pid
    echo "Redis server stopped."
fi

# Stop any running Django instances
echo "Stopping Django development server..."
pkill -f "python manage.py runserver" 2>/dev/null

echo "All services stopped successfully!"