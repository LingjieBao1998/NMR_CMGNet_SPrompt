#!/bin/bash

# Celery startup script for NMR Core Detection System

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Start Celery worker
echo "Starting Celery worker..."
celery -A nmr_core_system worker --loglevel=info --logfile=logs/celery.log --pidfile=logs/celery.pid

# Alternative: Start Celery beat as well (for periodic tasks)
# celery -A nmr_core_system worker --beat --loglevel=info --logfile=logs/celery.log --pidfile=logs/celery.pid