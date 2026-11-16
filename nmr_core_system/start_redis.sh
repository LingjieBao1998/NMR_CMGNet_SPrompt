#!/bin/bash

# Redis startup script for NMR Core Detection System

echo "Starting Redis server..."
redis-server --daemonize yes --port 6379 --logfile logs/redis.log --pidfile logs/redis.pid

echo "Redis server started successfully!"
echo "Check logs at: logs/redis.log"