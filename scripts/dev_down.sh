#!/bin/bash
# Shutdown script for EHC AI MVP stack

set -e

echo "=== Stopping EHC AI MVP Stack ==="
cd "$(dirname "$0")/.."

docker-compose -f docker/docker-compose.yml down

echo "Stack stopped successfully"
