#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo " Starting HealthTrack AI Full-Stack Platform"
echo "=========================================================="
echo " 1. Backend: FastAPI Gateway (http://localhost:8000)"
echo " 2. Frontend: Next.js Studio (http://localhost:3000)"
echo "=========================================================="

mkdir -p logs

# Start FastAPI Gateway in background
echo "[1/2] Launching FastAPI Gateway on port 8000..."
if command -v uv &> /dev/null; then
  uv run uvicorn apps.server:create_app --factory --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
elif [ -f ".venv/bin/python" ]; then
  .venv/bin/python -m uvicorn apps.server:create_app --factory --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
else
  python3 -m uvicorn apps.server:create_app --factory --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
fi
BACKEND_PID=$!
echo "      Backend running with PID: $BACKEND_PID (logs: logs/backend.log)"

cleanup() {
  echo ""
  echo "Shutting down servers..."
  kill "$BACKEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start Next.js Frontend
echo "[2/2] Launching Next.js Studio on port 3000..."
cd "$PROJECT_ROOT/frontend"
npm run dev

