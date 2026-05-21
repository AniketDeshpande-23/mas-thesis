#!/usr/bin/env bash
# setup_server.sh — Run this ONCE in the Jupyter Lab terminal to prepare the server.
# Usage: bash setup_server.sh

set -e

echo "=== [1/5] Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== [2/5] Starting Ollama in background ==="
nohup ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID (log: /tmp/ollama.log)"

# Give it a moment to initialise
sleep 5
curl -s http://localhost:11434/api/tags > /dev/null && echo "Ollama is up." || {
  echo "ERROR: Ollama did not start. Check /tmp/ollama.log"; exit 1
}

echo "=== [3/5] Setting up Python virtual environment ==="
python -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Python env ready."

echo "=== [4/5] Pulling models (this may take a while — ~50 GB total) ==="
echo "Pulling devstral-small-2 (~15 GB) ..."
ollama pull devstral-small-2:latest

echo "Pulling qwen3-coder (~18.6 GB) ..."
ollama pull qwen3-coder:latest

echo "Pulling qwen3.6 (~23.9 GB) ..."
ollama pull qwen3.6:latest

echo "=== [5/5] Verifying models ==="
ollama list

echo ""
echo "=== Setup complete! ==="
echo "To run a smoke test:"
echo "  source venv/bin/activate"
echo "  python main.py 2>&1 | tee run_log.txt"
echo ""
echo "NOTE: If you close this terminal, Ollama will stop."
echo "To restart Ollama in a new session:"
echo "  nohup ollama serve > /tmp/ollama.log 2>&1 &"
