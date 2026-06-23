# MAS vs Single LLM — Demo

Runs 3 SWE-bench tasks (easy / medium / hard) under both MAS and Single LLM architectures via JupyterHub Ollama. Outputs a CSV comparing patch quality scores.

No GPU, no Docker, no API keys needed.

## Setup

```bash
git clone https://github.com/AniketDeshpande-23/mas-thesis.git
cd mas-thesis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp demo/.env.example demo/.env  # Windows: copy demo\.env.example demo\.env
```

Edit `demo/.env`:

```
OLLAMA_BASE_URL=https://<hub-url>/user/<username>/proxy/11434
JUPYTERHUB_TOKEN=<token from JupyterHub > File > Hub Control Panel > Token>
```

## Run

```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
python demo/run_demo.py    # Windows: python demo\run_demo.py
```

6 pipeline runs (3 tasks × 2 modes). Results → `demo/demo_results/demo_TIMESTAMP.csv`. Runtime: 20–50 min.

## Supported models (auto-detected)

`qwen3-coder-next` · `qwen3-coder` · `qwen3.5:27b` · `qwen3.5:9b` · `gemma4:27b` · `gemma4:31b` · `glm-4.7-flash`

Pull one on the pod if needed: `ollama pull gemma4:27b`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Timeout errors | Raise nginx `proxy_read_timeout 900s` on the pod |
| No model found | `curl -H "Authorization: Bearer <token>" <OLLAMA_BASE_URL>/api/tags` |
