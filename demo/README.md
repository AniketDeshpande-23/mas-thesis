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
OLLAMA_BASE_URL=https://jupyterhub.ki-awz.iisys.de/user/<your-username>@hof-university.de/proxy/11434
JUPYTERHUB_TOKEN=<your-token>
```

Get your token: **JupyterHub → File → Hub Control Panel → Token → Request new API token**

## Verify connection

```bash
curl -H "Authorization: Bearer <your-token>" <OLLAMA_BASE_URL>/api/tags
```

Should return a JSON list of available models.

## Run

```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
python demo/run_demo.py    # Windows: python demo\run_demo.py
```

6 pipeline runs (3 tasks × 2 modes). Results → `demo/demo_results/demo_TIMESTAMP.csv`. Runtime: 20–50 min.

## Supported models (auto-detected)

`qwen3-coder-next` · `qwen3-coder` · `qwen3.5:27b` · `qwen3.5:9b` · `gemma4:27b` · `gemma4:31b` · `glm-4.7-flash`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| 500 on `/api/tags` | Wrong URL — pointing to a pod with no Ollama |
| 403 on `/api/tags` | Token rejected — generate a fresh one from Hub Control Panel |
| No model found | Model not pulled on that pod: `ollama pull gemma4:27b` |
| Timeout errors | Raise nginx `proxy_read_timeout 900s` on the pod |
