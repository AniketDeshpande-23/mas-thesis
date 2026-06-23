# MAS vs Single LLM — Reproducibility Demo

Standalone pipeline run connecting to the university JupyterHub pod Ollama.
Executes 3 SWE-bench tasks under both architectures (MAS and Single LLM) and
produces a results CSV for comparison.

**No local GPU, no Docker, no API keys required.**

---

## Setup

```bash
git clone https://github.com/AniketDeshpande-23/mas-thesis.git
cd mas-thesis

# Create virtual environment
python -m venv venv

# Activate — Linux/Mac:
source venv/bin/activate
# Activate — Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

Copy and configure the environment file:

```bash
# Linux/Mac:
cp demo/.env.example demo/.env

# Windows:
# copy demo\.env.example demo\.env
```

Edit `demo/.env` with two values:

```
OLLAMA_BASE_URL=https://<hub-url>/user/<username>/proxy/11434
JUPYTERHUB_TOKEN=<token from JupyterHub > File > Hub Control Panel > Token>
```

---

## Running

```bash
# Linux/Mac:
source venv/bin/activate
python demo/run_demo.py

# Windows:
# venv\Scripts\activate
# python demo\run_demo.py
```

The script connects to the pod, auto-selects the best available model, and runs
3 tasks x 2 modes = 6 pipeline executions. Results are saved to
`demo/demo_results/demo_TIMESTAMP.csv`.

**Expected runtime:** 20-50 minutes depending on model size.

---

## What the script runs

| | MAS Pipeline | Single LLM |
|-|---|---|
| **Agents** | Planner -> Developer -> Debugger -> DevRefine -> Reviewer | Solo Developer |
| **Iterations** | Difficulty-adaptive (easy: 2, medium: 3, hard: 4) | Same cap |
| **Evaluation** | Static Tester (no Docker in demo mode) | Same |

Tasks: 3 tasks from SWE-bench Lite (1 easy + 1 medium + 1 hard), loaded from a
parquet file shipped with the repository — **no internet download required**.

---

## Output metrics

| Column | Definition |
|--------|-----------|
| `patch_score` | Primary metric — 0.6 x file_recall + 0.4 x content_overlap |
| `file_recall` | Fraction of gold-patch files correctly identified |
| `content_overlap` | Token-level Jaccard similarity of changed lines vs gold patch |
| `debug_iterations` | How many Debugger->DevRefine cycles ran |

Full experiment results are in `results/` at the repo root.

---

## Notes

**No dataset download needed** — tasks load from `datasets/swebench_lite_test.parquet`
which is included in the repository. The script runs immediately after `pip install`.

**nginx proxy timeout must be raised before running** — the default `proxy_read_timeout`
is 60s. Model inference takes 90-600s per call. Increase to `proxy_read_timeout 900s`
in the nginx config on the pod (requires admin access).

---

## Troubleshooting

**Timeout errors** — increase `proxy_read_timeout 900s` in the nginx config on the pod.

**No model found** — check available models with:
```bash
curl -H "Authorization: Bearer <token>" <OLLAMA_BASE_URL>/api/tags
```
Pull a model on the pod if needed:
```bash
ollama pull gemma4:27b
# or
ollama pull qwen3-coder:latest
```

Supported models (auto-detected in priority order):
- `qwen3-coder-next`, `qwen3-coder`
- `qwen3.5:27b` / `qwen3.5-27b`, `qwen3.5:9b` / `qwen3.5-9b`
- `gemma4:31b` / `gemma4-31b`, `gemma4:27b` / `gemma4-27b`
- `glm-4.7-flash`
