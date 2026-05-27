# MAS vs Single LLM — Reproducibility Demo

Standalone pipeline run connecting to the university JupyterHub pod Ollama.
Executes 3 SWE-bench Pro tasks under both architectures (MAS and Single LLM) and
produces a results CSV for comparison.

**No local GPU, no Docker, no API keys required.**

---

## Setup

```bash
git clone https://github.com/AniketDeshpande-23/mas-thesis.git
cd mas-thesis
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copy and configure the environment file:

```bash
copy demo\.env.example demo\.env
```

Edit `demo\.env` — two values:

```
OLLAMA_BASE_URL=https://<hub-url>/user/<username>/proxy/11434
JUPYTERHUB_TOKEN=<token from JupyterHub > File > Hub Control Panel > Token>
```

---

## Running

```bash
venv\Scripts\activate
python demo\run_demo.py
```

The script connects to the pod, auto-selects the best available model, and runs
3 tasks × 2 modes = 6 pipeline executions. Results are saved to
`demo\demo_results\demo_TIMESTAMP.csv`.

**Expected runtime:** 20–50 minutes depending on model size.

---

## What the script runs

| | MAS Pipeline | Single LLM |
|-|---|---|
| **Agents** | Planner → Developer → Debugger → DevRefine → Reviewer | Solo Developer |
| **Iterations** | Difficulty-adaptive (easy: 2, medium: 3, hard: 4) | Same cap |
| **Evaluation** | Static Tester (no Docker in demo mode) | Same |

Tasks: 1 easy (NodeBB, JavaScript), 1 medium (ansible, Python), 1 hard (flipt, Go).

---

## Output metrics

| Column | Definition |
|--------|-----------|
| `patch_score` | Primary metric — `0.6 × file_recall + 0.4 × content_overlap` |
| `file_recall` | Fraction of gold-patch files correctly identified |
| `content_overlap` | Token-level Jaccard similarity of changed lines vs gold patch |
| `debug_iterations` | How many Debugger→DevRefine cycles ran |

Full experiment results (Runs E–M) are in `results/` at the repo root.

---

## Troubleshooting

**Timeout errors** — the nginx proxy has a default 60s `proxy_read_timeout`.
Inference on larger models (qwen3.5:27b, gemma4:31b) can take 90–600s.
Increase to `proxy_read_timeout 900s` in the nginx config on the pod.

**No model found** — check available models with:
```bash
curl -H "Authorization: Bearer <token>" <OLLAMA_BASE_URL>/api/tags
```
If the list is empty, pull a model on the pod: `ollama pull qwen3-coder:latest`
