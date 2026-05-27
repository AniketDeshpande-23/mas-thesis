# MAS vs Single LLM — Demo

Runs the full MAS and Single pipeline on 3 SWE-bench tasks using the university JupyterHub pod Ollama.
No Docker, no API keys, no local GPU required.

---

## Prerequisites

- **Python 3.11+** — download from [python.org/downloads](https://www.python.org/downloads/)
- **JupyterHub access** — the pod must be running with Ollama active

---

## Step 1 — Clone the repository

```
git clone https://github.com/AniketDeshpande-23/mas-thesis.git
cd mas-thesis
```

---

## Step 2 — Run setup

Double-click `demo\setup_windows.bat` — it creates the virtual environment and installs all dependencies.

Or from the command line at the repo root:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 3 — Configure the Ollama connection

Open `demo\.env` in any text editor and fill in two values:

**OLLAMA_BASE_URL** — the JupyterHub proxy URL for Ollama:
- Log in to JupyterHub in your browser
- Look at the URL bar: `https://<hub-url>/user/<your-username>/...`
- The Ollama URL is: `https://<hub-url>/user/<your-username>/proxy/11434`

**JUPYTERHUB_TOKEN** — your authentication token:
- In JupyterHub: File > Hub Control Panel > Token > Request new API token
- Copy and paste the token

Example `demo\.env`:
```
OLLAMA_BASE_URL=https://hub.hof-university.de/user/adeshpande2/proxy/11434
JUPYTERHUB_TOKEN=abc123def456...
OTEL_SDK_DISABLED=true
```

---

## Step 4 — Verify Ollama is reachable

Open a terminal and run (replace the URL and token with your values):
```
curl -H "Authorization: Bearer <your-token>" https://<hub-url>/user/<username>/proxy/11434/api/tags
```

You should see a JSON list of models. If it times out, the pod may need Ollama restarted or the proxy timeout needs to be increased (admin required).

If no models are listed, ask to have one pulled on the pod:
```
ollama pull qwen3-coder:latest
```

---

## Step 5 — Run the demo

```
venv\Scripts\activate
python demo\run_demo.py
```

The script will:
1. Connect to the pod Ollama and auto-detect which model to use
2. Load 3 SWE-bench tasks (1 easy, 1 medium, 1 hard)
3. Run each task through both MAS and Single pipelines
4. Print a result row for each run
5. Show a final summary comparing MAS vs Single

**Expected runtime:** 20–50 minutes depending on model size.

---

## Expected output

```
  Ollama  : https://hub.hof-university.de/user/.../proxy/11434
  Model   : qwen3-coder
  Tasks   : 3 (smoke test — 1 easy, 1 medium, 1 hard)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [mas   ] NodeBB-community__NodeBB-...   score=0.XXXX  status=COMPLETED
  [single] NodeBB-community__NodeBB-...   score=0.XXXX  status=COMPLETED
  [mas   ] ansible__ansible-...           score=0.XXXX  status=COMPLETED
  [single] ansible__ansible-...           score=0.XXXX  status=COMPLETED
  [mas   ] flipt-io__flipt-...            score=0.XXXX  status=COMPLETED
  [single] flipt-io__flipt-...            score=0.XXXX  status=COMPLETED

  ═══════════════════════════════════════════════════════
  MAS    avg patch_score = 0.XXXX
  Single avg patch_score = 0.XXXX
  MAS wins

  Results saved to: demo\demo_results\demo_20260527_XXXXXX.csv
```

---

## What the metrics mean

| Metric | Definition |
|--------|-----------|
| `patch_score` | Primary metric: `0.6 × file_recall + 0.4 × content_overlap` |
| `file_recall` | Did the agent target the correct source file(s)? |
| `content_overlap` | Token-level similarity between generated fix and the correct fix |
| `status` | COMPLETED = patch produced; FAILED = agent could not generate a patch |

---

## Troubleshooting

**"Cannot reach Ollama"** — check that the JupyterHub pod is running, the proxy URL is correct, and the token is valid.

**"No supported model found"** — run `ollama pull qwen3-coder:latest` on the pod terminal.

**Timeout errors** — the nginx proxy on the pod has a default 60s read timeout. For large models (inference takes 90–600s), this needs to be increased by an admin: set `proxy_read_timeout 900s` in the nginx config.

**Import errors** — make sure you ran `pip install -r requirements.txt` from the repo root (not from inside `demo\`).
