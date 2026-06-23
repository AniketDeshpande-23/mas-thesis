# MAS vs Single LLM — Demo

Runs 3 SWE-bench tasks (easy / medium / hard) under both MAS and Single LLM architectures via JupyterHub Ollama. Outputs a CSV comparing patch quality scores.

No GPU, no Docker, no API keys needed.

## Quick Start

**1. Install**
```bash
git clone https://github.com/AniketDeshpande-23/mas-thesis.git
cd mas-thesis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure** — copy `demo/.env.example` to `demo/.env` and set your credentials:
```
OLLAMA_BASE_URL=https://jupyterhub.ki-awz.iisys.de/user/<your-username>@hof-university.de/proxy/11434
JUPYTERHUB_TOKEN=<your-token>
```
Token: JupyterHub → File → Hub Control Panel → Token → Request new API token

**3. Run**
```bash
python demo/run_demo.py         # Windows: python demo\run_demo.py
```
6 pipeline runs (3 tasks × 2 modes). Results → `demo/demo_results/demo_TIMESTAMP.csv`. Runtime: 20–50 min.

## Already cloned

```bash
cd mas-thesis && git pull
# configure demo/.env, then:
python demo/run_demo.py
```

## What it runs

| | MAS Pipeline | Single LLM |
|-|---|---|
| **Agents** | Planner → Developer → Tester → Debugger → Reviewer | Solo Developer |
| **Iterations** | Difficulty-adaptive (easy: 2, medium: 3, hard: 4) | Same |
| **Evaluation** | Static Tester (no Docker) | Same |

## Output metrics

| Column | Definition |
|--------|-----------|
| `patch_score` | `0.6 × file_recall + 0.4 × content_overlap` |
| `file_recall` | Fraction of gold-patch files correctly identified |
| `content_overlap` | Token-level Jaccard similarity of changed lines |
| `debug_iterations` | Debugger→DevRefine cycles run |

## Supported models (auto-detected)

`qwen3-coder-next` · `qwen3-coder` · `qwen3.5:27b` · `qwen3.5:9b` · `gemma4:27b` · `gemma4:31b` · `glm-4.7-flash`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| 500 on `/api/tags` | Ollama not running on that pod |
| 403 on `/api/tags` | Token invalid — regenerate from Hub Control Panel |
| No model found | `ollama pull gemma4:27b` on the pod |
| Timeout errors | Raise nginx `proxy_read_timeout 900s` on the pod |
