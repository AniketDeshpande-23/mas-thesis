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

