"""
MAS vs Single LLM — demo run.
Runs 3 SWE-bench tasks in both modes. No Docker, no API keys needed.
Set OLLAMA_BASE_URL and JUPYTERHUB_TOKEN in demo/.env before running.
"""
import sys, os
from datetime import datetime
import csv, requests

# ── Repo root on sys.path so all project imports work ──────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

# ── Load .env from demo/ first, then repo root as fallback ─────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(_REPO, ".env"))

# ── Patch orchestrator BEFORE any crewai imports (order matters) ────────
from agents.litellm_patch import APPLIED  # noqa: F401  must be first agent import
import agents.orchestrator as _orch
_orch.USE_AGENT_TOOLS = False   # no MCP server on demo machine
_orch.AGENT_LLM_MAP   = None    # single model for all roles (simpler demo)

# ── Ollama: health check + auto-detect best available model ────────────
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TOKEN      = os.getenv("JUPYTERHUB_TOKEN", "")
HEADERS    = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

# Priority list — first model found on the pod wins
MODEL_PRIORITY = [
    ("qwen3-coder-next", "Qwen3CoderNextLocal"),
    ("qwen3-coder",      "Qwen3Coder"),
    ("qwen3.5:27b",      "Qwen35_27B"),
    ("qwen3.5:9b",       "Qwen35_9B"),
    ("gemma4:31b",       "Gemma4_31B"),
    ("gemma4:27b",       "Gemma4_27B"),
    ("glm-4.7-flash",    "GLM47Flash"),
]

try:
    resp   = requests.get(f"{OLLAMA_URL}/api/tags", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    pulled = {m["name"] for m in resp.json().get("models", [])}
except Exception as exc:
    print(f"\nERROR: Cannot reach Ollama at {OLLAMA_URL}")
    print(f"  Cause: {exc}")
    print("  Check OLLAMA_BASE_URL and JUPYTERHUB_TOKEN in demo/.env")
    sys.exit(1)

llm = None
for tag, cls_name in MODEL_PRIORITY:
    if any(tag in name for name in pulled):
        mod = __import__("agents.models", fromlist=[cls_name])
        llm = getattr(mod, cls_name)()
        break

if llm is None:
    print(f"\nERROR: No supported model found on Ollama at {OLLAMA_URL}")
    print(f"  Models currently pulled: {sorted(pulled) or '(none)'}")
    print("  Pull a model on the pod:  ollama pull qwen3-coder:latest")
    sys.exit(1)

print(f"  Ollama  : {OLLAMA_URL}")
print(f"  Model   : {llm.name}")

# ── Dataset: 3-task smoke test (NodeBB JS / ansible Python / flipt Go) ─
from data_loaders.swebench_pro import SWEBenchPro
DATASET = SWEBenchPro(
    sample_size=3,
    seed=42,
    use_test_subset=True,
    smoke_test=True,
    cache_dir=os.path.join(_REPO, "datasets"),
)
tasks = DATASET.load()
print(f"  Tasks   : {len(tasks)} (smoke test — 1 easy, 1 medium, 1 hard)")
print(f"{'━' * 60}")

# ── Pipeline imports ────────────────────────────────────────────────────
from agents.orchestrator import run_pipeline
from evaluation.patch_similarity import score_patch
from validation.swebench_pro_validator import validate

# ── Run loop — meta=None means no Docker, uses static Tester fallback ──
rows = []
for task in tasks:
    inst       = DATASET.get_instance(task.task_id)
    gold_patch = inst.patch if inst else ""

    for mode in ["mas", "single"]:
        print(f"\n  Running {mode.upper()} on {task.task_id[-50:]}")
        result = run_pipeline(task, llm, mode=mode, meta=None)
        report = validate(result, task.test_cases, task.function_name)
        sim    = score_patch(result.final_code, gold_patch)

        row = {
            "task_id":          task.task_id,
            "difficulty":       inst.difficulty.value if inst else "unknown",
            "mode":             mode,
            "model":            llm.name,
            "patch_score":      round(sim["patch_score"], 4),
            "file_recall":      round(sim["file_recall"], 4),
            "content_overlap":  round(sim["content_overlap"], 4),
            "status":           report.overall_status,
            "debug_iterations": result.debug_iterations,
        }
        rows.append(row)

        print(
            f"  [{mode:6s}] {task.task_id[-45:]:45s}  "
            f"score={row['patch_score']:.4f}  "
            f"file_recall={row['file_recall']:.4f}  "
            f"status={report.overall_status}"
        )

# ── Save results CSV ────────────────────────────────────────────────────
out_dir  = os.path.join(os.path.dirname(__file__), "demo_results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ── Summary ─────────────────────────────────────────────────────────────
mas_rows    = [r for r in rows if r["mode"] == "mas"]
single_rows = [r for r in rows if r["mode"] == "single"]
mas_avg     = sum(r["patch_score"] for r in mas_rows)    / len(mas_rows)
single_avg  = sum(r["patch_score"] for r in single_rows) / len(single_rows)

print(f"\n{'=' * 55}")
print(f"  MAS    avg patch_score = {mas_avg:.4f}")
print(f"  Single avg patch_score = {single_avg:.4f}")
print(f"  {'MAS wins' if mas_avg > single_avg else 'Single wins'}")
print(f"\n  Results saved to: {out_path}")
