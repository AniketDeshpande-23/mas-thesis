"""
Main entrypoint for MAS vs SINGLE experiments on different datasets.

Supports:
- BigCodeBench-Hard (function generation)
- SWE-bench Pro (patch generation)
"""
from agents.litellm_patch import APPLIED as _THINKING_DISABLED  # must be first import

import csv
import json
import os
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════════
# 1. DATASET
# ══════════════════════════════════════════════════════════════════════

# --- Option A: BigCodeBench-Hard (function generation) ---
# from data_loaders.bigcodebench_hard import BigCodeBenchHard
# DATASET = BigCodeBenchHard(sample_size=3, seed=42)

# --- Option B: SWE-bench Lite ---
# from data_loaders.swebench_lite import SWEBenchLite
# DATASET = SWEBenchLite(sample_size=15, seed=42, cache_dir="./datasets")

# --- Option C: SWE-bench Pro (ACTIVE) ---
# 15-task curated subset: 5 repos × 3 difficulties (easy/medium/hard)
# Repos: NodeBB(JS), ansible(Python), flipt-io(Go), teleport(Go), openlibrary(Python)
# use_test_subset=True  → loads the 15 IDs from TEST_SUBSET_IDS (curated, verified)
# use_test_subset=False → balanced random sample from all 731 instances
from data_loaders.swebench_pro import SWEBenchPro
DATASET = SWEBenchPro(
    sample_size=15,            # ignored when smoke_test=True
    seed=42,
    use_test_subset=True,
    smoke_test=True,           # True = 3 tasks (1 easy NodeBB/JS + 1 med ansible/Py + 1 hard flipt/Go)
    cache_dir="./datasets",    # False = full 15-task thesis run (set before VM run)
)

# ═══════════════════════════════════════════════
# 2. MODELS  
# ═══════════════════════════════════════════════

# ── Thesis models  ───────────────────────────
#
#  Model              Size         VRAM     Released   Pull command
#  glm-4.7-flash      30B/3B MoE   ~5 GB    Jan 2026   ollama pull glm-4.7-flash:latest
#  gemma4:31b         31B dense    ~19 GB   Apr 2026   ollama pull gemma4:31b
#  qwen3.5:27b        27B dense    ~17 GB   Feb 2026   ollama pull qwen3.5:27b

from agents.models import GeminiFlashLiteOpenRouter
LLMS = [GeminiFlashLiteOpenRouter()]   # Flash-Lite — Single baseline (Qwen3CoderNext breaks Single)


# ══════════════════════════════════════
# 3. MODES
# ══════════════════════════════════════
MODES = ["mas", "single"]

# ══════════════════════════════════════════════════════════════════════
# RESUME — set to an existing CSV to skip already-completed rows
#          and append results to that file instead of creating a new one.
#          Set to None to start a fresh run.
# ══════════════════════════════════════════════════════════════════════
RESUME_CSV = None

# ══════════════════════════════════════════════════════════════════════
# 4. VALIDATOR
# validator.py is only for BigCodeBench-Hard (function generation tasks).
# ══════════════════════════════════════════════════════════════════════
from data_loaders.swebench_lite import SWEBenchLite as _SWEBenchLite
try:
    from data_loaders.swebench_pro import SWEBenchPro as _SWEBenchPro
except ImportError:
    _SWEBenchPro = None  # type: ignore

_IS_SWEBENCH = isinstance(DATASET, _SWEBenchLite) or (
    _SWEBenchPro is not None and isinstance(DATASET, _SWEBenchPro)
)

if _IS_SWEBENCH:
    from validation.swebench_pro_validator import validate
else:
    # BigCodeBench-Hard only — generates Python functions, not patches
    from validation.validator import validate

# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════
from agents.orchestrator import run_pipeline
import requests
from evaluation.trulens_evaluator import EvaluationTracker
from evaluation.patch_similarity import score_patch, score_codebleu
from evaluation.pass_at_k import compute_metrics as compute_pass_at_k, format_summary as format_pass_summary

try:
    from docker_eval import check_docker as _check_docker
    _DOCKER_MODULE_AVAILABLE = True
except ImportError:
    _DOCKER_MODULE_AVAILABLE = False
    _check_docker = lambda: False  # type: ignore


def start_docker_if_needed(timeout: int = 90) -> bool:
    """
    If Docker daemon is not running, attempt to start Docker Desktop (Windows/Mac)
    and wait up to `timeout` seconds for it to become ready.
    Returns True if Docker is ready, False otherwise.
    """
    import subprocess, time, sys

    if _check_docker():
        return True  # already running

    print("  Docker  : not running — attempting to start Docker Desktop...")

    # Platform-specific launch commands
    candidates = []
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
            r"C:\Program Files (x86)\Docker\Docker\Docker Desktop.exe",
        ]
    elif sys.platform == "darwin":
        candidates = ["/Applications/Docker.app/Contents/MacOS/Docker"]

    launched = False
    for path in candidates:
        if os.path.exists(path):
            try:
                subprocess.Popen([path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                launched = True
                print(f"  Docker  : launched from {path}")
                break
            except Exception as exc:
                print(f"  Docker  : failed to launch ({exc})")

    if not launched and not candidates:
        print("  Docker  : auto-start not supported on this platform. Start Docker manually.")
        return False

    if not launched:
        print("  Docker  : Docker Desktop not found at expected paths. Start it manually.")
        return False

    # Poll until ready
    print(f"  Docker  : waiting up to {timeout}s for daemon to become ready...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(5)
        print(".", end="", flush=True)
        if _check_docker():
            print(" ready.")
            return True

    print(" timed out.")
    print("  Docker  : daemon did not start in time. Run will use static Tester fallback.")
    return False


def check_ollama_health() -> bool:
    """
    Verify Ollama is running AND that both experiment models are pulled.
    Returns False (and prints guidance) on any failure.
    """
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    _token = os.getenv("JUPYTERHUB_TOKEN", "")
    _headers = {"Authorization": f"Bearer {_token}"} if _token else {}

    # 1. Server reachable?
    try:
        resp = requests.get(f"{ollama_url}/api/tags", headers=_headers, timeout=10)
        resp.raise_for_status()
        pulled = {m["name"] for m in resp.json().get("models", [])}
    except requests.ConnectionError:
        print(f"\n  ERROR: Ollama server is not running at {ollama_url}")
        print("  Start it with:  ollama serve")
        return False
    except Exception as exc:
        print(f"\n  ERROR: Cannot reach Ollama ({exc})")
        return False

    print(f"  Ollama server reachable at {ollama_url}")

    # 2. Required local models present?
    required = {llm.crewai_llm.model.removeprefix("ollama/").removeprefix("openai/") for llm in LLMS if llm.backend == "local"}
    missing = []
    for model_tag in required:
        # Ollama /api/tags returns names like "qwen3.6:latest"
        if not any(model_tag in name for name in pulled):
            missing.append(model_tag)

    if missing:
        print(f"\n  ERROR: These models are not pulled in Ollama:")
        for m in missing:
            print(f"    ollama pull {m}")
        return False

    print(f"  Models verified: {sorted(required)}")
    return True


def main():
    os.makedirs("results", exist_ok=True)

    # Initialize evaluation tracker (no external dependencies)
    tracker = EvaluationTracker(experiment_name="mas_vs_single_comparison")

    # Auto-start Docker Desktop if not running
    if _IS_SWEBENCH and _DOCKER_MODULE_AVAILABLE:
        start_docker_if_needed()

    # Check Ollama connectivity (skip for cloud-only runs)
    has_local = any(llm.backend == "local" for llm in LLMS)
    if has_local and not check_ollama_health():
        print("\n❌ Cannot proceed without Ollama server. Exiting.")
        return

    # Check Docker availability for in-loop evaluation
    _docker_available = _IS_SWEBENCH and _DOCKER_MODULE_AVAILABLE and _check_docker()
    if _docker_available:
        print("  Docker  : available — real test eval will run in the debug loop")
    else:
        print("  Docker  : not available — using static Tester fallback")

    tasks = DATASET.load()

    backends = "/".join(sorted({llm.backend for llm in LLMS}))
    print(f"\n{'━'*60}")
    print(f"  Dataset : {DATASET.name}  ({len(tasks)} tasks)")
    print(f"  Models  : {[l.name for l in LLMS]}")
    print(f"  Backend : {backends}")
    print(f"  Modes   : {MODES}")
    print(f"  No-think: {_THINKING_DISABLED}")
    # Smoke-test: 3 runs × 3 tasks × 2 modes = 18 runs. Full thesis: 15 tasks × 3 runs × 4 conditions = 180 pts
    NUM_RUNS = 3

    print(f"  Runs    : {NUM_RUNS}x per model+mode")
    print(f"{'━'*60}")

    # ── Resume: build set of already-completed (run, model, mode, task_id) ──
    _completed: set = set()
    if RESUME_CSV and os.path.exists(RESUME_CSV):
        with open(RESUME_CSV, encoding="utf-8") as _rf:
            for _row in csv.DictReader(_rf):
                _completed.add((_row["run_number"], _row["model_name"], _row["mode"], _row["task_id"]))
        print(f"  Resume  : {len(_completed)} rows already done → skipping them")

    all_reports = []
    swebench_patches = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Append to existing CSV when resuming; create fresh otherwise
    if RESUME_CSV and os.path.exists(RESUME_CSV) and _completed:
        csv_out = RESUME_CSV
    else:
        csv_out = os.path.join("results", f"results_{ts}.csv")

    _inc_cols   = [
        "task_id", "model_name", "mode", "run_number",
        "backend", "difficulty", "repo", "language",
        "overall_status", "resolved", "syntax_valid", "code_extracted",
        "duration_seconds", "llm_calls", "total_latency_ms", "avg_latency_ms",
        "tester_approved", "tester_pass_iteration", "debug_iterations",
        "patch_changed_by_debug", "max_debug_iterations", "reviewer_approved",
        "patch_score", "file_recall", "content_overlap", "codebleu",
        "initial_patch_score", "initial_file_recall", "initial_content_overlap",
        "debug_improvement", "pattern_matched", "failure_mode",
        "files_correct", "gen_files_count",
        "gold_patch_size", "gold_hunks", "gold_files_changed",
        "tests_run", "tests_failed",
        "docker_resolved", "docker_status",
        "agent_model_breakdown",
    ]
    _is_resume  = bool(_completed)
    _csv_fh     = open(csv_out, "a" if _is_resume else "w", newline="", encoding="utf-8")
    _csv_writer = csv.DictWriter(_csv_fh, fieldnames=_inc_cols, extrasaction="ignore")
    if not _is_resume:
        _csv_writer.writeheader()
    _csv_fh.flush()
    print(f"  Incremental CSV → {csv_out}  ({'append' if _is_resume else 'new'})")

    # ── Run metadata sidecar (reproducibility) ────────────────────────
    try:
        import crewai as _crewai
        import platform as _platform
        _run_meta = {
            "timestamp": ts,
            "crewai_version": getattr(_crewai, "__version__", "unknown"),
            "python_version": _platform.python_version(),
            "models": [l.name for l in LLMS],
            "modes": MODES,
            "num_runs": NUM_RUNS,
            "dataset": DATASET.name,
            "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "temperature": 0.15,
            "max_tokens": 8192,
        }
        _meta_out = os.path.join("results", f"run_config_{ts}.json")
        with open(_meta_out, "w", encoding="utf-8") as _mf:
            json.dump(_run_meta, _mf, indent=2)
        print(f"  Run config     → {_meta_out}")
    except Exception:
        pass

    for run_num in range(1, NUM_RUNS + 1):
        for llm in LLMS:
            for mode in MODES:
                print(f"\n{'━'*60}")
                print(f"  RUN {run_num}/{NUM_RUNS}  |  {llm.name}  |  {mode.upper()}")
                print(f"{'━'*60}")

                for task_idx, task in enumerate(tasks, 1):
                    # ── Skip if already done in resumed run ───────────
                    if (str(run_num), llm.name, mode, task.task_id) in _completed:
                        print(f"  [SKIP] Run{run_num} {mode} {task.task_id[-25:]} — already completed")
                        continue

                    # ── Build Docker meta dict (if Docker available) ──
                    _inst_pre = DATASET.get_instance(task.task_id) if _IS_SWEBENCH else None
                    meta = None
                    if _docker_available and _inst_pre:
                        meta = {
                            "dockerhub_tag":              getattr(_inst_pre, "dockerhub_tag", ""),
                            "before_repo_set_cmd":        getattr(_inst_pre, "before_repo_set_cmd", ""),
                            "fail_to_pass":               getattr(_inst_pre, "fail_to_pass", []),
                            "pass_to_pass":               getattr(_inst_pre, "pass_to_pass", []),
                            "repo_language":              getattr(_inst_pre, "repo_language", "python"),
                            "selected_test_files_to_run": getattr(_inst_pre, "selected_test_files_to_run", []),
                            "test_patch":                 getattr(_inst_pre, "test_patch", ""),
                        }

                    # ── Run the pipeline  ─────────────────
                    result = run_pipeline(task, llm, mode=mode, meta=meta)

                    # ── Validate (auto-selects correct validator) ────
                    report = validate(result, task.test_cases, task.function_name)

                    # ── Print result ───────────────────────────────
                    icon = "✅" if report.overall_status == "PASSED" else (
                           "🔧" if report.overall_status == "COMPLETED" else (
                           "⚠️" if report.overall_status == "ERROR" else "❌"))
                    _docker_tag = (
                        f"  docker={'RESOLVED' if result.docker_resolved else 'failed'}"
                        if result.docker_status != "not_run" else ""
                    )
                    print(
                        f"\n  {icon}  [{task_idx}/{len(tasks)}]  Task     : {task.task_id}"
                        f"\n              Status   : {report.overall_status}"
                        f"\n              Syntax   : {report.syntax_valid}"
                        f"\n              Tester   : {'PASS' if result.tester_approved else 'FAIL'}"
                        f"  (iter {result.tester_pass_iteration})"
                        f"  debug_iters={result.debug_iterations}"
                        f"{_docker_tag}"
                        f"\n              Reviewer : {report.reviewer_approved}"
                        f"\n              Time     : {report.duration_seconds}s"
                    )

                    report_dict = report.to_dict()

                    # ── Add run number for tracking ──────────────────
                    report_dict["run_number"] = run_num
                    report_dict["resolved"] = 1 if report.overall_status in ("PASSED", "COMPLETED") else 0

                    # ── Pipeline fields not on ValidationReport ──────
                    # These come from PipelineResult directly; copy for CSV.
                    report_dict.setdefault("tester_approved", result.tester_approved)
                    report_dict.setdefault("tester_pass_iteration", result.tester_pass_iteration)
                    report_dict.setdefault("debug_iterations", result.debug_iterations)
                    report_dict.setdefault("patch_changed_by_debug", result.patch_changed_by_debug)
                    report_dict.setdefault("max_debug_iterations", result.max_debug_iterations)
                    report_dict.setdefault("backend", result.backend)
                    report_dict.setdefault("llm_calls", result.llm_calls)
                    report_dict.setdefault("total_latency_ms", result.total_latency_ms)
                    report_dict.setdefault("avg_latency_ms", result.avg_latency_ms)
                    report_dict.setdefault("pattern_matched", getattr(result, "pattern_matched", ""))
                    report_dict["docker_resolved"]        = getattr(result, "docker_resolved", False)
                    report_dict["docker_status"]          = getattr(result, "docker_status", "not_run")
                    report_dict["agent_model_breakdown"]  = getattr(result, "agent_model_breakdown", "")

                    # ── Log to evaluation tracker ────────────────────
                    tracker.log_evaluation(
                        model_name=llm.name,
                        mode=mode,
                        run_num=run_num,
                        report_dict=report_dict,
                        metrics={
                            "code_extracted": report.code_extracted,
                            "tests_run": report.tests_run,
                            "tests_failed": report.tests_failed,
                            "syntax_valid": report.syntax_valid,
                            "overall_status": report.overall_status,
                            "duration_seconds": report.duration_seconds,
                            "tester_approved": result.tester_approved,
                            "debug_iterations": result.debug_iterations,
                            "patch_changed_by_debug": result.patch_changed_by_debug,
                        }
                    )

                    # ── SWE-bench: enrich report + gold-patch similarity ─
                    if _IS_SWEBENCH:
                        inst = _inst_pre  # already fetched above
                        if inst:
                            report_dict["difficulty"] = inst.difficulty.value
                            report_dict["repo"] = inst.repo
                            report_dict["language"] = getattr(inst, "repo_language", "unknown")
                            report_dict["gold_patch_size"] = inst.patch_size
                            # hunks / files_changed: native on SWEBenchLite;
                            # computed from the gold patch for SWEBenchPro.
                            gold_patch = inst.patch or ""
                            report_dict["gold_hunks"] = getattr(
                                inst, "hunks",
                                gold_patch.count("\n@@"),
                            )
                            report_dict["gold_files_changed"] = getattr(
                                inst, "files_changed",
                                gold_patch.count("\ndiff --git"),
                            )

                            # ── Final patch similarity ─────────────────────
                            sim = score_patch(result.final_code, inst.patch)
                            report_dict["file_recall"]      = sim["file_recall"]
                            report_dict["content_overlap"]  = sim["content_overlap"]
                            report_dict["patch_score"]      = sim["patch_score"]
                            report_dict["files_correct"]    = sim["files_correct"]
                            report_dict["gen_files_count"]  = len(sim["gen_files"])

                            # ── Initial patch similarity (before debug loop) ─
                            # Measures whether the debug loop actually improved
                            # the patch. debug_improvement > 0 means debugging helped.
                            init_sim = score_patch(result.initial_code, inst.patch)
                            report_dict["initial_patch_score"]   = init_sim["patch_score"]
                            report_dict["initial_file_recall"]   = init_sim["file_recall"]
                            report_dict["initial_content_overlap"] = init_sim["content_overlap"]
                            report_dict["debug_improvement"]     = round(
                                sim["patch_score"] - init_sim["patch_score"], 4
                            )

                            # ── Failure mode categorisation ────────────────
                            _fr = sim["file_recall"]
                            _co = sim["content_overlap"]
                            if report.overall_status in ("PASSED", "COMPLETED") and _fr > 0:
                                _fmode = "ok"
                            elif not report.code_extracted:
                                _fmode = "no_extraction"
                            elif report.overall_status == "ERROR":
                                _fmode = "error"
                            elif _fr == 0 and _co > 0:
                                _fmode = "wrong_file"
                            elif _fr > 0 and _co == 0:
                                _fmode = "zero_content"
                            elif _fr == 0 and _co == 0:
                                _fmode = "wrong_file"
                            else:
                                _fmode = "unknown"
                            report_dict["failure_mode"] = _fmode

                            # ── CodeBLEU (code-aware similarity) ──────────
                            lang = report_dict.get("language", "python")
                            report_dict["codebleu"] = score_codebleu(
                                result.final_code, inst.patch, language=lang
                            )

                            print(
                                f"              Patch score : {sim['patch_score']:.2f}  "
                                f"(file_recall={sim['file_recall']:.2f}  "
                                f"content={sim['content_overlap']:.2f}  "
                                f"codebleu={report_dict['codebleu']:.4f})\n"
                                f"              Initial     : {init_sim['patch_score']:.2f}  "
                                f"debug_improvement={report_dict['debug_improvement']:+.4f}"
                            )

                        # Collect patches for Docker eval (run separately after)
                        swebench_patches.append({
                            "instance_id": task.task_id,
                            "patch": result.final_code,
                            "prefix": f"{llm.name}_{mode}_run{run_num}",
                        })

                    all_reports.append(report_dict)
                    # Incremental write — task is on disk the moment it finishes
                    _csv_writer.writerow(report_dict)
                    _csv_fh.flush()

    # ── Close incremental CSV ─────────────────────────────────────────
    _csv_fh.close()
    print(f"\n  CSV saved      → {csv_out}  ({len(all_reports)} rows)")

    # ── Full JSON save ────────────────────────────────────────────────
    out = os.path.join("results", f"results_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    print(f"  JSON saved     → {out}")

    # ── SWE-bench patches for Docker eval (always saved, run separately) ──
    # These patches can be fed to the official Docker eval script later.
    # Set DOCKERHUB_USER in .env and clone swebench_pro_ref/ to use it.
    if _IS_SWEBENCH and swebench_patches:
        patches_out = os.path.join("results", f"swebench_patches_{ts}.json")
        with open(patches_out, "w", encoding="utf-8") as f:
            json.dump(swebench_patches, f, indent=2)
        print(f"  Patches saved  → {patches_out}")
        print(
            "  NOTE: status=COMPLETED means heuristic checks passed only.\n"
            "  For ground-truth verification run Docker eval:\n"
            f"    python docker_eval.py --patches {patches_out} --merge --csv {csv_out}\n"
            f"  Or pre-fetch images first:\n"
            f"    python docker_eval.py --patches {patches_out} --pull-only"
        )

    _print_summary(all_reports)
    tracker.print_summary()


def _print_summary(reports: list):
    runs = max((r.get("run_number", 1) for r in reports), default=1)
    print(f"\n{'═'*80}")
    print(f"  SUMMARY: MAS vs SINGLE - {runs} RUN(S) PER MODEL+MODE")
    print(f"{'═'*80}")

    # ── Pass@k + Resolved Rate ────────────────────────────────────────
    pass_metrics = compute_pass_at_k(reports)
    print(f"\n  {'─'*76}")
    print(f"  RESOLVED RATE & PASS@K")
    print(f"  {'─'*76}")
    print(format_pass_summary(pass_metrics))
    print(f"  {'─'*76}")

    g = defaultdict(list)
    for r in reports:
        g[(r["model_name"], r["mode"])].append(r)

    for (model, mode), items in sorted(g.items()):
        # Group by run number
        by_run = defaultdict(list)
        for r in items:
            run = r.get("run_number", 0)
            by_run[run].append(r)

        # Print header
        print(f"\n  📊 {model.upper()} - {mode.upper()}")
        print(f"  {'─'*76}")

        all_passed = []
        all_times = []
        all_patch_scores = []
        all_codebleu = []

        for run_num in sorted(by_run.keys()):
            run_items = by_run[run_num]
            passed = sum(1 for r in run_items if r["overall_status"] in ("PASSED", "COMPLETED"))
            avg_t = sum(r["duration_seconds"] for r in run_items) / len(run_items) if run_items else 0
            avg_ps = (
                sum(r.get("patch_score", 0.0) for r in run_items) / len(run_items)
                if run_items else 0.0
            )
            avg_cb = (
                sum(r.get("codebleu", 0.0) for r in run_items) / len(run_items)
                if run_items else 0.0
            )

            all_passed.append(passed)
            all_times.append(avg_t)
            all_patch_scores.append(avg_ps)
            all_codebleu.append(avg_cb)

            if _IS_SWEBENCH:
                by_diff = defaultdict(lambda: [0, 0])
                for r in run_items:
                    d = r.get("difficulty", "?")
                    by_diff[d][1] += 1
                    if r["overall_status"] in ("PASSED", "COMPLETED"):
                        by_diff[d][0] += 1
                diff_str = "  ".join(
                    f"{d}:{p}/{t}" for d, (p, t) in sorted(by_diff.items())
                )
                print(
                    f"    Run {run_num}  │  {passed}/{len(run_items)} ok  │  "
                    f"patch_score={avg_ps:.2f}  codebleu={avg_cb:.4f}  │  "
                    f"[{diff_str}]  │  avg {avg_t:.1f}s"
                )
            else:
                print(
                    f"    Run {run_num}  │  {passed}/{len(run_items)} passed  │  "
                    f"avg {avg_t:.1f}s"
                )

        avg_passed  = sum(all_passed) / len(all_passed) if all_passed else 0
        avg_time    = sum(all_times) / len(all_times) if all_times else 0
        avg_ps_all  = sum(all_patch_scores) / len(all_patch_scores) if all_patch_scores else 0
        avg_cb_all  = sum(all_codebleu) / len(all_codebleu) if all_codebleu else 0
        print(f"  {'─'*76}")
        print(
            f"    Average  │  {avg_passed:.1f} tasks ok  │  "
            f"patch_score={avg_ps_all:.2f}  codebleu={avg_cb_all:.4f}  │  avg {avg_time:.1f}s"
        )

        # ── Difficulty breakdown ──────────────────────────────────
        if _IS_SWEBENCH:
            by_diff: dict = defaultdict(lambda: {"ok": 0, "total": 0, "ps": [], "fm": []})
            for r in items:
                d = r.get("difficulty", "?")
                by_diff[d]["total"] += 1
                if r.get("overall_status") in ("PASSED", "COMPLETED"):
                    by_diff[d]["ok"] += 1
                by_diff[d]["ps"].append(float(r.get("patch_score", 0)))
                fm = r.get("failure_mode", "")
                if fm:
                    by_diff[d]["fm"].append(fm)
            print(f"    {'─'*72}")
            print(f"    Difficulty breakdown:")
            for diff in ("easy", "medium", "hard"):
                if diff not in by_diff:
                    continue
                dv = by_diff[diff]
                avg_dps = sum(dv["ps"]) / len(dv["ps"]) if dv["ps"] else 0
                top_fm = max(set(dv["fm"]), key=dv["fm"].count) if dv["fm"] else "—"
                print(
                    f"      {diff:6}  {dv['ok']}/{dv['total']} ok  "
                    f"patch_score={avg_dps:.3f}  top_failure={top_fm}"
                )

    print(f"{'═'*80}\n")


if __name__ == "__main__":
    main()
    # # One-time shutdown after experiment completion
    # import os
    # os.system("shutdown /s /t 0")