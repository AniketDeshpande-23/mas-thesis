from __future__ import annotations
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from crewai import Crew, Process, Task

from agents.dataset import TaskEntry
from agents.models import BaseLLM
from agents.planner import make_planner
from agents.developer import make_developer, make_developer_refine, make_single_developer
from agents.tester import make_tester
from agents.debugger import make_debugger
from agents.reviewer import make_reviewer
from agents.tools import build_repo_tools
from agents.bug_patterns import classify_issue, format_hints, PatternAnalysisTool, PatchAnalyzerTool
from agents.schemas import TesterVerdict, ReviewerDecision
from validation.swebench_pro_validator import _extract_patch

try:
    from docker_eval import evaluate_patch as _docker_evaluate
except ImportError:
    _docker_evaluate = None  # type: ignore


# ─────────────────────────────────────────────────────────────
# TOOL SWITCH
# Set USE_AGENT_TOOLS = True only when BOTH of these are ready:
#   1. MCP server is running  (python mcp_server/dataset_server.py)
#   2. Repos are cloned       (./repos/<org>/<repo>)
# ─────────────────────────────────────────────────────────────
USE_AGENT_TOOLS = True

# Static Tester fallback iteration cap (used when Docker is unavailable).
MAX_DEBUG_ITERATIONS = 2

# Docker convergence loop cap — iterate Debugger→DevRefine until resolved or stuck.
MAX_ITERATIONS = 3

# Reform 5: difficulty-adaptive iteration caps
_ITER_CAP = {"easy": 2, "medium": 3, "hard": 4}

# HARD_TASK_LLM disabled — superseded by AGENT_LLM_MAP (all roles now explicitly assigned)
HARD_TASK_LLM: "BaseLLM | None" = None

# Heterogeneous model routing — different models for different agent roles.
# Set to None to use the default llm for all roles (backward-compatible).
from agents.models import (
    ClaudeOpusOpenRouter           as _ClaudeOpus,
    Gemini25FlashTesterOpenRouter  as _GeminiPro,
    GeminiFlashLiteOpenRouter      as _FlashLite,
    Qwen3CoderNext                 as _Qwen3Coder,
)
AGENT_LLM_MAP: "dict[str, BaseLLM] | None" = {
    "planner":    _ClaudeOpus(),    # strong reasoning — verified file paths + accurate plans
    "developer":  _Qwen3Coder(),   # coding specialist — 2s per diff, exact Go/Python/JS output
    "debugger":   _FlashLite(),    # pattern diagnosis — fast is sufficient
    "dev_refine": _Qwen3Coder(),   # surgical code corrections — same specialist as Developer
    "tester":     _GeminiPro(),    # static code review — stronger (no-Docker fallback only)
    "reviewer":   _ClaudeOpus(),   # final quality gate — strong reasoning
}


def _resolve_llm(role: str, task_llm: "BaseLLM") -> "BaseLLM":
    """Return per-role LLM from AGENT_LLM_MAP if configured, else fall back to task_llm."""
    if AGENT_LLM_MAP and role in AGENT_LLM_MAP:
        return AGENT_LLM_MAP[role]
    return task_llm


_MINI_CREW_RETRIES = 2
_RETRY_DELAY_SECS  = 10

_PATCH_REMINDER = (
    "Output ONLY a valid unified git diff — no prose, no markdown fences.\n"
    "Required structure:\n"
    "  diff --git a/<file> b/<file>\n"
    "  --- a/<file>\n"
    "  +++ b/<file>\n"
    "  @@ -N,M +N,M @@\n"
    "  -removed line\n"
    "  +added line\n"
    "The patch MUST start with 'diff --git'."
)

_DIFFICULTY_RE = re.compile(r"difficulty:\s*(\w+)", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def is_valid_patch(patch: str) -> bool:
    return (
        isinstance(patch, str)
        and patch.startswith("diff --git")
        and "--- a/" in patch
        and "+++ b/" in patch
        and "@@" in patch
    )


def _patch_converged(a: str, b: str, threshold: float = 0.95) -> bool:
    """Reform 3: Jaccard token similarity — True if patches are functionally identical."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def _get_difficulty(task: TaskEntry) -> str:
    m = _DIFFICULTY_RE.search(task.signature or "")
    return m.group(1).lower() if m else "medium"


def _parse_tester_verdict(output: str) -> str:
    """Regex fallback — extract PASS or FAIL from prose tester output."""
    for line in output.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("VERDICT:"):
            verdict = stripped.split(":", 1)[1].strip()
            if "PASS" in verdict:
                return "PASS"
            if "FAIL" in verdict:
                return "FAIL"
    return "FAIL"


def _parse_reviewer_decision(output: str) -> bool:
    """Regex fallback — True only if 'Approved: True' appears on its own line."""
    m = re.search(r"^approved:\s*(true|false)", output, re.IGNORECASE | re.MULTILINE)
    return m is not None and m.group(1).lower() == "true"


def _fallback_tester_verdict(raw: str) -> TesterVerdict:
    """Parse prose tester output into TesterVerdict when output_pydantic fails."""
    verdict = _parse_tester_verdict(raw)
    issues: list[str] = []
    in_issues = False
    for line in raw.splitlines():
        if line.strip().upper().startswith("ISSUES:"):
            in_issues = True
            rest = line.split(":", 1)[1].strip()
            if rest and rest.lower() != "none":
                issues.append(rest)
        elif in_issues and line.strip().startswith("-"):
            issues.append(line.strip().lstrip("- ").strip())
    return TesterVerdict(
        verdict=verdict,
        issues=issues,
        failure_types=classify_issue(raw),
    )


# ─────────────────────────────────────────────────────────────
# Mini-crew runner (used by single mode and Docker MAS phase)
# ─────────────────────────────────────────────────────────────

def _run_mini_crew(agent, task: Task, call_log: list) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(_MINI_CREW_RETRIES + 1):
        t0 = time.time()
        try:
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )
            crew.kickoff()
            call_log.append(round((time.time() - t0) * 1000, 1))
            return task.output.raw if task.output else ""

        except ValueError as exc:
            call_log.append(round((time.time() - t0) * 1000, 1))
            if "none or empty" in str(exc).lower():
                if attempt < _MINI_CREW_RETRIES:
                    print(
                        f"  [RETRY {attempt + 1}/{_MINI_CREW_RETRIES}] "
                        f"Empty LLM response — retrying in {_RETRY_DELAY_SECS}s..."
                    )
                    time.sleep(_RETRY_DELAY_SECS)
                    last_exc = exc
                else:
                    print(f"  [WARN] Empty response after {_MINI_CREW_RETRIES + 1} attempts — task will be marked ERROR")
                    return ""
            else:
                raise

        except Exception as exc:
            call_log.append(round((time.time() - t0) * 1000, 1))
            last_exc = exc
            if attempt < _MINI_CREW_RETRIES:
                print(
                    f"  [RETRY {attempt + 1}/{_MINI_CREW_RETRIES}] "
                    f"Crew failed: {exc}. Waiting {_RETRY_DELAY_SECS}s..."
                )
                time.sleep(_RETRY_DELAY_SECS)
            else:
                raise last_exc


def _extract_file_hints(task: TaskEntry) -> str:
    """Extract plausible file paths from the problem description and test names."""
    import re
    hints: set[str] = set()

    ext_re   = re.compile(r'[\w/.-]+\.(?:py|go|js|ts|java|rb|rs|yaml|yml|toml)\b')
    tick_re  = re.compile(r'`([^`\n]{4,80})`')
    test_re  = re.compile(r'([\w/.-]+\.(?:py|go|js|ts))\s*(?::|::|\|)')

    for src in (task.description or "", task.test_cases or "", task.signature or ""):
        for m in ext_re.findall(src):
            if len(m) > 5 and "/" in m:
                hints.add(m)
        for m in tick_re.findall(src):
            if "/" in m or m.endswith((".py", ".go", ".js", ".ts")):
                hints.add(m.strip())
    for m in test_re.findall(task.test_cases or ""):
        hints.add(m)

    if not hints:
        return ""
    lines = "\n".join(f"  - {h}" for h in sorted(hints)[:12])
    return f"Likely relevant files (inferred from issue and test names):\n{lines}"


def _docker_eval_helper(instance_id: str, patch: str, meta: dict, call_log: list) -> dict:
    """Run Docker evaluation and record latency. Returns the evaluate_patch result dict."""
    _empty = {"resolved": False, "docker_status": "error", "fail_output_tail": "",
              "fail_to_pass_passed": False, "fail_to_pass_count": 0, "error": ""}

    if _docker_evaluate is None:
        return {**_empty, "error": "docker_eval module not importable"}
    if not patch or not is_valid_patch(patch):
        return {**_empty, "docker_status": "skipped_invalid_patch", "error": "invalid patch"}

    t0 = time.time()
    try:
        dr = _docker_evaluate(instance_id, patch, meta, verbose=False)
    except Exception as exc:
        call_log.append(round((time.time() - t0) * 1000, 1))
        print(f"  [Docker] Eval failed: {exc}")
        return {**_empty, "error": str(exc)}

    call_log.append(round((time.time() - t0) * 1000, 1))
    status = "resolved" if dr.get("resolved") else "failed"
    fp = dr.get("fail_to_pass_passed", "?")
    fc = dr.get("fail_to_pass_count", "?")
    print(f"  [Docker] {instance_id}: {status}  | fail_to_pass: {fp}/{fc}")
    return dr


@dataclass
class PipelineResult:
    task_id: str
    model_name: str
    mode: str
    status: str
    backend: str
    final_code: str
    initial_code: str
    debug_iterations: int
    patch_changed_by_debug: bool
    tester_approved: bool
    tester_pass_iteration: int
    max_debug_iterations: int
    reviewer_approved: bool
    reviewer_feedback: str
    duration_seconds: float
    llm_calls: int
    total_latency_ms: float
    avg_latency_ms: float
    pattern_matched: str = ""
    error: Optional[str] = None
    docker_resolved: bool = False
    docker_status: str = "not_run"
    docker_fail_output: str = ""
    agent_model_breakdown: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


def _get_repo_from_task(task: TaskEntry) -> Optional[str]:
    if not task.signature:
        return None
    for line in task.signature.splitlines():
        if line.lower().startswith("repository:"):
            return line.split(":", 1)[1].strip()
    return None


def _safe_build_tools(task: TaskEntry) -> List:
    if not USE_AGENT_TOOLS:
        return []
    try:
        repo = _get_repo_from_task(task)
        if not repo:
            return []
        # Use repo-only tools (RepoSearch, RepoFileRead, RepoFileExists).
        # MCP tools (GetInstanceTool, RunTestsTool) require a running MCP server
        # and waste agent tool-call budget when the server is not available.
        return build_repo_tools(repo)
    except Exception as exc:
        print(f"[WARN] Tool build failed for {task.task_id}: {exc}")
        return []


def _safe_build_planner_tools(task: TaskEntry) -> List:
    """Repo search/read tools for Planner — verify file paths before writing the plan."""
    if not USE_AGENT_TOOLS:
        return []
    try:
        repo = _get_repo_from_task(task)
        if not repo:
            return []
        return build_repo_tools(repo)
    except Exception as exc:
        print(f"[WARN] Planner tool build failed for {task.task_id}: {exc}")
        return []


# ─────────────────────────────────────────────────────────────
# MAS pipeline
#
# Docker mode (meta is not None):
#   Phase 1: Planner + Developer (single Crew)
#   Phase 2: Docker eval
#   Phase 3: (if failed) Debugger mini-crew → DevRefine mini-crew → Docker eval 2
#   Phase 4: Reviewer mini-crew
#
# Static mode (meta is None):
#   All 7 agents run in ONE Crew.kickoff() with context chains.
# ─────────────────────────────────────────────────────────────

def _run_mas(task: TaskEntry, llm: BaseLLM, meta: dict | None = None) -> tuple:
    tools      = _safe_build_tools(task)
    difficulty = _get_difficulty(task)
    # Reform 7c: use stronger model for hard tasks if configured
    if difficulty == "hard" and HARD_TASK_LLM is not None:
        llm = HARD_TASK_LLM
        print(f"  [Reform 7c] Hard task — using boosted LLM: {llm.name}")
    call_log: list = []

    full_context = (
        f"{task.signature or ''}\n\n"
        f"## Problem\n{task.description or ''}\n\n"
        f"## Tests that must pass after the fix\n{task.test_cases or ''}"
    )

    planner_tools = _safe_build_planner_tools(task)   # Reform 6: Planner verifies paths via RepoSearch
    planner   = make_planner(_resolve_llm("planner", llm), planner_tools, difficulty=difficulty)
    developer = make_developer(_resolve_llm("developer", llm), tools, difficulty=difficulty)
    file_hints = _extract_file_hints(task)
    _test_src  = (((meta or {}).get("test_patch", "") or ""))[:2000]

    t_plan = Task(
        description=(
            f"{full_context}\n\n"
            + (f"{file_hints}\n\n" if file_hints else "")
            + "Produce a numbered implementation plan that will fix the bug.\n"
            "Each step must include a file path, function name, and specific change.\n"
            "Do NOT write any code."
        ),
        expected_output="Numbered implementation plan (steps only, no code).",
        agent=planner,
    )

    t_dev = Task(
        description=(
            "Implement the Planner's fix by producing a valid unified git diff.\n\n"
            + (f"{file_hints}\n\n" if file_hints else "")
            + (f"Failing test source (make these tests pass):\n{_test_src}\n\n" if _test_src else "")
            + f"{_PATCH_REMINDER}"
        ),
        expected_output="Valid unified git diff starting with 'diff --git'.",
        context=[t_plan],
        agent=developer,
    )

    # ══════════════════════════════════════════════════════════════
    # DOCKER MODE
    # ══════════════════════════════════════════════════════════════
    if meta is not None:

        # Phase 1: Planner + Developer
        t0 = time.time()
        phase1_crew = Crew(
            agents=[planner, developer],
            tasks=[t_plan, t_dev],
            process=Process.sequential,
            verbose=True,
        )
        try:
            phase1_crew.kickoff()
        except Exception as exc:
            print(f"  [WARN] Phase1 crew error: {exc} — extracting partial results.")
        call_log.append(round((time.time() - t0) * 1000, 1))

        initial_patch = _extract_patch(
            t_dev.output.raw if t_dev.output else ""
        ) or ""
        current_patch      = initial_patch
        patch_changed      = False
        docker_resolved    = False
        docker_fail_output = ""
        debug_iterations   = 0
        tester_pass_iteration = -1

        # Phase 2: Convergence loop — Docker eval → Debugger → DevRefine → repeat
        # Reform 5: difficulty-adaptive cap (easy=2, medium=3, hard=4)
        _iter_cap  = _ITER_CAP.get(difficulty, MAX_ITERATIONS)
        # Fix 7: lower Jaccard threshold for hard tasks (small but critical changes e.g. adding an
        # import should NOT be mistaken for convergence with the prior patch)
        _conv_thresh = 0.85 if difficulty == "hard" else 0.95
        fail_tests = ", ".join(meta.get("fail_to_pass", []))
        prev_docker_status = ""
        for _iter in range(_iter_cap):
            dr = _docker_eval_helper(task.task_id, current_patch, meta, call_log)
            docker_resolved    = dr.get("resolved", False)
            docker_status_code = dr.get("docker_status", "")
            _new_tail          = dr.get("fail_output_tail", "")
            if _new_tail:
                docker_fail_output = _new_tail

            if docker_resolved:
                tester_pass_iteration = _iter + 1
                print(f"  [MAS-Docker] RESOLVED on iteration {_iter + 1}/{_iter_cap}")
                break

            if _iter == _iter_cap - 1:
                print(f"  [MAS-Docker] Cap reached ({_iter + 1}/{_iter_cap}) — going to Reviewer.")
                break

            # Fix 2: zero-score recovery — if initial patch is invalid, run DevRefine to salvage
            # instead of breaking immediately (prevents 0.000 catastrophic failures)
            _patch_invalid = not is_valid_patch(current_patch)

            # Reform 2: delta context for Debugger
            _status_change = (
                f"Docker status: {prev_docker_status} → {docker_status_code}"
                if prev_docker_status else f"Docker status: {docker_status_code}"
            )

            if _patch_invalid:
                # Fix 2: zero-score recovery — Developer produced no valid patch.
                # Skip Debugger (nothing to diagnose), send DevRefine a fresh-start prompt.
                print(f"  [Fix2] Invalid initial patch on iter {_iter + 1} — DevRefine salvage attempt.")
                diagnosis = "No valid patch was produced. Write a new patch from scratch using the task description and failing test source."
            else:
                real_errors = docker_fail_output
                if docker_status_code == "apply_failed":
                    diagnose_desc = (
                        f"The patch FAILED TO APPLY inside Docker — the file paths are WRONG.\n"
                        f"{_status_change}\n\n"
                        f"Apply error:\n{real_errors}\n\n"
                        f"Current patch header lines:\n{current_patch[:400]}\n\n"
                        + (f"{file_hints}\n\n" if file_hints else "")
                        + (f"Failing test source:\n{_test_src}\n\n" if _test_src else "")
                        + f"Task test context:\n{task.test_cases}\n\n"
                        "Call 'Patch Analyzer Tool' to see the wrong file paths.\n"
                        "Identify the CORRECT file path from the hints and test names above.\n"
                        "Write DIAGNOSIS (wrong path → correct path) + FIX PLAN (change the diff --git header)."
                    )
                else:
                    diagnose_desc = (
                        f"Docker test execution FAILED on this patch (iteration {_iter + 1}/{_iter_cap}).\n"
                        f"{_status_change}\n\n"
                        f"Tests that must pass: {fail_tests}\n\n"
                        f"Actual test output (from Docker):\n{real_errors}\n\n"
                        f"Current patch:\n{current_patch}\n\n"
                        + (f"{file_hints}\n\n" if file_hints else "")
                        + (f"Failing test source:\n{_test_src}\n\n" if _test_src else "")
                        + f"Task test context:\n{task.test_cases}\n\n"
                        "Call 'Pattern Analysis Tool' with the test failure output above.\n"
                        "Call 'Patch Analyzer Tool' with the current patch.\n"
                        "Then write DIAGNOSIS + FIX PLAN."
                    )
                debugger = make_debugger(_resolve_llm("debugger", llm), tools)
                t_diagnose = Task(
                    description=diagnose_desc,
                    expected_output="Diagnosis Report with DIAGNOSIS and FIX PLAN.",
                    agent=debugger,
                )
                diagnosis = _run_mini_crew(debugger, t_diagnose, call_log)

            # Fix 3: when initial patch is invalid use fresh Developer (not surgical DevRefine)
            # Reform 2: patch delta context for DevRefine
            _prev_patch_summary = (
                f"Previous patch (iter {_iter}):\n{current_patch[:300]}...\n\n"
                if (_iter > 0 and not _patch_invalid) else ""
            )
            if _patch_invalid:
                # No existing patch to surgically correct — fresh Developer write
                _fixer = make_developer(_resolve_llm("developer", llm), tools, difficulty=difficulty)
                _refine_desc = (
                    f"Original problem:\n{task.description}\n\n"
                    + (f"{file_hints}\n\n" if file_hints else "")
                    + (f"Failing test source:\n{_test_src}\n\n" if _test_src else "")
                    + f"Note: {diagnosis}\n\n"
                    "Write a complete new patch from scratch.\n\n"
                    f"{_PATCH_REMINDER}"
                )
            else:
                _fixer = make_developer_refine(_resolve_llm("dev_refine", llm), tools)
                _refine_desc = (
                    f"Original problem:\n{task.description}\n\n"
                    + (f"{file_hints}\n\n" if file_hints else "")
                    + (f"Failing test source:\n{_test_src}\n\n" if _test_src else "")
                    + _prev_patch_summary
                    + f"Current patch (iteration {_iter + 1}, failed Docker tests):\n{current_patch}\n\n"
                    f"Docker status progression: {_status_change}\n\n"
                    + f"Debugger diagnosis:\n{diagnosis}\n\n"
                    "Apply SURGICAL corrections only. Keep all correct parts unchanged.\n\n"
                    f"{_PATCH_REMINDER}"
                )
            t_refine = Task(
                description=_refine_desc,
                expected_output="Valid unified git diff starting with 'diff --git'.",
                agent=_fixer,
            )
            refine_output = _run_mini_crew(_fixer, t_refine, call_log)
            new_patch = _extract_patch(refine_output)
            debug_iterations += 1
            prev_docker_status = docker_status_code

            # Fix 2: distinguish empty/invalid DevRefine output from genuine convergence.
            # Fix 7: use difficulty-adaptive Jaccard threshold (_conv_thresh).
            if not new_patch or not is_valid_patch(new_patch):
                print(f"  [WARN] Refine produced no valid patch on iter {_iter + 1} — retaining current patch.")
                # Don't break — let next iteration run Docker eval on existing patch
            elif _patch_converged(new_patch, current_patch, threshold=_conv_thresh):
                print(f"  [Early exit] Patch converged on iter {_iter + 1} (Jaccard≥{_conv_thresh}) — stopping MAS loop.")
                break
            else:
                current_patch = new_patch
                patch_changed = True

        # Phase 4: Reviewer — skip if Docker already resolved (tests passed = patch is correct)
        if docker_resolved:
            review_obj = ReviewerDecision(
                approved=True,
                feedback="Docker tests passed — auto-approved.",
                concerns=[],
            )
            print("  [MAS-Docker] Resolved=True — Reviewer skipped (auto-approved).")
        else:
            reviewer = make_reviewer(_resolve_llm("reviewer", llm), tools)
            t_review = Task(
                description=(
                    "Review the final patch. Docker tests did NOT pass.\n\n"
                    f"Final patch:\n{current_patch}\n\n"
                    f"Original problem:\n{task.description or ''}\n\n"
                    "Approve only if the patch clearly fixes the described bug,\n"
                    "is a valid unified diff, and does not break unrelated code.\n\n"
                    "Reply EXACTLY:\nApproved: True\nFeedback: <one sentence>\n\n"
                    "or:\nApproved: False\nFeedback: <reason>"
                ),
                expected_output="Approved: True/False\nFeedback: <one sentence>",
                agent=reviewer,
            )
            _run_mini_crew(reviewer, t_review, call_log)
            raw_review = t_review.output.raw if t_review.output else ""
            review_obj = (
                (t_review.output.pydantic if t_review.output and t_review.output.pydantic else None)
                or ReviewerDecision(
                    approved=_parse_reviewer_decision(raw_review),
                    feedback=raw_review[:200],
                    concerns=[],
                )
            )
            print(
                f"  [MAS-Docker] Resolved=False  "
                f"Reviewer={'APPROVED' if review_obj.approved else 'REJECTED'}"
            )

        all_patterns = classify_issue(docker_fail_output)
        return (
            current_patch, initial_patch,
            debug_iterations, patch_changed,
            docker_resolved, tester_pass_iteration,
            review_obj.approved, review_obj.feedback[:300],
            call_log, format_hints(all_patterns),
            docker_resolved, docker_fail_output,
        )

    # ══════════════════════════════════════════════════════════════
    # STATIC MODE: existing single-Crew flow (fallback when no Docker)
    # ══════════════════════════════════════════════════════════════
    tester_a   = make_tester(_resolve_llm("tester", llm), tools)
    debugger   = make_debugger(_resolve_llm("debugger", llm), tools)
    dev_refine = make_developer_refine(_resolve_llm("dev_refine", llm), tools)
    tester_b   = make_tester(_resolve_llm("tester", llm), tools)
    reviewer   = make_reviewer(_resolve_llm("reviewer", llm), tools)

    t_test1 = Task(
        description=(
            f"Review the Developer's patch for this problem:\n\n{full_context}\n\n"
            "Check: correct files, root cause addressed, logic errors, diff format, missing imports.\n"
            "Reply VERDICT: PASS/FAIL and ISSUES."
        ),
        expected_output="VERDICT: PASS or FAIL with ISSUES list.",
        output_pydantic=TesterVerdict,
        context=[t_plan, t_dev],
        agent=tester_a,
    )

    t_diagnose = Task(
        description=(
            f"Failing tests for context:\n{task.test_cases}\n\n"
            "Call PatternAnalysisTool and PatchAnalyzerTool, then write:\n"
            "DIAGNOSIS: root_cause, failure_class, current_files, correct_files\n"
            "FIX PLAN: numbered steps (no code, no diff)."
        ),
        expected_output="Diagnosis Report with DIAGNOSIS and FIX PLAN sections.",
        context=[t_plan, t_dev, t_test1],
        agent=debugger,
    )

    t_refine = Task(
        description=(
            f"Original problem:\n{task.description}\n\n"
            "Re-implement the patch following the Debugger's diagnosis and fix plan.\n\n"
            f"{_PATCH_REMINDER}"
        ),
        expected_output="Valid unified git diff starting with 'diff --git'.",
        context=[t_plan, t_dev, t_test1, t_diagnose],
        agent=dev_refine,
    )

    t_test2 = Task(
        description=(
            f"Review the revised patch for this problem:\n\n{full_context}\n\n"
            "If a prior PASS verdict is visible in context, output PASS with no issues.\n"
            "Otherwise check the revised patch and reply VERDICT: PASS/FAIL and ISSUES."
        ),
        expected_output="VERDICT: PASS or FAIL with ISSUES list.",
        output_pydantic=TesterVerdict,
        context=[t_plan, t_refine],
        agent=tester_b,
    )

    t_review = Task(
        description=(
            "Review the final patch. Approve only if it clearly fixes the described bug,\n"
            "is a valid unified diff, and does not break unrelated code.\n\n"
            "Reply EXACTLY:\nApproved: True\nFeedback: <one sentence>\n\n"
            "or:\nApproved: False\nFeedback: <reason>"
        ),
        expected_output="Approved: True/False\nFeedback: <one sentence>",
        context=[t_plan, t_dev, t_test1, t_diagnose, t_refine, t_test2],
        agent=reviewer,
    )

    # Single Crew kickoff — all agents, shared memory
    t0 = time.time()
    crew = Crew(
        agents=[planner, developer, tester_a, debugger, dev_refine, tester_b, reviewer],
        tasks=[t_plan, t_dev, t_test1, t_diagnose, t_refine, t_test2, t_review],
        process=Process.sequential,
        verbose=True,
    )
    try:
        crew.kickoff()
    except Exception as exc:
        print(f"  [WARN] Crew kickoff error: {exc} — extracting partial results.")
    call_log.append(round((time.time() - t0) * 1000, 1))

    # Extract patch outputs
    initial_patch = _extract_patch(
        t_dev.output.raw if t_dev.output else ""
    ) or ""
    refined_raw   = t_refine.output.raw if t_refine.output else ""
    refined_patch = _extract_patch(refined_raw)
    current_patch = (
        refined_patch if (refined_patch and is_valid_patch(refined_patch))
        else initial_patch
    )

    # Typed tester verdicts (with fallback)
    raw1 = t_test1.output.raw if t_test1.output else ""
    raw2 = t_test2.output.raw if t_test2.output else ""
    v1: TesterVerdict = t_test1.output.pydantic or _fallback_tester_verdict(raw1)
    v2: TesterVerdict = t_test2.output.pydantic or _fallback_tester_verdict(raw2)

    tester_approved      = v1.verdict == "PASS" or v2.verdict == "PASS"
    tester_pass_iteration = (
        0 if v1.verdict == "PASS" else
        1 if v2.verdict == "PASS" else -1
    )
    patch_changed = bool(current_patch) and current_patch != initial_patch

    # Typed reviewer decision (with fallback)
    raw_review = t_review.output.raw if t_review.output else ""
    review_obj: ReviewerDecision = t_review.output.pydantic or ReviewerDecision(
        approved=_parse_reviewer_decision(raw_review),
        feedback=raw_review[:200],
        concerns=[],
    )

    print(
        f"  [MAS] Tester1={v1.verdict}  Tester2={v2.verdict}  "
        f"Reviewer={'APPROVED' if review_obj.approved else 'REJECTED'}"
    )

    # Reviewer feedback loop — extra Developer pass if rejected
    if not review_obj.approved and current_patch:
        concerns_text = "; ".join(review_obj.concerns) if review_obj.concerns else "see feedback"
        dev_final = make_developer_refine(_resolve_llm("dev_refine", llm), tools)
        t_final = Task(
            description=(
                f"Reviewer rejected the patch.\n"
                f"Reason: {review_obj.feedback}\n"
                f"Concerns: {concerns_text}\n\n"
                f"Original problem:\n{task.description}\n\n"
                "Revise the patch to address the reviewer's concerns.\n\n"
                f"{_PATCH_REMINDER}"
            ),
            expected_output="Valid unified git diff starting with 'diff --git'.",
            context=[t_plan, t_review],
            agent=dev_final,
        )
        t0b = time.time()
        feedback_crew = Crew(
            agents=[dev_final], tasks=[t_final],
            process=Process.sequential,
            verbose=True,
        )
        try:
            feedback_crew.kickoff()
        except Exception as exc:
            print(f"  [WARN] Feedback crew error: {exc}")
        call_log.append(round((time.time() - t0b) * 1000, 1))
        revised = _extract_patch(t_final.output.raw if t_final.output else "")
        if revised and is_valid_patch(revised):
            current_patch = revised
            print("  [Reviewer loop] Patch revised after rejection.")

    # Pattern matching from tester verdicts
    all_patterns: list[str] = []
    for v in [v1, v2]:
        if v.failure_types:
            all_patterns.extend(f for f in v.failure_types if f not in all_patterns)
    if not all_patterns:
        combined = " ".join(i for v in [v1, v2] for i in (v.issues or []))
        all_patterns = classify_issue(combined)

    return (
        current_patch, initial_patch,
        1, patch_changed,
        tester_approved, tester_pass_iteration,
        review_obj.approved, review_obj.feedback[:300],
        call_log, format_hints(all_patterns),
        False, "",  # docker_resolved, docker_fail_output — not used in static mode
    )


# ─────────────────────────────────────────────────────────────
# Single-agent pipeline — mini-crew per step
# ─────────────────────────────────────────────────────────────

def _run_single(task: TaskEntry, llm: BaseLLM, meta: dict | None = None) -> tuple:
    tools      = _safe_build_tools(task)
    difficulty = _get_difficulty(task)
    # Reform 7c: use stronger model for hard tasks if configured
    if difficulty == "hard" and HARD_TASK_LLM is not None:
        llm = HARD_TASK_LLM
        print(f"  [Reform 7c] Hard task — using boosted LLM: {llm.name}")

    full_context = (
        f"{task.signature or ''}\n\n"
        f"## Problem\n{task.description or ''}\n\n"
        f"## Tests that must pass after the fix\n{task.test_cases or ''}"
    )

    developer  = make_single_developer(llm, tools, difficulty=difficulty)
    file_hints = _extract_file_hints(task)
    _test_src  = (((meta or {}).get("test_patch", "") or ""))[:2000]
    t_dev = Task(
        description=(
            f"{full_context}\n\n"
            + (f"{file_hints}\n\n" if file_hints else "")
            + (f"Failing test source (make these tests pass):\n{_test_src}\n\n" if _test_src else "")
            + "Analyse the problem, locate the root cause, and produce a patch.\n\n"
            f"{_PATCH_REMINDER}"
        ),
        expected_output="Valid unified git diff patch starting with 'diff --git'.",
        agent=developer,
    )
    call_log: list = []
    dev_output    = _run_mini_crew(developer, t_dev, call_log)
    initial_patch = _extract_patch(dev_output) or ""
    current_patch = initial_patch

    tester_approved       = False
    tester_pass_iteration = -1
    debug_iterations      = 0
    patch_changed         = False
    all_matched_patterns: list[str] = []
    docker_resolved    = False
    docker_fail_output = ""

    if meta is not None:
        # ── Docker convergence loop ───────────────────────────────────
        # Reform 5: difficulty-adaptive cap (easy=2, medium=3, hard=4)
        _iter_cap    = _ITER_CAP.get(difficulty, MAX_ITERATIONS)
        _conv_thresh = 0.85 if difficulty == "hard" else 0.95  # Fix 7
        fail_tests         = ", ".join(meta.get("fail_to_pass", []))
        prev_docker_status = ""
        for _iter in range(_iter_cap):
            dr       = _docker_eval_helper(task.task_id, current_patch, meta, call_log)
            resolved = dr.get("resolved", False)
            docker_status_code = dr.get("docker_status", "")

            if resolved:
                docker_resolved       = True
                tester_approved       = True
                tester_pass_iteration = _iter + 1
                break

            _new_tail = dr.get("fail_output_tail", "")
            if _new_tail:
                docker_fail_output = _new_tail
            all_matched_patterns.extend(classify_issue(docker_fail_output))

            if _iter == _iter_cap - 1:
                print(f"  [Single-Docker] Cap reached ({_iter + 1}/{_iter_cap}).")
                debug_iterations += 1
                break

            # Reform 2: status delta for DevRefine context
            _status_change = (
                f"Docker status: {prev_docker_status} → {docker_status_code}"
                if prev_docker_status else f"Docker status: {docker_status_code}"
            )
            _apply_fail  = docker_status_code == "apply_failed"
            _fail_prefix = (
                "The patch FAILED TO APPLY — the file paths in the patch are WRONG.\n"
                if _apply_fail else
                f"Docker test execution FAILED.\nTests that must pass: {fail_tests}\n"
            )
            # Reform 2: include previous patch snippet for delta awareness
            _prev_patch_summary = (
                f"Previous patch (iter {_iter}):\n{current_patch[:300]}...\n\n"
                if _iter > 0 else ""
            )
            dev_refine = make_single_developer(
                llm, [PatternAnalysisTool(), PatchAnalyzerTool()] + tools
            )
            t_refine = Task(
                description=(
                    f"{_fail_prefix}\n"
                    f"{_status_change}\n\n"
                    f"Actual output:\n{docker_fail_output}\n\n"
                    + (f"{file_hints}\n\n" if file_hints else "")
                    + (f"Failing test source:\n{_test_src}\n\n" if _test_src else "")
                    + _prev_patch_summary
                    + f"Current patch (iteration {_iter + 1}/{_iter_cap}):\n{current_patch}\n\n"
                    f"Problem context:\n{full_context}\n\n"
                    "Use your diagnostic tools to classify the failure, then produce "
                    "an improved patch with correct file paths.\n\n"
                    f"{_PATCH_REMINDER}"
                ),
                expected_output="Valid unified git diff patch starting with 'diff --git'.",
                agent=dev_refine,
            )
            refine_output = _run_mini_crew(dev_refine, t_refine, call_log)
            new_patch = _extract_patch(refine_output)
            debug_iterations += 1
            prev_docker_status = docker_status_code

            # Fix 2: distinguish empty/invalid from genuine convergence.
            # Fix 7: difficulty-adaptive Jaccard threshold.
            if not new_patch or not is_valid_patch(new_patch):
                print(f"  [WARN] Single refine produced no valid patch on iter {_iter + 1} — retaining current.")
            elif _patch_converged(new_patch, current_patch, threshold=_conv_thresh):
                print(f"  [Early exit] Patch converged on iter {_iter + 1} (Jaccard≥{_conv_thresh}) — stopping Single loop.")
                break
            else:
                patch_changed = True
                current_patch = new_patch

    else:
        # ── Static Tester fallback (no Docker) ───────────────────────
        for iteration in range(MAX_DEBUG_ITERATIONS):
            tester = make_tester(llm, tools)
            t_test = Task(
                description=(
                    f"Review this patch for the following problem:\n\n"
                    f"{full_context}\n\n"
                    f"Patch to review:\n{current_patch}\n\n"
                    "Reply with VERDICT: PASS/FAIL and ISSUES as instructed."
                ),
                expected_output="VERDICT: PASS or FAIL with ISSUES list.",
                agent=tester,
            )
            test_output = _run_mini_crew(tester, t_test, call_log)

            v: TesterVerdict = _fallback_tester_verdict(test_output)
            verdict = v.verdict
            print(f"  [Iter {iteration + 1}/{MAX_DEBUG_ITERATIONS}] Tester verdict: {verdict}")

            if verdict == "PASS":
                tester_approved       = True
                tester_pass_iteration = iteration
                break

            if iteration < MAX_DEBUG_ITERATIONS - 1:
                _matched = v.failure_types or classify_issue(test_output)
                all_matched_patterns.extend(m for m in _matched if m not in all_matched_patterns)

                dev_refine = make_single_developer(
                    llm, [PatternAnalysisTool(), PatchAnalyzerTool()] + tools
                )
                t_refine = Task(
                    description=(
                        f"Tester found issues:\n{test_output}\n\n"
                        f"Current patch:\n{current_patch}\n\n"
                        f"Problem context:\n{full_context}\n\n"
                        "Use your diagnostic tools to classify the failure, then produce "
                        "an improved patch.\n\n"
                        f"{_PATCH_REMINDER}"
                    ),
                    expected_output="Valid unified git diff patch starting with 'diff --git'.",
                    agent=dev_refine,
                )
                refine_output = _run_mini_crew(dev_refine, t_refine, call_log)
                new_patch = _extract_patch(refine_output)
                if new_patch and is_valid_patch(new_patch):
                    if new_patch != current_patch:
                        patch_changed = True
                        current_patch = new_patch
                    else:
                        print("  [Early exit] Patch unchanged after refinement — stopping loop.")
                        debug_iterations += 1
                        break
                debug_iterations += 1

    pattern_label = format_hints(all_matched_patterns)

    return (
        current_patch, initial_patch,
        debug_iterations, patch_changed,
        tester_approved, tester_pass_iteration,
        False, "Single-agent mode — no reviewer",
        call_log, pattern_label,
        docker_resolved, docker_fail_output,
    )


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def _latency_stats(call_log: list) -> tuple[int, float, float]:
    n     = len(call_log)
    total = round(sum(call_log), 1)
    avg   = round(total / n, 1) if n else 0.0
    return n, total, avg


def run_pipeline(
    task: TaskEntry,
    llm: BaseLLM,
    mode: str = "mas",
    meta: dict | None = None,
) -> PipelineResult:
    start = time.time()

    try:
        if mode == "mas":
            (
                final_patch, initial_patch,
                debug_iterations, patch_changed,
                tester_approved, tester_pass_iteration,
                reviewer_approved, reviewer_feedback,
                call_log, pattern_label,
                docker_resolved, docker_fail_output,
            ) = _run_mas(task, llm, meta=meta)
        else:
            (
                final_patch, initial_patch,
                debug_iterations, patch_changed,
                tester_approved, tester_pass_iteration,
                reviewer_approved, reviewer_feedback,
                call_log, pattern_label,
                docker_resolved, docker_fail_output,
            ) = _run_single(task, llm, meta=meta)

        llm_calls, total_lat, avg_lat = _latency_stats(call_log)
        duration = round(time.time() - start, 2)
        if mode == "single":
            _breakdown = {"all": llm.name}
        elif AGENT_LLM_MAP:
            _breakdown = {r: m.name for r, m in AGENT_LLM_MAP.items()}
        else:
            _breakdown = {"all": llm.name}
        _breakdown_str = str(_breakdown)
        if meta is None:
            docker_status = "not_run"
        elif docker_resolved:
            docker_status = "resolved"
        else:
            docker_status = "failed"

        if not final_patch or not is_valid_patch(final_patch):
            return PipelineResult(
                task_id=task.task_id, model_name=llm.name, mode=mode,
                status="ERROR", backend=llm.backend,
                final_code=final_patch or "", initial_code=initial_patch or "",
                debug_iterations=debug_iterations, patch_changed_by_debug=patch_changed,
                tester_approved=tester_approved, tester_pass_iteration=tester_pass_iteration,
                max_debug_iterations=MAX_DEBUG_ITERATIONS,
                reviewer_approved=False,
                reviewer_feedback="No valid patch extracted from agent output",
                duration_seconds=duration, llm_calls=llm_calls,
                total_latency_ms=total_lat, avg_latency_ms=avg_lat,
                pattern_matched=pattern_label,
                docker_resolved=docker_resolved,
                docker_status=docker_status,
                docker_fail_output=docker_fail_output,
                agent_model_breakdown=_breakdown_str,
            )

        return PipelineResult(
            task_id=task.task_id, model_name=llm.name, mode=mode,
            status="PASSED" if reviewer_approved else "COMPLETED",
            backend=llm.backend,
            final_code=final_patch, initial_code=initial_patch,
            debug_iterations=debug_iterations, patch_changed_by_debug=patch_changed,
            tester_approved=tester_approved, tester_pass_iteration=tester_pass_iteration,
            max_debug_iterations=MAX_DEBUG_ITERATIONS,
            reviewer_approved=reviewer_approved, reviewer_feedback=reviewer_feedback,
            duration_seconds=duration, llm_calls=llm_calls,
            total_latency_ms=total_lat, avg_latency_ms=avg_lat,
            pattern_matched=pattern_label,
            docker_resolved=docker_resolved,
            docker_status=docker_status,
            docker_fail_output=docker_fail_output,
            agent_model_breakdown=_breakdown_str,
        )

    except Exception as exc:
        print(f"[ERROR] Pipeline failed for {task.task_id}: {repr(exc)}")
        return PipelineResult(
            task_id=task.task_id, model_name=llm.name, mode=mode,
            status="ERROR", backend=llm.backend,
            final_code="", initial_code="",
            debug_iterations=0, patch_changed_by_debug=False,
            tester_approved=False, tester_pass_iteration=-1,
            max_debug_iterations=MAX_DEBUG_ITERATIONS,
            reviewer_approved=False, reviewer_feedback="",
            duration_seconds=round(time.time() - start, 2),
            llm_calls=0, total_latency_ms=0.0, avg_latency_ms=0.0,
            error=str(exc),
            docker_resolved=False,
            docker_status="error",
            docker_fail_output="",
        )
