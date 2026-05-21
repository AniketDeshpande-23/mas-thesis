"""
SWE-bench Pro Validator
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from validation.validator import PipelineResult, ValidationReport

logger = logging.getLogger(__name__)


# =====================================================================
#  Patch extraction from agent output
# =====================================================================


def _looks_like_fake_or_refusal(raw: str, patch: str) -> bool:
    """Heuristic: reject hypothetical or refusal-based patches."""
    text = (raw or "") + "\n" + (patch or "")
    lower = text.lower()

    banned_phrases = [
        "i'm sorry, as an ai model",
        "as an ai model",
        "i don't have access to",
        "i do not have access to",
        "as a language model",
        "hypothetical example",
        "this is a hypothetical patch",
        "might not work as expected",
        "these are placeholders",
        "placeholder",
        "replace `path/to",
        "replace 'path/to",
        "replace path/to",
        "replace the actual path",
    ]
    if any(p in lower for p in banned_phrases):
        return True

    # obvious placeholder paths
    if "path/to/" in text:
        return True

    # format-reminder placeholder — model copied the _PATCH_FORMAT example
    # e.g. "diff --git a/<file> b/<file>" — skip, real diff follows below
    if patch.startswith("diff --git a/<") or "diff --git a/<file>" in patch[:60]:
        return True

    return False


def _extract_patch(raw: str) -> Optional[str]:
    """
    Extract a unified diff patch from the agent's raw output.

    Tries (in order), but filters out refusals / hypothetical patches:
      1. ```diff ... ```  fenced block
      2. ```patch ... ``` fenced block
      3. ``` ... ```      generic fence containing 'diff --git'
      4. Raw text starting from first 'diff --git' line
      5. Raw text starting from first '---' / '+++' pair
    """
    if not raw or not raw.strip():
        return None

    raw = raw.replace("\r\n", "\n")  # normalize Windows line endings
    candidates: List[str] = []

    # 1. ```diff ... ```
    m = re.search(r"```diff\s*([\s\S]+?)```", raw, re.DOTALL)
    if m:
        candidates.append(m.group(1).strip())

    # 2. ```patch ... ```
    m = re.search(r"```patch\s*([\s\S]+?)```", raw, re.DOTALL)
    if m:
        candidates.append(m.group(1).strip())

    # 3. Generic fence containing diff --git
    m = re.search(r"```\s*([\s\S]+?)```", raw, re.DOTALL)
    if m and "diff --git" in m.group(1):
        candidates.append(m.group(1).strip())

    # 4. Raw text from each 'diff --git a/' at line-start (handles template prefix + real diff)
    for m in re.finditer(r"(?m)^diff --git a/", raw):
        candidates.append(raw[m.start():].strip())

    # 5. Raw text from first '---' / '+++' pair (headerless patch)
    m = re.search(r"(---\s+\S+.*?\n\+\+\+\s+\S+.*?\n@@[\s\S]+)", raw)
    if m:
        candidates.append(m.group(1).strip())

    # Pick the first candidate that does not look like a refusal/hypothetical patch
    for patch in candidates:
        if not _looks_like_fake_or_refusal(raw, patch):
            return patch

    return None


# =====================================================================
#  Patch format validation
# =====================================================================


@dataclass
class PatchCheck:
    """Lightweight patch format validation result."""
    is_valid: bool = False
    has_diff_header: bool = False
    num_files_changed: int = 0
    num_hunks: int = 0
    additions: int = 0
    deletions: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _validate_patch_format(patch: str) -> PatchCheck:
    """Check that a string is a well-formed unified diff."""
    result = PatchCheck()

    if not patch or not patch.strip():
        result.errors.append("Patch is empty")
        return result

    lines = patch.split("\n")

    # Diff headers
    diff_headers = [l for l in lines if l.startswith("diff --git ")]
    result.has_diff_header = len(diff_headers) > 0
    result.num_files_changed = len(diff_headers)

    # Hunks
    result.num_hunks = sum(1 for l in lines if l.startswith("@@"))

    # Additions / deletions
    result.additions = sum(
        1 for l in lines if l.startswith("+") and not l.startswith("+++")
    )
    result.deletions = sum(
        1 for l in lines if l.startswith("-") and not l.startswith("---")
    )

    # Binary content
    if re.search(r"^Binary files .* differ$", patch, re.MULTILINE):
        result.warnings.append("Contains binary hunks")
    if re.search(r"^GIT binary patch$", patch, re.MULTILINE):
        result.warnings.append("Contains binary hunks")

    # Merge conflict markers
    if "<<<<<<" in patch or "======" in patch or ">>>>>>" in patch:
        result.errors.append("Contains merge conflict markers")

    # Stronger structural checks
    if not result.has_diff_header:
        result.errors.append("No 'diff --git' header with file paths")
    if not any(l.startswith("--- a/") for l in lines):
        result.errors.append("Missing '--- a/...' file header")
    if not any(l.startswith("+++ b/") for l in lines):
        result.errors.append("Missing '+++ b/...' file header")
    if result.num_hunks == 0:
        result.errors.append("No hunk headers (@@ ... @@) found")
    if result.additions == 0 and result.deletions == 0:
        result.errors.append("Patch has no additions or deletions")

    result.is_valid = len(result.errors) == 0
    return result


# =====================================================================
#  Public API  —  same signature as validation.validator.validate()
# =====================================================================


def validate(result: PipelineResult, test_cases: str, function_name: str = None) -> ValidationReport:
    """
    Validate a SWE-bench Pro pipeline result.

    Drop-in replacement for validation.validator.validate().
    Returns the same ValidationReport so main.py doesn't change.

    Stages:
      1. Extract patch from agent output
      2. Validate patch format (replaces syntax check)
      3. Lightweight test matching (replaces pytest execution)
    """
    r = ValidationReport(
        task_id=result.task_id,
        model_name=result.model_name,
        mode=result.mode,
        reviewer_approved=result.reviewer_approved,
        reviewer_feedback=result.reviewer_feedback,
        debug_iterations=result.debug_iterations,
        duration_seconds=result.duration_seconds,
    )

    # ERROR status — skip all checks
    if result.status == "ERROR":
        r.overall_status = "ERROR"
        return r

    # ── Step 1: Extract patch ────────────────────────────────────
    patch = _extract_patch(result.final_code)
    if not patch:
        r.code_extracted = False
        r.overall_status = "FAILED"
        r.test_output = "Could not extract a unified diff patch from agent output."
        return r
    r.code_extracted = True

    # ── Step 2: Validate patch format (replaces syntax check) ────
    check = _validate_patch_format(patch)
    r.syntax_valid = check.is_valid
    if not check.is_valid:
        r.syntax_error = "; ".join(check.errors)
        r.overall_status = "FAILED"
        r.test_output = (
            f"Invalid patch format: {r.syntax_error}\n"
            f"Warnings: {check.warnings}"
        )
        return r

    # ── Step 3: Lightweight test matching ────────────────────────
    # We can't actually run the SWE-bench tests without Docker.
    # Instead we do heuristic checks against the test_cases info.

    test_files = _parse_test_files(test_cases)
    patch_files = _extract_patched_files(patch)

    n_checks = 0
    n_passed = 0

    # Check 1: patch is non-trivial
    n_checks += 1
    if check.additions > 0 or check.deletions > 0:
        n_passed += 1

    # Check 2: patch modifies source files (not just test files)
    n_checks += 1
    non_test_files = [f for f in patch_files if "test" not in f.lower()]
    if non_test_files:
        n_passed += 1

    # Check 3: patch doesn't only delete code
    n_checks += 1
    if check.additions > 0:
        n_passed += 1

    # Check 4: no binary content
    n_checks += 1
    if not any("binary" in w.lower() for w in check.warnings):
        n_passed += 1

    r.tests_run = n_checks
    r.tests_failed = n_checks - n_passed
    r.tests_passed = (n_passed == n_checks)

    r.test_output = (
        f"Patch format: valid\n"
        f"Files changed: {check.num_files_changed} ({', '.join(patch_files[:5])})\n"
        f"Hunks: {check.num_hunks}\n"
        f"Additions: +{check.additions}  Deletions: -{check.deletions}\n"
        f"Heuristic checks: {n_passed}/{n_checks} passed\n"
        f"NOTE: Full test execution requires Docker eval. "
        f"Run the official eval script on the saved patches."
    )

    # Mark as COMPLETED (not PASSED) since we can't verify tests without Docker
    if n_passed == n_checks:
        r.overall_status = "COMPLETED"
    else:
        r.overall_status = "FAILED"

    return r


# =====================================================================
#  Docker-based evaluation  (separate, run after all tasks)
# =====================================================================


@dataclass
class DockerEvalResult:
    """Result from running the official Docker evaluation."""
    instance_id: str
    resolved: bool = False
    fail_to_pass_passed: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_passed: int = 0
    pass_to_pass_total: int = 0
    error_message: str = ""
    time_seconds: float = 0.0


def run_docker_eval(
    patches_json_path: str,
    eval_repo_path: str = "./swebench_pro_ref",
    output_dir: str = "./results/swebench_pro_eval",
    use_local_docker: bool = True,
    num_workers: int = 4,
    timeout: int = 3600,
) -> Optional[Dict]:
    """
    Run the official SWE-bench Pro evaluation via Docker.

    Call this AFTER main.py finishes and has saved patches.
    """
    eval_repo = Path(eval_repo_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_repo / "external_hf_v2.csv"
    if not csv_path.exists():
        logger.warning(
            "CSV not found at %s — generating from HuggingFace...", csv_path
        )
        try:
            import importlib
            hf_datasets = importlib.import_module("datasets")
            import pandas as pd
            ds = hf_datasets.load_dataset("ScaleAI/SWE-bench_Pro", split="test")
            df = ds.to_pandas()
            df.to_csv(csv_path, index=False)
        except ImportError:
            logger.error("Need `datasets` and `pandas` to generate CSV.")
            return None

    cmd = [
        "python",
        str(eval_repo / "swe_bench_pro_eval.py"),
        f"--raw_sample_path={csv_path}",
        f"--patch_path={patches_json_path}",
        f"--output_dir={out_dir}",
        f"--scripts_dir={eval_repo / 'run_scripts'}",
        f"--dockerhub_username={os.getenv('DOCKERHUB_USER', 'your_dockerhub_username')}",
        f"--num_workers={num_workers}",
    ]
    if use_local_docker:
        cmd.append("--use_local_docker")

    logger.info("Starting Docker evaluation...")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(eval_repo),
        )
        if proc.returncode == 0:
            results_path = out_dir / "eval_results.json"
            if results_path.exists():
                with open(results_path) as f:
                    return json.load(f)
        else:
            logger.error("Docker eval failed: %s", proc.stderr[:500])
    except subprocess.TimeoutExpired:
        logger.error("Docker eval timed out after %ds", timeout)
    except FileNotFoundError:
        logger.error("Eval script not found at %s", eval_repo / "swe_bench_pro_eval.py")
    except Exception as e:
        logger.error("Docker eval error: %s", e)

    return None


# =====================================================================
#  Internal helpers
# =====================================================================


def _parse_test_files(test_cases: str) -> List[str]:
    """
    Extract test identifiers from the test_cases string.

    Handles two formats:
      - SWE-bench Lite/Pro: "FAIL_TO_PASS: [\"tests/...\", ...]"
      - Legacy format:      "Test files to run: tests/foo.py, tests/bar.py"
    """
    files: List[str] = []

    # Format 1: FAIL_TO_PASS / PASS_TO_PASS JSON lists (SWE-bench Lite + Pro)
    m = re.search(r"FAIL_TO_PASS:\s*(\[.*?\])", test_cases, re.DOTALL)
    if m:
        try:
            entries = json.loads(m.group(1))
            # Each entry is a test node-id like "tests/test_foo.py::TestClass::test_method"
            # Extract just the file path portion
            for entry in entries:
                path = entry.split("::")[0].strip()
                if path and path not in files:
                    files.append(path)
            return files
        except (json.JSONDecodeError, IndexError):
            pass

    # Format 2: legacy "Test files to run: ..." (kept for backwards compatibility)
    m = re.search(r"Test files to run:\s*(.+)", test_cases)
    if m:
        files = [f.strip() for f in m.group(1).split(",") if f.strip()]

    return files


def _extract_patched_files(patch: str) -> List[str]:
    """Extract file paths that the patch modifies."""
    files: List[str] = []
    for m in re.finditer(r"diff --git a/(.+?) b/", patch):
        files.append(m.group(1))
    if not files:
        # Fallback: look for --- a/... lines
        for m in re.finditer(r"--- a/(.+)", patch):
            files.append(m.group(1))
    return files