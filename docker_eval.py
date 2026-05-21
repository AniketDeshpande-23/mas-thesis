"""
docker_eval.py — Offline Docker evaluation for SWE-bench Pro patches.

Run this AFTER main.py completes, pointing it at the saved patches JSON.
It spins up one Docker container per patch, applies the patch, runs the
failing tests, and records whether the task was truly resolved.

Usage
-----
# Basic: evaluate all patches, write docker_eval_{ts}.csv
python docker_eval.py --patches results/swebench_patches_20260423_XXXXXX.json

# Pre-fetch all Docker images before the run (do this while pipeline is running)
python docker_eval.py --patches results/swebench_patches_20260423_XXXXXX.json --pull-only

# Merge resolved column into existing results CSV
python docker_eval.py --patches results/swebench_patches_20260423_XXXXXX.json \\
                      --csv     results/results_20260423_XXXXXX.csv \\
                      --merge

Requirements
------------
- Docker Desktop running (verify: docker info)
- Internet access for first-time image pull (~2–5 GB per task image)
- ~50 GB free disk for 15 task images
- Each evaluation takes ~3–8 minutes per patch
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
DOCKER_IMAGE_PREFIX = "jefzda/sweap-images"
REPO_WORKDIR        = "/testbed"          # standard workdir in SWE-bench Pro images
CONTAINER_TIMEOUT   = 300                 # seconds: max time for test execution
APPLY_TIMEOUT       = 60                  # seconds: max time for git apply
SETUP_TIMEOUT       = 120                 # seconds: max time for before_repo_set_cmd
CACHE_PATH          = "./datasets/swebench_pro_cache.json"


# ── Docker helpers ─────────────────────────────────────────────────────────────

def check_docker() -> bool:
    """Return True if Docker daemon is reachable."""
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def image_is_pulled(image: str) -> bool:
    r = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, timeout=15
    )
    return r.returncode == 0


def pull_image(image: str) -> bool:
    """Pull a Docker image. Returns True on success."""
    if image_is_pulled(image):
        print(f"    [cache] {image}")
        return True
    print(f"    [pull ] {image} ...")
    r = subprocess.run(["docker", "pull", image], timeout=600)
    return r.returncode == 0


# ── Test command builders ──────────────────────────────────────────────────────

def _pytest_command(tests: list[str], extra_files: list[str]) -> str:
    """Build a pytest command for Python tests."""
    if tests:
        # pytest accepts node IDs directly: path/to/test.py::Class::method
        node_ids = " ".join(f'"{t}"' for t in tests)
        return (
            f"cd {REPO_WORKDIR} && "
            f"python -m pytest {node_ids} -x --tb=short --no-header -q 2>&1"
        )
    if extra_files:
        files = " ".join(extra_files)
        return f"cd {REPO_WORKDIR} && python -m pytest {files} -x --tb=short -q 2>&1"
    return f"cd {REPO_WORKDIR} && python -m pytest -x --tb=short -q 2>&1"


def _go_command(tests: list[str]) -> str:
    """Build a go test command. Tests are function names (may include /subtest)."""
    if tests:
        # Extract base function names (before '/') for the -run pattern
        base_names = sorted({t.split("/")[0] for t in tests})
        pattern = "|".join(f"^{re.escape(n)}$" for n in base_names)
        return (
            f"cd {REPO_WORKDIR} && "
            f"go test ./... -run '{pattern}' -v -count=1 -timeout 180s 2>&1"
        )
    return f"cd {REPO_WORKDIR} && go test ./... -count=1 -timeout 180s 2>&1"


def _js_command(tests: list[str], test_files: list[str]) -> str:
    """
    Build a mocha/npm command for JS tests.
    SWE-bench Pro JS tests are formatted as:  'file.js | mocha test title'
    """
    if test_files:
        # Run only the relevant test files (faster than full npm test)
        files = " ".join(f'"{f}"' for f in test_files)
        return (
            f"cd {REPO_WORKDIR} && "
            f"./node_modules/.bin/mocha {files} --timeout 15000 --exit 2>&1"
        )
    return f"cd {REPO_WORKDIR} && npm test 2>&1"


def build_test_command(meta: dict, fail_only: bool = True) -> str:
    """Return the shell command to run the relevant tests inside the container."""
    lang       = meta.get("repo_language", "python")
    fail_tests = meta.get("fail_to_pass", [])
    pass_tests = meta.get("pass_to_pass", [])
    test_files = meta.get("selected_test_files_to_run", [])

    tests = fail_tests if fail_only else fail_tests + pass_tests

    if lang == "python":
        return _pytest_command(tests, test_files)
    if lang == "go":
        return _go_command(tests)
    if lang in ("js", "ts"):
        # For JS, extract test file paths from the 'file.js | title' format
        js_files = []
        for t in tests:
            if " | " in t:
                js_files.append(t.split(" | ")[0].strip())
        js_files = sorted(set(js_files)) or test_files
        return _js_command(tests, js_files)

    # Fallback
    return _pytest_command(tests, test_files)


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate_patch(
    instance_id: str,
    patch: str,
    meta: dict,
    verbose: bool = False,
) -> dict:
    """
    Apply patch inside a fresh Docker container and run tests.

    Returns a dict with at minimum:
        instance_id, resolved (bool), docker_status (str), error (str|None)
    """
    dockerhub_tag = meta.get("dockerhub_tag", "").strip()
    if not dockerhub_tag:
        return _result(instance_id, False, "no_image",
                       error="No dockerhub_tag in cache — cannot evaluate")

    image = f"{DOCKER_IMAGE_PREFIX}:{dockerhub_tag}"

    if not pull_image(image):
        return _result(instance_id, False, "pull_failed",
                       error=f"docker pull {image} failed")

    # Unique container name
    container = f"sweeval_{int(time.time() * 1000) % 99999999}"
    patch_tmp: Optional[str] = None

    try:
        # Write patch to a temp file on the host
        with tempfile.NamedTemporaryFile(
            suffix=".patch", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(patch)
            patch_tmp = f.name

        # ── Start container ────────────────────────────────────────────────
        r = subprocess.run(
            ["docker", "run", "-d", "--name", container,
             "--memory", "3g", "--cpus", "2",
             image, "sleep", str(CONTAINER_TIMEOUT + 120)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return _result(instance_id, False, "start_failed",
                           error=r.stderr.strip()[:300])

        # ── Copy patch into container ──────────────────────────────────────
        r = subprocess.run(
            ["docker", "cp", patch_tmp, f"{container}:/tmp/fix.patch"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return _result(instance_id, False, "copy_failed",
                           error=r.stderr.strip()[:300])

        # ── Reset repo to base commit (before_repo_set_cmd) ───────────────
        setup_cmd = (meta.get("before_repo_set_cmd") or "").strip()
        if setup_cmd:
            _exec(container, setup_cmd, timeout=SETUP_TIMEOUT)

        # ── Apply patch ────────────────────────────────────────────────────
        apply_cmd = (
            f"cd {REPO_WORKDIR} && "
            f"git apply --whitespace=fix /tmp/fix.patch 2>&1"
        )
        ar = _exec(container, apply_cmd, timeout=APPLY_TIMEOUT)

        if ar["returncode"] != 0:
            # Fallback: try 3-way merge
            apply_3way = (
                f"cd {REPO_WORKDIR} && "
                f"git apply --3way --whitespace=fix /tmp/fix.patch 2>&1"
            )
            ar2 = _exec(container, apply_3way, timeout=APPLY_TIMEOUT)
            if ar2["returncode"] != 0:
                apply_err = (ar2["output"] or ar["output"])[:600]
                return _result(
                    instance_id, False, "apply_failed",
                    error="git apply failed (tried --whitespace=fix and --3way)",
                    fail_output_tail=f"PATCH APPLY FAILED — file paths are likely wrong.\n{apply_err}",
                )

        # ── Run FAIL_TO_PASS tests ─────────────────────────────────────────
        fail_cmd = build_test_command(meta, fail_only=True)
        if verbose:
            print(f"      fail_cmd: {fail_cmd}")
        fr = _exec(container, fail_cmd, timeout=CONTAINER_TIMEOUT)
        fail_passed = fr["returncode"] == 0

        # ── Run PASS_TO_PASS tests (only if fail tests passed) ─────────────
        pass_passed = True
        pass_output = ""
        if fail_passed and meta.get("pass_to_pass"):
            pass_cmd = build_test_command(meta, fail_only=False)
            pr = _exec(container, pass_cmd, timeout=CONTAINER_TIMEOUT)
            pass_passed = pr["returncode"] == 0
            pass_output = pr["output"][-300:]

        resolved = fail_passed and pass_passed

        return {
            "instance_id":          instance_id,
            "resolved":             resolved,
            "docker_status":        "ok",
            "fail_to_pass_passed":  fail_passed,
            "pass_to_pass_passed":  pass_passed,
            "fail_to_pass_count":   len(meta.get("fail_to_pass", [])),
            "pass_to_pass_count":   len(meta.get("pass_to_pass", [])),
            "fail_output_tail":     fr["output"][-800:],
            "pass_output_tail":     pass_output,
            "error":                None,
        }

    except subprocess.TimeoutExpired:
        return _result(instance_id, False, "timeout",
                       error=f"Timed out after {CONTAINER_TIMEOUT}s")
    except Exception as exc:
        return _result(instance_id, False, "exception", error=str(exc))

    finally:
        subprocess.run(["docker", "rm", "-f", container],
                       capture_output=True, timeout=30)
        if patch_tmp and os.path.exists(patch_tmp):
            os.unlink(patch_tmp)


def _exec(container: str, cmd: str, timeout: int) -> dict:
    """Run a shell command inside a container and return {returncode, output}."""
    r = subprocess.run(
        ["docker", "exec", container, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return {
        "returncode": r.returncode,
        "output": (r.stdout + r.stderr).strip(),
    }


def _result(instance_id, resolved, status, error=None, **extra) -> dict:
    base = {
        "instance_id":          instance_id,
        "resolved":             resolved,
        "docker_status":        status,
        "fail_to_pass_passed":  False,
        "pass_to_pass_passed":  False,
        "fail_to_pass_count":   0,
        "pass_to_pass_count":   0,
        "fail_output_tail":     "",
        "pass_output_tail":     "",
        "error":                error,
    }
    base.update(extra)
    return base


# ── Prefix parsing ─────────────────────────────────────────────────────────────

def parse_prefix(prefix: str) -> dict:
    """
    Parse prefix like 'qwen3.6_mas_run1' → {model_name, mode, run_number}.
    Format built in main.py: f"{llm.name}_{mode}_run{run_num}"
    """
    try:
        model_mode, run_str = prefix.rsplit("_run", 1)
        parts = model_mode.rsplit("_", 1)
        return {
            "model_name": parts[0],
            "mode":       parts[1],
            "run_number": int(run_str),
        }
    except Exception:
        return {"model_name": prefix, "mode": "unknown", "run_number": 1}


# ── CSV merge ──────────────────────────────────────────────────────────────────

def merge_into_csv(csv_path: str, docker_results: list[dict]) -> str:
    """
    Add 'resolved', 'docker_status', 'fail_to_pass_passed' columns to an
    existing results CSV. Matches rows on task_id + model_name + mode + run_number.
    Writes a new file (original is untouched).
    """
    # Build lookup: (instance_id, model_name, mode, run_number) → docker result
    lookup: dict[tuple, dict] = {}
    for dr in docker_results:
        key = (
            dr["instance_id"],
            dr.get("model_name", ""),
            dr.get("mode", ""),
            str(dr.get("run_number", "")),
        )
        lookup[key] = dr

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        extra_cols = ["resolved", "docker_status",
                      "fail_to_pass_passed", "pass_to_pass_passed",
                      "fail_to_pass_count",  "pass_to_pass_count",
                      "docker_error"]
        new_fields = fieldnames + [c for c in extra_cols if c not in fieldnames]
        for row in reader:
            key = (
                row.get("task_id", ""),
                row.get("model_name", ""),
                row.get("mode", ""),
                str(row.get("run_number", "")),
            )
            dr = lookup.get(key, {})
            row["resolved"]             = dr.get("resolved", "")
            row["docker_status"]        = dr.get("docker_status", "not_evaluated")
            row["fail_to_pass_passed"]  = dr.get("fail_to_pass_passed", "")
            row["pass_to_pass_passed"]  = dr.get("pass_to_pass_passed", "")
            row["fail_to_pass_count"]   = dr.get("fail_to_pass_count", "")
            row["pass_to_pass_count"]   = dr.get("pass_to_pass_count", "")
            row["docker_error"]         = dr.get("error") or ""
            rows.append(row)

    merged_path = csv_path.replace(".csv", "_with_docker.csv")
    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return merged_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline Docker evaluation for SWE-bench Pro patches."
    )
    parser.add_argument(
        "--patches", required=True,
        help="Path to swebench_patches_{ts}.json saved by main.py",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to results_{ts}.csv to merge resolved column into (optional)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge resolved results into --csv (writes *_with_docker.csv)",
    )
    parser.add_argument(
        "--pull-only", action="store_true",
        help="Only pull Docker images, do not run evaluation",
    )
    parser.add_argument(
        "--cache", default=CACHE_PATH,
        help=f"Path to swebench_pro_cache.json (default: {CACHE_PATH})",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print test commands and full output",
    )
    args = parser.parse_args()

    # ── Validate inputs ────────────────────────────────────────────────────
    if not Path(args.patches).exists():
        print(f"ERROR: patches file not found: {args.patches}")
        sys.exit(1)
    if not Path(args.cache).exists():
        print(f"ERROR: cache not found: {args.cache}")
        sys.exit(1)
    if not check_docker():
        print("ERROR: Docker is not running. Start Docker Desktop and retry.")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────
    with open(args.patches, encoding="utf-8") as f:
        patches: list[dict] = json.load(f)

    with open(args.cache, encoding="utf-8") as f:
        cache_raw = json.load(f)
    cache: dict[str, dict] = {d["instance_id"]: d for d in cache_raw}

    print(f"\n{'━'*60}")
    print(f"  SWE-bench Pro Docker Evaluation")
    print(f"  Patches file : {args.patches}")
    print(f"  Entries      : {len(patches)}")
    print(f"{'━'*60}")

    # ── Pull-only mode ─────────────────────────────────────────────────────
    if args.pull_only:
        print("\n  Pre-fetching Docker images...\n")
        images = set()
        for entry in patches:
            meta = cache.get(entry["instance_id"], {})
            tag = meta.get("dockerhub_tag", "").strip()
            if tag:
                images.add(f"{DOCKER_IMAGE_PREFIX}:{tag}")

        print(f"  {len(images)} unique images to pull:\n")
        ok, failed = 0, []
        for img in sorted(images):
            if pull_image(img):
                ok += 1
            else:
                failed.append(img)

        print(f"\n  Pulled: {ok}/{len(images)}")
        if failed:
            print("  Failed:")
            for img in failed:
                print(f"    {img}")
        return

    # ── Evaluation ─────────────────────────────────────────────────────────
    results = []
    total = len(patches)

    for i, entry in enumerate(patches, 1):
        instance_id = entry["instance_id"]
        patch       = entry.get("patch", "")
        prefix      = entry.get("prefix", "")
        meta        = cache.get(instance_id, {})
        parsed      = parse_prefix(prefix)

        print(f"\n  [{i:02d}/{total}] {instance_id[:55]}")
        print(f"           prefix={prefix}  lang={meta.get('repo_language','?')}")

        if not patch or not patch.strip().startswith("diff --git"):
            print("           SKIP — empty or invalid patch")
            dr = _result(instance_id, False, "invalid_patch",
                         error="Patch is empty or does not start with 'diff --git'")
        else:
            t0 = time.time()
            dr = evaluate_patch(instance_id, patch, meta, verbose=args.verbose)
            elapsed = round(time.time() - t0, 1)
            status_icon = "✅" if dr["resolved"] else "❌"
            print(
                f"           {status_icon}  resolved={dr['resolved']}"
                f"  status={dr['docker_status']}"
                f"  time={elapsed}s"
            )
            if dr.get("error"):
                print(f"           error: {dr['error']}")

        # Attach prefix metadata so merge can match rows
        dr.update(parsed)
        results.append(dr)

    # ── Save docker_eval CSV ───────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("results", f"docker_eval_{ts}.csv")
    os.makedirs("results", exist_ok=True)

    docker_cols = [
        "instance_id", "model_name", "mode", "run_number",
        "resolved", "docker_status",
        "fail_to_pass_passed", "fail_to_pass_count",
        "pass_to_pass_passed", "pass_to_pass_count",
        "fail_output_tail", "error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=docker_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Docker eval saved → {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    resolved_count = sum(1 for r in results if r.get("resolved"))
    error_count    = sum(1 for r in results if r.get("docker_status") not in ("ok", "invalid_patch"))
    print(f"\n{'━'*60}")
    print(f"  RESOLVED : {resolved_count} / {total}")
    print(f"  ERRORS   : {error_count} / {total}")
    print(f"  RESOLVE RATE : {resolved_count/total*100:.1f}%")
    print(f"{'━'*60}")

    # ── Merge into existing CSV ────────────────────────────────────────────
    if args.merge:
        if not args.csv:
            print("\n  WARNING: --merge requires --csv <path>. Skipping merge.")
        elif not Path(args.csv).exists():
            print(f"\n  WARNING: CSV not found: {args.csv}. Skipping merge.")
        else:
            merged = merge_into_csv(args.csv, results)
            print(f"\n  Merged CSV saved → {merged}")


if __name__ == "__main__":
    main()
