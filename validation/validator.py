from __future__ import annotations
import ast, os, re, subprocess, sys, tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineResult:
    task_id: str
    model_name: str
    mode: str
    status: str
    final_code: str
    debug_iterations: int
    reviewer_approved: bool
    reviewer_feedback: str
    duration_seconds: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class ValidationReport:
    task_id: str
    model_name: str
    mode: str
    tests_passed: bool = False
    tests_run: int = 0
    tests_failed: int = 0
    test_output: str = ""
    syntax_valid: bool = False
    syntax_error: Optional[str] = None
    code_extracted: bool = False
    reviewer_approved: bool = False
    reviewer_feedback: str = ""
    debug_iterations: int = 0
    duration_seconds: float = 0.0
    overall_status: str = "FAILED"

    def to_dict(self) -> dict:
        return self.__dict__


# ── Code Extraction ───────────────────────────────────────────────────────────

def _extract_code(raw: str, fn: str = None) -> Optional[str]:
    """Extract code from raw LLM output.
    
    Handles:
    1. Regular functions (looks for specific function name)
    2. Git patches (diff format)
    3. Generic code blocks
    """
    # Check if this is a git patch (SWE-bench format)
    if "diff --git" in raw or "--- " in raw or "+++ " in raw:
        # Extract unified diff patch
        m = re.search(r"(diff --git[\s\S]+?)$", raw, re.MULTILINE)
        if m:
            return m.group(1).strip()
    
    # 1. ```python ... ```
    m = re.search(r"```python\s*([\s\S]+?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2. Plain ``` fence (only if it contains a def or patch)
    m = re.search(r"```\s*([\s\S]+?)```", raw, re.DOTALL)
    if m and ("def " in m.group(1) or "diff " in m.group(1)):
        return m.group(1).strip()

    # 3. If function name provided, search for specific function
    if fn:
        # 3a. Function with leading imports
        m = re.search(
            rf"((?:(?:import\s+\S+|from\s+\S+\s+import\s+[\s\S]*?\n))*def\s+{re.escape(fn)}[\s\S]+?)(?=\n\ndef\s|\Z)",
            raw,
        )
        if m:
            return m.group(1).strip()

        # 3b. Any def block matching the function name
        m = re.search(rf"(def\s+{re.escape(fn)}[\s\S]+?)(?=\n\ndef\s|\Z)", raw)
        if m:
            return m.group(1).strip()

    # 4. Last resort — everything from first import or def or diff
    if "def " in raw:
        idx = raw.find("import ")
        if idx == -1:
            idx = raw.find("def ")
        return raw[idx:].strip()
    
    if "diff " in raw or "--- " in raw:
        idx = raw.find("diff ")
        if idx == -1:
            idx = raw.find("--- ")
        return raw[idx:].strip()

    return None


# ── Syntax Check ─────────────────────────────────────────────────────────────

def _syntax_ok(code: str):
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


# ── Test Runner ───────────────────────────────────────────────────────────────

def _run_tests(code: str, test_str: str):
    if not test_str.strip():
        return False, 0, 0, "No tests provided."

    with tempfile.TemporaryDirectory() as tmp:
        # Write solution file
        with open(os.path.join(tmp, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)

        # Header imports solution into test namespace
        header = (
            f"import sys\n"
            f"sys.path.insert(0, r'{tmp}')\n"
            f"from solution import *\n\n"
        )

        # Use test_str as-is if it has proper test functions/unittest classes
        if (
            "def test_" in test_str
            or "class Test" in test_str
            or "unittest" in test_str
        ):
            body = test_str
        else:
            # Wrap raw assert lines in a pytest function
            lines = [
                l.strip()
                for l in test_str.splitlines()
                if l.strip().startswith("assert")
            ]
            if not lines:
                return False, 0, 0, "No assert statements found in test cases."
            body = "def test_auto():\n" + "\n".join(f"    {l}" for l in lines)

        with open(os.path.join(tmp, "test_solution.py"), "w", encoding="utf-8") as f:
            f.write(header + body)

        try:
            r = subprocess.run(
                [
                    sys.executable, "-m", "pytest",   # ← sys.executable = same Python running main.py
                    os.path.join(tmp, "test_solution.py"),
                    "-v", "--tb=short", "--timeout=30",
                ],
                capture_output=True, text=True, timeout=60, cwd=tmp,
            )
            out = r.stdout + r.stderr
            passed = int((re.findall(r"(\d+) passed", out) or [0])[0])
            failed = int((re.findall(r"(\d+) failed", out) or [0])[0])
            return r.returncode == 0, passed + failed, failed, out[-1500:]

        except subprocess.TimeoutExpired:
            return False, 0, 0, "Tests timed out after 60s."

        except Exception as e:
            # exec() fallback if pytest missing
            try:
                g: dict = {}
                exec(code, g)
                asserts = [
                    l.strip()
                    for l in test_str.splitlines()
                    if l.strip().startswith("assert")
                ]
                if not asserts:
                    return False, 0, 0, f"pytest unavailable and no assert lines found. Error: {e}"
                ok = 0
                for a in asserts:
                    try:
                        exec(a, g)
                        ok += 1
                    except Exception:
                        pass
                fail = len(asserts) - ok
                return (
                    fail == 0,
                    len(asserts),
                    fail,
                    f"{ok}/{len(asserts)} assertions passed (exec fallback — install pytest for full results)",
                )
            except Exception as e2:
                return False, 0, 0, f"exec fallback error: {e2}"


# ── Public API ────────────────────────────────────────────────────────────────

def validate(result: PipelineResult, test_cases: str, function_name: str = None) -> ValidationReport:
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

    # Step 1 — extract code (use provided function name or default fallback)
    code = _extract_code(result.final_code, function_name or "task_func")
    if not code:
        r.overall_status = "FAILED"
        return r
    r.code_extracted = True

    # Step 2 — syntax check
    ok, err = _syntax_ok(code)
    r.syntax_valid = ok
    r.syntax_error = err
    if not ok:
        r.overall_status = "FAILED"
        return r

    # Step 3 — run tests
    passed, n_run, n_fail, output = _run_tests(code, test_cases)
    r.tests_passed = passed
    r.tests_run = n_run
    r.tests_failed = n_fail
    r.test_output = output

    r.overall_status = "PASSED" if passed else "FAILED"
    return r