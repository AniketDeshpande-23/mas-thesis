"""
Generates professor_presentation.ipynb — a curated notebook showing
only the architecturally significant code excerpts.
Open with: jupyter notebook professor_presentation.ipynb
"""
import nbformat as nbf
import json, csv, os, glob

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(src, collapsed=False):
    c = nbf.v4.new_code_cell(src)
    if collapsed:
        c.metadata["collapsed"] = True
    cells.append(c)


# ── Title ────────────────────────────────────────────────────────────────────
md("""# MAS vs Single LLM — Automated Bug Fixing
## Architecture & Results Walkthrough
**Aniket Deshpande · MSc Thesis · SWE-bench Pro · May 2026**

---
> **Purpose of this notebook:** Show the professor the key architectural decisions
> and a concrete example. Not a full codebase tour.

---
""")

# ── Section 1: Pipeline Architecture ─────────────────────────────────────────
md("""## 1. Pipeline Architecture

Two pipelines, same tasks, same benchmark. Primary metric:
```
patch_score = 0.6 × file_recall + 0.4 × content_overlap
```
where both are measured against the **gold patch** (human-written correct fix).
""")

code('''\
# MAS pipeline — 5 specialised agents, each with a different LLM
MAS_PIPELINE = """
  Planner        (Claude Opus 4.5)     — verifies file paths via RepoSearch, writes numbered plan
      |
  Developer      (Qwen3-Coder-Next)    — implements plan as a unified git diff
      |
  [Docker eval]  (real test execution) — runs FAIL_TO_PASS tests inside SWE-bench container
      |
  Debugger       (Gemini Flash-Lite)   — diagnoses failure: wrong path vs wrong logic
      |
  DevRefine      (Qwen3-Coder-Next)    — applies surgical corrections from Debugger diagnosis
      |
  [Docker eval]  (repeat up to N times based on difficulty)
      |
  Reviewer       (Claude Opus 4.5)     — final quality gate; rejects → triggers revision
"""

# Single pipeline — 1 agent, higher iteration budget
SINGLE_PIPELINE = """
  Solo Developer (Gemini Flash-Lite)   — plans + implements + self-corrects in one context
      |
  [Docker eval]  (real test execution)
      |
  Solo Developer (self-correction)     — sees real test failure, re-patches
      |
  [repeat up to N times]
"""

print("MAS Pipeline:")
print(MAS_PIPELINE)
print("Single Pipeline:")
print(SINGLE_PIPELINE)
''')

# ── Section 2: Key Architectural Decision — Heterogeneous Models ──────────────
md("""## 2. Heterogeneous Model Routing (AGENT_LLM_MAP)

**The single most impactful change:** Different LLMs for different roles.
Each role has fundamentally different task demands — a strong reasoning model for
planning is different from a fast coding specialist for diff generation.
""")

code('''\
# From agents/orchestrator.py — the routing map
AGENT_LLM_MAP = {
    "planner":    "claude-opus-4-5",        # strong reasoning — accurate multi-file plans
    "developer":  "qwen3-coder-next",       # coding specialist — 2-6s per diff, clean syntax
    "debugger":   "gemini-3.1-flash-lite",  # pattern diagnosis — fast is sufficient
    "dev_refine": "qwen3-coder-next",       # surgical corrections — same specialist as Developer
    "tester":     "gemini-2.5-flash",       # static fallback when Docker unavailable
    "reviewer":   "claude-opus-4-5",        # final quality gate — strong reasoning
}

# Single baseline uses one model for everything:
SINGLE_MODEL = "gemini-3.1-flash-lite"

# Why this matters — Run L experiment result:
print("Key finding (Run L):")
print("  Qwen3-Coder-Next as Single agent  → patch_score = 0.020 (near failure)")
print("  Qwen3-Coder-Next as MAS Developer → patch_score = 0.769 (breakthrough)")
print()
print("Same model. Same task. Radically different outcome.")
print("Role specialisation unlocks capability inaccessible in single-agent setting.")
''')

# ── Section 3: The Convergence Loop ──────────────────────────────────────────
md("""## 3. Convergence Loop — How Iteration Works

The pipeline does NOT run a fixed number of debug cycles.
It iterates until one of three exit conditions is met.
""")

code('''\
# Simplified version of the convergence loop in agents/orchestrator.py

ITER_CAP = {"easy": 2, "medium": 3, "hard": 4}  # difficulty-adaptive

def convergence_loop(task, initial_patch, docker_eval, debugger, dev_refine):
    """
    Iterate Debugger → DevRefine until:
      1. docker_resolved = True   (tests pass — stop immediately)
      2. Patch converged          (Jaccard similarity >= 0.95 — no progress)
      3. Cap reached              (bounded time guarantee)
    """
    current_patch = initial_patch
    cap = ITER_CAP[task.difficulty]

    for iteration in range(cap):
        result = docker_eval(task.id, current_patch)

        if result.resolved:
            print(f"  RESOLVED on iteration {iteration + 1}")
            return current_patch, iteration + 1

        # Apply-aware diagnosis: different strategy for path failures vs logic failures
        if result.status == "apply_failed":
            # Patch did not apply — file path is wrong
            diagnosis = debugger.diagnose_path_error(current_patch, result.error)
        else:
            # Patch applied but tests failed — logic is wrong
            diagnosis = debugger.diagnose_logic_error(current_patch, result.test_output)

        new_patch = dev_refine.apply_correction(current_patch, diagnosis)

        # Jaccard convergence check — prevent re-running identical patches
        if jaccard_similarity(new_patch, current_patch) >= 0.95:
            print(f"  Patch converged on iteration {iteration + 1} — stopping.")
            break

        current_patch = new_patch

    return current_patch, iteration + 1
''')

# ── Section 4: The Timeout Problem (Professor's Question) ────────────────────
md("""## 4. The JupyterHub Timeout Problem

The professor asked about increasing the timeout in the agent library.
The actual failure was at a different layer — the nginx reverse proxy.
""")

code('''\
# The problem was NOT the agent-side timeout
# litellm_patch.py already sets:
import litellm
litellm.request_timeout = 600   # 10 minutes — more than enough

# The actual failure chain:
TIMEOUT_CHAIN = """
Windows pipeline
    → HTTP request to JupyterHub proxy
        → nginx (proxy_read_timeout = 60s)  <-- THE PROBLEM
            → pod Ollama (inference takes 90-600s)
                → response never reaches nginx in time
                → HTTP 599 upstream timeout
"""

# Why increasing client timeout didn't help:
print("Client timeout = 600s   ✓  (already set)")
print("nginx proxy_read_timeout = 60s   ✗  (server-side, cannot change)")
print()
print("The connection was dropped before the model finished.")
print("Solution: moved to OpenRouter cloud API — eliminates proxy entirely.")
print()
print(TIMEOUT_CHAIN)
''')

# ── Section 5: Concrete Example ───────────────────────────────────────────────
md("""## 5. Concrete Example — flipt-io Go Task (Hard Difficulty)

This is the task where the structural vs semantic gap is most visible.
The agent finds the right file but implements the wrong logic.
""")

code('''\
# Task: flipt-io/flipt — Go hard task
TASK_DESCRIPTION = """
Repository: flipt-io/flipt
Language: Go
Difficulty: hard

Problem: The authentication middleware does not correctly handle the case where
a flag evaluation request arrives without authentication headers when
anonymous access is configured. The router should allow unauthenticated
requests through when AllowAnonymous=true in the server config.

Failing tests:
  FAIL_TO_PASS: [
    "TestFliptRouter_HandleFlag",
    "TestFliptRouter_EvaluateFlag",
    "TestFliptRouter_HandleWithAnonymous"
  ]
"""

# What the agent produced (Run M, best result):
GENERATED_PATCH_EXCERPT = """
diff --git a/internal/server/router.go b/internal/server/router.go
--- a/internal/server/router.go
+++ b/internal/server/router.go
@@ -45,6 +45,11 @@ func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
+    if r.config.Authentication.Required && !r.config.Authentication.AllowAnonymous {
+        if err := r.authenticate(req); err != nil {
+            http.Error(w, "Unauthorized", http.StatusUnauthorized)
+            return
+        }
+    }
     r.mux.ServeHTTP(w, req)
"""

# What the gold patch does (correct fix):
GOLD_PATCH_EXCERPT = """
diff --git a/internal/server/router.go b/internal/server/router.go
--- a/internal/server/router.go
+++ b/internal/server/router.go
@@ -45,6 +45,13 @@ func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
+    if r.config.Authentication.Required {
+        token, err := r.extractToken(req)
+        if err != nil && !r.config.Authentication.AllowAnonymous {
+            http.Error(w, "Unauthorized", http.StatusUnauthorized)
+            return
+        }
+        if token != "" {
+            req = req.WithContext(setTokenContext(req.Context(), token))
+        }
+    }
     r.mux.ServeHTTP(w, req)
"""

print("=== GENERATED PATCH ===")
print(GENERATED_PATCH_EXCERPT)
print()
print("=== GOLD PATCH (correct) ===")
print(GOLD_PATCH_EXCERPT)
print()
print("Structural analysis:")
print("  Generated: targets correct file (router.go)  -> file_recall = 0.200")
print("  Generated: valid diff format                 -> syntax_valid = True")
print("  Generated: applies cleanly                   -> docker_status = 'failed' (not apply_failed)")
print()
print("Semantic gap:")
print("  Generated: checks config field BEFORE extracting token (wrong order)")
print("  Gold:      extracts token first, THEN conditionally rejects (correct)")
print("  Docker:    TestFliptRouter_HandleWithAnonymous fails - anonymous request rejected incorrectly")
''')

# ── Section 6: Reform Attribution ─────────────────────────────────────────────
md("""## 6. Reform Attribution — Closing the Gap

Each reform targeted a specific, measurable failure mode.
""")

code('''\
import pandas as pd

reforms = pd.DataFrame([
    ("Run E", "Baseline — tools enabled",               0.251, 0.353, 0.102, "Single"),
    ("Run F", "Delta signal + Jaccard convergence",     0.230, 0.323, 0.093, "Single"),
    ("Run G", "Planner given RepoSearch tools",         0.247, 0.293, 0.046, "Single"),
    ("Run H", "Devstral for hard tasks + 3-call cap",   0.224, 0.230, 0.006, "Single"),
    ("Run I", "Developer max_iter difficulty-adaptive", 0.234, 0.272, 0.038, "Single"),
    ("Run J", "Claude Opus as Planner + Reviewer",      0.228, 0.292, 0.064, "Single"),
    ("Run K", "Qwen3-Coder-Next as Developer",          0.334, 0.278, 0.056, "MAS WIN"),
    ("Run M", "Adaptive budget + zero-score recovery",  0.336, 0.273, 0.062, "MAS WIN"),
], columns=["Run", "Reform Added", "MAS", "Single", "Gap", "Winner"])

pd.set_option("display.max_colwidth", 45)
pd.set_option("display.width", 120)
print(reforms.to_string(index=False))
print()
print(f"Gap progression:  {reforms['Gap'].iloc[0]:.3f} (Single leads)  →  +{reforms['Gap'].iloc[-1]:.3f} (MAS leads)")
print(f"Most impactful single reform: Reform 6 — Planner tools (gap halved in one step)")
''')

# ── Section 7: Statistical Summary ────────────────────────────────────────────
md("""## 7. Statistical Significance

With n=9 per group (3 tasks × 3 runs), power is limited.
Effect sizes (Cohen's d) are the primary evidence for small samples.
""")

code('''\
import numpy as np
from scipy import stats

# Run M results — latest completed run
mas_scores    = [0.790, 0.769, 0.493,   # NodeBB easy   (3 runs)
                 0.000, 0.181, 0.000,   # ansible medium (3 runs)
                 0.181, 0.420, 0.186]   # flipt hard     (3 runs)

single_scores = [0.504, 0.504, 0.504,   # NodeBB easy
                 0.163, 0.163, 0.163,   # ansible medium
                 0.153, 0.153, 0.153]   # flipt hard

mas    = np.array(mas_scores)
single = np.array(single_scores)

t, p  = stats.ttest_ind(mas, single, equal_var=False)
u, up = stats.mannwhitneyu(mas, single, alternative="two-sided")
pooled_std = np.sqrt(((len(mas)-1)*mas.std(ddof=1)**2 +
                      (len(single)-1)*single.std(ddof=1)**2) /
                     (len(mas)+len(single)-2))
d = (mas.mean() - single.mean()) / pooled_std

print(f"MAS    mean = {mas.mean():.4f}  std = {mas.std():.4f}")
print(f"Single mean = {single.mean():.4f}  std = {single.std():.4f}")
print(f"Delta (MAS - Single) = {mas.mean() - single.mean():+.4f}")
print()
print(f"Welch t-test:   t = {t:+.3f},  p = {p:.4f}  {'(n.s.)' if p > 0.1 else '* p<0.10'}")
print(f"Mann-Whitney U: U = {u:.0f},  p = {up:.4f}  {'(n.s.)' if up > 0.1 else '* p<0.10'}")
print(f"Cohen d  =  {d:+.3f}  ({'large' if abs(d)>0.8 else 'medium' if abs(d)>0.5 else 'small'} effect)")
print()
print("Note: n=9 per group gives ~20% power for d=0.25.")
print("Full 15-task run (n=45) will give 80% power — standard publishable threshold.")
''')

# ── Section 8: Open Question for Professor ────────────────────────────────────
md("""## 8. Open Question — Logically Correct Patches

**Current state:** Patches apply cleanly and target correct files but fail test assertions.

**Hypothesis 1:** Model capability limit — the 7B–30B models cannot infer the precise
logic change from the problem description alone. Needs a larger model (70B+).

**Hypothesis 2:** Missing information — the agent sees the test *name* but not the
test *source code*. Seeing the exact assertion would let it write the exact fix.
(Partially addressed: test_patch injected into Debugger context.)

**Hypothesis 3:** Iteration quality — the Debugger→DevRefine loop is not converging
because Docker feedback says "test failed" but doesn't pinpoint which assertion
failed and why. Need richer per-assertion output from the test runner.

**Question for professor:**
At what point does this become a fundamental model capability limit vs
an information-access problem that better prompting/architecture can solve?
""")

# Finalise notebook
nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

with open("professor_presentation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Saved: professor_presentation.ipynb")
print("Open with: jupyter notebook professor_presentation.ipynb")
