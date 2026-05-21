from crewai import Agent
from agents.models import BaseLLM

# Injected into both developer variants — exact format the validator checks for.
_PATCH_FORMAT = """\
Output ONLY a valid unified git diff. Here is a concrete example of a correct patch:

diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -382,7 +382,8 @@
     def delete(self):
-        return self._raw_delete(self.db)
+        collector = Collector(using=self.db)
+        collector.collect(self)
+        return collector.delete()

Rules:
- Start with 'diff --git a/... b/...'
- Include '--- a/...' and '+++ b/...' lines
- Include at least one '@@ -LINE,COUNT +LINE,COUNT @@' hunk header
- Lines to remove start with '-', lines to add start with '+'
- Context lines (unchanged) start with a single space
- No markdown fences, no ``` — raw diff ONLY
- No explanations before or after the diff\
"""

_NO_FILE_ACCESS = """\
IMPORTANT: You do not have direct access to the repository files.
You must reason from the problem description alone to produce a patch:
  1. Read the problem statement carefully to identify the bug.
  2. Infer the most likely file path from the repository name and issue context
     (e.g. for repo 'django/django' a view bug is likely in 'django/views/*.py').
  3. Infer the function/class name from the error message or test names.
  4. Write only the changed lines — keep the diff minimal and targeted.
  5. Use realistic line numbers (e.g. @@ -45,7 +45,8 @@); they need not be exact
     but must follow the format.\
"""

_HAS_FILE_ACCESS = """\
You have access to repository tools (Repo Search Tool, Read file lines, List files in directory).
IMPORTANT: If the Planner already verified file paths, use those directly — skip RepoSearch.
Otherwise: call 'Repo Search Tool' with the failing test name or function to find the file.
Then call 'Read file lines' with start_line and end_line to read the exact function to patch.
Prefer reading actual source over guessing line numbers.
TOOL BUDGET: Make at most 3 tool calls total. After your 3rd tool call your NEXT output MUST
be the unified diff — no more tool calls after that, no explanations, just the diff.\
"""


_SCOPE_HINTS = {
    "easy":   "Scope hint: the fix is likely 1 file, 1–5 lines. Keep the patch minimal.",
    "medium": "Scope hint: the fix likely spans 2–4 files. Check related call sites.",
    "hard":   "Scope hint: the fix is large — multiple files/subsystems. Focus on the root cause first.",
}


def _context_instructions(tools) -> str:
    """Return the appropriate file-access instruction based on tool availability."""
    return _HAS_FILE_ACCESS if tools else _NO_FILE_ACCESS


_DEV_MAX_ITER = {"easy": 6, "medium": 8, "hard": 12}


def make_developer(llm: BaseLLM, tools=None, difficulty: str = "medium") -> Agent:
    """MAS Developer — patch generation from the Planner's implementation plan."""
    scope = _SCOPE_HINTS.get(difficulty, _SCOPE_HINTS["medium"])
    _max = _DEV_MAX_ITER.get(difficulty, 8) if tools else 4
    return Agent(
        role="Developer",
        goal=(
            "Implement the fix described in the Planner's plan by producing "
            "a valid unified git diff patch.\n\n"
            f"{scope}\n\n"
            f"{_context_instructions(tools)}\n\n"
            f"{_PATCH_FORMAT}"
        ),
        backstory=(
            "You are an expert software engineer specialising in bug fixes. "
            "You write minimal, targeted patches. "
            "You always output a valid unified diff and nothing else."
        ),
        llm=llm.crewai_llm,
        tools=tools or [],
        verbose=True,
        allow_delegation=False,
        use_system_prompt=True,
        respect_context_window=True,
        max_iter=_max,
    )


def make_developer_refine(llm: BaseLLM, tools=None) -> Agent:
    """
    MAS re-patch step — applies SURGICAL corrections to the existing patch guided
    by the Debugger's diagnosis. Does NOT rewrite from scratch.
    """
    return Agent(
        role="Developer",
        goal=(
            "Apply SURGICAL corrections to the existing patch based on the Debugger's diagnosis.\n\n"
            "You will receive:\n"
            "  1. The original problem description\n"
            "  2. The current patch (which failed tests)\n"
            "  3. The Debugger's DIAGNOSIS and FIX PLAN\n\n"
            "CRITICAL RULES — read carefully:\n"
            "  - Correct ONLY the specific lines, files, or logic the Debugger flagged\n"
            "  - Keep ALL other parts of the existing patch EXACTLY as-is\n"
            "  - Do NOT rewrite, restructure, or replace sections the Debugger did not flag\n"
            "  - Do NOT start from scratch — build on the existing patch\n\n"
            "Examples of surgical corrections:\n"
            "  - Debugger flagged wrong file path → only change the diff --git header\n"
            "  - Debugger flagged missing import → only add that one import line\n"
            "  - Debugger flagged wrong variable name → only fix that variable\n\n"
            f"{_context_instructions(tools)}\n\n"
            f"{_PATCH_FORMAT}"
        ),
        backstory=(
            "You are an expert software engineer who applies precise, targeted fixes. "
            "You never rewrite working code — you make the minimum change needed. "
            "You always output a valid unified diff and nothing else."
        ),
        llm=llm.crewai_llm,
        tools=tools or [],
        verbose=True,
        allow_delegation=False,
        use_system_prompt=True,
        respect_context_window=True,
        max_iter=6 if tools else 4,
    )


def make_single_developer(llm: BaseLLM, tools=None, difficulty: str = "medium") -> Agent:
    """
    Single-agent baseline — handles planning, implementation, and self-review alone.
    Given more iterations to compensate for having no team.
    """
    return Agent(
        role="Solo Developer",
        goal=(
            "Fix the described software bug entirely on your own.\n\n"
            f"{_SCOPE_HINTS.get(difficulty, _SCOPE_HINTS['medium'])}\n\n"
            "Follow this process:\n"
            "  Step 1 — Understand the bug: read the problem statement and failing tests.\n"
            "  Step 2 — Identify the root cause: which file and function contains the bug?\n"
            "  Step 3 — Plan the fix: what is the minimal code change needed?\n"
            "  Step 4 — Write the patch: produce a valid unified git diff.\n"
            "  Step 5 — Self-review: does the patch address the failing tests "
            "without breaking unrelated behaviour?\n\n"
            f"{_context_instructions(tools)}\n\n"
            f"{_PATCH_FORMAT}"
        ),
        backstory=(
            "You are a senior software engineer who works independently. "
            "You plan, implement, and review your own patches. "
            "You always produce a valid unified diff and nothing else."
        ),
        llm=llm.crewai_llm,
        tools=tools or [],
        verbose=True,
        allow_delegation=False,
        use_system_prompt=True,
        respect_context_window=True,
        max_iter=({"easy": 10, "medium": 15, "hard": 20}.get(difficulty, 15) if tools else 6),
    )
