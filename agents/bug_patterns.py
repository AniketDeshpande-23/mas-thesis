"""
agents/bug_patterns.py — Static bug-pattern library + diagnostic tools for the Debugger agent.

PatternAnalysisTool: classifies tester failure output into named bug patterns + hints.
PatchAnalyzerTool:   parses a unified diff to report targeted files, hunks, line counts.

Both tools are pure Python (no MCP / repo dependency) and are always available to the
Debugger regardless of the USE_AGENT_TOOLS flag in orchestrator.py.
"""
from __future__ import annotations

import re
from typing import NamedTuple, Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool


# ─────────────────────────────────────────────────────────────────────
# Pattern library
# ─────────────────────────────────────────────────────────────────────

class BugPattern(NamedTuple):
    name: str
    keywords: tuple[str, ...]
    hint: str


_PATTERNS: tuple[BugPattern, ...] = (
    BugPattern(
        name="wrong-file",
        keywords=(
            "wrong file", "incorrect file", "not the right file",
            "path/to", "wrong module", "wrong path", "different file",
            "unrelated file", "does not target",
        ),
        hint=(
            "WRONG-FILE: The patch targets the wrong source file. "
            "Re-derive the correct file path from the failing test module path and "
            "the function/class name in the error. "
            "Do NOT change any logic — only correct the 'diff --git a/...' path headers."
        ),
    ),
    BugPattern(
        name="missing-import",
        keywords=(
            "import", "undefined", "not defined", "nameerror",
            "modulenotfound", "no module named", "cannot find module",
            "is not defined", "importerror", "unresolved import",
        ),
        hint=(
            "MISSING-IMPORT: The patch introduces a symbol that is not imported. "
            "Scan every name on added (+) lines for undefined references and "
            "add the required import statement at the top of the patched file."
        ),
    ),
    BugPattern(
        name="off-by-one",
        keywords=(
            "off by one", "off-by-one", "index", "boundary", "out of range",
            "fencepost", "fence post", "range(", "len(", "slice",
            "index out", "out of bounds", "expected length",
        ),
        hint=(
            "OFF-BY-ONE: A loop bound, array index, or slice boundary is wrong by one. "
            "Check every numeric index on added lines: should the limit be < N or <= N? "
            "Should the slice end at N or N+1? Verify against the failing test's expected count."
        ),
    ),
    BugPattern(
        name="wrong-condition",
        keywords=(
            "condition", "logic error", "incorrect check", "always true",
            "always false", "opposite", "negated", "inverted",
            "should be", "incorrect comparison", "wrong comparison",
            "check is wrong", "reversed",
        ),
        hint=(
            "WRONG-CONDITION: A boolean condition has the wrong polarity. "
            "For every 'if' expression on added lines ask: should it be negated? "
            "Should == be !=, > be <, or should 'not' be removed/added?"
        ),
    ),
    BugPattern(
        name="missing-null-check",
        keywords=(
            "null", "none", "nil", "undefined", "attribute error",
            "attributeerror", "typeerror", "cannot read propert",
            "is not an object", "nullpointerexception", "npe", "dereference",
        ),
        hint=(
            "MISSING-NULL-CHECK: A variable may be null/None/nil when the added code uses it. "
            "Before the first use of each new variable add a guard: "
            "Python: 'if x is None: return/raise', "
            "Go: 'if x == nil { return err }', "
            "JS/TS: 'if (!x) { return; }'."
        ),
    ),
    BugPattern(
        name="incomplete-patch",
        keywords=(
            "partial fix", "not all", "also need", "additionally",
            "another file", "other location", "multiple", "several places",
            "missing case", "other call site", "other occurrence",
        ),
        hint=(
            "INCOMPLETE-PATCH: The fix is applied in one place but the same bug exists elsewhere. "
            "Search the patch context for other functions, files, or branches with the same "
            "erroneous pattern and add additional diff hunks for each location."
        ),
    ),
    BugPattern(
        name="diff-format-error",
        keywords=(
            "hunk", "invalid patch", "parse error", "malformed",
            "no additions", "no deletions", "missing header",
            "format error", "patch format", "context line",
            "invalid diff", "diff format",
        ),
        hint=(
            "DIFF-FORMAT-ERROR: The patch has a structural problem. "
            "Do NOT change any logic. Fix ONLY the diff structure: "
            "starts with 'diff --git a/<f> b/<f>', has '--- a/<f>' and '+++ b/<f>' lines, "
            "has at least one '@@ -N,M +N,M @@' hunk header, "
            "and unchanged context lines start with a single space."
        ),
    ),
    BugPattern(
        name="regression",
        keywords=(
            "regression", "broke", "breaks", "break", "unrelated",
            "existing behavior", "existing behaviour", "side effect",
            "other test", "was working", "previously", "no longer",
        ),
        hint=(
            "REGRESSION: The patch changes code unrelated to the stated bug, "
            "breaking existing behaviour. Make the patch more surgical: "
            "revert every added/removed line that is not directly required by the described bug."
        ),
    ),
    BugPattern(
        name="type-mismatch",
        keywords=(
            "type error", "typeerror", "type mismatch", "expected type",
            "got type", "cannot convert", "cast", "wrong type",
            "string expected", "integer expected", "bool expected",
            "type assertion", "cannot assign",
        ),
        hint=(
            "TYPE-MISMATCH: A value of the wrong type is passed, assigned, or returned. "
            "Check the declared type of every variable and parameter on added lines "
            "against the types of values assigned. Add an explicit cast or conversion."
        ),
    ),
    BugPattern(
        name="wrong-return-value",
        keywords=(
            "return", "returns", "returned value", "expected value",
            "incorrect value", "wrong result", "wrong output",
            "incorrect result", "wrong answer",
        ),
        hint=(
            "WRONG-RETURN-VALUE: The function returns an incorrect value. "
            "Trace the computation backward from the 'return' statement on added lines: "
            "which intermediate variable holds the wrong value, and where is it computed wrong?"
        ),
    ),
)

_NAME_TO_HINT: dict[str, str] = {p.name: p.hint for p in _PATTERNS}


# ─────────────────────────────────────────────────────────────────────
# Classification helpers (also used by orchestrator for CSV logging)
# ─────────────────────────────────────────────────────────────────────

def classify_issue(tester_output: str) -> list[str]:
    """
    Confidence-weighted keyword match against tester output.
    Returns pattern names sorted by hit count (most confident first).
    """
    lower = tester_output.lower()
    scored = [
        (sum(1 for kw in p.keywords if kw in lower), p.name)
        for p in _PATTERNS
    ]
    matched = [(score, name) for score, name in scored if score > 0]
    matched.sort(reverse=True)
    return [name for _, name in matched]


def format_hints(matched_names: list[str]) -> str:
    """Comma-joined pattern names for CSV logging. Returns 'none' if empty."""
    return ", ".join(matched_names) if matched_names else "none"


# ─────────────────────────────────────────────────────────────────────
# PatternAnalysisTool
# ─────────────────────────────────────────────────────────────────────

class _PatternInput(BaseModel):
    tester_output: str = Field(
        ...,
        description="The full VERDICT + ISSUES text produced by the Tester agent.",
    )


class PatternAnalysisTool(BaseTool):
    name: str = "Pattern Analysis Tool"
    description: str = (
        "Analyse the Tester's ISSUES text and return the detected failure class "
        "plus targeted fix strategies. Call this FIRST before attempting any correction."
    )
    args_schema: Type[BaseModel] = _PatternInput

    def _run(self, tester_output: str) -> str:
        lower = tester_output.lower()
        scored = [
            (sum(1 for kw in p.keywords if kw in lower), p)
            for p in _PATTERNS
        ]
        matched_scored = [(s, p) for s, p in scored if s > 0]
        matched_scored.sort(reverse=True)

        if not matched_scored:
            return (
                "No specific pattern detected from the tester output.\n"
                "Apply a general review: verify the patch targets the correct file, "
                "that all changed lines are syntactically valid, and that the diff "
                "format is correct (starts with 'diff --git')."
            )

        lines = [f"Detected {len(matched_scored)} failure pattern(s), ranked by confidence:\n"]
        for score, p in matched_scored:
            confidence = "HIGH" if score >= 3 else ("MEDIUM" if score >= 2 else "LOW")
            lines.append(f"[{confidence}] {p.hint}")
        lines.append("\nAddress HIGH confidence patterns first.")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# PatchAnalyzerTool
# ─────────────────────────────────────────────────────────────────────

class _PatchInput(BaseModel):
    patch: str = Field(
        ...,
        description="The unified git diff patch to analyse.",
    )


class PatchAnalyzerTool(BaseTool):
    name: str = "Patch Analyzer Tool"
    description: str = (
        "Parse the current unified diff patch and return structured info: "
        "which files are targeted, how many hunks, lines added/removed. "
        "Call this SECOND to understand what the patch is doing before correcting it."
    )
    args_schema: Type[BaseModel] = _PatchInput

    def _run(self, patch: str) -> str:
        files = re.findall(r"^diff --git a/(\S+)", patch, re.MULTILINE)
        hunks = len(re.findall(r"^@@", patch, re.MULTILINE))
        added = len(re.findall(r"^\+(?!\+\+)", patch, re.MULTILINE))
        removed = len(re.findall(r"^-(?!--)", patch, re.MULTILINE))

        if not files:
            return (
                "Patch structure: INVALID — no 'diff --git' header found.\n"
                "The patch must start with 'diff --git a/<file> b/<file>'."
            )

        file_list = ", ".join(files)
        return (
            f"Patch targets {len(files)} file(s): {file_list}\n"
            f"Hunks: {hunks}\n"
            f"Lines added: {added}  |  Lines removed: {removed}\n\n"
            "Cross-check: do these target files match what the failing test names suggest "
            "should be modified? If not, you likely have a wrong-file issue."
        )
