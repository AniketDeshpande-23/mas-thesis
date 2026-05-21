from crewai import Agent
from agents.models import BaseLLM

_DIFFICULTY_HINTS = {
    "easy":   "The fix is likely small — 1 file, 1–5 lines changed. Identify the single most likely file quickly.",
    "medium": "The fix touches 2–4 files and may span multiple functions. Think about related call sites.",
    "hard":   "The fix is large — multiple files and subsystems. Focus on the root cause first, not symptoms.",
}


# Tool call budget and max_iter scale with difficulty — more files expected = more searches needed
_TOOL_BUDGET = {"easy": 3, "medium": 5, "hard": 7}
_MAX_ITER    = {"easy": 6, "medium": 9, "hard": 12}


def _has_tools_instruction(budget: int) -> str:
    return (
        "You have access to repository tools. Use them BEFORE writing the plan.\n"
        f"IMPORTANT: Budget — make at most {budget} tool calls total, then write the plan immediately.\n"
        "  1. Call 'Repo Search Tool' with each failing test name or key symbol to find its file.\n"
        "     CRITICAL: if the test name (e.g. 'test_install_collection') is in a test file, also\n"
        "     search for the FUNCTION it tests (e.g. 'install_collection') to find the SOURCE module.\n"
        "     Test files expose bugs — the FIX goes in the source module, NOT the test file.\n"
        "  2. Call 'Read file lines' on relevant files (use start_line and end_line).\n"
        "  3. Call 'Repo File Exists Tool' to confirm uncertain paths.\n"
        f"After your {budget}th tool call (or sooner), write the plan immediately.\n"
        "Only include file paths you have verified exist in the repository.\n"
        "NEVER plan changes to test files — only plan changes to source/library files."
    )


_NO_TOOLS = "Infer file paths from the repository name, error messages, and failing test names."


def make_planner(llm: BaseLLM, tools=None, difficulty: str = "medium") -> Agent:
    scope_hint = _DIFFICULTY_HINTS.get(difficulty, _DIFFICULTY_HINTS["medium"])
    budget     = _TOOL_BUDGET.get(difficulty, 5)
    max_iter   = _MAX_ITER.get(difficulty, 9) if tools else 3
    file_instructions = _has_tools_instruction(budget) if tools else _NO_TOOLS
    return Agent(
        role="Planner",
        goal=(
            "Analyse the bug report and produce a concrete numbered implementation plan.\n\n"
            f"Scope guidance ({difficulty} task): {scope_hint}\n\n"
            f"{file_instructions}\n\n"
            "Each step MUST include:\n"
            "  - The exact verified file path\n"
            "  - The function or class to modify\n"
            "  - What specifically to change (add, remove, or replace what)\n\n"
            "Do NOT write any code. Output ONLY the numbered plan."
        ),
        backstory=(
            "You are a senior software engineer who excels at root-cause analysis. "
            "You use repository search tools to find the exact file and function before planning. "
            "You never guess a file path you haven't verified."
        ),
        llm=llm.crewai_llm,
        tools=tools or [],
        verbose=True,
        allow_delegation=False,
        use_system_prompt=True,
        respect_context_window=True,
        max_iter=max_iter,
    )
