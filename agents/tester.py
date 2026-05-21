from crewai import Agent
from agents.models import BaseLLM


def make_tester(llm: BaseLLM, tools=None) -> Agent:
    tools = tools or []
    has_runner = any(getattr(t, "name", "") == "Run SWE Tests" for t in tools)

    if has_runner:
        goal = (
            "Use the 'Run SWE Tests' tool to run the failing tests against the patch "
            "produced by the Developer in the previous step.\n"
            "Steps:\n"
            "  1. Extract the patch exactly from the Developer's output.\n"
            "  2. Call 'Run SWE Tests' with the patch and the instance_id.\n"
            "  3. Return the raw tool output verbatim — do NOT interpret or modify it."
        )
    else:
        goal = (
            "The automated test runner is not available. "
            "Perform a structural review of the patch.\n\n"
            "VERDICT: FAIL — ONLY for definite structural errors:\n"
            "  - Patch is empty or has no content at all\n"
            "  - Diff format is invalid (missing 'diff --git', '--- a/', '+++ b/', or '@@' lines)\n"
            "  - File paths are clearly placeholder ('path/to/file', 'example.py', 'TODO')\n"
            "  - Patch has no hunks (no lines starting with + or -)\n\n"
            "VERDICT: PASS — for all structurally valid patches, even if logic might be wrong.\n\n"
            "ADVISORY — a separate section for speculative concerns (logic errors, missing imports,\n"
            "wrong variable names, edge cases). Write these AFTER the VERDICT/ISSUES block.\n"
            "They are read by the Debugger but do NOT trigger a rewrite on their own.\n\n"
            "You MUST reply with EXACTLY this format:\n\n"
            "VERDICT: PASS\n"
            "ISSUES: none\n"
            "ADVISORY: <optional speculative concerns, or 'none'>\n\n"
            "or, if a definite structural error is found:\n\n"
            "VERDICT: FAIL\n"
            "ISSUES:\n"
            "- <specific structural error>\n"
            "ADVISORY: <optional speculative concerns, or 'none'>"
        )

    return Agent(
        role="Tester",
        goal=goal,
        backstory=(
            "You are a QA engineer and code reviewer. "
            "You report findings factually and concisely. "
            "You never guess or fabricate test results. "
            "You approve only patches that are correct and complete."
        ),
        llm=llm.crewai_llm,
        tools=tools,
        verbose=True,
        allow_delegation=False,
        use_system_prompt=True,
        respect_context_window=True,
        max_iter=3,
    )
