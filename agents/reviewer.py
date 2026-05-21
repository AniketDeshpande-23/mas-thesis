from crewai import Agent
from agents.models import BaseLLM


def make_reviewer(llm: BaseLLM, tools=None) -> Agent:
    return Agent(
        role="Reviewer",
        goal=(
            "Review the final patch and decide whether to approve it.\n"
            "Approve only if the patch:\n"
            "  - clearly targets the described bug (not a no-op or placeholder)\n"
            "  - is a valid unified diff (starts with 'diff --git')\n"
            "  - does not obviously break unrelated code\n\n"
            "You MUST reply with EXACTLY these two lines and nothing else:\n"
            "Approved: True\n"
            "Feedback: <one sentence>\n\n"
            "If the patch is empty, invalid, or a refusal, reply:\n"
            "Approved: False\n"
            "Feedback: <reason>"
        ),
        backstory=(
            "You are a strict senior code reviewer. "
            "You approve only patches that are coherent, minimal, and genuinely address the reported bug."
        ),
        llm=llm.crewai_llm,
        tools=tools or [],
        verbose=True,
        allow_delegation=False,
        use_system_prompt=True,
        respect_context_window=True,
        max_iter=2,
    )
