"""Test: qwen3-coder with tools and 16K context via A100 tunnel"""
import sys; sys.path.insert(0, '.')
import agents.litellm_patch
from dotenv import load_dotenv; load_dotenv()
from agents.models import Qwen3Coder
from agents.tools import RepoSearchTool, RepoFileReadTool
from crewai import Agent, Task, Crew, Process
import time

llm = Qwen3Coder().crewai_llm
print(f'Model: {llm.model}, Type: {type(llm).__name__}')

agent = Agent(
    role='Developer', goal='Fix bugs by writing diffs',
    backstory='Expert engineer. After reading code, always outputs diff --git format.',
    llm=llm,
    tools=[RepoSearchTool(repo_path='repos/flipt-io/flipt'), RepoFileReadTool(repo_path='repos/flipt-io/flipt')],
    max_iter=4, verbose=False
)
task = Task(
    description='Search for TestLoad in flipt repo, read authentication.go lines 1-40, write a minimal unified diff. Output ONLY diff --git format.',
    expected_output='Unified git diff starting with diff --git',
    agent=agent
)
crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
t0 = time.time()
try:
    crew.kickoff()
    out = task.output.raw or ''
    has_diff = 'diff --git' in out
    print(f'OK in {round(time.time()-t0,1)}s — has_diff={has_diff}')
    print(repr(out[:200]))
except Exception as e:
    print(f'FAIL in {round(time.time()-t0,1)}s: {type(e).__name__}: {str(e)[:200]}')
