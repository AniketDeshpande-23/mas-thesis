from __future__ import annotations

import os
import requests
from pathlib import Path
from typing import List, Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# ─────────────────────────────────────────────────────────────
# MCP CONFIG
# ─────────────────────────────────────────────────────────────

MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
MCP_TIMEOUT = int(os.getenv("MCP_TIMEOUT", "30"))


# ─────────────────────────────────────────────────────────────
# MCP TOOLS  (graceful failure when server is not running)
# ─────────────────────────────────────────────────────────────

class GetInstanceInput(BaseModel):
    instance_id: str = Field(..., description="SWE-bench instance id")


class GetInstanceTool(BaseTool):
    name: str = "Get SWE Instance"
    description: str = "Fetch the SWE-bench problem context (repo, issue, tests) for an instance id"
    args_schema: Type[BaseModel] = GetInstanceInput

    def _run(self, instance_id: str) -> str:
        try:
            response = requests.post(
                MCP_URL,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": "get_instance", "arguments": {"instance_id": instance_id}},
                    "id": 1,
                },
                timeout=MCP_TIMEOUT,
            )
            return str(response.json().get("result", {}))
        except requests.ConnectionError:
            return "[Get SWE Instance] MCP server not reachable. Use the problem context already in your task description."
        except Exception as exc:
            return f"[Get SWE Instance] Error: {exc}"


class RunTestsInput(BaseModel):
    patch: str = Field(..., description="Unified diff patch to test")
    instance_id: str = Field(..., description="SWE-bench instance id")


class RunTestsTool(BaseTool):
    name: str = "Run SWE Tests"
    description: str = "Apply a patch and run the SWE-bench fail-to-pass tests. Returns PASS/FAIL per test."
    args_schema: Type[BaseModel] = RunTestsInput

    def _run(self, patch: str, instance_id: str) -> str:
        try:
            response = requests.post(
                MCP_URL,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "run_tests",
                        "arguments": {"patch": patch, "instance_id": instance_id},
                    },
                    "id": 1,
                },
                timeout=MCP_TIMEOUT,
            )
            return str(response.json().get("result", {}))
        except requests.ConnectionError:
            return "[Run SWE Tests] MCP server not reachable. Cannot run tests automatically."
        except Exception as exc:
            return f"[Run SWE Tests] Error: {exc}"


# ─────────────────────────────────────────────────────────────
# LOCAL REPO TOOLS
# ─────────────────────────────────────────────────────────────

def resolve_repo_path(instance_repo: str, repos_root: str = "./repos") -> Path:
    return (Path(repos_root) / instance_repo).resolve()


class RepoSearchInput(BaseModel):
    query: str = Field(..., description="Keyword or symbol to search for across source files")


class RepoSearchTool(BaseTool):
    name: str = "Repo Search Tool"
    description: str = (
        "Search for a keyword, function name, or symbol across all source files in the repository. "
        "Use specific names (e.g. 'func Load' or 'TestServeHTTP') for best results. "
        "Returns up to 40 matching lines with file paths and line numbers."
    )
    args_schema: Type[BaseModel] = RepoSearchInput
    repo_path: str = ""

    def __init__(self, repo_path: str, **kwargs):
        super().__init__(repo_path=repo_path, **kwargs)

    def _run(self, query: str) -> str:
        matches = []
        exts = (
            ".py", ".js", ".ts", ".tsx", ".jsx",
            ".java", ".go", ".rb", ".php", ".rs",
            ".c", ".cpp", ".h", ".hpp",
        )
        for root, _, files in os.walk(self.repo_path):
            for fname in files:
                if not fname.endswith(exts):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        for line_no, line in enumerate(fh, 1):
                            if query in line:
                                matches.append(f"{fpath}:{line_no}: {line.rstrip()}")
                except Exception:
                    continue
        if not matches:
            return f"No matches found for: {query!r}"
        return "\n".join(matches[:40])


class RepoFileExistsInput(BaseModel):
    relative_path: str = Field(..., description="File path relative to the repo root")


class RepoFileExistsTool(BaseTool):
    name: str = "Repo File Exists Tool"
    description: str = "Check whether a file exists at the given path inside the repository"
    args_schema: Type[BaseModel] = RepoFileExistsInput
    repo_path: str = ""

    def __init__(self, repo_path: str, **kwargs):
        super().__init__(repo_path=repo_path, **kwargs)

    def _run(self, relative_path: str) -> str:
        path = Path(self.repo_path) / relative_path
        return "EXISTS" if path.exists() else f"NOT FOUND: {relative_path}"


class RepoFileReadInput(BaseModel):
    file_path: str = Field(..., description="Path to file (absolute, or relative to repo root)")
    start_line: int = Field(default=1, description="First line to read (1-indexed)")
    end_line: int = Field(default=80, description="Last line to read (inclusive)")


class RepoFileReadTool(BaseTool):
    name: str = "Read file lines"
    description: str = (
        "Read specific lines from a source file. "
        "Use start_line and end_line to read only the relevant section (e.g. a single function). "
        "Default reads lines 1–80. Always prefer targeted reads over reading whole files."
    )
    args_schema: Type[BaseModel] = RepoFileReadInput
    repo_path: str = ""

    def __init__(self, repo_path: str, **kwargs):
        super().__init__(repo_path=repo_path, **kwargs)

    def _run(self, file_path: str, start_line: int = 1, end_line: int = 80) -> str:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(self.repo_path) / file_path
        if not path.exists():
            return f"FILE NOT FOUND: {file_path}"
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            total = len(lines)
            s = max(1, start_line) - 1
            e = min(total, end_line)
            selected = lines[s:e]
            header = f"# {path.name} (lines {start_line}–{end_line} of {total})\n"
            return header + "\n".join(f"{s + i + 1:4d}: {l}" for i, l in enumerate(selected))
        except Exception as exc:
            return f"ERROR reading {file_path}: {exc}"


# ─────────────────────────────────────────────────────────────
# TOOL BUILDERS
# ─────────────────────────────────────────────────────────────

def build_repo_tools(instance_repo: str, repos_root: str = "./repos") -> List:
    """Return local file-system tools for the cloned repo. Raises FileNotFoundError if not cloned."""
    repo_path = resolve_repo_path(instance_repo, repos_root)
    if not repo_path.exists():
        raise FileNotFoundError(f"Repo not cloned locally: {repo_path}")
    return [
        RepoFileReadTool(repo_path=str(repo_path)),
        RepoSearchTool(repo_path=str(repo_path)),
        RepoFileExistsTool(repo_path=str(repo_path)),
    ]


def build_full_tools(instance_repo: str) -> List:
    """
    Full agent toolset: MCP tools + local repo tools.

    MCP tools are always included (they return a clear error message if the
    server is not running rather than crashing the agent).

    Repo tools are added only when the repo has been cloned locally to
    ./repos/<instance_repo>.  Missing repos are logged but do not prevent
    the MCP tools from being returned.
    """
    tools: List = [GetInstanceTool(), RunTestsTool()]

    try:
        tools.extend(build_repo_tools(instance_repo))
    except FileNotFoundError as exc:
        print(f"[WARN] Local repo tools skipped: {exc}")

    return tools
