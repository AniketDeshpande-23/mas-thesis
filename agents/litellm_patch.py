"""
Patches LLM call paths with two fixes:

1. Ollama: think=False, stream=True (nginx keepalive), timeout=600
   Applied via litellm.completion / litellm.acompletion patches.

2. OpenRouter (qwen3-coder-next): model hallucninates tool_calls in plain-text responses.
   CrewAI routes openrouter/ models through OpenAICompatibleCompletion which uses the
   OpenAI SDK directly (bypasses litellm entirely). The fix patches _handle_completion
   and _ahandle_completion in OpenAICompletion to convert tool_calls → text when
   the function name is not in available_functions (hallucinated call).
"""
import json
import os as _os
import re as _re
import litellm as _lt
from litellm import stream_chunk_builder as _scb

_orig       = _lt.completion
_orig_async = _lt.acompletion

# Global safety net
_lt.request_timeout = 600


# ── Fix 2: Patch CrewAI's OpenAI-compatible completion path ──────────────────
#
# CrewAI 1.14 routes openrouter/ models through OpenAICompatibleCompletion which
# uses the OpenAI Python SDK directly. When qwen3-coder-next returns tool_calls
# format (even with no tools defined), _handle_completion returns a list of
# tool_call objects. This propagates to TaskOutput(raw=list) → pydantic error.
#
# Fix: intercept after _handle_completion. If the result is a list of tool_calls
# whose function name is NOT in available_functions, convert to text.

def _tool_calls_to_text(tool_calls_list: list) -> str:
    """Extract text from a list of ChatCompletionMessageToolCall objects."""
    parts = []
    for tc in tool_calls_list:
        func = getattr(tc, "function", None)
        if func is None:
            continue
        args_str = getattr(func, "arguments", "") or ""
        try:
            args_obj = json.loads(args_str)
            best = max((v for v in args_obj.values() if isinstance(v, str)), key=len, default="")
            parts.append(best if best else args_str)
        except (json.JSONDecodeError, TypeError):
            parts.append(args_str)
    return "\n\n".join(parts) if parts else repr(tool_calls_list)


def _is_hallucinated_tool_call(result, params: dict) -> bool:
    """Return True if result is a tool_calls list returned when no tools were in the API call.

    CrewAI's executor always passes available_functions=None to llm.call() and handles
    tool_calls itself. So we can't use available_functions to detect hallucination.
    Instead, check params["tools"]: if no tools were sent to the API but the model
    returned tool_calls, it's a hallucination that must be converted to text.
    When tools WERE sent, the executor's _handle_native_tool_calls will process them.
    """
    if not isinstance(result, list) or not result:
        return False
    if not hasattr(result[0], "function"):
        return False
    # Only intercept when no tools were provided to the API request
    return not params.get("tools")


try:
    from crewai.llms.providers.openai.completion import OpenAICompletion as _OpenAICompletion

    _orig_handle_completion       = _OpenAICompletion._handle_completion
    _orig_ahandle_completion      = _OpenAICompletion._ahandle_completion

    def _patched_handle_completion(self, params, available_functions=None,
                                   from_task=None, from_agent=None, response_model=None):
        result = _orig_handle_completion(self, params, available_functions,
                                         from_task, from_agent, response_model)
        if _is_hallucinated_tool_call(result, params):
            print(f"[litellm_patch] Intercepted hallucinated tool_calls (no tools in call) → text")
            return _tool_calls_to_text(result)
        return result

    async def _patched_ahandle_completion(self, params, available_functions=None,
                                          from_task=None, from_agent=None, response_model=None):
        result = await _orig_ahandle_completion(self, params, available_functions,
                                                from_task, from_agent, response_model)
        if _is_hallucinated_tool_call(result, params):
            print(f"[litellm_patch] Intercepted hallucinated tool_calls (no tools in call) → text")
            return _tool_calls_to_text(result)
        return result

    _OpenAICompletion._handle_completion         = _patched_handle_completion
    _OpenAICompletion._ahandle_completion        = _patched_ahandle_completion
    OPENAI_COMPAT_PATCHED = True
    print("[litellm_patch] OpenAICompletion._handle_completion patched OK")

except ImportError:
    OPENAI_COMPAT_PATCHED = False
    print("[litellm_patch] WARNING: could not patch OpenAICompletion (crewai not installed?)")


def _is_ollama(kwargs: dict) -> bool:
    model    = str(kwargs.get("model", ""))
    base_url = str(kwargs.get("api_base", "") or kwargs.get("base_url", ""))
    return (model.startswith("ollama/")
            or "11434" in base_url
            or "11436" in base_url
            or "jupyterhub" in base_url.lower()
            or "trycloudflare" in base_url.lower())


def _is_openrouter(kwargs: dict) -> bool:
    model = str(kwargs.get("model", ""))
    return model.startswith("openrouter/")


_THINK_RE = _re.compile(r"<think>[\s\S]*?</think>", _re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by thinking models (qwen3-coder-next)."""
    if not text or "<think>" not in text:
        return text
    return _THINK_RE.sub("", text).strip()


def _fix_tool_calls(resp):
    """Convert tool_calls-format responses to plain text content.

    qwen3-coder-next on OpenRouter returns tool_call format even with tool_choice=none.
    This extracts the arguments and sets them as content, then CLEARS tool_calls so
    CrewAI's pydantic TaskOutput validator receives a plain string, not a list.
    Also strips <think>...</think> blocks from content.
    """
    if resp is None:
        return resp
    for choice in getattr(resp, "choices", []):
        msg = getattr(choice, "message", None)
        if msg is None:
            continue
        content = getattr(msg, "content", None)
        # Strip thinking tokens from content regardless of tool_calls
        if content and "<think>" in content:
            msg.content = _strip_thinking(content)
            content = msg.content
        tool_calls = getattr(msg, "tool_calls", None)
        if not content and tool_calls:
            parts = []
            for tc in tool_calls:
                func = getattr(tc, "function", None)
                if func is None:
                    continue
                args_str = getattr(func, "arguments", "") or ""
                try:
                    args_obj = json.loads(args_str)
                    # Extract the longest string value — likely contains the actual response
                    best = max((v for v in args_obj.values() if isinstance(v, str)), key=len, default="")
                    parts.append(best if best else args_str)
                except (json.JSONDecodeError, TypeError):
                    parts.append(args_str)
            msg.content = "\n\n".join(parts) if parts else repr(tool_calls)
            # Clear tool_calls so CrewAI treats this as a regular text response
            msg.tool_calls = None
            try:
                choice.finish_reason = "stop"
            except Exception:
                pass
    return resp


def _inject_jupyterhub_auth(kwargs: dict) -> None:
    base_url = str(kwargs.get("api_base", "") or kwargs.get("base_url", ""))
    if "jupyterhub" not in base_url.lower():
        return
    token = _os.getenv("JUPYTERHUB_TOKEN", "")
    if not token:
        return
    headers = dict(kwargs.get("extra_headers") or {})
    headers.setdefault("Authorization", f"Bearer {token}")
    kwargs["extra_headers"] = headers


def _completion(*args, **kwargs):
    if args:
        kwargs.setdefault("model", args[0])
        args = args[1:]
    if _is_ollama(kwargs):
        kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "think": False}
        kwargs.setdefault("timeout", 600)
        _inject_jupyterhub_auth(kwargs)
        if not kwargs.get("stream"):
            kwargs["stream"] = True
            stream = _orig(*args, **kwargs)
            chunks = list(stream)
            return _scb(chunks, messages=kwargs.get("messages", []))
    if _is_openrouter(kwargs):
        if kwargs.get("tools"):
            kwargs["tool_choice"] = "none"
        resp = _orig(*args, **kwargs)
        return _fix_tool_calls(resp)
    return _orig(*args, **kwargs)


async def _acompletion(*args, **kwargs):
    if args:
        kwargs.setdefault("model", args[0])
        args = args[1:]
    if _is_ollama(kwargs):
        kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "think": False}
        kwargs.setdefault("timeout", 600)
        _inject_jupyterhub_auth(kwargs)
        if not kwargs.get("stream"):
            kwargs["stream"] = True
            stream = await _orig_async(*args, **kwargs)
            chunks = [chunk async for chunk in stream]
            return _scb(chunks, messages=kwargs.get("messages", []))
    if _is_openrouter(kwargs):
        if kwargs.get("tools"):
            kwargs["tool_choice"] = "none"
        resp = await _orig_async(*args, **kwargs)
        return _fix_tool_calls(resp)
    return await _orig_async(*args, **kwargs)


_lt.completion  = _completion
_lt.acompletion = _acompletion

APPLIED = True
