"""
agents/models.py  —  Models for MAS vs Single thesis experiment


"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from crewai import LLM
import litellm

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Set global timeout — LLM inference via JupyterHub proxy can take 90s+
litellm.request_timeout = 600

LLM_TEMPERATURE = 0.15
LLM_MAX_TOKENS  = 8192
LLM_NUM_CTX     = 16384  # 16K — enough for tool results + task + plan; A100 has VRAM headroom


def _local_llm(model_tag: str, num_ctx: int = LLM_NUM_CTX, temperature: float = LLM_TEMPERATURE) -> LLM:
    return LLM(
        model=f"ollama/{model_tag}",
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        max_tokens=LLM_MAX_TOKENS,
        extra_body={"options": {"num_ctx": num_ctx}, "think": False},
    )


def _cloud_llm(model_tag: str, api_key_env: str, base_url: str | None = None) -> LLM:
    kwargs: dict = dict(
        model=model_tag,
        api_key=os.getenv(api_key_env, ""),
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    if base_url:
        kwargs["base_url"] = base_url
    return LLM(**kwargs)


class BaseLLM(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def crewai_llm(self) -> LLM: ...

    @property
    def backend(self) -> str:
        return "local"   # override in cloud models


# ─────────────────────────────────────────────────────────────────────
# Devstral Small 2  (Mistral, Dec 2025) — LOCAL
# 24B dense, 68% SWE-bench (highest of the three)
# Pull: ollama pull devstral-small-2:latest  (~15 GB)
# ─────────────────────────────────────────────────────────────────────
class DevstralSmall2(BaseLLM):
    @property
    def name(self) -> str:
        return "devstral-small-2"

    @property
    def backend(self) -> str:
        return "local"

    @property
    def crewai_llm(self) -> LLM:
        return _local_llm("devstral-small-2:latest")


# ─────────────────────────────────────────────────────────────────────
# GLM-4.7-Flash  (Zhipu AI, Jan 2026) — LOCAL via Ollama
# 30B/3B-active MoE — coding + agentic workflows, ~5 GB at Q4_K_M
# Pull: ollama pull glm-4.7-flash:latest
# ─────────────────────────────────────────────────────────────────────
class GLM47Flash(BaseLLM):
    @property
    def name(self) -> str:
        return "glm-4.7-flash"

    @property
    def backend(self) -> str:
        return "local"

    @property
    def crewai_llm(self) -> LLM:
        # MoE 3B active — fast, benefits from slightly higher temp for diversity
        return _local_llm("glm-4.7-flash:latest", num_ctx=8192, temperature=0.20)


# ─────────────────────────────────────────────────────────────────────
# Qwen3.5-27B  (Qwen team, ~Feb 2026) — LOCAL via Ollama
# 27B dense, hybrid Gated DeltaNet — 72.4% SWE-bench Verified
# Pull: ollama pull qwen3.5:27b
# ─────────────────────────────────────────────────────────────────────
class Qwen35_27B(BaseLLM):
    @property
    def name(self) -> str:
        return "qwen3.5-27b"

    @property
    def backend(self) -> str:
        return "local"

    @property
    def crewai_llm(self) -> LLM:
        return _local_llm("qwen3.5:27b")


# ─────────────────────────────────────────────────────────────────────
# Qwen3.5-9B  (Qwen team, ~Feb 2026) — LOCAL via Ollama
# 9B dense — lightweight smoke-test model, fast inference, fits easily
# Pull: ollama pull qwen3.5:9b
# ─────────────────────────────────────────────────────────────────────
class Qwen35_9B(BaseLLM):
    @property
    def name(self) -> str:
        return "qwen3.5-9b"

    @property
    def backend(self) -> str:
        return "local"

    @property
    def crewai_llm(self) -> LLM:
        # 9B dense — small model benefits from higher temp for exploration
        return _local_llm("qwen3.5:9b", num_ctx=8192, temperature=0.20)


# ─────────────────────────────────────────────────────────────────────
# Gemma4-27B  (Google DeepMind, Apr 2 2026) — LOCAL via Ollama
# 27B MoE — native tool-calling, strong agentic, within 5-month window
# Pull: ollama pull gemma4:27b
# ─────────────────────────────────────────────────────────────────────
class Gemma4_27B(BaseLLM):
    @property
    def name(self) -> str:
        return "gemma4-27b"

    @property
    def backend(self) -> str:
        return "local"

    @property
    def crewai_llm(self) -> LLM:
        return _local_llm("gemma4:27b")


# ─────────────────────────────────────────────────────────────────────
# Gemma4-31B  (Google DeepMind, Apr 2 2026) — LOCAL via Ollama
# 31B dense — strongest Gemma4 for coding, native tool-calling
# VRAM: ~19 GB at Q4_K_M (partial CPU offload on <24 GB GPUs)
# Pull: ollama pull gemma4:31b
# ─────────────────────────────────────────────────────────────────────
class Gemma4_31B(BaseLLM):
    @property
    def name(self) -> str:
        return "gemma4-31b"

    @property
    def backend(self) -> str:
        return "local"

    @property
    def crewai_llm(self) -> LLM:
        # 31B dense — VRAM constrained, lower temp for more deterministic output
        return _local_llm("gemma4:31b", num_ctx=4096, temperature=0.10)


# ─────────────────────────────────────────────────────────────────────
# Legacy models — kept for backward compatibility with 22 Apr 2026 run
# ─────────────────────────────────────────────────────────────────────
class Qwen3Coder(BaseLLM):
    @property
    def name(self) -> str:
        return "qwen3-coder"

    @property
    def crewai_llm(self) -> LLM:
        return _local_llm("qwen3-coder:latest", num_ctx=16384, temperature=0.15)


class Qwen3CoderNextLocal(BaseLLM):
    """qwen3-coder-next via local Ollama (A100 pod) — pull: ollama pull qwen3-coder-next:latest"""
    @property
    def name(self) -> str:
        return "qwen3-coder-next-local"

    @property
    def crewai_llm(self) -> LLM:
        return _local_llm("qwen3-coder-next:latest", num_ctx=16384, temperature=0.15)


class Qwen36(BaseLLM):
    @property
    def name(self) -> str:
        return "qwen3.6"

    @property
    def crewai_llm(self) -> LLM:
        return _local_llm("qwen3.6:latest")


# ─────────────────────────────────────────────────────────────────────
# OpenRouter — latest coding models via paid API
# ─────────────────────────────────────────────────────────────────────
class ClaudeOpusOpenRouter(BaseLLM):
    """Claude Opus 4.5 via OpenRouter — strong reasoning for Planner + Reviewer roles."""
    @property
    def name(self) -> str:
        return "claude-opus-4-5"

    @property
    def backend(self) -> str:
        return "cloud"

    @property
    def crewai_llm(self) -> LLM:
        return LLM(
            model="openrouter/anthropic/claude-opus-4-5",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )


class Gemini25FlashTesterOpenRouter(BaseLLM):
    """Gemini 2.5 Flash via OpenRouter — stronger Tester for static code review (no-Docker fallback).
    Gemini 3.1 Pro not available on OpenRouter; 2.5 Flash is the strongest confirmed working Gemini."""
    @property
    def name(self) -> str:
        return "gemini-2.5-flash-tester"

    @property
    def backend(self) -> str:
        return "cloud"

    @property
    def crewai_llm(self) -> LLM:
        return LLM(
            model="openrouter/google/gemini-2.5-flash",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )


class GeminiFlashLiteOpenRouter(BaseLLM):
    """Gemini 3.1 Flash-Lite via OpenRouter — fast, cheap, strong SWE-bench"""
    @property
    def name(self) -> str:
        return "gemini-3.1-flash-lite"

    @property
    def backend(self) -> str:
        return "cloud"

    @property
    def crewai_llm(self) -> LLM:
        return LLM(
            model="openrouter/google/gemini-3.1-flash-lite",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )


class Gemini25FlashOpenRouter(BaseLLM):
    """Gemini 2.5 Flash via OpenRouter — stronger model for hard tasks (Go, complex refactors)"""
    @property
    def name(self) -> str:
        return "gemini-2.5-flash"

    @property
    def backend(self) -> str:
        return "cloud"

    @property
    def crewai_llm(self) -> LLM:
        return LLM(
            model="openrouter/google/gemini-2.5-flash",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )


class DevstralOpenRouter(BaseLLM):
    """Devstral Small via OpenRouter — Mistral coding specialist, 68% SWE-bench (Dec 2025/Jan 2026)
    Purpose-built for software engineering: patch generation, bug fixing, repo navigation."""
    @property
    def name(self) -> str:
        return "devstral-small"

    @property
    def backend(self) -> str:
        return "cloud"

    @property
    def crewai_llm(self) -> LLM:
        return LLM(
            model="openrouter/mistralai/devstral-small",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )


class Qwen36OpenRouter(BaseLLM):
    """Qwen3.5-27B via OpenRouter — 27B dense model, non-thinking, reliable text output for patch generation"""
    @property
    def name(self) -> str:
        return "qwen3.5-27b-openrouter"

    @property
    def backend(self) -> str:
        return "cloud"

    @property
    def crewai_llm(self) -> LLM:
        return LLM(
            model="openrouter/qwen/qwen3.5-27b",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )


class Qwen3CoderNext(BaseLLM):
    """Qwen3-Coder-Next via OpenRouter — latest coding specialist"""
    @property
    def name(self) -> str:
        return "qwen3-coder-next"

    @property
    def backend(self) -> str:
        return "cloud"

    @property
    def crewai_llm(self) -> LLM:
        return LLM(
            model="openrouter/qwen/qwen3-coder-next",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )


# ─────────────────────────────────────────────────────────────────────
# Google AI Studio — FREE cloud models (no Ollama needed)
# Get API key: aistudio.google.com
# ─────────────────────────────────────────────────────────────────────
class GeminiFlashLite(BaseLLM):
    """Gemini 3.1 Flash-Lite — 78% SWE-bench, free 1500 req/day"""
    @property
    def name(self) -> str:
        return "gemini-3.1-flash-lite"

    @property
    def backend(self) -> str:
        return "cloud"

    @property
    def crewai_llm(self) -> LLM:
        return LLM(
            model="gemini/gemini-3.1-flash-lite-preview",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )


class Gemini25Flash(BaseLLM):
    """Gemini 2.5 Flash — strong coding, free 1500 req/day"""
    @property
    def name(self) -> str:
        return "gemini-2.5-flash"

    @property
    def backend(self) -> str:
        return "cloud"

    @property
    def crewai_llm(self) -> LLM:
        return LLM(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
