"""
agents/schemas.py — Typed Pydantic schemas for inter-agent communication.

Used as output_pydantic on CrewAI Tasks so agents exchange structured objects
instead of free text that needs regex parsing downstream.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class TesterVerdict(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    issues: list[str]           # each identified issue as one sentence
    failure_types: list[str]    # matched bug pattern names (may be empty)


class ReviewerDecision(BaseModel):
    approved: bool
    feedback: str               # one sentence
    concerns: list[str]         # specific concerns (may be empty)


class DiagnosisReport(BaseModel):
    root_cause: str
    failure_class: list[str]    # matched pattern names
    current_files: list[str]    # files the current patch targets
    correct_files: list[str]    # inferred correct files
    fix_steps: list[str]        # numbered fix instructions
