"""
evaluation/patch_similarity.py

Gold-patch comparison metrics for the MAS vs Single thesis experiment.

All metrics operate on raw patch strings — no Docker, no internet required.
The SWEBenchLite and SWEBenchPro datasets both provide the gold patch in
inst.patch.

Metrics
-------
file_recall      : fraction of gold-patched files that the generated patch
                   also modifies.  1.0 = touched exactly the right files.
                   Core quality indicator — if the model doesn't know WHICH
                   file to change, the patch is useless.

content_overlap  : Token-level Jaccard similarity of the changed lines
                   (additions + deletions) between the generated and gold patches.
                   Tokens are whitespace-split and lowercased, making the metric
                   robust to minor formatting/naming differences.

patch_score      : composite (0.0–1.0) weighted average:
                     0.6 * file_recall + 0.4 * content_overlap
                   Higher weight on file_recall because identifying the
                   right file is the hardest and most informative step.

Usage
-----
from evaluation.patch_similarity import score_patch

scores = score_patch(generated_patch, gold_patch)
# scores = {"file_recall": 0.5, "content_overlap": 0.2, "patch_score": 0.38,
#            "gen_files": [...], "gold_files": [...]}
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

import math
import collections

# Patch score weights — must sum to 1.0
# File recall weighted higher: identifying the correct file is the hardest step.
WEIGHT_FILE_RECALL     = 0.6
WEIGHT_CONTENT_OVERLAP = 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_files(patch: str) -> Set[str]:
    """Return the set of file paths modified by a unified diff patch."""
    files: Set[str] = set()
    for m in re.finditer(r"diff --git a/(.+?) b/", patch):
        files.add(m.group(1).strip())
    if not files:
        # Headerless diffs — fall back to '--- a/...' lines
        for m in re.finditer(r"--- a/(.+)", patch):
            path = m.group(1).strip()
            if path != "/dev/null":
                files.add(path)
    return files


def _extract_changed_tokens(patch: str) -> Set[str]:
    """
    Return the set of tokens (whitespace-split, lowercased) from all added/removed
    lines in a diff. Token-level comparison tolerates whitespace and minor naming
    differences that would cause line-exact Jaccard to score 0.0 on logically
    similar fixes.
    """
    tokens: Set[str] = set()
    for line in patch.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            for tok in line[1:].strip().lower().split():
                if tok:
                    tokens.add(tok)
    return tokens


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _extract_added_lines(patch: str) -> str:
    """Return only the added lines from a diff (without the leading '+')."""
    lines = []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def score_patch(generated: str, gold: str) -> Dict:
    """
    Compare a generated patch against the gold (expert) patch.

    Parameters
    ----------
    generated : str
        The unified diff produced by the agent pipeline.
    gold : str
        The gold patch from the dataset (inst.patch).

    Returns
    -------
    dict with keys:
        file_recall      float  0.0–1.0
        content_overlap  float  0.0–1.0
        patch_score      float  0.0–1.0  (weighted composite)
        gen_files        list   files the generated patch touches
        gold_files       list   files the gold patch touches
        files_correct    list   files in both (true positives)
    """
    if not generated or not gold:
        return {
            "file_recall": 0.0,
            "content_overlap": 0.0,
            "patch_score": 0.0,
            "gen_files": [],
            "gold_files": [],
            "files_correct": [],
        }

    gen_files  = _extract_files(generated)
    gold_files = _extract_files(gold)
    correct    = gen_files & gold_files

    # File recall: how many gold files did we correctly identify?
    file_recall = len(correct) / len(gold_files) if gold_files else 0.0

    # Content overlap: token-level Jaccard on changed lines
    gen_lines  = _extract_changed_tokens(generated)
    gold_lines = _extract_changed_tokens(gold)
    content_overlap = _jaccard(gen_lines, gold_lines)

    # Composite score
    patch_score = round(WEIGHT_FILE_RECALL * file_recall + WEIGHT_CONTENT_OVERLAP * content_overlap, 4)

    return {
        "file_recall":     round(file_recall, 4),
        "content_overlap": round(content_overlap, 4),
        "patch_score":     patch_score,
        "gen_files":       sorted(gen_files),
        "gold_files":      sorted(gold_files),
        "files_correct":   sorted(correct),
    }


def score_codebleu(generated: str, gold: str, language: str = "python") -> float:
    """
    Compute a code-aware BLEU score between the added lines of generated and
    gold patches.

    Implements n-gram BLEU (n=1..4) with brevity penalty on tokenised code
    lines, equivalent to the first component of CodeBLEU (Chen et al. 2022).
    No external dependencies required — the AST/dataflow components of the
    full CodeBLEU metric are not included as they require a C++ compiler on
    Windows.

    Returns 0.0 if either patch is empty.
    """
    if not generated or not gold:
        return 0.0

    gen_code  = _extract_added_lines(generated)
    gold_code = _extract_added_lines(gold)
    if not gen_code or not gold_code:
        return 0.0

    # Tokenise: split on whitespace and common code punctuation
    def _tokenize(text: str) -> List[str]:
        import re as _re
        tokens = _re.findall(r"[A-Za-z_]\w*|[0-9]+|[^\s\w]", text)
        return [t.lower() for t in tokens if t.strip()]

    hyp  = _tokenize(gen_code)
    ref  = _tokenize(gold_code)

    if not hyp or not ref:
        return 0.0

    # Brevity penalty
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))

    # Clipped n-gram precision for n = 1..4
    log_avg = 0.0
    for n in range(1, 5):
        hyp_ngrams = collections.Counter(
            tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1)
        )
        ref_ngrams = collections.Counter(
            tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)
        )
        clipped = sum(min(c, ref_ngrams[g]) for g, c in hyp_ngrams.items())
        total   = max(len(hyp) - n + 1, 0)
        if total == 0 or clipped == 0:
            return 0.0
        log_avg += math.log(clipped / total)

    return round(bp * math.exp(log_avg / 4), 4)
