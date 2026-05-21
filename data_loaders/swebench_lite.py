from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import pandas as pd

from agents.dataset import TaskEntry  # canonical definition — do not redefine here


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class SWEBenchLiteInstance:
    instance_id: str
    repo: str
    base_commit: str
    created_at: str
    version: str
    problem_statement: str
    hints_text: str
    patch: str
    test_patch: str
    fail_to_pass: List[str]
    pass_to_pass: List[str]
    environment_setup_commit: str
    difficulty: Difficulty
    patch_size: int
    hunks: int
    files_changed: int


class SWEBenchLite:
    name = "SWE-bench-Lite"

    def __init__(
        self,
        sample_size: int = 15,
        seed: int = 42,
        cache_dir: str = "./datasets",
        use_fixed_thesis_subset: bool = True,
    ):
        self.sample_size = sample_size
        self.seed = seed
        self.cache_dir = Path(cache_dir)
        self.use_fixed_thesis_subset = use_fixed_thesis_subset

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._all_instances: List[SWEBenchLiteInstance] = []
        self._selected_instances: List[SWEBenchLiteInstance] = []

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def load(self) -> List[TaskEntry]:
        selected = self._get_selected_instances()
        return [self._to_task_entry(inst) for inst in selected]

    def get_instance(self, task_id: str) -> Optional[SWEBenchLiteInstance]:
        self._ensure_loaded()
        for inst in self._all_instances:
            if inst.instance_id == task_id:
                return inst
        return None

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------

    def _ensure_loaded(self):
        if self._all_instances:
            return

        parquet_path = self.cache_dir / "swebench_lite_test.parquet"
        if not parquet_path.exists():
            self._download_test_parquet(parquet_path)

        df = pd.read_parquet(parquet_path)
        self._all_instances = [self._row_to_instance(row) for _, row in df.iterrows()]
        print(f"[Dataset] Loaded {len(self._all_instances)} SWE-bench Lite test instances")

    def _download_test_parquet(self, out_path: Path):
        api_url = "https://huggingface.co/api/datasets/SWE-bench/SWE-bench_Lite/parquet"
        meta = json.load(urllib.request.urlopen(api_url))
        test_url = meta["default"]["test"][0]
        urllib.request.urlretrieve(test_url, out_path)
        print(f"[Dataset] Downloaded SWE-bench Lite test parquet → {out_path}")

    # -----------------------------------------------------------------
    # Selection
    # -----------------------------------------------------------------

    def _get_selected_instances(self) -> List[SWEBenchLiteInstance]:
        self._ensure_loaded()

        if self._selected_instances:
            return self._selected_instances

        if self.use_fixed_thesis_subset:
            selected = self._fixed_15_task_subset()
        else:
            selected = self._balanced_sample(self.sample_size)

        self._selected_instances = selected

        print(f"[Dataset] Using SWE-bench Lite subset ({len(selected)} tasks)")
        for inst in selected:
            print(
                f"[SWE-lite] {inst.instance_id} | {inst.repo} | "
                f"{inst.difficulty.value} | patch={inst.patch_size} | "
                f"fail_tests={len(inst.fail_to_pass)} | files={inst.files_changed}"
            )

        return selected

    def _fixed_15_task_subset(self) -> List[SWEBenchLiteInstance]:
        """
        Fixed thesis subset: 5 easy + 5 medium + 5 hard tasks.
        Spread across django, sphinx, sympy, astropy, matplotlib — four different
        repos — to avoid repo-specific memorisation bias.
        These are real instances from the official SWE-bench Lite test split.
        """
        target_ids = [
            # ── EASY (5) ──────────────────────────────────────────────
            "django__django-11179",
            "django__django-13230",
            "sphinx-doc__sphinx-8595",
            "sphinx-doc__sphinx-8721",
            "sympy__sympy-21627",
            # ── MEDIUM (5) ────────────────────────────────────────────
            "astropy__astropy-6938",
            "astropy__astropy-12907",
            "django__django-15814",
            "django__django-12908",
            "matplotlib__matplotlib-25433",
            # ── HARD (5) ──────────────────────────────────────────────
            "astropy__astropy-14182",
            "django__django-14672",
            "django__django-11099",
            "matplotlib__matplotlib-23913",
            "matplotlib__matplotlib-26020",
        ]

        by_id = {inst.instance_id: inst for inst in self._all_instances}
        selected = [by_id[i] for i in target_ids if i in by_id]

        if len(selected) != len(target_ids):
            missing = [i for i in target_ids if i not in by_id]
            raise ValueError(f"Missing expected SWE-bench Lite instances: {missing}")

        return selected[: self.sample_size]

    def _balanced_sample(self, sample_size: int) -> List[SWEBenchLiteInstance]:
        rng = random.Random(self.seed)

        buckets = {
            Difficulty.EASY: [],
            Difficulty.MEDIUM: [],
            Difficulty.HARD: [],
        }
        for inst in self._all_instances:
            buckets[inst.difficulty].append(inst)

        per_bucket = max(1, sample_size // 3)
        selected: List[SWEBenchLiteInstance] = []

        for diff in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
            items = buckets[diff][:]
            rng.shuffle(items)
            selected.extend(items[:per_bucket])

        remaining = sample_size - len(selected)
        if remaining > 0:
            leftovers = [x for x in self._all_instances if x not in selected]
            rng.shuffle(leftovers)
            selected.extend(leftovers[:remaining])

        return selected[:sample_size]

    # -----------------------------------------------------------------
    # Conversion helpers
    # -----------------------------------------------------------------

    def _row_to_instance(self, row) -> SWEBenchLiteInstance:
        fail_to_pass = self._safe_json_list(row.get("FAIL_TO_PASS"))
        pass_to_pass = self._safe_json_list(row.get("PASS_TO_PASS"))
        patch = row.get("patch") or ""

        difficulty = self._heuristic_difficulty(
            patch=patch,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            problem_statement=row.get("problem_statement") or "",
        )

        return SWEBenchLiteInstance(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            created_at=row.get("created_at", ""),
            version=row.get("version", ""),
            problem_statement=row.get("problem_statement", ""),
            hints_text=row.get("hints_text", ""),
            patch=patch,
            test_patch=row.get("test_patch", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            environment_setup_commit=row.get("environment_setup_commit", ""),
            difficulty=difficulty,
            patch_size=self._patch_lines(patch),
            hunks=self._hunk_count(patch),
            files_changed=self._files_changed(patch),
        )

    def _to_task_entry(self, inst: SWEBenchLiteInstance) -> TaskEntry:
        hints_section = (
            f"\n\nHints (from maintainers):\n{inst.hints_text.strip()}"
            if inst.hints_text and inst.hints_text.strip()
            else ""
        )
        description = f"""
    Repository: {inst.repo}

    Problem:
    {inst.problem_statement}{hints_section}

    You must generate a VALID git patch.

    STRICT REQUIREMENTS:
    - Output ONLY a unified git diff
    - MUST start with: diff --git
    - MUST include: --- a/ and +++ b/
    - MUST include at least one @@ hunk
    - NO explanations
    - NO markdown
    """

        test_cases = (
            f"FAIL_TO_PASS: {json.dumps(inst.fail_to_pass)}\n"
            f"PASS_TO_PASS: {json.dumps(inst.pass_to_pass)}"
        )

        signature = (
            f"Repository: {inst.repo}\n"
            f"Base commit: {inst.base_commit}\n"
            f"Difficulty: {inst.difficulty.value}"
        )

        return TaskEntry(
            task_id=inst.instance_id,
            description=description,
            signature=signature,
            function_name="generate_patch",   
            test_cases=test_cases,
        )

    # -----------------------------------------------------------------
    # Heuristics
    # -----------------------------------------------------------------

    def _heuristic_difficulty(
        self,
        patch: str,
        fail_to_pass: List[str],
        pass_to_pass: List[str],
        problem_statement: str,
    ) -> Difficulty:
        patch_lines = self._patch_lines(patch)
        hunks = self._hunk_count(patch)
        files_changed = self._files_changed(patch)
        issue_words = len((problem_statement or "").split())

        score = 0.0
        score += min(patch_lines, 60) / 15
        score += min(hunks, 6) * 1.5
        score += min(len(fail_to_pass), 6) * 1.5
        score += min(len(pass_to_pass), 20) / 10
        score += min(issue_words, 240) / 60
        score += max(files_changed - 1, 0) * 2

        if score < 6:
            return Difficulty.EASY
        elif score < 11:
            return Difficulty.MEDIUM
        return Difficulty.HARD

    @staticmethod
    def _safe_json_list(value) -> List[str]:
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    @staticmethod
    def _patch_lines(patch: str) -> int:
        if not isinstance(patch, str):
            return 0
        return sum(
            1
            for line in patch.splitlines()
            if (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
        )

    @staticmethod
    def _hunk_count(patch: str) -> int:
        if not isinstance(patch, str):
            return 0
        return sum(1 for line in patch.splitlines() if line.startswith("@@"))

    @staticmethod
    def _files_changed(patch: str) -> int:
        if not isinstance(patch, str):
            return 0
        return sum(1 for line in patch.splitlines() if line.startswith("diff --git"))

    @staticmethod
    def _extract_test_files(test_names: List[str]) -> List[str]:
        files = []
        for t in test_names:
            if "::" in t:
                files.append(t.split("::", 1)[0])
            elif "." in t and "/" in t:
                files.append(t)
        return sorted(set(files))