
from __future__ import annotations

import ast
import json
import logging
import random
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from agents.dataset import BaseDataset, TaskEntry

logger = logging.getLogger(__name__)


# ── Difficulty enum (based on gold-patch size terciles) ──────────────

class Difficulty(Enum):
    EASY = "easy"      # patch_len <= ~5300 chars
    MEDIUM = "medium"  # patch_len ~5300-11300
    HARD = "hard"      # patch_len > ~11300 chars


# ── Internal rich dataclass (kept for validation / eval later) ───────

@dataclass
class SWEBenchProInstance:
    """Full SWE-bench Pro instance — superset of what TaskEntry carries."""
    instance_id: str
    repo: str
    repo_language: str
    base_commit: str
    problem_statement: str
    patch: str                          # gold patch (for evaluation)
    test_patch: str
    requirements: Optional[str]
    interface: Optional[str]
    fail_to_pass: List[str]
    pass_to_pass: List[str]
    issue_specificity: List[str]
    issue_categories: List[str]
    before_repo_set_cmd: str
    selected_test_files_to_run: List[str]
    dockerhub_tag: str
    difficulty: Difficulty = Difficulty.MEDIUM
    patch_size: int = 0

    @property
    def num_tests_to_fix(self) -> int:
        return len(self.fail_to_pass)

    @property
    def num_tests_to_preserve(self) -> int:
        return len(self.pass_to_pass)


# ── Pre-selected 21-task test subset ─────────────────────────────────
# 7 easy + 7 medium + 7 hard, across 7 repos, 4 languages, seed=42.

# 15-task curated thesis subset: 5 repos × 3 difficulty levels.
# Repos chosen for language diversity and consistent presence across difficulties:
#   NodeBB/NodeBB          → JavaScript
#   ansible/ansible        → Python
#   flipt-io/flipt         → Go
#   gravitational/teleport → Go
#   internetarchive/openlibrary → Python
# element-hq and future-architect excluded (redundant language coverage).
# All 15 IDs verified present in swebench_pro_cache.json (seed=42).

# 3-task smoke subset: 1 easy + 1 medium + 1 hard, across 3 repos/languages.
# Used for quick pipeline validation before full VM run.
#   NodeBB  (JS,     easy)   → fast patch, JS coverage
#   ansible (Python, medium) → multi-file Python, typical thesis task
#   flipt   (Go,     hard)   → large patch, Go coverage
SMOKE_SUBSET_IDS: List[str] = [
    "instance_NodeBB__NodeBB-da0211b1a001d45d73b4c84c6417a4f1b0312575-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e",
    "instance_ansible__ansible-deb54e4c5b32a346f1f0b0a14f1c713d2cc2e961-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5",
    "instance_flipt-io__flipt-0fd09def402258834b9d6c0eaa6d3b4ab93b4446",
]

TEST_SUBSET_IDS: List[str] = [
    # ── EASY (5) ── patch ~1900–3900 chars ──────────────────────────
    "instance_NodeBB__NodeBB-da0211b1a001d45d73b4c84c6417a4f1b0312575-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e",
    "instance_ansible__ansible-0ea40e09d1b35bcb69ff4d9cecf3d0defa4b36e8-v30a923fb5c164d6cd18280c02422f75e611e8fb2",
    "instance_flipt-io__flipt-2eac0df47b5ecc8bb05002d80383ceb08ab3620a",
    "instance_gravitational__teleport-1a77b7945a022ab86858029d30ac7ad0d5239d00-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_internetarchive__openlibrary-03095f2680f7516fca35a58e665bf2a41f006273-v8717e18970bcdc4e0d2cea3b1527752b21e74866",
    # ── MEDIUM (5) ── patch ~6500–11000 chars ───────────────────────
    "instance_NodeBB__NodeBB-8ca65b0c78c67c1653487c02d1135e1b702185e1-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e",
    "instance_ansible__ansible-deb54e4c5b32a346f1f0b0a14f1c713d2cc2e961-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5",
    "instance_flipt-io__flipt-af7a0be46d15f0b63f16a868d13f3b48a838e7ce",
    "instance_gravitational__teleport-1330415d33a27594c948a36d9d7701f496229e9f",
    "instance_internetarchive__openlibrary-757fcf46c70530739c150c57b37d6375f155dc97-ve8c8d62a2b60610a3c4631f5f23ed866bada9818",
    # ── HARD (5) ── patch ~11500–37000 chars ────────────────────────
    "instance_NodeBB__NodeBB-b398321a5eb913666f903a794219833926881a8f-vd59a5728dfc977f44533186ace531248c2917516",
    "instance_ansible__ansible-1bd7dcf339dd8b6c50bc16670be2448a206f4fdb-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5",
    "instance_flipt-io__flipt-0fd09def402258834b9d6c0eaa6d3b4ab93b4446",
    "instance_gravitational__teleport-3fa6904377c006497169945428e8197158667910-v626ec2a48416b10a88641359a169d99e935ff037",
    "instance_internetarchive__openlibrary-d40ec88713dc95ea791b252f92d2f7b75e107440-v13642507b4fc1f8d234172bf8129942da2c2ca26",
]


# =====================================================================
#  Public API — implements BaseDataset
# =====================================================================

class SWEBenchPro(BaseDataset):
    """
    SWE-bench Pro dataset loader.

    Maps every SWE-bench Pro instance to a TaskEntry so the existing
    CrewAI orchestrator can consume it identically to BigCodeBench-Hard.

    Mapping to TaskEntry fields:
        task_id        → instance_id
        description    → full agent prompt (repo info + issue + instructions)
        function_name  → "generate_patch" (constant — output is a diff)
        signature      → repo context summary (repo, lang, commit, docker tag)
        test_cases     → fail_to_pass test list + pass_to_pass test list
    """

    name = "SWE-bench-Pro"

    def __init__(
        self,
        sample_size: int = 21,
        seed: int = 42,
        use_test_subset: bool = True,
        smoke_test: bool = False,
        cache_dir: str = "./datasets",
    ):
        self.sample_size = sample_size
        self.seed = seed
        self.use_test_subset = use_test_subset
        self.smoke_test = smoke_test          # True → use SMOKE_SUBSET_IDS (1 easy+1 med+1 hard)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._all_instances: Optional[List[SWEBenchProInstance]] = None
        self._index: Dict[str, SWEBenchProInstance] = {}

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def load(self) -> List[TaskEntry]:
        """Load and return TaskEntry list for the orchestrator."""
        raw = self._get_selected_instances()

        tasks: List[TaskEntry] = []
        for inst in raw:
            tasks.append(self._to_task_entry(inst))

        for t in tasks:
            print(f"\n  [Task] {t.task_id}")
            print(f"  Desc : {t.description[:120]}...")

        return tasks

    # ------------------------------------------------------------------
    # Rich instance access (for validation / detailed eval)
    # ------------------------------------------------------------------

    def get_instance(self, instance_id: str) -> Optional[SWEBenchProInstance]:
        """Retrieve the full SWEBenchProInstance by ID (for validation)."""
        self._ensure_loaded()
        return self._index.get(instance_id)

    def get_all_instances(self) -> List[SWEBenchProInstance]:
        """All 731 raw instances."""
        self._ensure_loaded()
        return self._all_instances

    # ------------------------------------------------------------------
    # Mapping SWEBenchProInstance → TaskEntry
    # ------------------------------------------------------------------

    @staticmethod
    def _to_task_entry(inst: SWEBenchProInstance) -> TaskEntry:
        """
        Convert a SWEBenchProInstance into a TaskEntry that the existing
        CrewAI pipeline can consume.

        Key differences from BigCodeBench:
            - description is the full issue + repo context + instructions
            - function_name is always "generate_patch"
            - signature carries repo metadata
            - test_cases lists the fail-to-pass and pass-to-pass tests
        """

        # ── description: full prompt for the planner agent ───────
        desc_parts = [
            f"## Repository: {inst.repo}",
            f"## Language: {inst.repo_language}",
            f"## Base Commit: {inst.base_commit}",
            f"## Difficulty: {inst.difficulty.value}",
            "",
            "## Issue Description",
            inst.problem_statement,
        ]
        if inst.requirements:
            desc_parts += ["", "## Requirements", inst.requirements]
        if inst.interface:
            desc_parts += ["", "## Interface / API", inst.interface]
        if inst.issue_categories:
            desc_parts += ["", "## Issue Category", ", ".join(inst.issue_categories)]

        if inst.pass_to_pass:
            desc_parts += [
                "",
                "## Regression tests (must REMAIN passing — do not break these)",
            ]
            for t in inst.pass_to_pass[:8]:
                desc_parts.append(f"  - {t}")
            if len(inst.pass_to_pass) > 8:
                desc_parts.append(f"  ... and {len(inst.pass_to_pass) - 8} more")

        desc_parts += [
            "",
            "## Task",
            "Generate a git patch (unified diff format) that resolves the issue above.",
            "The patch should:",
            "  1. Fix the described problem",
            "  2. Not break any existing tests",
            f"  3. Make the following {inst.num_tests_to_fix} failing test(s) pass:",
        ]
        for t in inst.fail_to_pass:
            desc_parts.append(f"     - {t}")

        description = "\n".join(desc_parts)

        # ── signature: repo metadata for context ─────────────────
        signature = (
            f"Repository: {inst.repo}\n"
            f"Language: {inst.repo_language}\n"
            f"Base Commit: {inst.base_commit}\n"
            f"Docker Image: jefzda/sweap-images:{inst.dockerhub_tag}\n"
            f"Difficulty: {inst.difficulty.value}\n"
            f"Gold patch size: {inst.patch_size} chars"
        )

        # ── test_cases: structured test info ─────────────────────
        test_lines = ["## Tests that must PASS after the patch (fail_to_pass):"]
        for t in inst.fail_to_pass:
            test_lines.append(f"  - {t}")
        test_lines.append("")
        test_lines.append("## Tests that must REMAIN passing (pass_to_pass):")
        for t in inst.pass_to_pass:
            test_lines.append(f"  - {t}")
        test_lines.append("")
        test_lines.append(f"## Test files to run: {', '.join(inst.selected_test_files_to_run)}")

        test_cases = "\n".join(test_lines)

        return TaskEntry(
            task_id=inst.instance_id,
            description=description,
            function_name="generate_patch",
            signature=signature,
            test_cases=test_cases,
        )

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._all_instances is None:
            self._all_instances = self._load_all()
            self._assign_difficulty(self._all_instances)
            self._index = {i.instance_id: i for i in self._all_instances}

    def _get_selected_instances(self) -> List[SWEBenchProInstance]:
        """Return the subset that the user configured."""
        self._ensure_loaded()

        if self.smoke_test:
            selected = [self._index[iid] for iid in SMOKE_SUBSET_IDS if iid in self._index]
            print(
                f"[Dataset] Loading {self.name} — SMOKE TEST subset "
                f"({len(selected)} tasks: 1 easy + 1 medium + 1 hard)"
            )
            return selected

        if self.use_test_subset and self.sample_size <= 21:
            # Use the curated 21-task subset
            ids = TEST_SUBSET_IDS[: self.sample_size]
            selected = [self._index[iid] for iid in ids if iid in self._index]
            print(
                f"[Dataset] Loading {self.name} — curated test subset "
                f"({len(selected)} of {len(self._all_instances)} total instances)"
            )
            return selected

        # Random balanced sample
        print(
            f"[Dataset] Loading {self.name} — sampling {self.sample_size} "
            f"of {len(self._all_instances)} instances (seed={self.seed})"
        )
        return self._balanced_sample(self.sample_size, self.seed)

    def _load_all(self) -> List[SWEBenchProInstance]:
        """Load all 731 instances from local cache or HuggingFace."""
        cache_path = self.cache_dir / "swebench_pro_cache.json"

        if cache_path.exists():
            print(f"[Dataset] Loading from cache: {cache_path}")
            return self._load_from_cache(cache_path)

        # Fallback: download from HuggingFace
        print("[Dataset] Cache not found — downloading from HuggingFace...")
        instances = self._download_and_parse()
        self._save_to_cache(cache_path, instances)
        return instances

    def _download_and_parse(self) -> List[SWEBenchProInstance]:
        """Download from HuggingFace (fallback if cache missing)."""
        try:
            import importlib
            hf_datasets = importlib.import_module("datasets")
            if not hasattr(hf_datasets, "load_dataset"):
                raise ImportError("Not the HuggingFace datasets library")
            load_dataset = hf_datasets.load_dataset
        except (ImportError, AttributeError):
            raise ImportError(
                "Cache file not found and HuggingFace `datasets` not installed.\n"
                "Either place swebench_pro_cache.json in ./datasets/ or:\n"
                "  pip install datasets"
            )

        ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
        return [self._row_to_instance(row) for row in ds]

    @staticmethod
    def _row_to_instance(row: dict) -> SWEBenchProInstance:
        def _parse_list(v):
            if not v:
                return []
            if isinstance(v, list):
                return v
            try:
                result = ast.literal_eval(v)
                return result if isinstance(result, list) else [result]
            except (ValueError, SyntaxError):
                return [v] if v else []

        return SWEBenchProInstance(
            instance_id=row["instance_id"],
            repo=row["repo"],
            repo_language=row["repo_language"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"] or "",
            patch=row["patch"] or "",
            test_patch=row["test_patch"] or "",
            requirements=row.get("requirements"),
            interface=row.get("interface"),
            fail_to_pass=_parse_list(row["fail_to_pass"]),
            pass_to_pass=_parse_list(row["pass_to_pass"]),
            issue_specificity=_parse_list(row.get("issue_specificity")),
            issue_categories=_parse_list(row.get("issue_categories")),
            before_repo_set_cmd=row.get("before_repo_set_cmd", "") or "",
            selected_test_files_to_run=_parse_list(row.get("selected_test_files_to_run")),
            dockerhub_tag=row.get("dockerhub_tag", "") or "",
            patch_size=len(row["patch"]) if row.get("patch") else 0,
        )

    # ── Subset sampling ──────────────────────────────────────────

    def _balanced_sample(self, n: int, seed: int) -> List[SWEBenchProInstance]:
        rng = random.Random(seed)
        per_diff = n // 3
        remainder = n - per_diff * 3
        selected: List[SWEBenchProInstance] = []

        for i, diff in enumerate([Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]):
            count = per_diff + (1 if i < remainder else 0)
            pool = [inst for inst in self._all_instances if inst.difficulty == diff]

            by_repo: Dict[str, List[SWEBenchProInstance]] = {}
            for inst in pool:
                by_repo.setdefault(inst.repo, []).append(inst)

            picked: List[SWEBenchProInstance] = []
            repos = list(by_repo.keys())
            rng.shuffle(repos)
            for repo in repos:
                if len(picked) >= count:
                    break
                picked.append(rng.choice(by_repo[repo]))

            remaining = [inst for inst in pool if inst not in picked]
            rng.shuffle(remaining)
            while len(picked) < count and remaining:
                picked.append(remaining.pop())

            selected.extend(picked[:count])

        return selected

    # ── Difficulty assignment ────────────────────────────────────

    @staticmethod
    def _assign_difficulty(instances: List[SWEBenchProInstance]):
        sorted_inst = sorted(instances, key=lambda x: x.patch_size)
        n = len(sorted_inst)
        for i, inst in enumerate(sorted_inst):
            if i < n // 3:
                inst.difficulty = Difficulty.EASY
            elif i < 2 * n // 3:
                inst.difficulty = Difficulty.MEDIUM
            else:
                inst.difficulty = Difficulty.HARD

    # ── Cache I/O ────────────────────────────────────────────────

    @staticmethod
    def _save_to_cache(path: Path, instances: List[SWEBenchProInstance]):
        data = []
        for inst in instances:
            data.append({
                "instance_id": inst.instance_id,
                "repo": inst.repo,
                "repo_language": inst.repo_language,
                "base_commit": inst.base_commit,
                "problem_statement": inst.problem_statement,
                "patch": inst.patch,
                "test_patch": inst.test_patch,
                "requirements": inst.requirements,
                "interface": inst.interface,
                "fail_to_pass": inst.fail_to_pass,
                "pass_to_pass": inst.pass_to_pass,
                "issue_specificity": inst.issue_specificity,
                "issue_categories": inst.issue_categories,
                "before_repo_set_cmd": inst.before_repo_set_cmd,
                "selected_test_files_to_run": inst.selected_test_files_to_run,
                "dockerhub_tag": inst.dockerhub_tag,
                "patch_size": inst.patch_size,
            })
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[Dataset] Cached {len(data)} instances to {path}")

    def _load_from_cache(self, path: Path) -> List[SWEBenchProInstance]:
        with open(path) as f:
            data = json.load(f)
        return [
            SWEBenchProInstance(
                instance_id=d["instance_id"],
                repo=d["repo"],
                repo_language=d["repo_language"],
                base_commit=d["base_commit"],
                problem_statement=d["problem_statement"],
                patch=d["patch"],
                test_patch=d["test_patch"],
                requirements=d.get("requirements"),
                interface=d.get("interface"),
                fail_to_pass=d.get("fail_to_pass", []),
                pass_to_pass=d.get("pass_to_pass", []),
                issue_specificity=d.get("issue_specificity", []),
                issue_categories=d.get("issue_categories", []),
                before_repo_set_cmd=d.get("before_repo_set_cmd", ""),
                selected_test_files_to_run=d.get("selected_test_files_to_run", []),
                dockerhub_tag=d.get("dockerhub_tag", ""),
                patch_size=d.get("patch_size", 0),
            )
            for d in data
        ]
