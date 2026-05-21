"""
BigCodeBench-Hard dataset loader.
sample_size=1  → 1 task for pipeline testing
sample_size=25 → full thesis run
"""
from __future__ import annotations
import random
from typing import List
from agents.dataset import BaseDataset, TaskEntry


class BigCodeBenchHard(BaseDataset):
    name = "BigCodeBench-Hard"

    def __init__(self, sample_size: int = 1, seed: int = 42):
        self.sample_size = sample_size
        self.seed = seed

    def load(self) -> List[TaskEntry]:
        from datasets import load_dataset

        print(f"[Dataset] Loading {self.name} (sample_size={self.sample_size})...")
        try:
            ds_full = load_dataset("bigcode/bigcodebench", split="v0.1.4")
            ds_hard = load_dataset("bigcode/bigcodebench-hard", split="v0.1.2")
            hard_ids = {t["task_id"] for t in ds_hard}
            pool = [t for t in ds_full if t["task_id"] in hard_ids]
            print(f"[Dataset] {len(pool)} hard tasks available.")
        except Exception as e:
            print(f"[Dataset] Hard split unavailable ({e}). Using full dataset.")
            pool = list(load_dataset("bigcode/bigcodebench", split="v0.1.4"))

        random.seed(self.seed)
        sample = random.sample(pool, min(self.sample_size, len(pool)))

        tasks = [
            TaskEntry(
                task_id=t["task_id"],
                description=t["instruct_prompt"],
                function_name=t.get("entry_point", "task_func"),
                signature=t.get("code_prompt", ""),
                test_cases=t.get("test", ""),
            )
            for t in sample
        ]

        for task in tasks:
            print(f"\n  [Task] {task.task_id} | {task.function_name}")
            print(f"  Desc : {task.description[:120]}...")

        return tasks
    