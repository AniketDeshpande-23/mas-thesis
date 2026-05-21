from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TaskEntry:
    task_id: str
    description: str
    signature: str
    test_cases: str
    # Optional — used by BigCodeBench; SWE-bench tasks use "generate_patch" as a dummy
    function_name: Optional[str] = None

class BaseDataset(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self) -> List[TaskEntry]: ...