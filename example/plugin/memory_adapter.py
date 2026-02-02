"""Memory adapter interface."""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory_system import MemorySystem


class MemoryAdapter(ABC):
    """Abstract adapter interface for memory update and reconstruction."""

    @abstractmethod
    def update(
        self,
        memory: MemorySystem,
        user_id: str,
        session_dialogue: str,
        failed_qa: List[Dict[str, Any]],
        failed_reasons: Optional[List[str]] = None,
    ) -> str:
        """Filter and update dialogue based on evaluation results. Returns filtered dialogue."""
        pass

    @abstractmethod
    def reconstruct(
        self,
        memory: MemorySystem,
        user_id: str,
        filtered_dialogue: str,
        failed_qa: List[Dict[str, Any]],
        batch_size: int = 20,
        max_retries: int = 5,
    ) -> bool:
        """Reconstruct memory with filtered dialogue and error corrections. Returns success status."""
        pass
