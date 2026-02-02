"""Memory system interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemorySystem(ABC):
    """Abstract memory system interface."""

    @abstractmethod
    def search(
        self,
        user_id: str,
        query: str,
        *,
        top_k_memories: Optional[int] = None,
        search_method: str = "hybrid",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search memories."""
        pass

    def build_memory(
        self,
        user_id: str,
        dialogue: str,
    ) -> Dict[str, Any]:
        """Build memory from dialogue. Returns dict with status and other info."""
        raise NotImplementedError("build_memory must be implemented by subclass")
