"""Simple adapter implementation."""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plugin.memory_adapter import MemoryAdapter
from plugin.memory_system import MemorySystem


class SimpleMemoryAdapter(MemoryAdapter):
    """Simple adapter that filters messages and reconstructs memory."""

    def update(
        self,
        memory: MemorySystem,
        user_id: str,
        session_dialogue: str,
        failed_qa: List[Dict[str, Any]],
        failed_reasons: Optional[List[str]] = None,
    ) -> str:
        """Filter dialogue based on failed QA pairs."""
        STOP_WORDS = {
            "how", "what", "when", "where", "why", "who", "which", "is", "are", "was", "were",
            "in", "on", "at", "for", "with", "to", "and", "or", "but", "the", "a", "an",
        }
        failed_qa = failed_qa or []
        if not failed_qa:
            return session_dialogue

        key_keywords: set[str] = set()
        for qa in failed_qa:
            true_answer = qa.get("true_answer", "").strip()
            question = qa.get("question", "").strip()
            for text in [true_answer, question]:
                if not text:
                    continue
                words = re.findall(r"\b\w+\b", text.lower())
                for word in words:
                    if len(word) >= 3 and word not in STOP_WORDS:
                        key_keywords.add(word)

        if not key_keywords:
            return session_dialogue

        lines = session_dialogue.split("\n")
        filtered_lines = []
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in key_keywords):
                filtered_lines.append(line)

        filtered_dialogue = "\n".join(filtered_lines) if filtered_lines else session_dialogue
        if not filtered_lines:
            lines_to_keep = max(1, len(lines) // 3)
            filtered_dialogue = "\n".join(lines[:lines_to_keep])
        
        return filtered_dialogue

    def reconstruct(
        self,
        memory: MemorySystem,
        user_id: str,
        filtered_dialogue: str,
        failed_qa: List[Dict[str, Any]],
        batch_size: int = 20,
        max_retries: int = 5,
    ) -> bool:
        """Reconstruct memory with filtered dialogue by summarizing and adding corrections."""
        try:
            from simple_memory_system import SimpleMemorySystem
            if isinstance(memory, SimpleMemorySystem):
                dialogue = filtered_dialogue
                if failed_qa:
                    corrections = "\n".join([
                        f"Q: {qa.get('question', '')} A: {qa.get('true_answer', '')}"
                        for qa in failed_qa
                    ])
                    dialogue += f"\n\nCorrections:\n{corrections}"
                
                result = memory.build_memory(user_id, dialogue)
                
                if result.get("status") == "built":
                    correction_count = memory.add_correction_memories(user_id, failed_qa)
                    if correction_count > 0:
                        print(f"Added {correction_count} correction memories from failed QA pairs")
                    return True
                return False
            return False
        except Exception as e:
            print(f"Reconstruction error: {e}")
            return False
