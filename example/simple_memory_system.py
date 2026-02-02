"""Simple memory system implementation using JSON file storage."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from openai import OpenAI

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import API_BASE_URL, API_KEY, LLM_MODEL
from plugin.memory_system import MemorySystem


class SimpleMemorySystem(MemorySystem):
    """Simple memory system that stores memories in JSON files."""

    def __init__(
        self,
        storage_dir: str = "memory_storage",
        api_key: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._memories: Dict[str, List[Dict[str, Any]]] = {}
        self.api_key = api_key or API_KEY
        self.api_base = api_base or API_BASE_URL
        self.model = model or LLM_MODEL
        self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)

    def search(
        self,
        user_id: str,
        query: str,
        *,
        top_k_memories: Optional[int] = None,
        search_method: str = "hybrid",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search memories - simple keyword matching."""
        memories = self._load_user_memories(user_id)
        query_lower = query.lower()
        matched = []
        for mem in memories:
            content = mem.get("content", "").lower()
            if query_lower in content or any(word in content for word in query_lower.split() if len(word) > 2):
                matched.append(mem)
        top_k = top_k_memories or 10
        return {
            "memories": matched[:top_k],
        }

    def _load_user_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Load memories for a user from JSON file."""
        if user_id in self._memories:
            return self._memories[user_id]
        file_path = self.storage_dir / f"{user_id}_memories.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self._memories[user_id] = json.load(f)
            except Exception:
                self._memories[user_id] = []
        else:
            self._memories[user_id] = []
        return self._memories[user_id]

    def _save_user_memories(self, user_id: str, memories: List[Dict[str, Any]]) -> None:
        """Save memories for a user to JSON file."""
        self._memories[user_id] = memories
        file_path = self.storage_dir / f"{user_id}_memories.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2, ensure_ascii=False)

    def save_to_file(self, user_id: str, file_path: str) -> None:
        """Save memories to a specific file path."""
        memories = self._load_user_memories(user_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2, ensure_ascii=False)

    def load_from_file(self, user_id: str, file_path: str) -> None:
        """Load memories from a specific file path."""
        if Path(file_path).exists():
            with open(file_path, "r", encoding="utf-8") as f:
                self._memories[user_id] = json.load(f)
                self._save_user_memories(user_id, self._memories[user_id])

    def add_correction_memories(
        self, user_id: str, failed_qa: List[Dict[str, Any]]
    ) -> int:
        """Add correction memories from failed QA pairs."""
        if not failed_qa:
            return 0
        
        existing_memories = self._load_user_memories(user_id)
        current_time = datetime.now().isoformat()
        added_count = 0
        
        for qa in failed_qa:
            question = qa.get("question", "").strip()
            true_answer = qa.get("true_answer", "").strip()
            if not question or not true_answer:
                continue
            
            correction_content = f"Question: {question}. Correct Answer: {true_answer}."
            
            memory_entry = {
                "memory_id": str(uuid.uuid4()),
                "user_id": user_id,
                "content": correction_content,
                "timestamp": current_time,
                "created_at": current_time,
                "metadata": {
                    "source": "correction",
                    "question": question,
                    "true_answer": true_answer,
                    "type": "qa_correction",
                },
            }
            existing_memories.append(memory_entry)
            added_count += 1
        
        if added_count > 0:
            self._save_user_memories(user_id, existing_memories)
        
        return added_count

    def build_memory(
        self, user_id: str, dialogue: str
    ) -> Dict[str, Any]:
        """Build memory from dialogue using LLM prompt summarization and store as JSON."""
        try:
            prompt = f"""You are a memory summarization expert. Please analyze the following dialogue content, extract key information and summarize it into memories.

Dialogue content:
{dialogue}

Please summarize the dialogue into structured memories. Return a JSON object containing the following fields:
{{
    "summary": "Brief summary of the dialogue (1-2 sentences)",
    "memories": [
        {{
            "content": "Detailed memory content including key facts, people, events, time, etc. Use third-person narrative and ensure all important details are included.",
            "timestamp": "Timestamp in YYYY-MM-DDTHH:MM:SS format (extracted or inferred from dialogue)"
        }}
    ]
}}

Requirements:
1. The summary should include all key information from the dialogue: people, events, time, location, decisions, emotions, etc.
2. Extract timestamps from the dialogue; if no explicit time is available, use the current time
3. Memory content should be detailed and searchable, including specific keywords
4. If the dialogue involves multiple topics, you can generate multiple memory entries
5. Use third-person narrative
6. Ensure time information is precise to the hour

Return only the JSON object, do not add any other text:"""

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            content = (response.choices[0].message.content or "").strip()
            if content.startswith("```"):
                content = content.split("```")[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()

            data = json.loads(content)
            summary = data.get("summary", "")
            memories_data = data.get("memories", [])

            existing_memories = self._load_user_memories(user_id)
            current_time = datetime.now().isoformat()
            for mem_data in memories_data:
                memory_entry = {
                    "memory_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "content": mem_data.get("content", ""),
                    "timestamp": mem_data.get("timestamp", current_time),
                    "created_at": mem_data.get("timestamp", current_time),
                    "metadata": {
                        "source": "dialogue_summarization",
                        "summary": summary,
                    },
                }
                existing_memories.append(memory_entry)

            self._save_user_memories(user_id, existing_memories)

            return {
                "status": "built",
                "summary": summary,
                "memories_count": len(memories_data),
                "total_memories": len(existing_memories),
            }

        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"Failed to parse JSON response: {e}",
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to build memory: {e}",
            }
