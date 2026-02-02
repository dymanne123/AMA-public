"""Memory challenger interface."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import API_BASE_URL, API_KEY, LLM_MODEL

logger = logging.getLogger("memory_challenger")


class MemoryChallenger:
    """Challenger for generating QA pairs from dialogue."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
    ):
        self.api_key = api_key or API_KEY
        self.api_base = api_base or API_BASE_URL
        self.model = model or LLM_MODEL
        self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)

    def generate_qa_pairs(self, session_dialogue: str, num_qa: int = 10) -> List[Dict[str, Any]]:
        """Generate QA pairs from dialogue session."""
        prompt = f"""Based on the following dialogue session, generate exactly {num_qa} QA pairs 
that test whether the key information from this session is properly remembered. Answer with fewest words!!!

Dialogue:
{session_dialogue}

Core Requirements (MUST follow strictly):
1. Questions must cover key facts, characters, events
2. **CRITICAL: The answer must be the core factual keywords/phrases EXACTLY extracted from the Original Messages (verbatim if possible).**
3. **CRITICAL: Your answer must match the style of the dataset's annotated answers (contain details).**
4. Return ONLY valid JSON, no extra text, no comments, no code blocks

Examples of valid QA pair:
{{"question": "What instruments does Melanie play?", "answer": "Adoption agencies"}}
{{"question": "When did Melanie paint a sunrise?", "answer": "2022"}}

Return as JSON only, with no extra text outside the JSON object:
{{
    "qa_pairs": [
        {{"question": "Question about this session", "answer": "Shortest possible answer"}},
        {{"question": "Question about this session", "answer": "Shortest possible answer"}},
    ]
}}
"""

        content = ""
        try:
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
            qa_pairs: List[Dict[str, Any]] = []
            for item in data.get("qa_pairs", []):
                if "question" in item and "answer" in item:
                    qa_pairs.append({
                        "question": item["question"].strip(),
                        "answer": item["answer"].strip(),
                    })
            return qa_pairs[:num_qa]

        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response from LLM: %s. Raw: %s", e, content)
            return []
        except Exception as e:
            logger.error("Failed to generate QA pairs: %s", e)
            return []
