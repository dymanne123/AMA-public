"""Memory evaluator interface."""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from jinja2 import Template
from openai import OpenAI

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import API_BASE_URL, API_KEY, LLM_MODEL, EMBEDDING_MODEL
from .memory_challenger import MemoryChallenger
from .memory_system import MemorySystem

PASS_RATE_THRESHOLD = 70.0

logger = logging.getLogger("memory_evaluator")

ANSWER_PROMPT = Template(
    """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    6. The answer should be in the form of a short phrase for the following question, less than 6-7 words.
    7. **Critical**: Use exact keywords/phrases from memories (verbatim if possible), avoid paraphrasing.
    8. **Critical**: Match question dimensions—plural questions require plural answers; country-level questions avoid cities; no missing key details.
    9. **Critical**: Answer in minimum words (remove redundant modifiers), no extra descriptions.
    10. If no relevant evidence, output "Unanswerable" (exact wording).

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories—use original words, match question dimensions
    6. Double-check that your answer directly addresses the question asked, no omissions
    7. Ensure your final answer is specific, avoids vague time references, and meets word limit

    Memories:
    {{ memories }}

    Question: {{ question }}

    Answer:
    """
)

from config import API_BASE_URL, API_KEY, LLM_MODEL, EMBEDDING_MODEL


class MemoryEvaluator:
    """Evaluator for memory quality assessment."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        challenger: "MemoryChallenger | None" = None,
    ):
        self.api_key = api_key or API_KEY
        self.api_base = api_base or API_BASE_URL
        self.model = model or LLM_MODEL
        self.openai_client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        self.embedding_model = EMBEDDING_MODEL
        self.similarity_threshold = 0.8
        self.challenger = challenger or MemoryChallenger(api_key=self.api_key, model=self.model, api_base=self.api_base)

    def _compute_cosine_similarity(self, text1: str, text2: str) -> float:
        try:
            response = self.openai_client.embeddings.create(
                input=[text1.strip(), text2.strip()],
                model=self.embedding_model,
            )
            emb1 = np.array(response.data[0].embedding)
            emb2 = np.array(response.data[1].embedding)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(emb1, emb2) / (norm1 * norm2))
        except Exception:
            return 0.0

    def retrieve_answer(self, memory: MemorySystem, user_id: str, question: str) -> str:
        """Retrieve answer from memory for a question."""
        try:
            search_results = memory.search(
                user_id,
                question,
                top_k_memories=20,
                search_method="vector",
            )
            memories_list = search_results.get("memories", [])
            if not memories_list:
                memories_list = search_results.get("episodic", []) + search_results.get("semantic", [])
            memories_content = "\n".join([item.get("content", "") for item in memories_list])
            prompt = ANSWER_PROMPT.render(
                question=question,
                memories=memories_content,
            )
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def evaluate_session_memories(
        self,
        memory: MemorySystem,
        user_id: str,
        session_dialogue: str,
    ) -> Tuple[Dict[str, Any], bool]:
        """Evaluate session memory quality. Returns (result, need_reconstruct)."""
        start_time = time.time()
        qa_pairs = self.challenger.generate_qa_pairs(session_dialogue)
        if not qa_pairs:
            return {"error": "No valid QA pairs generated", "summary": {}}, False

        logger.info(f"Generated {len(qa_pairs)} QA pairs for evaluation")
        print(f"\n[QA Evaluation] Generated {len(qa_pairs)} questions:")
        print("=" * 80)

        passed_count = 0
        failed_qa: List[Dict[str, Any]] = []

        for idx, qa in enumerate(qa_pairs, 1):
            q = qa["question"]
            true_answer = qa["answer"]
            retrieved_answer = self.retrieve_answer(memory, user_id, q)
            similarity = self._compute_cosine_similarity(true_answer, retrieved_answer)
            is_pass = similarity >= self.similarity_threshold

            result_item = {
                "question": q,
                "true_answer": true_answer,
                "retrieved_answer": retrieved_answer,
                "similarity": round(similarity, 4),
                "is_pass": is_pass,
            }
            
            status = "✓ PASS" if is_pass else "✗ FAIL"
            print(f"\nQuestion {idx}: {q}")
            print(f"  True Answer: {true_answer}")
            print(f"  Retrieved Answer: {retrieved_answer}")
            print(f"  Similarity: {similarity:.4f} | Status: {status}")
            
            if is_pass:
                passed_count += 1
            else:
                failed_qa.append(result_item)
        
        print("=" * 80)

        pass_rate = passed_count / len(qa_pairs) * 100 if qa_pairs else 0
        need_reconstruct = pass_rate < PASS_RATE_THRESHOLD

        result = {
            "summary": {
                "qa_pairs_count": len(qa_pairs),
                "passed": passed_count,
                "failed": len(failed_qa),
                "pass_rate": pass_rate,
                "need_reconstruct": need_reconstruct,
            },
            "failed_qa": failed_qa,
            "error_analysis": {
                "error_count": len(failed_qa),
                "error_types": ["missing info", "content deviation"] if failed_qa else [],
            },
            "duration": time.time() - start_time,
        }
        return result, need_reconstruct
