"""
Evaluator Module - Evaluates memory quality
"""

import json
from typing import List, Dict, Any
from openai import OpenAI


class MemoryAnswerer:
    """Generates answers based on memory content."""
    
    def __init__(
        self,
        api_key: str,
        model: str = 'gpt-4o-mini',
        api_base: str = None
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        ) if api_base else OpenAI(api_key=api_key)
        self.model = model
    
    def answer_from_memory(
        self,
        memory: List[Dict],
        question: str
    ) -> str:
        """
        Generate an answer based on memory content.
        
        Args:
            memory: List of memory entries
            question: The question to answer
        
        Returns:
            memory_answer: The answer generated from memory
        """
        if not memory:
            return "No relevant memories found."
        
        user_prompt = f"""
You are an AI assistant that answers questions based on the provided memory information.

Memory Context:
{memory}

Question: {question}

Instructions:
1. Use ONLY the information from the Memory Context above to answer the question.
2. If the question cannot be answered using the provided memory, say "I cannot answer this question based on the available memory."
3. Keep your answer concise and directly relevant to the question.
4. Do not make up or assume any information not present in the memory.

Answer:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating answer from memory: {e}")
            return ""


class QualityEvaluator:
    """Evaluates the quality of memory-based answers."""
    
    def __init__(
        self,
        api_key: str,
        model: str = 'gpt-4o-mini',
        api_base: str = None
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        ) if api_base else OpenAI(api_key=api_key)
        self.model = model
    
    def evaluate_qa_quality(
        self,
        question: str,
        original_answer: str,
        memory_answer: str,
        session_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a QA pair.
        
        Args:
            question: The generated question
            original_answer: The correct/expected answer
            memory_answer: The answer generated from memory
            session_idx: Session index
        
        Returns:
            Evaluation result dictionary
        """
        prompt = f"""Compare the original answer with the memory-based answer:

Question: {question}

Original Answer (Ground Truth):
{original_answer}

Memory-Based Answer:
{memory_answer}

Evaluate in JSON:
{{
    "missing_summary": "Describe missing facts (only contain factual information itself).",
    "scores": {{
        "completeness": 0.0-1.0,
        "accuracy": 0.0-1.0,
        "overall": 0.0-1.0
    }},
    "analysis": {{
        "missing": "what is missing",
        "error_type": "incomplete|incorrect|partial|match"
    }},
    "is_pass": true/false
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You must respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "question": question,
                "original_answer": original_answer,
                "memory_answer": memory_answer,
                "missing_summary": result.get("missing_summary", ""),
                "scores": result.get("scores", {}),
                "analysis": result.get("analysis", {}),
                "is_pass": result.get("is_pass", False),
                "session_idx": session_idx
            }
            
        except Exception as e:
            return {
                "question": question,
                "original_answer": original_answer,
                "memory_answer": memory_answer,
                "missing_summary": "",
                "scores": {"overall": 0.0},
                "analysis": {"error_type": "error"},
                "is_pass": False,
                "error": str(e),
                "session_idx": session_idx
            }