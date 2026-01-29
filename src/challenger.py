"""
Challenger Module - Generates QA pairs from dialogue
"""

import json
import uuid
from typing import List, Dict, Optional
from openai import OpenAI


class QuestionGenerator:
    """Generates QA pairs from original dialogue"""
    
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
    
    def generate_session_qa_pairs(
        self,
        session_dialogue: str,
        session_idx: int,
        k: int = 5,
        question_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate QA pairs for a single session.
        
        Args:
            session_dialogue: The dialogue text for this session
            session_idx: Index of the session
            k: Number of QA pairs to generate
            question_type: Optional (not used, kept for interface compatibility)
        
        Returns:
            List of QA pairs with session metadata
        """
        prompt = f"""Based on the following dialogue session, generate {k} QA pairs 
that test whether the key information from this session is properly remembered.

Dialogue:
{session_dialogue}

Focus on:
1. Key facts and information mentioned
2. Important opinions or preferences expressed
3. Specific details that should be remembered

Return as JSON:
{{
    "qa_pairs": [
        {{
            "question": "Question about this session",
            "answer": "Answer from this session's dialogue",
            "category": "fact|opinion|relationship|plan|other",
            "focus_area": "what information this tests"
        }}
    ]
}}

Only return JSON."""
        print(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You must respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1000
            )
            print(response)
            content = response.choices[0].message.content
            result = json.loads(content)
            
            qa_pairs = result.get("qa_pairs", [])
            for qa in qa_pairs:
                qa["session_idx"] = session_idx
                qa["qa_id"] = str(uuid.uuid4())
            
            return qa_pairs
            
        except Exception as e:
            print(f"Error generating session QA pairs: {e}")
            return []
