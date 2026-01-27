"""
Adapter Module - Generates update strategies
"""

import json
from typing import List, Dict, Any
from openai import OpenAI


class UpdateStrategyGenerator:
    """Generates update strategies based on evaluation results."""
    
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
    
    def generate_session_strategy(
        self,
        evaluations: List[Dict],
        error_analysis: Dict,
        session_idx: int
    ) -> Dict[str, Any]:
        """Generate update strategy for a single session."""
        failed = [e for e in evaluations if not e.get("is_pass", True)]
        
        prompt = f"""Generate memory update strategy for session {session_idx}.

Failed QA pairs: {len(failed)}
Error types: {error_analysis.get('error_types', {})}

For each failed QA, suggest memory improvements:
{chr(10).join([f"- Q: {e.get('question', '')[:100]}  Missing: {e.get('analysis', {}).get('missing', 'N/A')}" for e in failed[:3]])}

Return JSON:
{{  
    "memory_updates": [
        {{
            "reason": "why this update is needed",
            "proposed_content": "new memory content to add",
            "priority": "high|medium|low"
        }}
    ],
    "extraction_improvements": ["improvement1", "improvement2"]
    "imporve_instructions": "overall instructions to improve memory extraction"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You must respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
                max_tokens=1000
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}