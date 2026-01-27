"""
AMA Pipeline - Automated Memory Assessment Pipeline
"""

from typing import List, Dict
from collections import defaultdict

from challenger import QuestionGenerator
from evaluator import MemoryAnswerer, QualityEvaluator
from adapter import UpdateStrategyGenerator


class SessionMemoryQA:
    """Session-level memory quality evaluation and update."""
    
    def __init__(
        self,
        api_key: str,
        model: str = 'gpt-4o-mini',
        api_base: str = None
    ):
        self.question_generator = QuestionGenerator(api_key, model, api_base)
        self.memory_answerer = MemoryAnswerer(api_key, model, api_base)
        self.quality_evaluator = QualityEvaluator(api_key, model, api_base)
        self.strategy_generator = UpdateStrategyGenerator(api_key, model, api_base)
    
    def evaluate_session_memories(
        self,
        session_dialogue: str,
        entry_loader: List[Dict],
        session_idx: int = 0
    ) -> tuple:
        """Evaluate memory quality for a single session."""
        result = {
            "session_idx": session_idx,
            "qa_pairs": [],
            "evaluations": [],
            "error_analysis": None,
            "update_strategy": None,
            "summary": {},
            "imporve_instructions": ""
        }
        
        # Step 1: Generate QA pairs for this session
        qa_pairs = self.question_generator.generate_session_qa_pairs(
            session_dialogue, session_idx
        )
        result["qa_pairs"] = qa_pairs
        
        if not qa_pairs:
            return result, False, ""
        
        entries = entry_loader
        missing_summary = ""
        
        for qa in qa_pairs:
            question = qa["question"]
            original_answer = qa["answer"]
            print("question:", question)
            print("original_answer:", original_answer)
            
            print("session_memories:", entries)
            memory_answer = self.memory_answerer.answer_from_memory(entries, question)
            print("memory_answer:", memory_answer)
            
            evaluation = self.quality_evaluator.evaluate_qa_quality(
                question=question,
                original_answer=original_answer,
                memory_answer=memory_answer,
                session_idx=session_idx
            )
            print("evaluation:", evaluation)
            result["evaluations"].append(evaluation)
            missing_summary = missing_summary + "\n" + evaluation["missing_summary"]
        
        # Step 4: Analyze error patterns
        error_analysis = self._analyze_session_errors(result["evaluations"])
        result["error_analysis"] = error_analysis
        print("error_analysis:", error_analysis)
        
        reconstruct = False
        
        # Step 5: Generate update strategy
        if error_analysis.get("failed", 0) > 2:
            update_strategy = self.strategy_generator.generate_session_strategy(
                result["evaluations"],
                error_analysis,
                session_idx
            )
            result["update_strategy"] = update_strategy
            result["imporve_instructions"] = update_strategy.get("improve_instructions", "")
            reconstruct = True
        print("update_strategy:", result["update_strategy"])
        
        # Generate summary
        result["summary"] = self._generate_session_summary(result)
        print("summary:", result["summary"])
        
        return result, reconstruct, missing_summary
    
    def _analyze_session_errors(self, evaluations: List[Dict]) -> Dict:
        """Analyze error patterns for session."""
        if not evaluations:
            return {"total": 0, "passed": 0, "failed": 0}
        
        passed = sum(1 for e in evaluations if e.get("is_pass", False))
        failed = len(evaluations) - passed
        
        error_types = defaultdict(int)
        for e in evaluations:
            if not e.get("is_pass", True):
                error_types[e.get("analysis", {}).get("error_type", "unknown")] += 1
        
        return {
            "total": len(evaluations),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(evaluations) * 100 if evaluations else 0,
            "error_types": dict(error_types)
        }
    
    def _generate_session_summary(self, result: Dict) -> Dict:
        """Generate summary for session evaluation."""
        evaluations = result.get("evaluations", [])
        error_analysis = result.get("error_analysis", {})
        
        return {
            "qa_pairs_count": len(evaluations),
            "passed": error_analysis.get("passed", 0),
            "failed": error_analysis.get("failed", 0),
            "pass_rate": error_analysis.get("pass_rate", 0),
            "needs_update": error_analysis.get("failed", 0) > 0
        }