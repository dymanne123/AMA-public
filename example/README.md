# Memory System Evaluation Framework

A decoupled memory evaluation framework that allows you to build, evaluate, and improve memory systems through adaptive reconstruction.

## Overview

This framework provides a modular interface for:
- **Building memories** from dialogue sessions
- **Evaluating memory quality** using QA pair generation and similarity scoring
- **Adaptive reconstruction** when memory quality is below threshold
- **Correction memory addition** to improve retrieval accuracy

## Architecture

The framework consists of four main interfaces:

### 1. MemorySystem

Abstract interface for memory storage and retrieval.

**Key Methods:**
- `search(user_id, query, top_k_memories, search_method)` - Search memories
- `build_memory(user_id, dialogue)` - Build memory from dialogue

**Implementation:**
- `SimpleMemorySystem` - JSON file-based storage with LLM summarization

### 2. MemoryAdapter

Abstract interface for memory update and reconstruction.

**Key Methods:**
- `update(memory, user_id, session_dialogue, failed_qa, failed_reasons)` - Filter dialogue based on failed QA pairs
- `reconstruct(memory, user_id, filtered_dialogue, failed_qa)` - Reconstruct memory with corrections

**Implementation:**
- `SimpleMemoryAdapter` - Keyword-based filtering and memory reconstruction

### 3. MemoryChallenger

Generates QA pairs from dialogue to test memory quality.

**Key Methods:**
- `generate_qa_pairs(session_dialogue, num_qa)` - Generate question-answer pairs

### 4. MemoryEvaluator

Evaluates memory quality by testing retrieval accuracy.

**Key Methods:**
- `evaluate_session_memories(memory, user_id, session_dialogue)` - Evaluate and return pass rate
- `retrieve_answer(memory, user_id, question)` - Retrieve answer from memory

## Installation

```bash
pip install openai numpy jinja2
```

## Quick Start

### Basic Usage

```python
from simple_memory_system import SimpleMemorySystem
from simple_adapter import SimpleMemoryAdapter
from plugin.memory_challenger import MemoryChallenger
from plugin.memory_evaluator import MemoryEvaluator
from example_session import process_session

# Initialize components
memory = SimpleMemorySystem(storage_dir="memory_storage")
adapter = SimpleMemoryAdapter()
challenger = MemoryChallenger()
evaluator = MemoryEvaluator(challenger=challenger)

# Process a session
session_dialogue = """user: Hello, I'm planning a trip to Japan.
assistant: That sounds exciting! When are you planning to go?
user: We're leaving on March 15th, 2024."""

result = process_session(
    memory=memory,
    adapter=adapter,
    evaluator=evaluator,
    challenger=challenger,
    user_id="user_001",
    session_id="session_001",
    session_dialogue=session_dialogue,
    output_dir="output"
)

print(f"Pass rate: {result['pass_rate']:.2f}%")
print(f"Reconstructed: {result['reconstructed']}")
```

### Running the Example

```bash
# Use default example session
python example_session.py

# Use custom dialogue file
python example_session.py --dialogue dialogue.json --user-id user_001 --session-id session_001

# Specify output directory
python example_session.py --output-dir results
```

## Usage Examples

### Example 1: Basic Memory Building

```python
from simple_memory_system import SimpleMemorySystem

memory = SimpleMemorySystem(storage_dir="memory_storage")

dialogue = """user: I love reading books about history.
assistant: That's interesting! What's your favorite period?
user: I'm particularly interested in ancient Rome."""

result = memory.build_memory("user_001", dialogue)
print(f"Status: {result['status']}")
print(f"Summary: {result['summary']}")
print(f"Memories created: {result['memories_count']}")
```

### Example 2: Memory Evaluation

```python
from simple_memory_system import SimpleMemorySystem
from plugin.memory_challenger import MemoryChallenger
from plugin.memory_evaluator import MemoryEvaluator

memory = SimpleMemorySystem(storage_dir="memory_storage")
challenger = MemoryChallenger()
evaluator = MemoryEvaluator(challenger=challenger)

# Build memory first
memory.build_memory("user_001", dialogue)

# Evaluate memory quality
eval_result, need_reconstruct = evaluator.evaluate_session_memories(
    memory, "user_001", dialogue
)

print(f"Pass rate: {eval_result['summary']['pass_rate']:.2f}%")
print(f"Passed: {eval_result['summary']['passed']}/{eval_result['summary']['qa_pairs_count']}")
print(f"Need reconstruction: {need_reconstruct}")
```

### Example 3: Complete Workflow with Reconstruction

```python
from simple_memory_system import SimpleMemorySystem
from simple_adapter import SimpleMemoryAdapter
from plugin.memory_challenger import MemoryChallenger
from plugin.memory_evaluator import MemoryEvaluator
from example_session import process_session, EXAMPLE_SESSION_DATA

# Use example session data
session_data = EXAMPLE_SESSION_DATA

# Initialize all components
memory = SimpleMemorySystem(storage_dir="memory_storage")
adapter = SimpleMemoryAdapter()
challenger = MemoryChallenger()
evaluator = MemoryEvaluator(challenger=challenger)

# Process the session
result = process_session(
    memory=memory,
    adapter=adapter,
    evaluator=evaluator,
    challenger=challenger,
    user_id=session_data["user_id"],
    session_id=session_data["session_id"],
    session_dialogue=session_data["session_dialogue"],
    output_dir="output"
)

# Check results
if result["reconstructed"]:
    print(f"Initial pass rate: {result['initial_pass_rate']:.2f}%")
    print(f"After reconstruction: {result['after_reconstruct']['pass_rate']:.2f}%")
    improvement = result['after_reconstruct']['pass_rate'] - result['initial_pass_rate']
    print(f"Improvement: {improvement:+.2f}%")
```

### Example 4: Custom Session Data

```python
import json
from example_session import process_session
from simple_memory_system import SimpleMemorySystem
from simple_adapter import SimpleMemoryAdapter
from plugin.memory_challenger import MemoryChallenger
from plugin.memory_evaluator import MemoryEvaluator

# Define custom session
custom_session = {
    "user_id": "user_002",
    "session_id": "session_2024_02_01",
    "session_dialogue": """user: I need help planning my vacation.
assistant: I'd be happy to help! Where would you like to go?
user: I'm thinking about visiting Paris in May.
assistant: Paris is beautiful in May! How long are you planning to stay?
user: About a week, from May 15th to May 22nd."""
}

# Initialize components
memory = SimpleMemorySystem(storage_dir="memory_storage")
adapter = SimpleMemoryAdapter()
challenger = MemoryChallenger()
evaluator = MemoryEvaluator(challenger=challenger)

# Process session
result = process_session(
    memory=memory,
    adapter=adapter,
    evaluator=evaluator,
    challenger=challenger,
    user_id=custom_session["user_id"],
    session_id=custom_session["session_id"],
    session_dialogue=custom_session["session_dialogue"],
    output_dir="output"
)

# Save result to file
with open("result.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
```

## Workflow

The framework follows this workflow:

1. **Build Memory**: Extract and store key information from dialogue
2. **Generate QA Pairs**: Create test questions from the dialogue
3. **Evaluate**: Test memory retrieval accuracy for each question
4. **Reconstruct** (if needed): If pass rate < 90%:
   - Filter dialogue to focus on failed questions
   - Rebuild memory with corrections
   - Add correction memories directly
5. **Re-evaluate**: Test improved memory quality

## Configuration

### API Configuration

All API settings are centralized in `config.py`:

```python
# config.py
API_BASE_URL = "YOUR_API_BASE_URL"
API_KEY = "YOUR_API_KEY"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
```

You can modify these values directly in `config.py`, or override them when initializing classes:

```python
memory = SimpleMemorySystem(
    storage_dir="memory_storage",
    api_key="custom-key",
    api_base="custom-url",
    model="custom-model"
)
```

### Evaluation Threshold

Default pass rate threshold is 90%. Modify in `plugin/memory_evaluator.py`:

```python
PASS_RATE_THRESHOLD = 90.0
```

### Similarity Threshold

Default similarity threshold for answer matching is 0.8. Modify in `plugin/memory_evaluator.py`:

```python
self.similarity_threshold = 0.8  # In MemoryEvaluator.__init__ method
```

### Pass Rate Threshold

Default pass rate threshold for reconstruction is 90%. Modify in `memory_evaluator.py`:

```python
PASS_RATE_THRESHOLD = 90.0  # At module level
```

## Output Files

The framework generates:

- `{user_id}_memories.json` - User's memory storage
- `{user_id}_{session_id}_original_memory.json` - Original memory (if pass rate >= PASS_RATE_THRESHOLD)
- `{user_id}_{session_id}_reconstructed_memory.json` - Reconstructed memory (if pass rate < PASS_RATE_THRESHOLD)

## Memory Format

Memories are stored as JSON with the following structure:

```json
{
  "memory_id": "uuid",
  "user_id": "user_001",
  "content": "Memory content in third-person narrative",
  "timestamp": "2024-01-15T14:30:00",
  "created_at": "2024-01-15T14:30:00",
  "metadata": {
    "source": "dialogue_summarization",
    "summary": "Brief summary"
  }
}
```

Correction memories have additional metadata:

```json
{
  "memory_id": "uuid",
  "user_id": "user_001",
  "content": "Question: What is X? Correct Answer: Y.",
  "timestamp": "2024-01-15T14:30:00",
  "metadata": {
    "source": "correction",
    "question": "What is X?",
    "true_answer": "Y",
    "type": "qa_correction"
  }
}
```

## Extending the Framework

### Custom MemorySystem

```python
from memory_system import MemorySystem

class CustomMemorySystem(MemorySystem):
    def search(self, user_id, query, top_k_memories=None, search_method="hybrid"):
        # Your implementation
        pass
    
    def build_memory(self, user_id, dialogue):
        # Your implementation
        return {"status": "built", "summary": "...", "memories_count": 1}
```

### Custom MemoryAdapter

```python
from memory_adapter import MemoryAdapter

class CustomMemoryAdapter(MemoryAdapter):
    def update(self, memory, user_id, session_dialogue, failed_qa, failed_reasons):
        # Your filtering logic
        return filtered_dialogue
    
    def reconstruct(self, memory, user_id, filtered_dialogue, failed_qa):
        # Your reconstruction logic
        return True
```

