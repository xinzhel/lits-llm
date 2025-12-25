"""
Demo: Understanding Inherited Memory Context in LiTS-Mem

This demo illustrates why isolated step content (like "Sub-answer: 4") is still useful
even without explicit context during recording. The key insight is that context is
reconstructed during RETRIEVAL, not during RECORDING.

Run this demo:
    python -m unit_test.memory.demo_inherited_memory_context

The demo walks through:
1. Recording isolated facts at different tree depths
2. Showing how inherited_units reconstructs the full context during retrieval
3. Demonstrating cross-trajectory retrieval with missing_units

=============================================================================
CALL CHAIN: What happens under the hood
=============================================================================

When you call manager.build_augmented_context(trajectory):

    LiTSMemoryManager.build_augmented_context(trajectory)     # lits/memory/manager.py
        │
        ├── list_inherited_units(trajectory)                  # lits/memory/manager.py
        │       │
        │       └── _ensure_cache(search_id)                  # Fetches all units from backend
        │               │
        │               └── backend.list_all_units(search_id) # lits/memory/backends.py
        │
        └── search_related_trajectories(trajectory)           # lits/memory/manager.py
                │
                └── TrajectorySearchEngine.search(...)        # lits/memory/retrieval.py
                        │
                        ├── normalize_pair(...)               # lits/memory/normalizer.py
                        │       Normalizes memory sets by depth and cardinality
                        │
                        ├── Compute overlap:
                        │       overlap_keys = ref_signatures & candidate_signatures
                        │
                        ├── Compute score:
                        │       score = len(overlap_keys) / max(1, len(ref_signatures))
                        │
                        └── select_new_units(...)             # lits/memory/normalizer.py
                                Returns facts in candidate but not in current trajectory

=============================================================================
HOW SIMILARITY SCORE IS COMPUTED (see lits/memory/retrieval.py)
=============================================================================

The similarity score measures how much memory overlap exists between trajectories.
Formula: score = |overlap| / |Mem(t)|

Implementation in TrajectorySearchEngine.search():
    1. Normalize current trajectory's memories (trim by depth and cardinality)
    2. For each candidate trajectory, normalize its memories
    3. Compute overlap: overlap_keys = ref_signatures & candidate_signatures
    4. Calculate score: score = len(overlap_keys) / max(1, len(ref_signatures))
    5. Filter by similarity_threshold (default 0.3)

Example from this demo:
    - Mem(q/0/0) = {"What is 15 * 12?", "Break into 15 * 10 + 15 * 2", "150 + 30 = 180"}
    - Mem(q/1) = {"What is 15 * 12?", "Use 12 * 15 = 12 * 10 + 12 * 5"}
    - overlap = {"What is 15 * 12?"} (1 shared fact)
    - score = 1 / 3 = 0.33

=============================================================================
HOW THE FORMATTED PROMPT IS CONSTRUCTED
=============================================================================

The prompt is built by AugmentedContext.to_prompt_blocks() in lits/memory/manager.py:

Part 1: Inherited Memories Section
    - Collects all memories from ancestor nodes via list_inherited_units()
    - Format: "# Inherited memories\\n- fact1\\n- fact2\\n..."
    - Source: AugmentedContext.to_prompt_blocks() in lits/memory/manager.py (lines 68-70):
        ```python
        if include_inherited and self.inherited_units:
            inherited_text = "\\n".join(f"- {unit.text}" for unit in self.inherited_units)
            blocks.append(f"# Inherited memories\\n{inherited_text}")
        ```
    
    Output:
    
    1. INHERITED UNITS (from ancestors q, q/0):
        [q] What is 15 * 12?
        [q/0] Break into 15 * 10 + 15 * 2
        [q/0/0] 150 + 30 = 180

Part 2: Cross-Trajectory Results Section  
    - For each similar trajectory found by search_related_trajectories()
    - Format: "Trajectory {path} (score={score})\\n- missing_fact1\\n..."
    - Source: TrajectorySimilarity.to_prompt_section() in lits/memory/types.py (lines 147-152):
        ```python
        def to_prompt_section(self) -> str:
            header = f"Trajectory {self.trajectory_path} (score={self.score:.2f})"
            details = "\\n".join(f"- {unit.text}" for unit in self.missing_units)
            return f"{header}\\n{details}".strip()
        ```
    - Called by to_prompt_blocks() in lits/memory/manager.py (lines 72-73):
        ```python
        for result in self.retrieved_trajectories:
            blocks.append(result.to_prompt_section())
        ```
    
    Output:
        
    2. CROSS-TRAJECTORY RESULTS:
        Similar trajectory: q/1 (score=0.33)
        Missing units (insights from sibling):
            - Use 12 * 15 = 12 * 10 + 12 * 5
        Similar trajectory: q/1/0 (score=0.33)
        Missing units (insights from sibling):
            - Use 12 * 15 = 12 * 10 + 12 * 5
            - 120 + 60 = 180


Final output example:
    # Inherited memories
    - What is 15 * 12?
    - Break into 15 * 10 + 15 * 2
    - 150 + 30 = 180

    Trajectory q/1 (score=0.33)
    - Use 12 * 15 = 12 * 10 + 12 * 5
"""

from __future__ import annotations

import os
import sys
import uuid

TEST_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_ROOT, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mem0.configs.base import MemoryConfig
from mem0.configs.vector_stores.qdrant import QdrantConfig
from mem0.memory.main import Memory

from lits.memory import (
    LiTSMemoryConfig,
    LiTSMemoryManager,
    Mem0MemoryBackend,
    TrajectoryKey,
)

# Model settings
EMBEDDING_MODEL = "cohere.embed-english-v3"  # Bedrock model ID (without bedrock/ prefix)
GENERATION_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # Bedrock model ID
VECTOR_SIZE = 1024


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def create_memory_manager(qdrant_dir: str) -> LiTSMemoryManager:
    """Create a LiTSMemoryManager with Mem0 backend using local Qdrant and Bedrock models."""
    config = MemoryConfig()
    
    # Embedder config - using Bedrock Cohere
    config.embedder.provider = "aws_bedrock"
    config.embedder.config = {"model": EMBEDDING_MODEL}

    # Qdrant vector store config
    qdrant_config = QdrantConfig(
        collection_name=f"lits_demo_{uuid.uuid4().hex[:6]}",
        embedding_model_dims=VECTOR_SIZE,
        client=None,
        host=None,
        port=None,
        path=qdrant_dir,
        url=None,
        api_key=None,
        on_disk=True,
    )
    config.vector_store.provider = "qdrant"
    config.vector_store.config = qdrant_config

    # LLM config - using Bedrock Claude for fact extraction
    config.llm.provider = "aws_bedrock"
    config.llm.config = {
        "model": GENERATION_MODEL,
        "temperature": 0.2,
        "max_tokens": 2000
    }

    memory = Memory(config=config)
    backend = Mem0MemoryBackend(memory)
    
    manager_config = LiTSMemoryConfig(
        similarity_threshold=0.3,
        max_retrieved_trajectories=2,
        max_augmented_memories=10,
    )
    return LiTSMemoryManager(backend=backend, config=manager_config)


def demo_basic_inheritance(manager: LiTSMemoryManager, search_id: str):
    """
    Steps 1-4: Demonstrates basic inherited memory retrieval.
    
    Records isolated facts at different tree depths and shows how
    inherited_units reconstructs the full context during retrieval.
    
    Tree structure:
        q (root) - "What is 15 * 12?"
        └── q/0 (left branch) - "Break into 15 * 10 + 15 * 2"
            └── q/0/0 - "150 + 30 = 180"
    """
    # =========================================================================
    # STEP 1: Record facts at ROOT (q)
    # =========================================================================
    print_separator("STEP 1: Recording at ROOT (q)")
    
    root = TrajectoryKey(search_id=search_id, indices=())
    print(f"Recording at trajectory: {root.path_str}")
    print(f"Content: 'What is 15 * 12?'")
    
    manager.record_action(
        root,
        messages=[{"role": "user", "content": "What is 15 * 12?"}],
        infer=False,  # Don't use LLM extraction, store as-is
    )
    
    # =========================================================================
    # STEP 2: Record facts at LEFT BRANCH (q/0) - isolated content!
    # =========================================================================
    print_separator("STEP 2: Recording at LEFT BRANCH (q/0)")
    
    left = TrajectoryKey(search_id=search_id, indices=(0,))
    print(f"Recording at trajectory: {left.path_str}")
    print(f"Content: 'Break into 15 * 10 + 15 * 2' (ISOLATED - no question context!)")
    
    manager.record_action(
        left,
        messages=[{"role": "assistant", "content": "Break into 15 * 10 + 15 * 2"}],
        infer=False,
    )
    
    # =========================================================================
    # STEP 3: Record facts at LEFT LEAF (q/0/0) - more isolated content!
    # =========================================================================
    print_separator("STEP 3: Recording at LEFT LEAF (q/0/0)")
    
    left_leaf = TrajectoryKey(search_id=search_id, indices=(0, 0))
    print(f"Recording at trajectory: {left_leaf.path_str}")
    print(f"Content: '150 + 30 = 180' (ISOLATED - just the answer!)")
    
    manager.record_action(
        left_leaf,
        messages=[{"role": "assistant", "content": "150 + 30 = 180"}],
        infer=False,
    )
    
    # =========================================================================
    # STEP 4: Now let's see what happens during RETRIEVAL
    # =========================================================================
    print_separator("STEP 4: RETRIEVAL - The Magic of Inherited Memories")
    
    print("When we retrieve context for a NEW node at q/0/0/0 (child of left_leaf):")
    print()
    
    new_node = left_leaf.child(0)  # q/0/0/0
    print(f"New node trajectory: {new_node.path_str}")
    print(f"Ancestors: {new_node.ancestry_paths}")
    print()
    
    # Get inherited units - THIS IS WHERE CONTEXT IS RECONSTRUCTED!
    # See lits/memory/manager.py: list_inherited_units() collects all memories
    # where unit.origin_path is a prefix of the current trajectory path
    inherited = manager.list_inherited_units(new_node)
    
    print("INHERITED MEMORIES (from all ancestors):")
    print("-" * 40)
    for unit in inherited:
        print(f"  [{unit.origin_path}] {unit.text}")
    print()
    
    print(">>> Notice: Even though each fact was recorded in ISOLATION,")
    print(">>> the inherited_units gives us the FULL CONTEXT:")
    print(">>>   - The original question (from q)")
    print(">>>   - The reasoning approach (from q/0)")
    print(">>>   - The calculation result (from q/0/0)")
    print()
    
    return left_leaf


def demo_cross_trajectory(manager: LiTSMemoryManager, search_id: str, left_leaf: TrajectoryKey):
    """
    Steps 5-7: Demonstrates cross-trajectory retrieval.
    
    Adds a sibling branch and shows how cross-trajectory search
    finds related trajectories and returns missing_units.
    
    Key code references:
    - Cross-trajectory search: manager.search_related_trajectories() in lits/memory/manager.py
    - Score computation: TrajectorySearchEngine.search() in lits/memory/retrieval.py
      Formula: score = len(overlap_keys) / max(1, len(ref_signatures))
    - Missing units selection: select_new_units() in lits/memory/normalizer.py
    - Prompt formatting: AugmentedContext.to_prompt_blocks() in lits/memory/manager.py
    
    Tree structure after this:
        q (root) - "What is 15 * 12?"
        ├── q/0 (left branch) - "Break into 15 * 10 + 15 * 2"
        │   └── q/0/0 - "150 + 30 = 180"
        └── q/1 (right branch) - "Use 12 * 15 = 12 * 10 + 12 * 5"
            └── q/1/0 - "120 + 60 = 180"
    """
    # =========================================================================
    # STEP 5: Add a RIGHT BRANCH to show cross-trajectory retrieval
    # =========================================================================
    print_separator("STEP 5: Adding RIGHT BRANCH (q/1) for cross-trajectory demo")
    
    right = TrajectoryKey(search_id=search_id, indices=(1,))
    print(f"Recording at trajectory: {right.path_str}")
    print(f"Content: 'Use 12 * 15 = 12 * 10 + 12 * 5' (different approach)")
    
    manager.record_action(
        right,
        messages=[{"role": "assistant", "content": "Use 12 * 15 = 12 * 10 + 12 * 5"}],
        infer=False,
    )
    
    right_leaf = TrajectoryKey(search_id=search_id, indices=(1, 0))
    print(f"Recording at trajectory: {right_leaf.path_str}")
    print(f"Content: '120 + 60 = 180'")
    
    manager.record_action(
        right_leaf,
        messages=[{"role": "assistant", "content": "120 + 60 = 180"}],
        infer=False,
    )
    
    # =========================================================================
    # STEP 6: Show cross-trajectory retrieval
    # Score computation (see lits/memory/retrieval.py):
    #   - Mem(q/0/0) has 3 facts: question + approach + result
    #   - Mem(q/1) has 2 facts: question + different approach
    #   - overlap = 1 (shared question "What is 15 * 12?")
    #   - score = 1/3 = 0.33
    # =========================================================================
    print_separator("STEP 6: Cross-Trajectory Retrieval")
    
    print(f"Searching for related trajectories from {left_leaf.path_str}...")
    print()
    
    # Build full augmented context
    # See lits/memory/manager.py: build_augmented_context() combines:
    #   1. inherited_units from list_inherited_units()
    #   2. retrieved_trajectories from search_related_trajectories()
    context = manager.build_augmented_context(left_leaf)
    
    print("AUGMENTED CONTEXT for q/0/0:")
    print("-" * 40)
    print()
    print("1. INHERITED UNITS (from ancestors q, q/0):")
    for unit in context.inherited_units:
        print(f"   [{unit.origin_path}] {unit.text}")
    print()
    
    print("2. CROSS-TRAJECTORY RESULTS:")
    print("   (Score = |overlap| / |Mem(current)|, see retrieval.py)")
    if context.retrieved_trajectories:
        for result in context.retrieved_trajectories:
            print(f"   Similar trajectory: {result.trajectory_path} (score={result.score:.2f})")
            print(f"   Missing units (insights from sibling):")
            for unit in result.missing_units:
                print(f"      - {unit.text}")
    else:
        print("   (No similar trajectories found above threshold)")
    print()
    
    # =========================================================================
    # STEP 7: Show the formatted prompt that would go to the policy
    # See lits/memory/manager.py: AugmentedContext.to_prompt_blocks()
    # Constructs: "# Inherited memories\n- fact1\n...\n\nTrajectory path (score=X)\n- missing1\n..."
    # =========================================================================
    print_separator("STEP 7: Formatted Prompt for Policy")
    
    prompt = context.to_prompt_blocks()
    print("This is what gets injected into the policy prompt:")
    print("(See AugmentedContext.to_prompt_blocks() in lits/memory/manager.py)")
    print("-" * 40)
    print(prompt if prompt else "(empty - no memories)")
    print("-" * 40)
    print()
    
    print(">>> KEY INSIGHT:")
    print(">>> Even though we recorded isolated facts like '150 + 30 = 180',")
    print(">>> the policy sees the FULL CONTEXT through inherited memories!")
    print(">>> Plus, it gets insights from sibling trajectories (cross-trajectory).")
    
    return context


def demo_inherited_memory_context(basic_only: bool = False):
    """
    Full demo: Demonstrates how inherited memories provide context during retrieval.
    
    Args:
        basic_only: If True, only run Steps 1-4 (basic inheritance demo).
                   If False, run all steps including cross-trajectory demo.
    
    Tree structure we'll build:
    
        q (root) - "What is 15 * 12?"
        ├── q/0 (left branch) - "Break into 15 * 10 + 15 * 2"
        │   └── q/0/0 - "150 + 30 = 180"
        └── q/1 (right branch) - "Use 12 * 15 = 12 * 10 + 12 * 5"  (only if basic_only=False)
            └── q/1/0 - "120 + 60 = 180"
    """
    # Setup Qdrant directory
    qdrant_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "qdrant_local"))
    os.makedirs(qdrant_dir, exist_ok=True)
    
    manager = create_memory_manager(qdrant_dir)
    search_id = f"math-demo-{uuid.uuid4().hex[:6]}"
    
    # Run Steps 1-4: Basic inheritance demo
    left_leaf = demo_basic_inheritance(manager, search_id)
    
    # Run Steps 5-7: Cross-trajectory demo (optional)
    if not basic_only:
        demo_cross_trajectory(manager, search_id, left_leaf)


def demo_step_to_messages():
    """
    Shows how step.to_messages() provides richer content for memory recording.
    """
    print_separator("BONUS: How step.to_messages() Works")
    
    from lits.structures.qa import SubQAStep, ThoughtStep
    from lits.structures.env_grounded import EnvStep, EnvAction
    
    print("1. SubQAStep.to_messages():")
    step1 = SubQAStep(sub_question="What is 2+2?", sub_answer="4")
    print(f"   Input: sub_question='{step1.sub_question}', sub_answer='{step1.sub_answer}'")
    print(f"   Output: {step1.to_messages()}")
    print()
    
    print("2. ThoughtStep.to_messages():")
    step2 = ThoughtStep(action="Let me break this down step by step")
    print(f"   Input: action='{step2.action}'")
    print(f"   Output: {step2.to_messages()}")
    print()
    
    print("3. EnvStep.to_messages():")
    step3 = EnvStep(action=EnvAction("move(block1, table)"), next_state="block1 is on table")
    print(f"   Input: action='{step3.action}', next_state='{step3.next_state}'")
    print(f"   Output: {step3.to_messages()}")
    print()
    
    print(">>> These messages are what get passed to memory_manager.record_action()")
    print(">>> The richer format (action + observation) helps mem0 extract better facts.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo: Inherited Memory Context in LiTS-Mem")
    parser.add_argument("--basic-only", action="store_true", 
                       help="Only run Steps 1-4 (basic inheritance demo)")
    parser.add_argument("--skip-step-demo", action="store_true",
                       help="Skip the step.to_messages() demo")
    args = parser.parse_args()
    
    demo_inherited_memory_context(basic_only=args.basic_only)
    
    if not args.skip_step_demo:
        demo_step_to_messages()
    
    print("\n" + "="*60)
    print("  DEMO COMPLETE")
    print("="*60)
