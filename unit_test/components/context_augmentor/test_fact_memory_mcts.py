"""MCTS + FactMemoryAugmentor integration test.

Mirrors the real MCTS execution path to verify that FactMemoryAugmentor
integrates correctly with the tree-search loop. The data flow under test:

    loader.py::setup_inference_logging  →  attach InferenceLogger to models
    factory.py::create_components_language_grounded  →  (world_model, policy, reward_model)
    backends.py::LocalMemoryBackend  →  in-process fact extraction
    manager.py::LiTSMemoryManager  →  record_action / build_augmented_context
    fact_memory.py::FactMemoryAugmentor  →  analyze / retrieve
    search_base.py::BaseTreeSearch.__init__  →  accepts augmentors=[]
    search_base.py::BaseTreeSearch.run  →  _setup → search → _teardown
    mcts.py::MCTSSearch.search  →  setup_augmentors() → MCTS loop
    augmentor_setup.py::setup_augmentors  →  (on_step_complete, on_trajectory_complete)
    base.py::Policy.set_dynamic_notes_fn  →  memory notes injected into sys_prompt
    base.py::Policy.set_llm_call_fn  →  prompt interception for inspection

Breakpoints (skip with PYTHONBREAKPOINT=0):
    1. llm_call_log        — every policy prompt with sys_prompt + user_prompt
    2. backend._units      — raw extracted facts in LocalMemoryBackend
    3. result.root         — full MCTS tree with rewards and trajectory keys
    4. sys_prompt_variants — system prompt evolution (with/without memory notes)

Usage:
    python -m unit_test.components.context_augmentor.test_fact_memory_mcts
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lits.lm import get_lm
from lits.lm.loader import setup_inference_logging
from lits.components.factory import create_components_language_grounded
from lits.components.context_augmentor.fact_memory import FactMemoryAugmentor
from lits.memory.manager import LiTSMemoryManager
from lits.memory.backends import LocalMemoryBackend
from lits.agents.tree.mcts import MCTSSearch, MCTSConfig

MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"


def run():
    temp_dir = tempfile.mkdtemp(prefix="test_fact_mem_mcts_")
    print(f"Temp dir: {temp_dir}")

    # ── 1. Load models (mirrors loader.py::load_models) ──────────────
    print("\n=== 1. Load models ===")
    base_model = get_lm(MODEL_NAME)
    eval_base_model = get_lm(MODEL_NAME)
    # NOTE: inference_logger setup deferred to step 3 after memory_llm is created,
    # so all three models are attached in a single setup_inference_logging call.
    print(f"  policy model: {MODEL_NAME}")

    # ── 2. Create components (mirrors factory.py::create_components_language_grounded) ──
    print("\n=== 2. Create components (ReST-MCTS*, language_grounded) ===")
    search_args = {"n_actions": 2, "max_steps": 4}
    component_args = {}
    world_model, policy, reward_model = create_components_language_grounded(
        base_model=base_model,
        eval_base_model=eval_base_model,
        task_name="gsm8k",
        search_args=search_args,
        component_args=component_args,
        search_framework="rest",
    )
    print(f"  world_model:  {type(world_model).__name__}")
    print(f"  policy:       {type(policy).__name__}")
    print(f"  reward_model: {type(reward_model).__name__}")

    # ── 3. Create FactMemoryAugmentor (mirrors cli/search.py::setup_memory_manager) ──
    print("\n=== 3. Create FactMemoryAugmentor ===")
    # mirrors backends.py::LocalMemoryBackend — in-process LLM fact extraction
    memory_llm = get_lm(MODEL_NAME)
    # mirrors loader.py::setup_inference_logging — single logger shared across all models
    inference_logger = setup_inference_logging(
        base_model, eval_base_model, memory_llm, root_dir=temp_dir, override=True
    )
    print(f"  inference log: {inference_logger.filepath}")
    backend = LocalMemoryBackend(llm=memory_llm)
    # mirrors manager.py::LiTSMemoryManager — record_action + build_augmented_context
    memory_manager = LiTSMemoryManager(backend)
    # mirrors fact_memory.py::FactMemoryAugmentor — wraps manager as ContextAugmentor
    fact_augmentor = FactMemoryAugmentor(memory_manager=memory_manager)
    print(f"  backend:    {type(backend).__name__}")
    print(f"  manager:    {type(memory_manager).__name__}")
    print(f"  augmentor:  {type(fact_augmentor).__name__}")

    # ── 4. Setup logging + prompt interceptor ────────────────────────
    #    setup_logging mirrors cli/search.py — writes execution.log to temp_dir.
    #    The llm_call_fn interceptor captures sys_prompt + user_prompt at each
    #    policy LLM call — setup_logging does NOT capture prompt content.
    print("\n=== 4. Setup logging + prompt interceptor ===")
    from lits.log import setup_logging
    run_logger = setup_logging(
        "execution", temp_dir,
        add_console_handler=True, verbose=True, override=True
    )

    llm_call_log = []

    def intercept_policy_calls(prompt, response, query_idx=None, from_phase=None, **kwargs):
        """Capture policy LLM calls with full system prompt + user prompt."""
        sys_prompt = getattr(base_model, "sys_prompt", None)
        llm_call_log.append({
            "call_idx": len(llm_call_log),
            "query_idx": query_idx,
            "from_phase": from_phase,
            "sys_prompt": sys_prompt,
            "user_prompt": prompt if isinstance(prompt, str) else str(prompt)[:500],
            "response_preview": response.text[:200] if hasattr(response, "text") else str(response)[:200],
        })
        return None  # keep original response

    # mirrors base.py::Policy.set_llm_call_fn
    policy.set_llm_call_fn(intercept_policy_calls)
    print(f"  run_logger: {temp_dir}/execution.log")
    print(f"  interceptor installed on {type(policy).__name__}")

    # ── 5. Configure and run MCTS ────────────────────────────────────
    #    Mirrors mcts.py::MCTSSearch instantiation and
    #    search_base.py::BaseTreeSearch.run (template: _setup → search → _teardown).
    #    Inside search(), mcts.py::MCTSSearch.search calls
    #    augmentor_setup.py::setup_augmentors to wire callbacks.
    print("\n=== 5. Run MCTS with FactMemoryAugmentor ===")
    query = (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
        "every morning and bakes muffins for her friends every day with four. "
        "She sells every duck egg at the farmers' market daily for $2. "
        "How much in dollars does she make every day at the farmers' market?"
    )
    query_idx = 0

    # mirrors mcts.py::MCTSConfig — minimal config for inspection
    # n_iters=4 ensures at least one iteration expands AFTER facts are recorded
    # by on_step_complete in earlier iterations, so _combined_retrieve() can
    # inject memory notes into the policy prompt (has_memory_notes=True).
    config = MCTSConfig(
        policy_model_name=MODEL_NAME,
        eval_model_name=MODEL_NAME,
        n_actions=2,
        max_steps=4,
        n_iters=4,
        roll_out_steps=2,
        w_exp=1.0,
        force_terminating_on_depth_limit=True,
        terminate_on_terminal_node=False,
        output_trace_in_each_iter=True,
    )

    # mirrors search_base.py::BaseTreeSearch.__init__ — augmentors param
    searcher = MCTSSearch(
        config=config,
        world_model=world_model,
        policy=policy,
        reward_model=reward_model,
        augmentors=[fact_augmentor],
        checkpoint_dir=temp_dir,
    )

    print(f"  query: {query[:80]}...")
    print(f"  n_iters={config.n_iters}, n_actions={config.n_actions}, max_steps={config.max_steps}")
    print(f"  Running search_base.py::BaseTreeSearch.run ...")

    # mirrors search_base.py::BaseTreeSearch.run → _setup → search → _teardown
    result = searcher.run(query, query_idx)

    print(f"\n  MCTS completed.")
    print(f"  terminal nodes: {len(result.terminal_nodes_collected)}")
    print(f"  iterations traced: {len(result.trace_in_each_iter)}")

    # ── 6. Inspect policy LLM call log ───────────────────────────────
    #    Each entry captures what base.py::Policy._call_model sent to the LLM.
    #    sys_prompt includes dynamic notes from augmentor_setup.py::_combined_retrieve
    #    → base.py::Policy._get_dynamic_notes → base.py::Policy.set_system_prompt.
    print(f"\n=== 6. Policy LLM call log ({len(llm_call_log)} calls) ===")
    for i, entry in enumerate(llm_call_log):
        sp = entry["sys_prompt"] or ""
        # "Additional Notes" — base.py::Policy._get_dynamic_notes() wrapper prefix
        # "Insights from"   — types.py::TrajectorySimilarity.to_prompt_section() cross-trajectory header
        # "Known facts"     — manager.py::AugmentedContext.to_prompt_blocks() inherited memory header
        has_memory_notes = "Additional Notes" in sp or "Insights from" in sp or "Known facts" in sp
        print(f"  [{i}] phase={entry['from_phase']}, "
              f"sys_prompt_len={len(sp)}, "
              f"has_memory_notes={has_memory_notes}, "
              f"response={entry['response_preview'][:60]}...")

    breakpoint()  # inspect: llm_call_log, entry['sys_prompt'], entry['user_prompt']

    # ── 7. Inspect memory backend state ──────────────────────────────
    #    backend._units mirrors backends.py::LocalMemoryBackend internal storage.
    #    Each MemoryUnit has .origin_path, .text (extracted fact), .content_hash.
    print(f"\n=== 7. Memory backend state (backends.py::LocalMemoryBackend._units) ===")
    print(f"  search_ids with units: {list(backend._units.keys())}")
    for search_id, units in backend._units.items():
        print(f"  [{search_id}] {len(units)} units:")
        for j, unit in enumerate(units):
            print(f"    [{j}] origin={unit.origin_path}, "
                  f"hash={unit.content_hash[:8] if unit.content_hash else 'N/A'}, "
                  f"text='{unit.text[:80]}'")
    breakpoint()  # inspect: len(backend._units['q_0']), backend._vectors['q_0'].shape

    # ── 8. Inspect MCTS tree structure ──────────────────────────────
    #    result.root is the MCTSNode tree built by mcts.py::MCTSSearch.search.
    #    Each node has .trajectory_key, .fast_reward, .reward, .children, .step.
    print(f"\n=== 8. Tree structure (mcts.py::MCTSResult) ===")

    def walk_tree(node, depth=0):
        indent = "  " * depth
        traj_key = node.trajectory_key.path_str if node.trajectory_key else "?"
        fast_r = f"fast_r={node.fast_reward:.3f}" if hasattr(node, "fast_reward") and node.fast_reward != -1 else "fast_r=?"
        action_preview = str(node.action)[:60] if node.action else "(root)"
        terminal = " [TERMINAL]" if node.is_terminal else ""
        print(f"{indent}Node {node.id} ({traj_key}) {fast_r}{terminal}: {action_preview}")
        for child in (node.children or []):
            walk_tree(child, depth + 1)

    walk_tree(result.root)

    breakpoint()  # inspect: result.root, result.trace_of_nodes, result.trace_in_each_iter

    # ── 9. System prompt evolution across policy calls ───────────────
    #    Groups unique system prompts to show when memory notes first appear.
    #    The prompt is built by base.py::Policy.set_system_prompt which calls
    #    base.py::Policy._get_dynamic_notes → augmentor_setup.py::_combined_retrieve.
    print(f"\n=== 9. System prompt evolution ===")
    sys_prompt_variants = {}
    for i, entry in enumerate(llm_call_log):
        sp = entry["sys_prompt"] or ""
        sp_hash = hash(sp)
        if sp_hash not in sys_prompt_variants:
            sys_prompt_variants[sp_hash] = {"first_seen_at_call": i, "count": 1, "prompt": sp}
        else:
            sys_prompt_variants[sp_hash]["count"] += 1

    for info in sys_prompt_variants.values():
        # "Additional Notes" — base.py::Policy._get_dynamic_notes() wrapper prefix
        # "Insights from"   — types.py::TrajectorySimilarity.to_prompt_section() cross-trajectory header
        # "Known facts"     — manager.py::AugmentedContext.to_prompt_blocks() inherited memory header
        has_memory_notes = "Additional Notes" in info["prompt"] or "Insights from" in info["prompt"] or "Known facts" in info["prompt"]
        print(f"  variant (first@call {info['first_seen_at_call']}, "
              f"seen {info['count']}x, len={len(info['prompt'])}, "
              f"has_memory_notes={has_memory_notes})")
        if has_memory_notes:
            for marker in ["Additional Notes", "Insights from", "Known facts"]:
                idx = info["prompt"].find(marker)
                if idx >= 0:
                    print(f"    ...{info['prompt'][max(0, idx - 20):idx + 200]}...")

    breakpoint()  # inspect: sys_prompt_variants — compare with/without memory notes

    # ── Done ─────────────────────────────────────────────────────────
    print(f"\n=== Done ===")
    print(f"  Temp dir (not deleted for inspection): {temp_dir}")


if __name__ == "__main__":
    run()
