# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Starting from v0.2.11, version numbers in this changelog are kept in sync with `pyproject.toml`.


## 2026-02-15 v0.2.11

### Changed
- Lazy-import `transformers`/`torch`/`numpy`/`huggingface_hub` in `lits/lm/base.py` and `lits/lm/__init__.py`

### Changed (See .kiro/specs/x-minor-chain-result-dir)
- Consolidate result directory logic: `get_model_dir_prefix()` and `MODEL_NAME_TO_DIR_PREFIX` in `lits/agents/base.py`
- `BaseConfig.setup_directories(run_id)` (`lits/agents/base.py`)
- Remove standalone `get_result_dir()` from `lits/cli/chain.py`
- `ExperimentConfig.get_result_dir()` uses shared `get_model_dir_prefix()`

### Added (See .kiro/specs/minor-output-dir-cli)
- `--output-dir` / `-o` CLI flag (`lits/cli/args.py`)
- `output_dir` field on `BaseConfig` and `ExperimentConfig`
- `output_dir` override in `BaseConfig.setup_directories()` and `ExperimentConfig.setup_directories()`

## 2026-02-14  v0.2.11 (Too-Use Registry)

### Fixed
- `lits_benchmark/mapeval.py`: inline dataset loading (fix broken `_load_mapeval_examples` call)
- `lits/benchmarks/registry.py`: remove stale `"examples"` references from `@register_resource` docstrings
- `unit_test/search/test_mcts_tool_use_integration.py`: use `load_dataset()` instead of `tool_use_spec["examples"]`
- `unit_test/components/test_benchmark_registry.py`: import modules in `setUpClass` instead of removed fallbacks
- `unit_test/components/test_auto_registration.py`: remove stale `mapeval` (non-sql) from docstring

### Removed
- `load_resource`/`load_dataset_examples` from `lits_benchmark/__init__.py`


## 2026-02-13 v0.2.11  (Too-Use Registry)
### Added
- `@register_resource` decorator for tool-use benchmarks (`lits/benchmarks/registry.py`)

## 2026-02-12 v0.2.10 (CLI Commands)
### Added
- `lits-search`, `lits-eval`, `lits-chain`, `lits-eval-chain` CLI entry points (`pyproject.toml`)
- `lits/cli/search.py` — tree search entry point
- `lits/cli/eval_search.py` — tree search evaluation entry point
- `lits/cli/chain.py` — chain agent entry point
- `lits/cli/eval_chain.py` — chain agent evaluation entry point

### Changed
- `examples/` scripts converted to thin backward-compat wrappers

### Removed
- `lits/cli/commands.py` placeholder stubs

## 2026-02-11 v0.2.9 (BlocksWorld: From built-in to external)
### Changed
- Moved `BlocksWorldTransition` from `lits/components/transition/blocksworld.py` to `lits_benchmark/blocksworld.py`
- Moved BlocksWorld prompts (policy, reward, transition) to decorator-based registration in `lits_benchmark/blocksworld.py`

## 2026-02-09 v0.2.9 (Search Registry)
### Added
- `BaseTreeSearch` ABC and `SearchResult` dataclass (`lits/agents/tree/search_base.py`)
- `InferenceLogger.log_context()` for binding search-phase metadata to LLM call records
- `MCTSSearch` and `BFSSearch` classes inheriting from `BaseTreeSearch`
- `--policy`, `--transition`, `--reward` CLI flags wired through component factory
- `TrajectoryKey` to BFS root node (`lits/agents/tree/bfs.py`)
- Thinking mode support for `TGIChatModel` (Qwen3) (`lits/lm/tgi.py`)

### Changed
- `@register_search` now supports class-based registration (`lits/agents/registry.py`)
- Simplified `main_search.py`: unified `search_kwargs`, `algo_output.to_paths()`, single `TreeToJsonl`
- Updated `lits/agents/__init__.py` exports: `MCTSSearch`, `BFSSearch`, `BaseTreeSearch`, `SearchResult`
- `create_components_language_grounded` and `create_components_env_grounded` accept override params (`lits/components/factory.py`)
- `ExperimentConfig`: added `policy`, `transition`, `reward` override fields (`lits/config.py`)

### Fixed
- TGI `running_time`: prefer server-side `x-compute-time` over client wallclock (`lits/lm/tgi.py`)

### Docs
- Added `log_context()` section to `INFERENCE_LOGGER.md`

## 2026-02-08 v0.2.8
### Docs
- Split `CHAIN.md` into `ENV_CHAIN.md` and `ReAct.md` (`docs/agents/`)

## 2026-02-06 v0.2.8
### Fixed
- Bedrock support for Opus 4.5 (`lits/lm/base.py`)

## 2026-02-04 v0.2.8
### Changed
- Moved `examples/math_qa/main_cot.py` to `examples/main_cot.py`
- `main_cot.py`: added `--override` flag, fixed `sys.path` and `.env` path for new location
- `main_cot.py`: fixed `extract_numerical_answer` to clean up literal `""` responses
- `main_cot.py`: `process_model_output` always wraps in `<think>...</think>` for correct parsing
- `main_cot.py`: skip extraction call when solution is empty

### Fixed
- `_parse_value` in `lits/cli/args.py` now handles list syntax `[5]` or `[1,2,3]`

## 2026-02-02 v0.2.8
### Added
- `dataset_kwargs` saved to config.json by `main_search.py` for reproducible dataset loading
- `--dataset-arg levels=[5]` to all math500 examples in `run_configs.sh`
- `docs/cli/search.md` - CLI documentation for main_search.py and eval_search.py
- CLI argument parsing in `examples/math_qa/main_cot.py` (`--dataset`, `--model`, `--dataset-arg`)

### Changed
- `main_search.py` uses `load_dataset()` from registry for language_grounded tasks
- `eval_search.py` uses `load_dataset()` from registry for language_grounded tasks
- `eval_search.py` loads `dataset_kwargs` from config.json and passes to dataset loader
- Accuracy calculation uses `eval_output()` for numeric comparison instead of exact string match

### Fixed
- Dataset index mismatch in `eval_search.py`: `query_idx` now correctly indexes filtered dataset
- Bedrock empty response handling in `eval_output()`
- `_parse_value` in `lits/cli/args.py` now handles list syntax `[5]` or `[1,2,3]`

## 2026-02-01 v0.2.8
### Added
- `SubQAStep.to_dict()` and `from_dict()` for proper serialization (`lits_benchmark/formulations/rap/structures.py`)
- `--override` cleanup: removes stale files in `checkpoints/`, `terminal_nodes/`, and `treetojsonl*.jsonl` (`examples/main_search.py`)

### Changed
- `llm_calls.jsonl` logging now skipped for `language_grounded` tasks (only enabled for `env_grounded`/`tool_use` where duplicate detection is meaningful)
- Refactor logging ssystem (see .kiro/specs/logging-system-refactoring for details)

### Fixed
- RAP `_step` TypeError: extract action string from Step object (`lits_benchmark/formulations/rap/transition.py`)
- RAP `_fast_reward`: handle Step objects and access `SubQAStep.sub_question` instead of tuple unpacking (`lits_benchmark/formulations/rap/reward.py`)
- CLI `--search-arg` overwrite bug: multiple `--search-arg` flags overwrite instead of append due to `nargs="+"` (e.g., `--search-arg a=1 --search-arg b=2` keeps only `b=2`); fixed `run_configs.sh` and added `or` fallback for `None` values

## 2026-01-31 v0.2.8
### Added
- `TGIChatModel` for TGI's `/v1/chat/completions` endpoint (`lits/lm/tgi.py`)
- `tgi-chat://` URL prefix for chat models via TGI
- `TGIModel.is_chat_model` property using `infer_chat_model()`
- `TGIModel.get_next_token_logits()` using TGI grammar constraints (`lits/lm/tgi.py`)
- `TGIModel.get_top5_logits()` using TGI's `top_n_tokens` feature (`lits/lm/tgi.py`)

### Fixed
- RAP state type mismatch with `TrajectoryState` (`lits_benchmark/formulations/rap/policy.py`, `transition.py`)
- RAP `_generate_prompt` now handles `TGIModel`
- `TGIModel` stop parameter: convert string to list for TGI API (`lits/lm/tgi.py`)
- `TGIModel` error handling: include response body in 422 errors

### Changed
- Generic type parameters in base classes: `Generic[StateT, ActionT]` → `Generic[StateT, StepT]` (`lits/components/base.py`)

## 2026-01-29 v0.2.8 (Search Registry)
see .kiro/specs/register-search-decorator for details

### Added
- `AgentRegistry` and `@register_search` decorator for custom search algorithms (`lits/agents/registry.py`)

### Changed
- Unified search invocation in `main_search.py` using `AgentRegistry.get_search()`

## 2026-01-29 v0.2.8 (Codebase Refactoring for Language-Grounded Tasks)
See .kiro/specs/lang_grounded_refactoring for details

### Added
- `--policy-model`, `--eval-model`, `--transition-model`, `--bn-model` CLI flags (`lits/cli/args.py`)
- `--search-arg` and `--component-arg` CLI flags for parameter passing
- `--help-config` flag for parameter discovery
- `from_config()` pattern for components (`GenerativePRM`, `ThinkPRM`, `ConcatPolicy`, `ConcatTransition`)
- `TGIModel` class for remote TGI completion models (`lits/lm/tgi.py`)
- `Step.verbalize_state()` class method for extensible state verbalization
- RAP formulation as external module (`lits_benchmark/formulations/rap/`)
- `ComponentRegistry` for custom formulation registration

### Changed
- Refactored `ExperimentConfig` from ~50 fields to minimal ~15 fields (`lits/config.py`)
- Moved `ExperimentConfig` to `lits/config.py`
- Moved `component_factory.py` to `lits/components/factory.py`
- Moved `model_loader.py` to `lits/lm/loader.py`
- Renamed `terminal_gen_model_name` to `transition_model_name`
- Factory now uses registry-driven component lookup

### Removed
- `reasoning_method` from `BaseConfig`
- RAP components from core package (moved to `lits_benchmark/formulations/rap/`)


## 2026-01-22 v0.2.8

### Added
- `docs/agents/TREE_SEARCH_GUIDE.md`: FAQ section explaining MCTS termination behavior

## 2026-01-21  v0.2.8

### Added
- `create_llm_call_logger()` factory for incremental LLM call logging (`lits/eval/llm_call_logger.py`)
- `load_llm_calls()`, `get_diversity_stats()`, `print_diversity_report()` analysis functions
- `normalize_crosswords_action()`, `parse_crosswords_correct_actions()` for crosswords analysis
- LLM call logging integration in `main_search.py`

### Fixed
- `eval_search.py`: Sort terminal nodes by cumulative reward before selecting best for env_grounded tasks
- `main_search.py`: Map `--override` CLI flag to `config.override_log_result`

## 2026-01-20  v0.2.8

### Added
- `Step.terminate` field to signal fatal errors that should stop trajectory generation (`lits/structures/base.py`)
- `EnvGroundedPolicy._generate_action_with_retry()` with temperature escalation for duplicate action avoidance (`lits/components/policy/env_grounded.py`)
  - Note: BlocksWorld MCTS results were generated with `max_retries=1` without temperature-incremented re-generation
- `Policy.set_llm_call_fn()` callback for intercepting LLM calls (`lits/components/base.py`)

### Changed
- `EnvStep.from_dict()` deserializes `terminate` field (`lits/structures/env_grounded.py`)
- `EnvGroundedPolicy` fallback behavior: finite action space uses unselected valid action; infinite action space sets `terminate=True`
- `EnvChain.run()` now checks `step.terminate` to stop trajectory on fatal errors (`lits/agents/chain/env_chain.py`)
- `bfs._expand()` and `bfs._expand_with_existing()` check `step.terminate` to mark nodes as terminal (`lits/agents/tree/bfs.py`)
- `mcts._expand()` checks `step.terminate` to mark nodes as terminal (`lits/agents/tree/mcts.py`)

## 2026-01-19  v0.2.8
## Fixed
- `goal_check` was incorrectly parsing answers from Filled/Changed sections
      - Now extracts board from "Current Board:" section and computes 10 answers

## 2026-01-18  v0.2.8 (Config & CLI Refactoring)

### Added
- `BaseConfig`: `benchmark`, `import_modules`, `dataset_kwargs` fields for experiment metadata
- `ChainConfig`: `temperature` field (0.0 = deterministic)
- `lits/cli/args.py`: `parse_script_vars()` for `--var` arguments

### Changed
- CLI arguments refactored: `--cfg` for config fields, `--var` for script variables
- `apply_config_overrides()` uses `--cfg` args (field names match config exactly)
- `EnvChainConfig` simplified: removed `offset`, `limit`, `override`, `temperature` (moved to parents)
- `main_env_chain.py` uses new CLI pattern with `--cfg`/`--var`/`--dataset-arg`

### Removed
- `offset`, `limit`, `override` from config classes (now script-local variables)

## 2026-01-13  v0.2.8 (Crosswords Domain & Error-Annotated Checkpoints)

### Added
- `lits_benchmark/crosswords.py` - CrosswordsTransition with `@register_transition("crosswords")`
- `CrosswordsTransition.validate_action()` for infinite action space validation
- `load_crosswords()` dataset loader with `@register_dataset("crosswords")`
- Crosswords-specific prompts registered via `@register_system_prompt`/`@register_user_prompt`
- `lits/prompts/policy/env_grounded.py` - generic fallback policy prompt
- `lits/prompts/reward/env_grounded.py` - generic fallback reward prompts
- `EnvState.__repr__()` showing step count and truncated current state
- Error-annotated checkpoints: `EnvStep(action=invalid_action, error=error_msg)` preserved in trajectory

### Changed
- `EnvGroundedPolicy` now always returns a step (error step if validation fails)
- `EnvGroundedPolicy._get_actions()` refactored into helper methods for readability
- `env_chain.py` handles `step.error` uniformly: appends error step, saves checkpoint, breaks
- `EnvGroundedTransition.generate_actions()` now non-abstract, returns `[]` by default
- `component_factory.py` uses `getattr` for optional `generate_actions` and `validate_action`

### Action Space Contract
- Finite action space (BlocksWorld): `generate_actions` returns exhaustive list
- Infinite action space (Crosswords): `validate_action` validates LLM-generated actions

## 2026-01-11  v0.2.7 (Component Registry)

### Added
- `lits/components/registry.py` - ComponentRegistry with `register_transition`, `register_policy`, `register_reward_model` decorators
- `lits/benchmarks/registry.py` - BenchmarkRegistry with `register_dataset`, `load_dataset`, `infer_task_type`
- `lits/prompts/registry.py` - `register_system_prompt`, `register_user_prompt` decorators
- `lits/components/transition/env_grounded.py` - EnvGroundedTransition base class with abstract `goal_check`, `generate_actions` static methods
- `lits/registry.py` - unified import entry re-exporting all registry decorators and utilities
- `import_custom_modules()` and `load_config_from_result_dir()` CLI utilities in `lits/registry.py`
- `import_modules` field to `BaseSearchConfig` for persisting custom module imports

### Changed
- `BlocksWorldTransition` now extends `EnvGroundedTransition` with `@register_transition("blocksworld")`
- `lits_benchmark/blocksworld.py` - `load_blocksworld` registered with `@register_dataset("blocksworld")`
- `lits_benchmark/math_qa.py` - registered gsm8k, math500, spart_yn datasets
- `lits_benchmark/mapeval.py` - registered mapeval, mapeval-sql datasets
- `lits_benchmark/crosswords.py` - registered crosswords dataset (placeholder)
- `component_factory.py` - uses registry-first with fallback pattern for env_grounded components
- `main_search.py` - added `--import` arg, saves `import_modules` to config
- `main_env_chain.py` - added `--import`/`--benchmark` args, uses registry lookup
- `eval_search.py` - auto-loads `import_modules` from config, uses registry for env_grounded detection
- `eval_env_chain.py` - auto-loads `import_modules`/`benchmark` from config

### Removed
- `lits_benchmark/main.py` - consolidated into `lits/benchmarks/registry.py`

## 2026-01-04  v0.2.6 (Env-Grounded Task: Main Scripts)
### Added
- `lits_llm/examples/main_env_chain.py` - unified env-grounded chain agent runner
- `lits_llm/examples/eval_env_chain.py` - evaluation script for env-grounded chain results

### Changed
- Refactored `main_env_chain.py` to use consistent result directory structure (`{model}_results/{benchmark}_chain/run_{version}/`)

### Fixed
- `EnvChain.run()` now passes `init_state_str` as keyword argument to `world_model.init_state()`
- `EnvChain.run()` now calls `is_terminal(state, query_or_goals)` with correct argument order


## 2026-01-04  v0.2.6 (Inference Report Generation)
### Added
- `lits/eval/inference_report.py` with `generate_report()` for formatted inference usage reports
- `InferenceLogger` multi-group aggregation: `get_metrics_by_component()`, `get_metrics_by_phase()`, `get_metrics_by_instance()`, `get_metrics_by_component_and_phase()`
- `_get_grouped_metrics()` core reader for efficient single-pass aggregation
- `calculate_cost()` utility in `lits/lm/base.py`
- `docs/lm/INFERENCE_LOGGER.md` documentation

### Changed
- `print_metrics_for_mcts_phases()` and `print_metrics_for_all_role_prefixes()` now use single file read
- `eval_search.py` uses `generate_report()` instead of `report_metrics_from_dir()`

### Removed
- `report_metrics_from_dir()` from `lits/lm/base.py`

## 2026-01-02  v0.2.6 (Hide create_role from subclass implementations)
### Changed
- Added `_call_model()` helper to `Policy`, `RewardModel`, `LlmTransition` base classes
- Added `step()`/`_step()` and `is_terminal()`/`_is_terminal()` wrapper pattern to `LlmTransition`
- Added `_batch_call_model()` and `_sample_binary_output()` helpers to `LlmTransition`
- Updated Policy subclasses (`ConcatPolicy`, `RAPPolicy`, `ToolUsePolicy`, `EnvGroundedPolicy`) to use `_call_model()`
- Updated RewardModel subclasses (`RapPRM`, `GenerativePRM`, `SelfConsistencyRM`) to use `_call_model()` / `_call_model_logits()`
- Updated LlmTransition subclasses (`BlocksWorldTransition`, `RAPTransition`, `ConcatTransition`) to use `_step()` / `_is_terminal()`
- `ConcatTransition` now extends `LlmTransition` instead of `Transition`


## 2025-12-24 - 12-25  v0.2.6
- Memory-MCTS integration: See `.kiro/specs/lits-mem-mcts-integration` 

- AugmentedContext.to_prompt_blocks() default include_inherited=False (lits/memory/manager.py)
- `TrajectorySearchEngine` in `lits/memory/retrieval.py`
  - filters redundant ancestor trajectories when descendant is retrieved
  - skips candidate trajectories that are ancestors of current trajectory

## 2025-12-22  v0.2.5 (3-level fallback of Prompt Registry)
### Changed
- Refactored prompt registry: renamed `math_qa` to `language_grounded` as task_type
- `PromptRegistry.get()` and `get_usr()` now use 3-level fallback: task_name → TASK_TYPE → default
- `BlocksWorldTransition.TASK_TYPE` set to `None` to prevent generic prompt fallback
- Updated `Policy`, `RewardModel`, `LlmTransition` base classes to pass `TASK_TYPE` for prompt lookup
- Integrated `PROMPT_INJECTION_DESIGN.md` content into `LITS_DESIGN.md`

### Removed
- `lits_llm/docs/PROMPT_INJECTION_DESIGN.md` (merged into LITS_DESIGN.md)


## 2025-12-21 (Morning)  v0.2.5 (Evaluation for Env-Grounded Task: BlocksWorld)
### Added
- eval_search.py: BlocksWorld evaluation support with goal satisfaction checking
- eval_chain.py: Logging support via setup_logging with file output

### Changed
- eval_search.py: Updated docstring with BlocksWorld usage examples and multi-task support
- eval_chain.py: Added comprehensive docstring with usage examples


## 2025-12-20  v0.2.5 (CiT for Env-Grounded Task: BlocksWorld)
### Added
- `BNEvaluatorEnv` for environment-grounded tasks (`lits/components/bn_evaluator/bn_evaluator_env.py`)
- Added `terminate_on_first_solution` to `ExperimentConfig` with the change of MCTS

### Changed
- create_bn_evaluator now supports task_type parameter for env_grounded tasks (lits_benchmark/experiments/component_factory.py)
- EnvGroundedPolicy._get_actions supports allow_duplicates parameter (
env_grounded.py
)
- sample_actions_with_existing passes allow_duplicates=True during continuation phase (lits/agents/tree/common.py)

### Fixed
- Removed max_steps from `BlocksWorldTransition` and `BlocksWorldTransition` creation in `component_factory.py`.
- `EnvGroundedPolicy._get_actions`
  - Added retry limit (max 5) and numeric index parsing to prevent infinite LLM calls
  - Added fallback to first valid action when max retries exceeded


## 2025-12-19  v0.2.5 (Tree Search for Env-Grounded Task: BlocksWorld)
### Added

- Incremental checkpoint saving in MCTS (`lits/agents/tree/mcts.py`)

- `checkpoint_dir` parameter in `mcts()` function

- `override_checkpoint` parameter in `mcts()` function

### Changed

- Enhanced `_back_propagate()` docstring with detailed example (`lits/agents/tree/mcts.py`)



## 2025-12-17  v0.2.5 (Tree Search for Env-Grounded Task: BlocksWorld)

### Added
- `init_state_kwargs` parameter in MCTS and BFS tree search agents

  - Allows passing example-specific data to `Transition.init_state(**kwargs)`
  - Subclasses extract what they need (e.g., BlocksWorld extracts init_state_str)
  - Files: `lits/agents/tree/mcts.py`, `lits/agents/tree/bfs.py`

- Comprehensive docstrings for component base classes
  - Documented `init_state_kwargs` convention in Transition class
  - Documented `query_idx` and `from_phase` parameters in RewardModel._fast_reward
  - Includes example code showing `create_role()` usage for inference logging
  - File: `lits/components/base.py`

### Updated
- Updated `BlocksworldTransition.init_state`
  - Now accepts `**kwargs` and extracts `init_state_str` from example data
  - Follows the unified `init_state_kwargs` convention
  - File: `lits/components/transition/blocksworld.py`


### Refactor
- `EnvGroundedPRM._fast_reward` to use sampled binary outputs with `n_sample`
  - Relevant Change: `LanguageModel.sample_binary_output` with unknown token support and last-word matching 
- Renamed task_type param to task_name for prompt lookup in components
  - Relevant Change: Added TASK_TYPE class constant to Policy, RewardModel, LlmTransition

### Fixed
- PDDL file path resolution in load_blocksworld
  - Added automatic base_dir inference from `data_file` path
  - File: `lits_benchmark/blocksworld.py`


## 2025-12-13  v0.2.5
### Added
- **Evaluation**:
  - Created `examples/blocksworld/eval_chain.py` for evaluating BlocksWorld chain checkpoints.
  - Added `report_metrics_from_dir` in `lits/lm/base.py` for standardized inference metric reporting.
- **Chain Agents**:
  - Added `BaseChainAgent` in `lits/agents/chain/base.py` to provide common functionality for chain-based agents (ReAct, EnvChain).
  - Added `init_state` and `next_state` fields to `EnvState` and `EnvStep` respectively.
  - Added `__type__` field to `EnvState` serialization for correct polymorphic deserialization.

### Refactor
- **State Structure**:
  - Unified `EnvState` with `TrajectoryState` by making `EnvState` inherit from `TrajectoryState`.
  - Integrated `env_state`, `last_env_state`, `step_idx` logic into `EnvState` properties, removing redundant fields.
  - Promoted `render_history` and `to_messages` from `ToolUseState` to `TrajectoryState` base class for shared usage.
  - Removed `StepConcatState` from `lits/structures/qa.py` as it is now fully covered by `TrajectoryState`.
- **Logging**:
  - Refactored `examples/eval_search.py` to use `report_metrics_from_dir` for cleaner metric reporting.
  - Refactored `setup_inference_logging` in `lits_benchmark` to support optional arguments.
- **BlocksWorld Transition & Reward**:
  - Renamed `GenerativeBwPRM` to `EnvGroundedPRM` and updated prompt registry accordingly.
  - Removed `buffered_action` and related logic from `EnvState` and other classes.
  - Removed reward logic from `EnvStep` to separate concerns.
  - Updated `EnvChain` and `ReAct` agents to support state overriding/resuming with query handling.
  - Removed duplicate `goal_check` in `lits/components/transition/blocksworld.py`; now uses `lits_benchmark.blocksworld.goal_check`.
  - Updated `BlocksWorldTransition.step` to strictly accept `query_or_goals` as `str` to resolve ambiguity.
  - Aligned `BlocksWorldTransition.step` arguments with `EnvChain` interface (renamed `action` to `step_or_action`, `goals` to `query_or_goals`).
  - Updated `GenerativeBwPRM` to support prompt-based scoring (Yes/No) for Bedrock models and accept `query_or_goals` in `_fast_reward`.

### Fixes
- Fixed `EnvChain` execution loop to correctly pass arguments to `world_model.step`.

## 2025-12-11  v0.2.5

### Fixed
* Made lits.benchmarks a separate package: `lits-benchmark`

## 2025-12-10  v0.2.5

### Added
- STATE_REGISTRY for State type deserialization
- Token usage metrics logging in eval_search.py
- Comprehensive serialization documentation in docs/eval/SERIALIZATION.md

### Fixed
- Build state dictionary including kwargs
- ToolUseAction and StringAction registered in TYPE_REGISTRY
- Action deserialization in SearchNode.from_dict()
- TrajectoryState serialization format updated with type information
  - TrajectoryState.to_dict() returns dict with __type__ and steps fields (was list)
  - TrajectoryState.save() includes query in serialized dict
  - TrajectoryState.load() extracts query from serialized dict
- `_serialize_obj` checks for custom to_dict() method before dataclass serialization
- `_deserialize_obj` handles State types with from_dict() method


### Removed 

## 2025-12-08  v0.2.5
### Added
- ToolUsePRM._complete_trajectory
  - saves rollout trajectories to result_dir/tool_use_prm_rollouts
  - saves with unique filenames like rollout_{query_idx}_{idx_rollout}.jsonl,

### Fixed
- ToolUsePRM
  - use max_length evaluation
  - use `from_phase+="_prm"` for policy call

## 2025-12-07  v0.2.5
### Changed
- Transition.step() interface redesigned to accept Step objects instead of just Actions
- ToolUseTransition now handles answer/error/malformed steps directly without execution
- ReActChat updated to pass full steps to transition model
- ToolUsePRM updated to require ToolUseStep (no longer accepts raw ToolUseAction)
- ToolUsePRM avoids redundant tool execution if step already has observation

## 2025-12-07  v0.2.5 (Unify tree search for tool use)
### Changed
- SearchNode serialization includes full step information (answer, error fields)
- Tree search components pass steps to both transition.step() and reward_model.fast_reward() (backward compatible with actions for QA tasks), including
  - _world_modeling passes steps
  - _assign_fast_reward() helper centralizes fast_reward assignment in _expand and _world_modeling

### Added
- ToolUsePRM saves rollout trajectories with unique filenames (rollout_{query_idx}_{idx_rollout}.jsonl)
- ToolUsePRM caches reward scores to avoid redundant evaluations


## 2025-12-06  v0.2.5
### Added

- ToolUsePRM (`lits/components/reward/tool_use.py`)

## 2025-12-05  v0.2.5

### Added
- **Post-Generation Callbacks**: Added `Policy.set_post_generation_fn()` for action validation after generation
  - Called after actions generated but before returned to agent
  - Receives `(steps, context)` with full generation context including `policy_model_name` and `task_type`
  - Use for validation, logging, or saving results without modifying agent logic
- **Automatic Learning Loop**: ReActChat now supports automatic setup with evaluators
  - Pass `step_evaluators` for per-step validation (e.g., SQLValidator)
  - Pass `trajectory_evaluators` for post-trajectory analysis (e.g., SQLErrorProfiler)
  - Automatically sets up dynamic notes (input enhancement) and validation (output validation)
  - Creates feedback loop: past issues → prompt → generate → validate → save → repeat
- **Demo Scripts**: 
  - `demo_learning_loop.py`: Complete learning loop example with both evaluator types
- **Documentation**: 
  - Added `docs/agents/LEARNING_LOOP.md` for automatic learning loop setup and usage

### Changed
**Some improvements during debugging the verbal evaluation feedback loop (BEGIN)**
- `get_dynamic_notes` directly returns the concatenated prompt string from `VerbalEvaluator.load_eval_as_prompt` for clean code and to retain the prefix of each section of issues
- **VerbalEvaluator Improvements**: Enhanced base class with unified interface and intelligent ranking
  - **Unified `evaluate()` method**: All evaluators implement consistent interface returning `Optional[str]`
  - **Standardized output**: All evaluators save `issues` as list of strings (removed `reasoning` field from SQLValidator)
  - **Improved `load_eval_as_prompt()`**: 
    - Issue-level granularity: Treats each issue as individual item (not records)
    - Score-based ranking: Prioritizes worst issues (lower scores) for maximum learning impact
    - `max_items` now refers to number of individual issues, not records
- **SQL Validator**
  - Saved `sql_query` for the result
  - **State Context for Validation**: Step-level evaluators now receive trajectory history (action-observation pairs via `ToolUseState.render_history()`) as context for more informed SQL validation decisions
**Some improvements during debugging the verbal evaluation feedback loop (END)**
- **ReActChat**: Added support for step-level and trajectory-level evaluators
  - Added `step_evaluators` parameter for per-step validation
  - Added `trajectory_evaluators` parameter for post-trajectory analysis
  - Added `_setup_learning_loop()` method for automatic callback configuration
  - Added `_evaluate_trajectory()` method called after `run()` completes
- **create_tool_use_agent()**: Updated parameters for clearer evaluator separation
  - Added `step_evaluators` parameter (replaces generic `evaluators`)
  - Added `trajectory_evaluators` parameter for trajectory-level analysis
  - Automatically configures learning loop when evaluators provided
  - Passes `policy_model_name` and `task_type` to ReActChat for callback context


## 2025-12-04  v0.2.5

### Added
- **Verbal Evaluator Base Class**: Created abstract `VerbalEvaluator` base class providing unified architecture for all verbal evaluators
  - Unified file management: All evaluators for same policy/task save to single file
  - Automatic metadata: Adds `evaluator_type` and `timestamp` to all records
  - Result filtering: Load results by evaluator type
  - Shared methods: `_get_result_saver()`, `_save_eval()`, `load_results()`
- **Unified Data Format**: Standardized output format across all verbal evaluators
  - All evaluators now save `issues` as a list for consistency
  - Unified schema: `{evaluator_type, query_idx, timestamp, issues, ...}`
  - Pandas-friendly format for easy DataFrame conversion
- **Unified `load_eval_as_prompt()`**: Moved to base class with evaluator-type grouping
  - Single implementation works for all evaluators
  - Groups results by `evaluator_type` for clear feedback
  - Optional `include_all_evaluators` parameter to load from all evaluators
- **Model Name Utility**: Added `get_clean_model_name()` in `lits.lm` module
  - Extracts clean, abbreviated model names for file naming
  - Example: `"bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"` → `"v1_0"`
- **Demo Script**: Added `demo_load_evaluations.py` to demonstrate unified storage
  - Loads and displays results as pandas DataFrame
  - Shows formatted prompts for policy injection
  - Verifies unified format across evaluators
- **Documentation**: 
  - Updated `docs/components/verbal_evaluator/VERBAL_EVALUATOR.md` with base class architecture
  - Added `docs/components/callback.md` for dynamic notes injection

### Changed
- **Refactored SQL Validator**: Now inherits from `VerbalEvaluator` base class
  - Removed duplicate `_get_result_saver()` and `_save_eval()` methods
  - Uses unified `load_eval_as_prompt()` from base class
  - Saves `issues` as list instead of single string
- **Refactored SQL Error Profiler**: Now inherits from `VerbalEvaluator` base class
  - Removed duplicate file management methods
  - Uses unified `load_eval_as_prompt()` from base class
  - Already used list format for `issues`
- **Unified File Storage**: Changed from separate files per evaluator to single shared file
  - Old: `resultdicttojsonl_{model}_{task}_validator.jsonl` and `resultdicttojsonl_{model}_{task}_profiler.jsonl`
  - New: `resultdicttojsonl_{model}_{task}.jsonl` (both evaluators)
  - Records distinguished by `evaluator_type` field


### Fixed
- Fixed test files to use unified `max_items` parameter instead of `max_issues`/`max_profiles`

### Benefits
- **Simplified Architecture**: Single base class eliminates code duplication
- **Unified Storage**: All evaluators share files, easier to manage and analyze
- **Consistent Format**: Pandas-friendly schema across all evaluators
- **Extensible**: Easy to create new evaluators by inheriting from base class
- **Maintainable**: Changes to file management or loading logic in one place

## 2025-12-03  v0.2.5

### Added
- **Dynamic Notes Injection**: Added callback-based system for injecting dynamic notes from external sources into policy system prompts
  - Added `Policy.set_dynamic_notes_fn()` method to register callback functions that return `List[str]` of notes
  - Added `Policy._get_dynamic_notes()` helper that retrieves and formats notes as bullet points
  - Added documentation in `docs/components/callback.md` with usage examples, best practices, and integration patterns
- **Policy System Prompt**: Refactored to handle dynamic notes in base class
  - `set_system_prompt()` now automatically appends dynamic notes
  - Subclasses only implement `_build_system_prompt()` for base prompt

## 2025-12-02  v0.2.5

### Added
- **SQL Validator Component**: Added LLM-based SQL query validation in `lits/components/verbal_evaluator/sql_validator.py`
- **SQL Error Profiler Component**: Added trajectory-level SQL error analysis in `lits/components/verbal_evaluator/sql_error_profiler.py`
- **Incremental Evaluation**: Added `evaluate_incremental()` to `GeneralEvaluator` f
- **ResultDictToCSV**: New result saver
- **Documentation**: Added `docs/eval/INCREMENTAL_EVALUATION.md` and `docs/eval/RESULT_SAVERS.md`

### Changed
- **Refactored `run_tree_search()`**: Separated tree search execution from evaluation; terminal nodes now saved to checkpoint files for post-processing
- **Error Handling**: Improved Bedrock API error messages to be concise (truncated from 100k+ to ~100 characters)

### Fixed
- Fixed `eval_result` not initialized when `include_input=False` in `GeneralEvaluator.evaluate()`

## 2025-12-01  v0.2.5

### Fixed
- Fixed `transition: ToolUseTransition` with tool exection 

### Added
- Added `transition: ToolUseTransition` parameter 
- **Documentation**: 
  - Added `docs/LITS_DESIGN.md`
  - Added `docs/components/transitions/TOOL_USE_TRANSITION.md` 

### Changed
- `ReActChat`: Updated to use explicit `ToolUseTransition` component

## 2025-11-30  v0.2.5

### Fixed
- Standardized all parameter names from `example_idx` to `query_idx` 
- **Argument Order**: Corrected `policy.get_actions()` call signature in tree search
  - Changed from `get_actions(example, node.state, ...)` to `get_actions(node.state, query=example, ...)`
  - Fixed in `bfs.py` and `common.py` helper functions

- **Type Parameter Fix**: Fixed `ToolUseTransition` class definition
  - Changed from `Transition[ToolUseState, ToolUseStep, str]` (3 params) to `Transition[ToolUseState, ToolUseStep]` (2 params)
  - Base `Transition` class only accepts 2 type parameters

- **Search Config Fix**: Fixed parameter mapping in `ExperimentConfig.to_search_config_dict()`
  - Added mapping: `policy_model_name` → `model_name` for `BaseConfig` compatibility
  - Ensures `BFSConfig` and `MCTSConfig` receive correct parameter names

- **Component Factory Fix**: Added `task_type` parameter to all LLM-based component instantiations
  - Updated: `RAPTransition`, `RAPPolicy`, `RapPRM`, `ConcatPolicy`, `GenerativePRM`, `ToolUsePolicy`, `ToolUsePRM`
  - Enables task-specific prompt loading from registry

- **Token Counting Fix**: Added cross-platform token counting support
  - Created `count_tokens()` utility function supporting HF models, OpenAI/Bedrock (via tiktoken), and fallback estimation
  - Fixed `ConcatPolicy` to work with non-HF models (Bedrock, OpenAI) that don't have tokenizer attribute

- **Policy Kwargs Fix**: Added `**kwargs` to policy `_get_actions()` signatures
  - Updated `ConcatPolicy` and `ToolUsePolicy` to accept optional parameters
  - Prevents errors when extra parameters are passed through base class

- **Policy Return Type Fix**: Fixed policies to return `Step` objects instead of raw strings
  - `ConcatPolicy`: Now returns `list[ThoughtStep]` wrapping action strings
  - `RAPPolicy`: Now returns `list[SubQAStep]` with sub_question, empty sub_answer, and 0.0 confidence
  - Updated `bfs_topk` and `mcts` to extract actions from Step objects using `.get_action()`
  - Ensures compliance with base `Policy` class contract expecting `List[StepT]`

### Changed
- **ConcatPolicy Refactoring**: Improved code organization and readability
  - Extracted constants: `SIMILARITY_THRESHOLD`, `MAX_TOKEN_LENGTH`, `MAX_RETRY_REPEAT`, `STEP_PREFIX_PATTERN`
  - Modularized monolithic `_get_actions()` into focused methods:
    - `_validate_task_prompt_spec()` - validation
    - `_generate_single_output()` - LLM calls
    - `_clean_output_text()` - text cleaning
    - `_is_valid_output()` - validation
    - `_log_validation_failure()` - logging
    - `_generate_action_with_retry()` - retry logic
    - `_check_similarity()` - similarity checking
  - Added comprehensive docstrings for all methods
  - Improved maintainability and testability

### Added
- **Debug Logging**: Added comprehensive debug logging for troubleshooting
  - Type checking and validation in `Policy.__init__`
  - Query type logging in `ConcatPolicy._generate_msg()`
  - Error details in `verbalize_concat_state()` for type mismatches

## 2025-11-28  v0.2.4

### Added
- **Config Refactoring**: Unified configuration system across all agent types
  - Added common attributes (`model_name`, `gpu_device`, `max_length`, `max_steps`) to `BaseConfig`
  - Replaced `depth_limit` with `max_steps` across `BaseSearchConfig`, `EnvChainConfig`, and `ReactChatConfig` for semantic consistency
  - Unified `to_dict()` and `save_config()` methods - now inherited from `BaseConfig` using `asdict()`
  - Updated all tree search algorithms (MCTS, BFS) and policies to use `max_steps`
  - Added `unit_test/test_config_refactoring.py` with 15 passing tests

- **Unified MCTS & BFS Interface**: Created consistent interface for tree search methods
  - Unified result structures: both `MCTSResult` and `BFSResult` now include:
    - `root` - root node of the search tree
    - `trace_of_nodes` - best path from root to terminal
    - `terminal_nodes_collected` - all terminal nodes found during search
  - Made `mcts()` and `bfs_topk()` function signatures nearly identical (only parameter names differ)
  - Extracted shared post-processing function: `extract_answers_from_terminal_nodes()` works for both MCTS and BFS
  - Unified `main_search_refactored.py` - same code handles both search methods transparently
  - Users no longer need to know which search method is used - identical interface for both

- **Tree Visualization Module** (`lits/visualize.py`): Comprehensive visualization tools for search trees
  - High-level functions:
    - `visualize_mcts_result()` - direct MCTS tree visualization
    - `visualize_bfs_result()` - direct BFS tree visualization
  - Mid-level functions:
    - `get_tree_from_result()` - extract paths from either result type
    - `plot_save_tree()` - visualize tree from paths
    - `buckets_to_paths()` - convert BFS buckets (breadth-wise) to paths (depth-wise)
  - Low-level functions:
    - `path_to_dict()` - convert node paths to dictionaries
    - `build_anytree_from_paths()` - build anytree structure with automatic node deduplication
    - `_make_label()` - generate node labels with symbols (⏹ terminal, ⇲ expanded, ∼ simulated)
  - Supports multiple output formats: PDF, PNG, SVG, DOT
  - Uses `anytree` and `graphviz` for publication-quality figures
  - Added `unit_test/test_visualization_demo.py` - creates real visualizations with mock trees
  - Added `unit_test/test_tree_visualization.py` - 10 passing tests

- **Documentation**: Comprehensive user guides and API references
  - Created `docs/agents/TREE_SEARCH_GUIDE.md` - complete user guide covering:
    - Quick start examples
    - Configuration for both MCTS and BFS
    - Result structure and post-processing
    - Visualization usage
    - Switching between methods
    - Advanced topics and best practices
  - Created `docs/TREE_VISUALIZATION.md` - visualization documentation with:
    - Installation requirements
    - Usage examples
    - API reference
    - Integration with search algorithms
  - Updated `README.md` with visualization section

- **Test Suite**: Comprehensive testing for new features
  - `unit_test/test_config_refactoring.py` - config system tests (15 tests)
  - `unit_test/test_tree_visualization.py` - visualization tests (10 tests)
  - `unit_test/test_tree_common.py` - shared utility tests
  - `unit_test/test_visualization_demo.py` - generates actual PDF visualizations
  - All tests passing ✅

### Changed
- **Removed Redundancy**: Eliminated duplicate code and storage
  - Removed `bucket_saver` - BFS now saves paths like MCTS for consistent storage format
  - Removed duplicate `to_dict()` methods from `EnvChainConfig` and `ReactChatConfig`
  - Extracted `buckets_to_paths()` to eliminate code duplication between visualization functions
  - Moved post-processing logic outside search algorithms for cleaner separation of concerns
  
- **Simplified BFS Implementation**:
  - Removed `retrieve_answer` and `return_buckets` parameters from `bfs_topk()`
  - Simplified `BFSResult` to only include core data: `root`, `terminal_nodes_collected`, `buckets_with_terminal`
  - Post-processing (vote_answers, answer_rewards) now done via shared `extract_answers_from_terminal_nodes()`

- **Updated MCTS Implementation**:
  - Added `terminal_nodes_collected` to `MCTSResult` for consistency with BFS
  - MCTS now collects terminal nodes recursively from the tree structure
  - Uses same post-processing function as BFS

### Fixed
- Fixed import errors:
  - `lits/agents/tree/continuation.py` - corrected `tree_search` → `tree`
  - `lits/agents/__init__.py` - corrected `react_chat` → `react`
  - `lits/agents/main.py` - corrected `EnvPolicy` → `EnvGroundedPolicy`
  - `lits/agents/chain/env_chain.py` - corrected `EnvPolicy` → `EnvGroundedPolicy`

### Benefits
- **Unified Interface**: Switch between MCTS and BFS without code changes
- **Consistent Storage**: Both methods use `TreeToJsonl` for path storage
- **Cleaner Code**: Eliminated duplication, separated concerns
- **Better Visualization**: Publication-quality tree visualizations with minimal code
- **Easier Maintenance**: Shared utilities reduce code duplication
- **Improved Testing**: Comprehensive test coverage for all new features

## 2025-11-27 (pdf_query_tool)
### Added
- Implemented PDF Query Tool for retrieving relevant content from PDF documents via URL
  - Added `PDFClient` in `lits/clients/pdf_client.py` for PDF downloading, parsing, chunking, and vector storage
  - Added `PDFQueryTool` in `lits/tools/pdf_tools.py` for agent integration
  - Uses Qdrant local vector database with SentenceTransformer embeddings
  - Supports configurable embedding model via `EMBEDDING_MODEL_NAME` environment variable
  - Implements UUID5-based point IDs for Qdrant compatibility
  - Validates PDF content with magic byte checking
  - Caches indexed documents for efficient repeated queries
- Added comprehensive documentation in `docs/PDF_QUERY_TOOL.md`
- Added unit tests in `unit_test/tools/test_pdf_tools.py` with real PDF downloads from arXiv
- Added dependencies: `pypdf`, `qdrant-client` to `pyproject.toml`

### Changed
- Updated `lits/tools/__init__.py` to register PDF tools via `build_tools(benchmark_name="pdf")`

## 2025-11-26 v0.2.4 (prompt_injection)
### Added
- Implemented dual-registry prompt injection system with separate registries for system prompts (`task_prompt_spec`) and user message templates (`usr_prompt_spec`)
- Added `PromptRegistry.register_usr()` method for registering user message templates
- Added `PromptRegistry.get_usr()` method for retrieving user message templates
- Added `LlmTransition` base class for LLM-based transitions with prompt management
- Updated `Policy` base class to load both `task_prompt_spec` and `usr_prompt_spec` independently from registries
- Updated `RewardModel` base class to load both prompt types independently from registries
- Updated `LlmTransition` base class to load both prompt types independently from registries
- Updated all Policy implementations to support independent prompt loading:
  - `RAPPolicy` - Updated `__init__` with new parameters
  - `ToolUsePolicy` - Updated to load PromptTemplate from registry and format with tools
  - `ConcatPolicy` - Updated `__init__` with new parameters
  - `EnvGroundedPolicy` - Updated `__init__` with new parameters
- Updated all RewardModel implementations to support independent prompt loading:
  - `RapPRM` - Updated `__init__` with new parameters
  - `GenerativePRM` - Updated `__init__` with new parameters
  - `SelfConsistencyRM` - Updated `__init__` with new parameters
  - `RLHFlowPRM` - Updated `__init__` with new parameters
- Updated all Transition implementations to inherit from `LlmTransition`:
  - `RAPTransition` - Now inherits from `LlmTransition`
  - `BlocksWorldTransition` - Now inherits from `LlmTransition`
  - `ConcatTransition` - Now inherits from `LlmTransition`
- Registered existing `usr_prompt_spec` entries in `load_default_prompts()`:
  - `lits.prompts.policy.rap.usr_prompt_spec_math_qa`
  - `lits.prompts.transition.rap.usr_prompt_spec_math_qa`
  - `lits.prompts.transition.blocksworld.usr_prompt_spec`
- Added comprehensive docstrings to base classes explaining:
  - `task_prompt_spec`: System-level instructions (system message)
  - `usr_prompt_spec`: User message structure (user message template)
  - Independent loading behavior with no priority between them
- Added unit tests in `unit_test/components/`:
  - `test_policy_reward_prompts.py` - Base class prompt loading tests
  - `test_llm_policy_components.py` - All Policy implementations tests
  - `test_llm_reward_components.py` - All RewardModel implementations tests
  - `test_llm_transition.py` - All Transition implementations tests
- Added comprehensive documentation:
  - `docs/PROMPT_INJECTION_DESIGN.md` - Complete design document with usage guide, code examples, best practices
  - Updated `README.md` with prompt injection section

### Changed
- Modified `PromptRegistry.clear()` to clear both `_registry` and `_usr_registry`
- Updated `load_default_prompts()` to register both system prompts and user templates
- Organized component test files into `unit_test/components/` folder for better organization

### Fixed
- Fixed `ToolUsePolicy` to properly pass `task_type` to parent class for registry loading
- Added missing `PromptTemplate` import in `lits/prompts/policy/tool_use.py`

## 2025-11-23 v0.2.3 (env_grounded)
### Added
- Added `EnvPolicy` class in `lits/components/policy/env_grounded.py` for environment-grounded action generation with comprehensive docstrings
- Added `EnvChain` agent in `lits/agents/chain/env_chain.py` for chain-like invocation of environment policies
- Added `create_env_chain_agent()` factory function in `lits/agents/main.py`
- Added trajectory tracking to `EnvState` via `history` field that accumulates all `EnvStep` objects
- Added serialization support for `EnvState` with `to_dict()`, `from_dict()`, `save()`, and `load()` methods
- Added unit tests:
  - `unit_test/test_state_serialization.py` - TrajectoryState serialization
  - `unit_test/test_env_state_serialization.py` - EnvState serialization
  - `unit_test/test_env_chain_trajectory.py` - Trajectory tracking
  - `unit_test/test_env_chain_history_accumulation.py` - History accumulation with new states

### Changed
- Moved `type_registry.py` from `lits/agents/tree_search/` to `lits/` to resolve circular import issues
- Made `lits/agents/__init__.py` use lazy imports via `__getattr__()` to prevent circular dependencies
- Moved serialization methods (`to_dict()`, `from_dict()`, `save()`, `load()`) from base `State` class to `TrajectoryState` class
- Updated `Step.to_dict()` to include `__type__` field for polymorphic deserialization
- Updated `State.from_dict()` to use type registry for dynamic Step subclass instantiation
- Updated `EnvChain.run()` to explicitly copy history from previous state to new state, ensuring trajectory accumulation regardless of world model implementation

### Fixed
- Fixed circular import between `lits.structures` and `lits.agents` by relocating type registry
- Fixed `EnvState` serialization error ("'EnvState' object is not iterable") by implementing state-specific serialization
- Fixed trajectory loss in `EnvChain` where only the most recent step was saved instead of the full action history

## 2025-11-21 v0.2.2 (memory)
### Added
- Implemented the LiTS-Mem subpackage (`lits/memory/`) containing `LiTSMemoryManager`, trajectory-aware configs, mem0/local backends, and retrieval utilities so tree-search agents can inherit and augment cross-trajectory memories.
- Added Unit tests
  - `unit_test/test_lits_memory_manager.py`
- Added `lits.clients.als_client.py` and `lits.tools.aws_geocode.AWSGeocodeTool`
- Added `lits.eval.general_eval.GeneralEvaluator` with flexible eval-block prompting plus `EvalPerspective` helpers, alongside `unit_test/test_general_eval.py`, to cover prompt construction, JSON parsing, and retry behaviour.
- Added `TrajectoryState` and `EnvState` base classes
  -  Further changed other `***State`, e.g., `ToolUseState` now inherits from `TrajectoryState`. `BWState` inherits from `EnvState`."
- Added `StringAction` base class in `base.py` for string-based actions. 
  - `ToolUseAction` and `EnvAction` now inherit from `StringAction`

### Changed
- Simplified `Policy` subclass implementation by using `**kwargs` in `__init__` and `_get_actions` method signature.

## v0.2.1
### Added
- Added `OpenAIChatModel`, `BedrockChatModel` support
- Added `get_fn_retrieve_answer_from_tool_use_state` 

### Changed
- Changed `make_retrieve_answer` to `get_fn_retrieve_answer_from_concat_state` 

## 2025-11-2 v0.2.0
### Added
- Introduced **unified `Tool` and `BaseClient` abstractions** for all tool-use scenarios:
  - `lits.tools.base.Tool`: defines the common interface (`name`, `description`, `args_schema`, `_run`).
  - `lits.clients.base_client.BaseClient`: defines the common client interface (`request`, `ping`).
- Added a new **`lits/clients/` subpackage** to separate backend connectivity from tool logic:
  - `web_client.py` for generic REST APIs.
  - `sql_client.py` for relational DBs.
  - `geosql_client.py` for spatial databases (GeoAlchemy).
  - `mapeval_client.py` as a specialized subclass of `WebServiceClient`.
- Refactored **SQL and geospatial tools** to align with the unified `Tool` ABC:
  - `lits.tools.sql_tools` wraps LangChain SQL tools while inheriting from `Tool`.
  - `lits.tools.geosql_tools` implements spatial functions and unique-value tools with unified semantics.

### Fixed
- Fixed Pydantic attribute conflict and Method Resolution Order (MRO) in multiple inheritance (`lits/tools/sql_tools.py`)
  - e.g., when subclassing LangChain’s `LCQuerySQLDatabaseTool` (a Pydantic `BaseModel`) together with our custom `BaseTool`

### Changed
- Changed package name: `langagent` -> `lits`. 
- Add `lits_llm` to build and distribute the package (LITS-LLM) on PyPI
- Add `examples` and `lits` into `lits_llm`

## 2025-10-21 v0.1.9
### Added
- Added `langagent/components/structures/trace.py` supplying `serialize_state`, `deserialize_state`, `log_state`, and `replay_state` so every reasoning paradigm can log and replay traces uniformly.
- Allowed `ReActChatPolicy` and `create_sys_msg` to accept an optional `tool_context` string so prompts can include high-level background before the tool list
- Compatible role interface and data structure fro tool use (**BIG Update**)
  - Introduced `ToolUsePolicy` so ReAct tool selection can be accessed through the generic `Policy` interface used by tree-search components
  - Refactored `ReActChat` and the CLUE/MapEval examples to delegate action generation to `ToolUsePolicy`, aligning the agent with the new abstractions
  - Centralized all step/state definitions under the new `langagent.components.structures` package and renamed `RapStep`→`SubQAStep`, `RestStep`→`ThoughtStep`, and `ReactState`→`ToolUseState`. 
    - Note: `langagent.reasoner_base` does not re-export these aliases for backward compatibility
- Extracted shared ReAct data structures and prompt helpers into `langagent.components.tool_use` for reuse across agents and policies

### Fixed
- Fixed the log file from the fixed `mcts_debug.log` to `f"{run_id}.log"` in `log.py`
- Prevented log handlers from accumulating across runs by scoping `setup_logging` to a per-run logger
- Hardened `verb_tool` to skip schema rendering when `args_schema` is not a Pydantic model class, preventing crashes from third-party tools

### Changed
- Added `langagent.agents.search.type_registry` and updated the search stack to consume it
  - Triggered by: circular imports in `agents.search.node` -> `components.structure.core` -> `type_registry` in `agents.search.node` 

## 2025-10-14 - v0.1.8 
### Added 
- Added `benchmarks` and put `langreason` directory into it
- Added `react_chat.py` for React Chat Agent (**BIG Update: ReAct**)
  - Added Checkpoints, i.e., resumable ReAct runs by checkpointing `ReactState` and reloading conversation history when a `checkpoint_path` is provided
  - Centralized tag parsing in `ReactStep` with configurable extractors and simplified `main_react.py` by delegating think/action/answer extraction
- Added `examples/mapeval` including original implemenation (`origin.ipynb`), from-scratch ReAct implementation (`main_react.py`), evaluation code (`eval.py`),  unit testing code (`test.ipynb`)
- Add `agents` directory and put `search` directory into it
- Added `tools` directory for tool inspection
  - Added MapEval tools in `tools/mapeval_tools.py`

### Modified
- Modified `BaseLLM` to set `max_new_tokens` and `max_length` with `_get_gen_legnth`, coordinating the default values from `__init__` and other invocation methods
- Updated `HfChatModel.tokenize` to allow multi-turn chatting
  - Purpose: preserve structured chat roles during ReAct loops and avoid hallucinated observations
  - But this is also a standardized implementations of ChatLLM

## - v0.1.7
### Added
- Added `examples/main_search.py` for general support (Untested)
- Added `examples/spatial_qa`
- Added unified dataset loading interface via huggingface datasets in `load_qa_dataset`
- Added yes-no evaluation to `eval_output`
  - Added `normalize_number_pair`, `normalize_yn` to `common.py`
- Added a new dataset `spart_yn` to `load_qa_dataset`
- Added `save_dir` to `RestEvaluator` to determine whether to save the correctness and usefulness results
- Added `spatial_qa` directory to `examples`
- Added `is_mcts_method` to `mcts_utils.py`


## - v0.1.6
### Added
- Added BN-SC2
  - `check_overlap_with_context` and `BNEvaluator.sc_eval()` to `mcts_utils.py`
- Added `runtime_limit_before_iter` to `BaseSearchConfig` for all the search methods

### Fixed
- Fixed the `expand_func` in `continuation` by adding `assign_rewards=False` 
- Used `verbalize_rap_state` and `verbalize_rest_state` in `mcts_utils.retrieve_answer` and remove `question` parameter from `common.extract_numerical_answer` 
- Fixed the issue in `mcts_utils.RAPWorldModel.step()` where `retrieve_answer_from_last_step`is called but returns empty string when the answer is not found

## 2025-09-20 - v0.1.4

### Added
- Added `verbalize_rap_state` to `mcts_utils.py`
- Added the RAP support for `BNEvaluator`

### Fixed (**BIG Update: RAP**)
- Fixed `RapEvaluator`, `RAPPolicy`
- Fixed `RAPPolicy`
  - Defined the prefix ( "Now we can answer the question: ") in `usr_msg_dict` from `framework_config.gsm8k_rap.actor_dynamics["overall_question_prefix"]`
- Fixed `BNEvaluator`
  - Fixed `entropy_eval` to only try a fixed number of times (Previously, although `success` was set to exit after the maxiumn number of tries, it was not used)
- Fixed `common.py` related to mcts
  - Fixed the issue that `fast_reward` is not assigned when calling `world_model.step` for mcts (unlike bfs, `fast_reward` and `reward` are required for backpropagration)
- Fixed `mcts.py`
  - `_output_iter` is set to `[]` (originally `None`) if no path leading to terminal state is found to avoid the issue when saved to JSON
  
### Removed
- Removed `terminal_state` from `MCTSResult`

### Added
- Added runtime limit for MCTS
- Added `bn_model_name` to `BaseSearchConfig` to allow users to customize the model for `BNEvaluator`

## 2025-09-17 - v0.1.3
### Added
- Added `mcts.py`
  - Added terimated paths after selection, continuation to `trace_in_each_iter`
  - Added `unselected_simulate` to save the unselected terminal paths during simulate
- Added `MCTSNode.from_continuation` to `node.py`

### Fixed
- Fixed `mcts.py` 
  - Remove `from_simulate` and use `from_phase` instead
  - Replace `is_expanded` with `from_expand`
- Fixed `bfs_topk` to correctly add terminal nodes to `terminal_nodes` (prevent duplicates)
- Fixed `bfs_topk` to not perform expansion if the last continuous node is a terminal node

### Added
- Added the function to return tree of nodes in `bfs_topk`
  - Added `return_buckets` in `bfs_topk`
  - Added `buckets_with_terminal` in `bfs_topk` to return buckets with all the nodes
- Added additional attributes to `BaseSearchConfig`
  - Added `bn_method`, `max_new_tokens_for_bn_eval`, `max_try_for_bn_eval`, and `package_version`

### Removed
- Removed the return of `reward_details` from `QAEvaluator.reward()` ({'r_useful': r_useful, 'r_conf': r_conf})

## 2025-09-15 - v0.1.2
### Fixed (**BIG Update**)
- Fixed `bfs_topk` for depth alignment of continuation and expansion

## 2025-09-12 - v0.1.1

### Fixed 
- Fixed `create_role` to handle `example_idx` correctly
    - Previously, `example_idx` was not added when it is 0
    - Previously, role in `RestWorldModel.is_terminal` has double underscores due to an additional `example_idx = f"_{example_idx}" if example_idx is not None else ''` before calling `create_role`
- Fixed the inclusion of `from_phase` for `role` in `RestEvaluator2`
- Fixed the inclusion of `from_phase="expand"` when calling `_expand_with_existing` in `bfs.bfs_topk`

### Added
- Added `bfs.py` (**BIG Update**)
    - Added `_expand_with_existing` for BFS with continuous phase
- Added `metrics.py` 
    - Added `get_inference_cost_metrics`

## 2025-09-09 - v0.1.0

### Fixed 
- Moved the answer-extraction logic from `langagent.search.mcts` to `mcts_utils` (`extract_answer_from_aggregation`
`extract_answer_from_dfs_path`)
- Only accepted non-terminal nodes for next-level search in `bfs_topk`
- Correctly added terminal nodes to `terminal_nodes` in `bfs_topk`
- Fixed `RestEvaluator` to use `get_next_token_logits` when generated score does not work

### Added
- Added `BNEvaluator` 
    - Added `entropy` (bne) and `direct` scoring
    - Added `extract_bne_output` to avoid the issue that the LLM returns a string wrapped around the target data structure (`list[dict]`)
- Added `langreason`
- Added `exploration.md`
- Added termination strategies:  `reward_threshold`, `binary_sampling`, `verify`
- Added critic
    - TODO: `incorrect_terminate`
- Added step/phase-wise memory (e.g., rest-mcts* critic) and instance-level memory (e.g.,  termination verification in `is_terminal` can modify the state, specifically the last `RestStep` in a state)
- Added `BaseSearchConfig` to share common parameters
- Added `BFSConfig` and `bfs_topk`

### Removed
- Removed `verbose` to control log writing in LangAgent (No Need since we have a gloabl control via `setup_logging`)

### TODO
- Prevented simulation nodes from being selected during the simulation phase 
