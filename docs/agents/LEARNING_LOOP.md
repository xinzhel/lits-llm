# Learning Loop: Automatic Feedback from Verbal Evaluators

## Overview

The Learning Loop is an automatic feedback mechanism that enables agents to learn from their mistakes over time. It combines **input enhancement** (injecting past errors into prompts) with **output validation** (catching and saving new errors) to create a continuous improvement cycle.

## Quick Start

### Enable Learning Loop in 3 Lines

```python
from lits.agents.main import create_tool_use_agent
from lits.components.verbal_evaluator import SQLValidator
from lits.lm import get_lm

# 1. Create evaluator
validator = SQLValidator(base_model=get_lm("gpt-4"), sql_tool_names=['sql_db_query'])

# 2. Create agent with evaluators
agent = create_tool_use_agent(
    tools=tools,
    model_name="gpt-4",
    task_type="spatial_qa",
    evaluators=[validator]  # That's it! Learning loop is automatic
)

# 3. Run - learning happens automatically
state = agent.run(query="Find priority sites near Melbourne")
```

**What happens automatically:**
- ✓ Past SQL errors are loaded and injected into the prompt
- ✓ New SQL queries are validated after generation
- ✓ Validation results are saved for future iterations
- ✓ Agent learns from mistakes over time

## How It Works

### The Learning Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                     Learning Loop                            │
│                                                              │
│  1. Load Past Issues                                        │
│     ↓                                                        │
│  2. Inject into Prompt (Dynamic Notes)                      │
│     ↓                                                        │
│  3. LLM Generates Action                                    │
│     ↓                                                        │
│  4. Validate Action (Post-Generation)                       │
│     ↓                                                        │
│  5. Save Issues                                             │
│     ↓                                                        │
│  6. Next Iteration (back to step 1)                         │
└─────────────────────────────────────────────────────────────┘
```

### Example: SQL Validation Loop

**Iteration 1:**
```python
# No past issues yet
# LLM generates: SELECT * FROM users WHERE ST_DWithin(geom, point, 10)
# Validator catches: "Using geometry in EPSG:4326 with meter-level distance..."
# Issue saved to file
```

**Iteration 2:**
```python
# Past issue loaded and injected into prompt:
# "Previous SQL Errors to Avoid:
#  1. Using geometry in EPSG:4326 with meter-level distance..."
# LLM sees the note and generates correct SQL with ST_Transform
# No new issues!
```

## Usage Examples

### Single Evaluator

```python
from lits.components.verbal_evaluator import SQLValidator

validator = SQLValidator(
    base_model=get_lm("gpt-4"),
    sql_tool_names=['sql_db_query', 'sql_db_schema']
)

agent = create_tool_use_agent(
    tools=tools,
    evaluators=[validator]
)
```

### Multiple Evaluators

The learning loop supports two types of evaluators with different granularities:

**Step-Level Evaluators** validate each generated action:
- Called after each `policy.get_actions()`
- Example: `SQLValidator` checks individual SQL queries
- Use for real-time validation and immediate feedback

**Trajectory-Level Evaluators** analyze complete trajectories:
- Called after `agent.run()` completes
- Example: `SQLErrorProfiler` identifies patterns across all steps
- Use for holistic analysis and pattern detection

```python
from lits.components.verbal_evaluator import SQLValidator, SQLErrorProfiler

# Step-level: validates each SQL query as it's generated
validator = SQLValidator(
    base_model=get_lm("gpt-4"),
    sql_tool_names=['sql_db_query', 'sql_db_schema']
)

# Trajectory-level: analyzes complete trajectory for error patterns
profiler = SQLErrorProfiler(base_model=get_lm("gpt-4"))

# Both work together in the learning loop
agent = create_tool_use_agent(
    tools=tools,
    model_name="gpt-4",
    task_type="spatial_qa",
    step_evaluators=[validator],        # Per-step validation
    trajectory_evaluators=[profiler],   # Post-trajectory analysis
)

# Run agent - both evaluators work automatically:
# 1. Past issues from both evaluators → injected into prompt
# 2. Each SQL query → validated by SQLValidator
# 3. Complete trajectory → analyzed by SQLErrorProfiler
# 4. All results saved to same file (distinguished by evaluator_type)
state = agent.run(query)
```

**Benefits of Using Both:**
- **Immediate feedback**: Step-level catches errors as they happen
- **Pattern detection**: Trajectory-level identifies recurring issues
- **Unified storage**: Both save to same file with `evaluator_type` field
- **Comprehensive learning**: Agent learns from both individual mistakes and patterns

### Real-World Example: SQL Query Refinement

Here's a concrete example showing how the learning loop improves SQL queries over multiple steps:

**Step 1**: Non-SQL action (geocoding)
- Action: `AWS_Geocode` converts address to coordinates
- No validation needed (not SQL)

**Step 2**: First SQL query (no prior feedback)
```sql
SELECT * FROM enviro_audit_point 
WHERE ST_DWithin(geometry, ST_SetSRID(ST_MakePoint(144.38078, -38.08395), 4326), 100)
```
- **Validation Result**: ❌ Invalid (score: 0.7)
- **Issues Saved**:
  - Mixing geographic coords with planar distance
  - Using degrees with meter-implied distance

**Step 3**: Schema query
- Action: `sql_db_schema` queries table structure
- No validation needed (not a data query)

**Step 4**: Second SQL query (with feedback from Step 2)
```sql
SELECT * FROM enviro_audit_point 
WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(144.38078, -38.08395), 4283), 0.001)
```
- **Dynamic Notes Injected**: Issues from Step 2 loaded into prompt
- **Validation Result**: ❌ Invalid (score: 0.7)
- **Issues Saved**:
  - Distance value without units
  - Degrees as units not meaningful

**Step 5**: Third SQL query (with accumulated feedback)
```sql
SELECT * FROM enviro_audit_polygon 
WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(144.38078, -38.08395), 4283), 0.001)
```
- **Dynamic Notes Injected**: Issues from Steps 2 & 4 (max 5 items)
- **Validation Result**: ✅ Valid
- **Agent learned from feedback!**

**Key Observations:**
1. Issues are saved with `evaluator_type: "sqlvalidator"` for tracking
2. Each validation includes `sql_query` field for reference
3. Score-based ranking prioritizes worst issues (lower scores first)
4. `max_items=5` limits prompt size while focusing on critical errors
5. Agent progressively improves through accumulated feedback

### Custom Evaluator

```python
from lits.components.verbal_evaluator.base import VerbalEvaluator

class MyCustomEvaluator(VerbalEvaluator):
    def evaluate(self, step, **kwargs):
        # Your validation logic
        if self._has_issue(step):
            issue = self._extract_issue(step)
            # Save automatically
            if kwargs.get('policy_model_name'):
                self._save_eval(
                    {'issues': [issue]},
                    kwargs.get('query_idx'),
                    kwargs['policy_model_name'],
                    kwargs['task_type']
                )
            return issue
        return None
    
    def load_eval_as_prompt(self, policy_model_name, task_type, **kwargs):
        # Inherited from base class - works automatically
        return super().load_eval_as_prompt(policy_model_name, task_type, **kwargs)

# Use it
evaluator = MyCustomEvaluator(base_model=get_lm("gpt-4"))
agent = create_tool_use_agent(tools=tools, evaluators=[evaluator])
```

## Configuration Options

### Control Number of Past Issues

The learning loop loads the 5 most recent issues by default. This is configured in `ReActChat._setup_learning_loop()`:

```python
# To change, modify the agent after creation:
agent.policy.set_dynamic_notes_fn(
    lambda: [
        note
        for evaluator in agent.evaluators
        for note in evaluator.load_eval_as_prompt(
            agent.policy_model_name,
            agent.task_type,
            max_items=10  # Load more issues
        ).split('\n')
        if note.strip()
    ][:10]  # Limit total notes
)
```

### Disable Learning Loop

If you want to use evaluators without the automatic loop:

```python
# Create agent without evaluators
agent = create_tool_use_agent(tools=tools)

# Manually set up validation only (no dynamic notes)
validator = SQLValidator(base_model=get_lm("gpt-4"))

def validate_only(steps, context):
    for step in steps:
        validator.evaluate(step, **context)

agent.policy.set_post_generation_fn(validate_only)
```

## Monitoring the Learning Loop

### Check Saved Issues

```python
from pathlib import Path
from lits.lm import get_clean_model_name

model_clean = get_clean_model_name("gpt-4")
task_type = "spatial_qa"
file_path = Path.home() / ".lits_llm" / "verbal_evaluator" / f"resultdicttojsonl_{model_clean}_{task_type}.jsonl"

print(f"Issues saved to: {file_path}")
print(f"File exists: {file_path.exists()}")
```

### View as DataFrame

```python
import pandas as pd

# Load results
results = validator.load_results("gpt-4", "spatial_qa")
df = pd.DataFrame(results)

# View by evaluator type
print(df.groupby('evaluator_type').size())

# View recent issues
print(df[['evaluator_type', 'issues', 'timestamp']].tail(10))
```

### Load Formatted Feedback

```python
# What the LLM sees in the prompt
feedback = validator.load_eval_as_prompt("gpt-4", "spatial_qa", max_items=5)
print(feedback)
```

## Technical Implementation

### Architecture Overview

The learning loop is implemented through two callback mechanisms in the `Policy` base class:

1. **`set_dynamic_notes_fn()`** - Input enhancement (before generation)
2. **`set_post_generation_fn()`** - Output validation (after generation)

### Component Interaction

```
create_tool_use_agent()
    ↓
    Creates: Policy, Transition, ReActChat
    ↓
    If evaluators provided:
        ReActChat.__init__()
            ↓
            Calls: _setup_learning_loop()
                ↓
                Creates: get_dynamic_notes() function
                    ↓
                    Calls: policy.set_dynamic_notes_fn()
                ↓
                Creates: validate_generations() function
                    ↓
                    Calls: policy.set_post_generation_fn()
```

### Execution Flow

```
agent.run(query)
    ↓
    ReActChat.update_state()
        ↓
        policy.get_actions()
            ↓
            [1] policy.set_system_prompt()
                ↓
                Calls: _get_dynamic_notes()
                    ↓
                    Calls: _dynamic_notes_fn()  # get_dynamic_notes()
                        ↓
                        Loads past issues from evaluators
                        ↓
                        Returns: List[str] of notes
                    ↓
                    Formats as bullet points
                    ↓
                    Appends to system prompt
            ↓
            [2] policy._get_actions()
                ↓
                LLM generates actions with enhanced prompt
            ↓
            [3] Post-generation callback
                ↓
                Calls: _post_generation_fn()  # validate_generations()
                    ↓
                    For each evaluator:
                        evaluator.evaluate(step, ...)
                            ↓
                            Validates step
                            ↓
                            If issue found:
                                _save_eval()
                                    ↓
                                    Saves to unified file
            ↓
            Returns: validated steps
```

### Key Methods

#### `ReActChat._setup_learning_loop()`

Located in `lits/agents/chain/react.py`:

```python
def _setup_learning_loop(self):
    """Setup the learning loop with dynamic notes and validation."""
    
    all_evaluators = self.step_evaluators + self.trajectory_evaluators
    
    # Input enhancement: Load past issues as formatted prompt
    def get_dynamic_notes():
        """Load and concatenate prompts from all evaluators."""
        all_prompts = []
        for evaluator in all_evaluators:
            try:
                prompt = evaluator.load_eval_as_prompt(
                    self.policy_model_name,
                    self.task_type,
                    max_items=5  # Top 5 worst issues per evaluator
                )
                if prompt:
                    all_prompts.append(prompt)
            except Exception as e:
                logger.error(f"Error loading prompt from {evaluator.__class__.__name__}: {e}")
        
        # Return concatenated prompts (preserves section headers)
        return '\n\n'.join(all_prompts) if all_prompts else []
    
    self.policy.set_dynamic_notes_fn(get_dynamic_notes)
    
    # Output validation: Step-level evaluators
    if self.step_evaluators:
        def validate_steps(steps, context):
            for evaluator in self.step_evaluators:
                try:
                    for step in steps:
                        evaluator.evaluate(
                            step,
                            query_idx=context.get('query_idx'),
                            policy_model_name=context.get('policy_model_name'),
                            task_type=context.get('task_type')
                        )
                except Exception as e:
                    logger.error(f"Error in {evaluator.__class__.__name__}.evaluate(): {e}")
        
        self.policy.set_post_generation_fn(validate_steps)
```

**Key Changes:**
1. **Direct prompt concatenation**: Returns formatted prompt string instead of parsing into list
2. **Preserves section headers**: Maintains "Previous SQL Errors" and "Previous SQL Error Patterns" headers
3. **Separate evaluator lists**: Distinguishes `step_evaluators` from `trajectory_evaluators`
4. **Score-based ranking**: `load_eval_as_prompt()` internally ranks by score (lower = higher priority)

#### `Policy.get_actions()` - Callback Execution

Located in `lits/components/base.py`:

```python
def get_actions(self, state, query, ...):
    # [1] Set system prompt with dynamic notes
    self.set_system_prompt()  # Calls _get_dynamic_notes() internally
    
    # [2] Generate actions
    outputs = self._get_actions(state, n_actions, temperature, ...)
    
    # [3] Execute post-generation callback
    if self._post_generation_fn is not None:
        context = {
            'query': query,
            'query_idx': query_idx,
            'policy_model_name': kwargs.get('policy_model_name'),
            'task_type': kwargs.get('task_type'),
            ...
        }
        self._post_generation_fn(outputs, context)
    
    return outputs
```

#### `VerbalEvaluator.evaluate()` - Validation & Saving

Located in `lits/components/verbal_evaluator/base.py`:

```python
def evaluate(self, step, query_idx, policy_model_name, task_type):
    # Subclass implements validation logic
    result = self._validate(step)
    
    # Save if issue found
    if result and policy_model_name and task_type:
        self._save_eval(result, query_idx, policy_model_name, task_type)
    
    return result.get('issue')
```

### Data Flow

```
File: ~/.lits_llm/verbal_evaluator/resultdicttojsonl_{model}_{task}.jsonl

Record Format (SQLValidator):
{
    "evaluator_type": "sqlvalidator",
    "query_idx": 11,
    "timestamp": "2025-12-05 17:30:27",
    "issues": [
        "Mixing geographic coordinates with planar distance...",
        "Using ST_DWithin with degrees-based coordinates..."
    ],
    "is_valid": false,
    "score": 0.7,
    "sql_query": "SELECT * FROM enviro_audit_point WHERE..."
}

Record Format (SQLErrorProfiler):
{
    "evaluator_type": "sqlerrorprofiler",
    "query_idx": 11,
    "timestamp": "2025-12-05 17:35:00",
    "error_type": "CRS mismatch errors",
    "issues": [
        "Using geometry with meter-based distances...",
        "ST_Transform required for coordinate conversion..."
    ]
}

Loading:
    VerbalEvaluator.load_results()
        ↓
    Reads file, filters by evaluator_type (optional)
        ↓
    Returns: List[dict]

Formatting:
    VerbalEvaluator.load_eval_as_prompt()
        ↓
    Calls load_results() to get all records
        ↓
    Groups by evaluator_type
        ↓
    Extracts individual issues from each record
        ↓
    Ranks issues by score (lower = higher priority)
        ↓
    Selects top max_items issues (not records!)
        ↓
    Formats as prompt sections with headers
        ↓
    Returns: str (formatted prompt)

Key Improvements:
- Issue-level granularity: max_items refers to individual issues, not records
- Score-based ranking: Prioritizes worst issues (lower scores = higher priority)
- Unified format: All evaluators save 'issues' as List[str]
- SQL query tracking: SQLValidator saves 'sql_query' field for reference
```

### Extension Points

#### Custom Evaluator

Inherit from `VerbalEvaluator` and implement:

```python
class MyEvaluator(VerbalEvaluator):
    def evaluate(self, input, **kwargs):
        # Required: Validation logic
        pass
    
    # Optional: load_eval_as_prompt() inherited from base
```

#### Custom Note Formatting

Override the dynamic notes function:

```python
def custom_notes():
    # Your custom logic
    return ["Custom note 1", "Custom note 2"]

agent.policy.set_dynamic_notes_fn(custom_notes)
```

#### Custom Validation Logic

Override the post-generation function:

```python
def custom_validation(steps, context):
    # Your custom logic
    pass

agent.policy.set_post_generation_fn(custom_validation)
```

## Best Practices

### 1. Use Appropriate Evaluators

- **SQLValidator**: Step-level SQL validation
- **SQLErrorProfiler**: Trajectory-level pattern analysis
- **Custom**: Domain-specific validation

### 2. Limit Number of Notes

Too many notes can overwhelm the prompt:

```python
# Good: 5-10 notes per evaluator
max_items=5

# Avoid: Too many notes
max_items=50  # Prompt becomes too long
```

### 3. Monitor File Growth

Evaluation files grow over time. Periodically clean old entries:

```python
from pathlib import Path

file_path = Path.home() / ".lits_llm" / "verbal_evaluator" / "resultdicttojsonl_*.jsonl"
# Implement rotation or cleanup strategy
```

### 4. Use Specific Task Types

Use descriptive task types for better organization:

```python
# Good
task_type="spatial_qa_postgis"

# Avoid
task_type="task1"
```

### 5. Test Evaluators Separately

Before enabling the loop, test evaluators independently:

```python
# Test validator
result = validator.evaluate(step, query_idx=0, 
                            policy_model_name="gpt-4",
                            task_type="test")
print(result)

# Then enable loop
agent = create_tool_use_agent(tools=tools, evaluators=[validator])
```

## Troubleshooting

### No Issues Being Saved

Check:
1. `policy_model_name` and `task_type` are provided
2. Evaluator's `evaluate()` method returns issues
3. File permissions for `~/.lits_llm/verbal_evaluator/`

### Notes Not Appearing in Prompt

Check:
1. Issues exist in the file
2. `load_eval_as_prompt()` returns non-empty string
3. Dynamic notes function is set correctly

### Too Many API Calls

Each evaluator makes LLM calls. To reduce:
1. Use fewer evaluators
2. Implement caching in custom evaluators
3. Use deterministic temperature (0.0)

## See Also

- [Verbal Evaluator Framework](../components/verbal_evaluator/VERBAL_EVALUATOR.md)
- [Dynamic Notes Injection](../components/callback.md)
- [ReActChat Agent](./REACT_CHAT.md)
- [Policy Base Class](../components/policy/POLICY.md)
