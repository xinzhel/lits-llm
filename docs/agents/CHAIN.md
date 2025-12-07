# Chain Agents

This directory contains chain-style agent implementations for sequential reasoning and planning.

## Available Agents

### ReActChat
ReAct-style reasoning-and-acting loop for tool-augmented LLMs. Iteratively generates thoughts, actions, and observations until reaching a final answer.

**Use Case**: Question answering, tool use, interactive problem solving

### EnvChain
Environment-grounded chain agent for planning tasks. Iteratively generates and executes actions in an environment until a goal is reached.

**Use Case**: Planning tasks (BlocksWorld, logistics, robotics), sequential decision making

## Usage

### EnvChain Example

```python
from lits.agents import create_env_chain_agent

# Define action generator
def generate_all_actions(env_state):
    # Return list of valid actions for current state
    return ["action1", "action2", ...]

# Define goal checker
def goal_check(state, query):
    # Return True if goal is reached
    return is_goal_satisfied(state, query)

# Define prompts
prompts = {
    "policy": "State: <init_state>\nGoals: <goals>\nActions:\n<action>\nSelect:"
}

# Create agent
agent = create_env_chain_agent(
    prompt_templates=prompts,
    generate_all_actions=generate_all_actions,
    world_model=your_world_model,
    goal_check=goal_check,
    max_steps=15
)

# Run agent
final_state = agent.run(
    query="achieve goal X",
    problem_instance=problem_data
)
```

### ReActChat Example

```python
from lits.agents import create_tool_use_agent

# Create agent
agent = create_tool_use_agent(
    tools=tool_list,
    max_iter=50
)

# Run agent
state = agent.run(query="What is the weather in Paris?")
```

## Key Differences

| Feature | ReActChat | EnvChain |
|---------|-----------|----------|
| Domain | Tool use, QA | Planning, environments |
| State | Conversation history | Environment state |
| Actions | Tool calls | Environment actions |
| Termination | Final answer | Goal reached |
| Validation | Tool execution | Action validity |
