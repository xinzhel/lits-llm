# ReActChat Agent

ReAct-style reasoning-and-acting loop for tool-augmented LLMs. Iteratively generates thoughts, actions, and observations until reaching a final answer.

## Use Case

Question answering, tool use, interactive problem solving

## Usage

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

## Key Characteristics

| Feature | Description |
|---------|-------------|
| Domain | Tool use, QA |
| State | Conversation history |
| Actions | Tool calls |
| Termination | Final answer |
| Validation | Tool execution |
