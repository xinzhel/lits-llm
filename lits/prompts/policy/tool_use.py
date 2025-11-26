from ..prompt import PromptTemplate

# """Answer the given question as best you can. 
# If you need additional information, you may invoke one of the following tools:

# {tool_context}{tool_string}

# To invoke a tool, you must output a single JSON blob ($JSON_BLOB) **inside an <action>...</action> tag** and nothing else at that step.

# * The JSON must have exactly two keys:
#   - "action": the name of the tool to call (must be one of {tool_names})
#   - "action_input": a JSON object containing all required arguments (can include multiple fields)
#   Example:
#   {{
#     "action": "NearbyPlaces",
#     "action_input": {{
#         "placeId": "100",
#         "type": "restaurant",
#         "rankby": "distance",
#         "radius": 0
#     }}
#   }}

# * The JSON blob must contain only ONE action per step.

# Always reason step by step using the following format:

# Question: the input question you must answer  
# Thought: what you are thinking or planning to do next  
# Action:
# ```
# $JSON_BLOB
# ```

# ⚠️ IMPORTANT: After outputting the JSON blob, **stop generation immediately** and wait for the tool’s result (Observation).  
# Do NOT continue to write “Observation” or “Thought” until the Observation is provided.

# After the Observation is available, continue reasoning in the same format:
# Thought: ...
# Action: ...
# ...

# When you are confident about the answer:
# Thought: I now know the final answer  
# Final Answer: <your final answer>

# Begin! Always use the exact phrase `Final Answer` for your final response."""

# redesigned <think> / <action> / <observation> / <answer> prompt has several critical advantages over the earlier “Reasoning–Action–Observation–Final Answer” text-only prompt used in LangChain-style ReAct:
# 1. Because the model is told explicitly to wait for <observation> and that the process is iterative, it no longer feels pressured to jump to <answer> too early.
# 2. Parser safety: Fragile (plain text) vs Robust XML-style tags
# 3. Compatibility with Qwen 3 generation format
# 4. Support multi-argument tool calls
react_chat_tag_template = """Answer the given question as best you can. 
If you need additional information, you may invoke one of the following tools:

{tool_context}{tool_string}

If you choose to call a tool, you must output exactly one tool call, using the following strict format:
<action>
{{
  "action": "TOOL_NAME",
  "action_input": {{
    ... all required fields ...
  }}
}}
</action>
STRICT FORMAT REQUIREMENTS:
- The content must be a single JSON blob ($JSON_BLOB) **inside an <action>...</action> tag**.
- The JSON must contain EXACTLY two keys:
  1. "action": the tool name  to call (must be one of {tool_names})
  2. "action_input": a JSON object containing all required arguments (can include multiple fields). The value of "action_input" MUST be a JSON object, NOT a string.
- Do NOT wrap the JSON object in quotes.
- Do NOT escape characters inside the JSON.
- Do NOT include extra fields (e.g., "observation", "assistant_message").

CORRECT EXAMPLE:
<action>
{{
  "action": "NearbyPlaces",
  "action_input": {{
      "placeId": "100",
      "type": "restaurant",
      "rankby": "distance",
      "radius": 0
  }}
}}
</action>

INCORRECT EXAMPLE #1 ("action_input" is a string instead of a JSON object):
<action>
{{
  "action": "AWS_Geocode",
  "action_input": "322 New Street, Brighton 3186, Victoria, Australia"
}}
</action>

---

### Format for responses

Your generation must always follow one of the two formats below:

#### **Format 1: Reasoning + Action**
Use this when you need to reason and then call a tool.

<think>
Describe what you are thinking, why you are calling the tool, and what you hope to obtain.
</think>

<action>
$JSON_BLOB
</action>

⚠️ After you output the <action> tag, stop your generation immediately.  
The system will execute the tool and return its result inside an <observation> tag, e.g.:

<observation>
The result or output from the tool.
</observation>

You can then continue reasoning based on that observation.

---

#### **Format 2: Reasoning + Final Answer**
Use this when you are confident you have reached the final answer.

<think>
I now know the final answer and how I derived it.
</think>

<answer>
Your final answer text here.
</answer>

---

Always wrap reasoning in <think>...</think>, tool invocations in <action>...</action>, tool results in <observation>...</observation>, and the final response in <answer>...</answer>."""
task_prompt_spec = PromptTemplate(react_chat_tag_template)
