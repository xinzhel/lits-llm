# Native Tool Use Policy — Known Issues

## Garbled LLM Output → ValidationException on Next Call

**Observed**: 2026-05-08, Terminal-Bench run, example 56 (`caffe-cifar-10`), attempt 4.

**Conditions**: Haiku 3.5, temperature=0.9. Observed with short context (5K input tokens, 4 steps). Not caused by long context — appears to be a stochastic failure mode of the model at high temperature.

**Symptom**:
```
RuntimeError: Bedrock Converse API call failed: An error occurred (ValidationException)
when calling the Converse operation: 2 validation errors detected:
Value at 'messages.4.member.content.2.member.toolUse.name' failed to satisfy constraint...
```

**Root cause**:

1. LLM returns degenerate output — garbled XML-like tags mixed with broken text:
   ```
   ./train_quick.sh 2>& 1 | tee //appffe//training_output.txt</parametertml:parameterameter>
   \n</antmlx:>tml:function>\n\nLet me verify the the the and training output:
   \n\n<:function_calls>\n<invoke n name="_shell">\n<parameter name="parameter command...
   ```

2. Bedrock's Converse API returns this as a `toolUse` block in the response (since native tool use mode is enabled). The `name` field contains garbled text.

3. This malformed `toolUse` step gets appended to the conversation state.

4. On the **next** `policy.get_actions()` call, the state is serialized into messages and sent to Bedrock. Bedrock validates the `toolUse.name` field in the conversation history and rejects it → `ValidationException`.

**Call chain**:
```
NativeReAct.run() loop
  → policy.get_actions(state)           # step N+1
    → NativeToolUsePolicy._get_actions()
      → _call_model(messages)           # messages contain garbled step N
        → bedrock_chat._converse_api()
          → self.client.converse(**params)  ← Bedrock rejects here
```

**Why it happens**:
- High temperature (0.9) can cause Haiku to produce degenerate output even with short context
- The Converse API's native tool use mode still returns a `toolUse` block even when the content is garbled
- No validation of `toolUse.name` after receiving the response

**Impact**: Process crashes. On resume, the corrupted checkpoint causes the same error (state contains the garbled step). Requires manual intervention (write reward file to skip).

**Current workaround**: Manually write `{example_idx}_a{attempt}_reward.json` with `reward: 0.0` to skip the corrupted attempt.

**Potential fixes** (not yet implemented):
1. Validate `toolUse.name` after parsing LLM response — if invalid, treat as a terminal error step (no tool call, just append error to state and let agent continue or terminate).
2. Catch `ValidationException` in the ReAct loop and terminate the attempt gracefully (write reward=0).
3. Truncate conversation history when it exceeds a threshold to reduce degenerate output probability.
