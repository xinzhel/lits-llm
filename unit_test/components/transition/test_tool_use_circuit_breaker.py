"""Unit test for the cross-example circuit-breaker counter in
`lits/components/transition/tool_use.py::ToolUseTransition`.

What this test covers (subtask 1.5 of
`.kiro/specs/lits_mem/0522-minor-tool-failure-circuit-breaker/tasks.md`):

i.   case_i_counter_increments_below_threshold:
     With a fake tool that always raises `urllib.error.URLError(...)`, two
     consecutive `step()` calls increment `consecutive_server_down` from 0→1
     then 1→2 without raising. The appended step's `observation` carries the
     "unreachable" phrasing and the attempt number — Layer (c) preserves
     evidence in state for case-study analysis.

ii.  case_j_threshold_reached_re_raises_and_appends_trip_step:
     Continuing from (i) (counter=2, threshold=3), the third call re-raises
     `ToolServerDownError`. Before the raise, `step()` appends a trip step
     whose observation contains the phrase
     "Final attempt before circuit-breaker abort" so post-hoc analysis can
     distinguish circuit-breaker aborts from natural max-steps termination.

iii. case_iii_counter_does_NOT_reset_across_examples:
     Counter is run-scoped, not per-example. After 1 failure in "example 0"
     and `init_state()` (which simulates moving to "example 1"), a single
     failure in example 1 brings the counter to 2 — not 1.

iv.  case_iv_successful_execution_resets_counter:
     Successful tool call is positive evidence the backend is alive. After
     a failure (counter=1), swapping in a tool that returns "OK" causes the
     next `step()` to succeed and reset the counter to 0. A subsequent
     failure starts at attempt=1/3 again.

v.   case_v_unrelated_exception_does_NOT_reset_counter:
     Unrelated exceptions (e.g. `ValueError` from a malformed action) are
     transparent to the breaker — they neither prove the backend is alive
     nor that it's down. After counter=2, an unrelated failure leaves the
     counter at 2 (not 0). The next ToolServerDownError trips at counter=3.

vi.  case_vi_init_state_does_NOT_reset_counter:
     The `init_state()` hook is called per example, but the counter is
     run-scoped. Setting counter=2 manually and calling `init_state()`
     leaves the counter at 2.

vii. case_vii_qa_q8_intermittent_scenario:
     Reproduces the silent-burn scenario in spec QA Q8.
        example 0: 1 failure → 1 success      (counter: 0→1→0)
        example 1: 2 failures, no success     (counter: 0→1→2)
        example 2: 1 failure                  (counter: 2→3 → trip)
     A v2-style per-example-reset breaker would never trip here; v3 does.

Implementation references:
- Transition under test: `lits/components/transition/tool_use.py::ToolUseTransition.step`
  (server-down branch, success-path counter reset), `::ToolUseTransition.init_state`
  (no longer resets the counter — see QA Q8).
- Server-down classification: `lits/tools/utils.py::execute_tool_action`
  (whitelist + string-marker scan).

Run (no breakpoints, batch-friendly):

    PYTHONBREAKPOINT=0 python -m unit_test.components.transition.test_tool_use_circuit_breaker
"""

import json
import urllib.error

from lits.components.transition.tool_use import ToolUseTransition
from lits.structures import ToolUseState, ToolUseStep, ToolUseAction
from lits.tools import ToolServerDownError


# ---------------------------------------------------------------------------
# Fakes — minimal tool stand-ins. `execute_tool_action` looks for `.name` and
# invokes either `_run(**kwargs)` or `__call__(**kwargs)`.
# ---------------------------------------------------------------------------

class _FakeRaisingTool:
    """Tool stand-in that always raises the given exception from `_run`."""

    def __init__(self, name: str, raise_exc: BaseException):
        self.name = name
        self._raise_exc = raise_exc

    def _run(self, **kwargs):
        raise self._raise_exc


class _FakeReturningTool:
    """Tool stand-in that returns a fixed string from `_run` (no raise)."""

    def __init__(self, name: str, return_value: str):
        self.name = name
        self._return_value = return_value

    def _run(self, **kwargs):
        return self._return_value


def _action_step() -> ToolUseStep:
    """Build a fresh ToolUseStep targeting `fake_tool` with empty args."""
    return ToolUseStep(
        action=ToolUseAction(json.dumps({"action": "fake_tool", "action_input": {}})),
    )


# ---------------------------------------------------------------------------
# Test cases (sequentially executable, no pytest)
# ---------------------------------------------------------------------------

def case_i_counter_increments_below_threshold():
    """First two server-down attempts return without raising; counter goes 0→1→2.
    The trajectory state preserves a per-step "unreachable" observation
    (Layer c — evidence in checkpoints / terminal_nodes JSON).
    """
    tool = _FakeRaisingTool("fake_tool", urllib.error.URLError("connection refused"))
    transition = ToolUseTransition(tools=[tool], tool_failure_threshold=3)
    state = transition.init_state()

    # Attempt 1 — counter 0 → 1.
    state, aux = transition.step(state, _action_step(), query_idx=0)
    if transition.consecutive_server_down != 1:
        print("[case i] FAIL — expected counter=1 after first failure, got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    if not state or "unreachable" not in (state[-1].observation or ""):
        print("[case i] FAIL — expected 'unreachable' in observation, got",
              repr(state[-1].observation if state else None))
        raise SystemExit(1)
    if "1/3" not in state[-1].observation:
        print("[case i] FAIL — expected attempt 1/3 in observation, got",
              repr(state[-1].observation))
        raise SystemExit(1)
    if aux.get("confidence") != 0.0:
        print("[case i] FAIL — expected confidence 0.0 below threshold, got", aux)
        raise SystemExit(1)

    # Attempt 2 — counter 1 → 2.
    state, aux = transition.step(state, _action_step(), query_idx=0)
    if transition.consecutive_server_down != 2:
        print("[case i] FAIL — expected counter=2 after second failure, got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    if "2/3" not in (state[-1].observation or ""):
        print("[case i] FAIL — expected attempt 2/3 in observation, got",
              repr(state[-1].observation))
        raise SystemExit(1)

    print("[case i] OK — counter=2, last observation:", repr(state[-1].observation))
    return transition, state


def case_ii_threshold_reached_re_raises_and_appends_trip_step(transition, state):
    """Third consecutive server-down failure: counter goes 2 → 3, the trip
    step is appended to state (Layer c), then `ToolServerDownError` is
    re-raised so the CLI top-level handler can abort the run.
    """
    raised = False
    try:
        transition.step(state, _action_step(), query_idx=0)
    except ToolServerDownError as e:
        raised = True
        print("[case ii] OK — got ToolServerDownError on threshold:", e)

    if not raised:
        print("[case ii] FAIL — expected ToolServerDownError to propagate at threshold")
        raise SystemExit(1)
    if transition.consecutive_server_down != 3:
        print("[case ii] FAIL — expected counter=3 at trip, got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    print("[case ii] counter at trip:", transition.consecutive_server_down)


def _verify_trip_step_observation_contents():
    """Direct-evidence check for case (ii): rebuild from scratch, capture the
    new_state mutated before re-raise via a side-channel append patch.

    This complements case_ii by confirming the trip step's observation string
    contains the special "Final attempt before circuit-breaker abort" phrase.
    """
    tool = _FakeRaisingTool("fake_tool", urllib.error.URLError("connection refused"))
    transition = ToolUseTransition(tools=[tool], tool_failure_threshold=3)
    state = transition.init_state()
    state, _ = transition.step(state, _action_step(), query_idx=0)  # 1
    state, _ = transition.step(state, _action_step(), query_idx=0)  # 2

    captured = {}
    original_append = ToolUseState.append

    def _capturing_append(self, item):
        captured["last"] = item
        return original_append(self, item)

    ToolUseState.append = _capturing_append
    try:
        transition.step(state, _action_step(), query_idx=0)
    except ToolServerDownError:
        pass
    finally:
        ToolUseState.append = original_append

    obs = captured.get("last").observation if captured.get("last") else None
    if not obs or "Final attempt before circuit-breaker abort" not in obs:
        print("[case ii+] FAIL — trip step observation missing trip phrase, got",
              repr(obs))
        raise SystemExit(1)
    print("[case ii+] OK — trip step observation:", repr(obs))


def case_iii_counter_does_NOT_reset_across_examples():
    """Cross-example accumulation: 1 fail in example 0, then `init_state()`
    (simulating example 1 start), then 1 fail in example 1 → counter=2.
    A v2-style per-example-reset breaker would have counter=1 here.
    """
    tool = _FakeRaisingTool("fake_tool", urllib.error.URLError("connection refused"))
    transition = ToolUseTransition(tools=[tool], tool_failure_threshold=3)

    # Example 0: 1 failure.
    state0 = transition.init_state()
    state0, _ = transition.step(state0, _action_step(), query_idx=0)
    if transition.consecutive_server_down != 1:
        print("[case iii] FAIL — expected counter=1 after example 0 failure, got",
              transition.consecutive_server_down)
        raise SystemExit(1)

    # Move to example 1: init_state must NOT reset the counter.
    state1 = transition.init_state()
    if transition.consecutive_server_down != 1:
        print("[case iii] FAIL — init_state reset counter (should not), got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    if len(state1) != 0:
        print("[case iii] FAIL — init_state should return empty state, got len",
              len(state1))
        raise SystemExit(1)

    # Example 1: 1 failure → counter goes to 2 (cross-example accumulation).
    state1, _ = transition.step(state1, _action_step(), query_idx=1)
    if transition.consecutive_server_down != 2:
        print("[case iii] FAIL — expected counter=2 after example 1 failure (cross-example), got",
              transition.consecutive_server_down)
        raise SystemExit(1)

    print("[case iii] OK — counter accumulated across examples: counter=2 (1 from ex0, 1 from ex1)")


def case_iv_successful_execution_resets_counter():
    """A successful tool execution is positive evidence the backend is alive
    and resets the counter to 0. After reset, the next failure starts at
    attempt=1/3.
    """
    failing_tool = _FakeRaisingTool("fake_tool", urllib.error.URLError("connection refused"))
    transition = ToolUseTransition(tools=[failing_tool], tool_failure_threshold=3)
    state = transition.init_state()
    state, _ = transition.step(state, _action_step(), query_idx=0)  # counter=1

    if transition.consecutive_server_down != 1:
        print("[case iv] FAIL — setup expected counter=1, got",
              transition.consecutive_server_down)
        raise SystemExit(1)

    # Swap in a successful tool. The transition holds the tool list by
    # reference, so we replace the entry in place.
    successful_tool = _FakeReturningTool("fake_tool", "OK")
    transition.tools[:] = [successful_tool]

    state, aux = transition.step(state, _action_step(), query_idx=0)
    if transition.consecutive_server_down != 0:
        print("[case iv] FAIL — expected counter=0 after success (reset), got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    if state[-1].observation != "OK":
        print("[case iv] FAIL — expected observation 'OK', got",
              repr(state[-1].observation))
        raise SystemExit(1)
    if aux.get("confidence") != 1.0:
        print("[case iv] FAIL — expected confidence 1.0 on success, got", aux)
        raise SystemExit(1)

    # Subsequent failure should restart at attempt=1/3.
    transition.tools[:] = [failing_tool]
    state, _ = transition.step(state, _action_step(), query_idx=0)
    if transition.consecutive_server_down != 1:
        print("[case iv] FAIL — expected counter=1 after post-reset failure, got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    if "1/3" not in (state[-1].observation or ""):
        print("[case iv] FAIL — expected attempt 1/3 after reset, got",
              repr(state[-1].observation))
        raise SystemExit(1)
    print("[case iv] OK — success reset works; post-reset observation:",
          repr(state[-1].observation))


def case_v_unrelated_exception_does_NOT_reset_counter():
    """An unrelated exception raised by `execute_tool_action` itself (not by
    a tool's `_run`) is transparent to the circuit breaker — counter neither
    increments nor resets.

    Subtlety: when a tool's `_run` raises a non-network exception (e.g.
    `ValueError`), `execute_tool_action` catches it, classifies it as NOT
    server-down, and returns a `PREFIX_FOR_ERROR_OBSERVATION`-prefixed
    observation string. From the transition's perspective, this looks like
    a successful return — and the counter resets in the `else` branch. This
    is correct behavior: the tool wrapper getting far enough to format an
    error string IS positive evidence the backend is alive (otherwise the
    network exception would have triggered `ToolServerDownError`).

    The transition's `except Exception` branch only fires when
    `execute_tool_action` itself raises — e.g. `ValueError("No tool found
    with name 'X'")` when the LLM picks a non-existent tool name. The
    backend isn't even called in that case, so it's neither evidence of
    backend health nor of backend death. Counter preserved.
    """
    failing_tool = _FakeRaisingTool("fake_tool", urllib.error.URLError("connection refused"))
    transition = ToolUseTransition(tools=[failing_tool], tool_failure_threshold=3)
    state = transition.init_state()
    state, _ = transition.step(state, _action_step(), query_idx=0)  # counter=1
    state, _ = transition.step(state, _action_step(), query_idx=0)  # counter=2
    if transition.consecutive_server_down != 2:
        print("[case v] FAIL — setup expected counter=2, got",
              transition.consecutive_server_down)
        raise SystemExit(1)

    # Trigger transition's `except Exception` by referring to a non-existent
    # tool — `execute_tool_action` raises `ValueError("No tool found ...")`
    # before its inner try block.
    bad_action = ToolUseStep(
        action=ToolUseAction(json.dumps({"action": "nonexistent_tool", "action_input": {}})),
    )
    raised = False
    try:
        state, _ = transition.step(state, bad_action, query_idx=0)
    except Exception as exc:
        raised = True
        print("[case v] FAIL — transition should swallow ValueError into observation, got", exc)

    if raised:
        raise SystemExit(1)
    if transition.consecutive_server_down != 2:
        print("[case v] FAIL — expected counter=2 after unrelated exception (unchanged), got",
              transition.consecutive_server_down)
        raise SystemExit(1)

    # Now another ToolServerDownError → counter goes to 3 → trip.
    raised_trip = False
    try:
        transition.step(state, _action_step(), query_idx=0)
    except ToolServerDownError:
        raised_trip = True
    if not raised_trip:
        print("[case v] FAIL — expected trip after counter reaches 3, no raise")
        raise SystemExit(1)
    if transition.consecutive_server_down != 3:
        print("[case v] FAIL — expected counter=3 at trip, got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    print("[case v] OK — unrelated `execute_tool_action` exception left counter at 2; "
          "next ToolServerDownError tripped at 3")


def case_vi_init_state_does_NOT_reset_counter():
    """`init_state()` is a per-example hook but the counter is run-scoped.
    Setting counter=2 manually and calling `init_state()` leaves counter=2.
    """
    tool = _FakeRaisingTool("fake_tool", urllib.error.URLError("connection refused"))
    transition = ToolUseTransition(tools=[tool], tool_failure_threshold=3)
    transition.consecutive_server_down = 2
    new_state = transition.init_state()
    if transition.consecutive_server_down != 2:
        print("[case vi] FAIL — expected counter=2 after init_state (no reset), got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    if len(new_state) != 0:
        print("[case vi] FAIL — expected empty state from init_state, got len",
              len(new_state))
        raise SystemExit(1)
    print("[case vi] OK — init_state preserved counter=2 and returned empty state")


def case_vii_qa_q8_intermittent_scenario():
    """Reproduces the silent-burn scenario in spec QA Q8.

    Trajectory:
        example 0: 1 fail → 1 success     (counter: 0 → 1 → 0)
        example 1: 2 fails                (counter: 0 → 1 → 2)
        example 2: 1 fail → trip          (counter: 2 → 3 → raise)

    A v2-style per-example-reset breaker would never trip here. v3 does.
    """
    failing_tool = _FakeRaisingTool("fake_tool", urllib.error.URLError("connection refused"))
    successful_tool = _FakeReturningTool("fake_tool", "OK")
    transition = ToolUseTransition(tools=[failing_tool], tool_failure_threshold=3)

    # --- example 0: 1 failure, then 1 success (success resets counter to 0) ---
    state0 = transition.init_state()
    state0, _ = transition.step(state0, _action_step(), query_idx=0)
    if transition.consecutive_server_down != 1:
        print("[case vii] FAIL — ex0 step1 expected counter=1, got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    transition.tools[:] = [successful_tool]
    state0, _ = transition.step(state0, _action_step(), query_idx=0)
    if transition.consecutive_server_down != 0:
        print("[case vii] FAIL — ex0 step2 expected counter=0 after success, got",
              transition.consecutive_server_down)
        raise SystemExit(1)

    # --- example 1: 2 failures, no success ---
    transition.tools[:] = [failing_tool]
    state1 = transition.init_state()
    if transition.consecutive_server_down != 0:
        print("[case vii] FAIL — init_state should not change counter; expected 0, got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    state1, _ = transition.step(state1, _action_step(), query_idx=1)  # counter=1
    state1, _ = transition.step(state1, _action_step(), query_idx=1)  # counter=2
    if transition.consecutive_server_down != 2:
        print("[case vii] FAIL — ex1 expected counter=2, got",
              transition.consecutive_server_down)
        raise SystemExit(1)

    # --- example 2: 1 failure → trip ---
    state2 = transition.init_state()
    if transition.consecutive_server_down != 2:
        print("[case vii] FAIL — init_state should not change counter; expected 2, got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    raised = False
    try:
        transition.step(state2, _action_step(), query_idx=2)
    except ToolServerDownError as e:
        raised = True
        print("[case vii] OK — tripped on example 2 first failure:", e)
    if not raised:
        print("[case vii] FAIL — expected trip on example 2, did not raise")
        raise SystemExit(1)
    if transition.consecutive_server_down != 3:
        print("[case vii] FAIL — expected counter=3 at trip, got",
              transition.consecutive_server_down)
        raise SystemExit(1)
    print("[case vii] OK — intermittent-failures scenario correctly trips at cross-example counter=3")


if __name__ == "__main__":
    transition, state = case_i_counter_increments_below_threshold()
    case_ii_threshold_reached_re_raises_and_appends_trip_step(transition, state)
    _verify_trip_step_observation_contents()
    case_iii_counter_does_NOT_reset_across_examples()
    case_iv_successful_execution_resets_counter()
    case_v_unrelated_exception_does_NOT_reset_counter()
    case_vi_init_state_does_NOT_reset_counter()
    case_vii_qa_q8_intermittent_scenario()
    print("\nALL CASES PASSED")
