"""Sequential test for server-down retry-with-backoff in execute_tool_action.

Mirrors usage of ``lits/tools/utils.py::execute_tool_action`` and the opt-in
attribute ``base.py::BaseTool.server_down_retry_delays``. Uses a fake tool that
fails a fixed number of times with a network exception, then succeeds — so we
can observe that retries ride out a transient blip without tripping the breaker.

Run: ``python -m unit_test.tools.test_server_down_retry``
Use ``PYTHONBREAKPOINT=0`` to skip breakpoints in batch runs.
"""
import json
import time
import urllib.error

from lits.tools.base import BaseTool
from lits.tools.utils import execute_tool_action, ToolServerDownError


class _FlakyTool(BaseTool):
    """Fails with URLError for the first ``fail_times`` calls, then returns ok.

    ``server_down_retry_delays`` is small so the test runs fast.
    """

    name = "flaky"
    description = "test tool"
    args_schema = None
    server_down_retry_delays = (0, 0, 0)  # 3 retries, no real sleeping

    def __init__(self, fail_times: int):
        object.__setattr__(self, "fail_times", fail_times)
        object.__setattr__(self, "calls", 0)

    def _run(self, **kwargs) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise urllib.error.URLError("tunnel down")
        return "[real KB result]"


class _NoRetryTool(_FlakyTool):
    """Same flaky backend but opts out of retry (default empty schedule)."""

    name = "noretry"
    server_down_retry_delays = ()


def _action(name: str) -> str:
    return json.dumps({"action": name, "action_input": {}})


def main():
    # Case 1: 2 transient failures then success → retry rides it out, returns ok.
    t1 = _FlakyTool(fail_times=2)
    obs1 = execute_tool_action(_action("flaky"), [t1])
    print(f"Case1 obs={obs1!r} calls={t1.calls}")  # expect '[real KB result]', calls=3

    # Case 2: failures exceed retry budget (3 retries -> 4 attempts) → raises.
    t2 = _FlakyTool(fail_times=10)
    raised = False
    try:
        execute_tool_action(_action("flaky"), [t2])
    except ToolServerDownError as e:
        raised = True
        print(f"Case2 raised ToolServerDownError after calls={t2.calls}: {e}")
    print(f"Case2 raised={raised}")  # expect True, calls=4 (1 + 3 retries)

    # Case 3: no-retry tool fails immediately on first URLError (default off).
    t3 = _NoRetryTool(fail_times=1)
    raised3 = False
    t0 = time.time()
    try:
        execute_tool_action(_action("noretry"), [t3])
    except ToolServerDownError:
        raised3 = True
    elapsed = time.time() - t0
    print(f"Case3 raised={raised3} calls={t3.calls} elapsed={elapsed:.3f}s")
    # expect raised=True, calls=1, elapsed ~0 (no sleeps)

    breakpoint()  # inspect: t1.calls, t2.calls, t3.calls, obs1


if __name__ == "__main__":
    main()
