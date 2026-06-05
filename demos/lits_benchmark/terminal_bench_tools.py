"""Shell tool for Terminal-Bench integration with LiTS.

Provides ``ShellTool``, a LiTS ``BaseTool`` subclass that executes bash
commands inside a ``TerminalBenchEnv`` Docker container. This is the single
tool exposed to the LLM agent for Terminal-Bench tasks.

Design follows CLI tool use pattern (see design.md § QA):
- Single tool (``shell``), minimal schema overhead
- Output truncation follows terminus-2's ``_limit_output_length``:
  10 KB max, keep first/last halves

Usage::

    from lits_benchmark.terminal_bench import TerminalBenchEnv
    from lits_benchmark.terminal_bench_tools import ShellTool

    env = TerminalBenchEnv(task_dir)
    env.start()
    tool = ShellTool(env)
    print(tool._run(command="ls /app"))
    env.stop()
"""

from pydantic import BaseModel, Field

from lits.tools.base import BaseTool
from .terminal_bench import TerminalBenchEnv


class ShellInput(BaseModel):
    """Input schema for the shell tool."""
    command: str = Field(description="The bash command to execute in the container.")


class ShellTool(BaseTool):
    """Execute a bash command inside a Terminal-Bench Docker container.

    This is the only tool the LLM agent sees for Terminal-Bench tasks.
    The agent outputs a bash command string, we execute it via
    ``TerminalBenchEnv.exec_sync()``, and return stdout+stderr.

    Args:
        env: A started ``TerminalBenchEnv`` instance.
    """
    name: str = "shell"
    description: str = (
        "Execute a bash command in the task's Docker container. "
        "Returns stdout and stderr. Use this to explore the filesystem, "
        "install packages, run scripts, compile code, etc."
    )
    args_schema = ShellInput

    # The shell tool's output is arbitrary command stdout/stderr — it routinely
    # contains words like "connect", "error", or "404" (e.g. from wget/curl/ping)
    # that are normal task content, not signals that the Docker backend is down.
    # Opt out of string-return server-down classification to avoid false-positive
    # circuit-breaker trips. A genuinely dead container surfaces as a raised
    # exception from exec_sync(), which is still classified normally.
    classify_string_result_as_server_down: bool = False

    def __init__(self, env: TerminalBenchEnv):
        # Skip BaseTool.__init__ which expects a client arg;
        # same pattern as KG tools (kgqa_tools.py::_KGToolBase)
        object.__setattr__(self, "env", env)

    def _run(self, command: str) -> str:
        """Execute command and return combined output.

        Output truncation follows terminus-2's _limit_output_length:
        10 KB max bytes, keep first half + last half with omission notice.
        """
        result = self.env.exec_sync(command)
        output = result.stdout or ""
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return _limit_output(output) or "(no output)"


def _limit_output(output: str, max_bytes: int = 10000) -> str:
    """Truncate output keeping first and last portions.

    Follows terminus-2's ``_limit_output_length``: 10 KB max, byte-level
    split, keep first half + last half with omission notice in the middle.

    Args:
        output: Raw command output string.
        max_bytes: Maximum allowed bytes (default 10000, matching terminus-2).

    Returns:
        Original output if under limit, or truncated with middle omitted.
    """
    raw = output.encode("utf-8")
    if len(raw) <= max_bytes:
        return output
    half = max_bytes // 2
    first = raw[:half].decode("utf-8", errors="ignore")
    last = raw[-half:].decode("utf-8", errors="ignore")
    omitted = len(raw) - len(first.encode("utf-8")) - len(last.encode("utf-8"))
    return (
        f"{first}\n[... output limited to {max_bytes} bytes; "
        f"{omitted} interior bytes omitted ...]\n{last}"
    )
