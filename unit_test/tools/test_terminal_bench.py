"""Test Terminal-Bench integration: TerminalBenchEnv, ShellTool, dataset loader, verifier.

Mirrors usage patterns from:
- terminal_bench.py::TerminalBenchEnv (start, exec_sync, verify, stop)
- terminal_bench.py::load_terminal_bench (dataset loader)
- terminal_bench_tools.py::ShellTool (_run, _limit_output)

Run with: python -m unit_test.tools.test_terminal_bench
Requires: Docker running, Harbor cache populated
(run 'harbor run --dataset terminal-bench@2.0 --agent oracle --n-tasks 1' first)

Uses breakpoint() for manual inspection per project rules.
Skip all breakpoints with: PYTHONBREAKPOINT=0 python -m unit_test.tools.test_terminal_bench
"""

import glob
from pathlib import Path


def find_task(name: str = "break-filter-js-from-html") -> Path:
    """Find a cached task by name."""
    matches = glob.glob(str(Path.home() / f".cache/harbor/tasks/*/{name}"))
    assert matches, f"Task '{name}' not found in Harbor cache. Run harbor first."
    return Path(matches[0])


def test_dataset_loader():
    """terminal_bench.py::load_terminal_bench — verify dataset loads correctly."""
    from demos.lits_benchmark.terminal_bench import load_terminal_bench

    # Load all
    all_tasks = load_terminal_bench()
    print(f"[dataset_loader] Total tasks: {len(all_tasks)}")
    breakpoint()  # inspect: len(all_tasks), all_tasks[0].keys(), set of categories

    # Check keys
    t = all_tasks[0]
    assert "question" in t
    assert "task_id" in t
    assert "task_dir" in t
    assert "metadata" in t
    assert "difficulty" in t["metadata"]
    assert "category" in t["metadata"]

    # Filter by category
    se_tasks = load_terminal_bench(category="software-engineering")
    print(f"[dataset_loader] Software engineering tasks: {len(se_tasks)}")

    # Filter by difficulty
    hard_tasks = load_terminal_bench(difficulty="hard")
    print(f"[dataset_loader] Hard tasks: {len(hard_tasks)}")

    categories = sorted(set(t["metadata"]["category"] for t in all_tasks))
    print(f"[dataset_loader] Categories: {categories}")
    print("[dataset_loader] PASSED")


def test_env_lifecycle():
    """terminal_bench.py::TerminalBenchEnv — start, exec_sync, get_instruction, stop."""
    from demos.lits_benchmark.terminal_bench import TerminalBenchEnv

    task_dir = find_task()
    env = TerminalBenchEnv(task_dir, container_name="tb-unittest-lifecycle")

    env.start()
    print(f"[env_lifecycle] Container started: {env.container_name}")

    # exec_sync
    r = env.exec_sync("echo hello")
    print(f"[env_lifecycle] echo: stdout={r.stdout.strip()!r}, rc={r.return_code}")
    assert r.return_code == 0
    assert "hello" in r.stdout

    # ls /app
    r = env.exec_sync("ls /app")
    print(f"[env_lifecycle] ls /app: {r.stdout.strip()}")
    breakpoint()  # inspect: r.stdout, r.stderr, r.return_code

    # get_instruction
    instruction = env.get_instruction()
    print(f"[env_lifecycle] instruction (first 80 chars): {instruction[:80]}...")
    assert len(instruction) > 10

    env.stop()
    print("[env_lifecycle] PASSED")


def test_shell_tool():
    """terminal_bench_tools.py::ShellTool — _run with real container."""
    from demos.lits_benchmark.terminal_bench import TerminalBenchEnv
    from demos.lits_benchmark.terminal_bench_tools import ShellTool

    task_dir = find_task()
    env = TerminalBenchEnv(task_dir, container_name="tb-unittest-shell")
    env.start()

    tool = ShellTool(env)
    print(f"[shell_tool] name={tool.name}, description={tool.description[:40]}...")

    # Normal command
    result = tool._run(command="echo ShellTool works")
    print(f"[shell_tool] echo: {result}")
    assert "ShellTool works" in result

    # Command with stderr
    result = tool._run(command="ls /nonexistent 2>&1")
    print(f"[shell_tool] error: {result}")
    breakpoint()  # inspect: result (should contain "No such file")

    env.stop()
    print("[shell_tool] PASSED")


def test_verify_with_oracle():
    """terminal_bench.py::TerminalBenchEnv.verify — run oracle solution then verify."""
    from demos.lits_benchmark.terminal_bench import TerminalBenchEnv

    task_dir = find_task()
    env = TerminalBenchEnv(task_dir, container_name="tb-unittest-verify")
    env.start()

    # Read and execute the oracle solution
    solution_path = task_dir / "solution" / "solve.sh"
    if not solution_path.exists():
        print("[verify] No solution/solve.sh found, skipping verify test")
        env.stop()
        return

    solution = solution_path.read_text()
    print(f"[verify] Running oracle solution ({len(solution)} chars)...")
    r = env.exec_sync(f"bash -c '{solution.replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'",
                       timeout=300)
    print(f"[verify] Oracle rc={r.return_code}")
    breakpoint()  # inspect: r.stdout[-200:], r.stderr[-200:]

    # Now verify
    reward = env.verify()
    print(f"[verify] Reward: {reward}")
    breakpoint()  # inspect: reward (should be 1.0 if oracle is correct)

    env.stop()
    print("[verify] PASSED")


def test_limit_output():
    """terminal_bench_tools.py::_limit_output — truncation logic."""
    from demos.lits_benchmark.terminal_bench_tools import _limit_output

    # Short string passes through
    short = "hello world"
    assert _limit_output(short) == short

    # Long string gets truncated
    long_str = "x" * 20000
    truncated = _limit_output(long_str)
    assert len(truncated.encode("utf-8")) < 15000
    assert "output limited to 10000 bytes" in truncated
    assert "interior bytes omitted" in truncated

    # Exact boundary
    exact = "y" * 10000
    assert _limit_output(exact) == exact

    # One byte over
    over = "z" * 10001
    assert "output limited to" in _limit_output(over)

    print("[limit_output] PASSED")


if __name__ == "__main__":
    test_limit_output()
    test_dataset_loader()
    test_env_lifecycle()
    test_shell_tool()
    test_verify_with_oracle()
    print("\n=== All Terminal-Bench tests passed ===")
