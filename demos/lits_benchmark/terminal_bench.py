"""Terminal-Bench 2.0 integration for LiTS framework.

Provides ``TerminalBenchEnv`` for Docker container lifecycle management,
dataset loader, evaluator, and resource registration.

Prerequisites:
    - Docker running locally
    - Harbor installed (``uv tool install harbor --python 3.12``)
    - Tasks cached via ``harbor run --dataset terminal-bench@2.0 ...`` (first run downloads)

Usage::

    from lits_benchmark.terminal_bench import TerminalBenchEnv

    env = TerminalBenchEnv(task_dir="/path/to/task")
    env.start()
    result = env.exec_sync("ls /app")
    print(result.stdout)
    reward = env.verify()
    env.stop()

See design.md § LiTS Integration Architecture for architecture overview.
"""

import logging
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    """Result of a command execution inside a Docker container."""
    stdout: str
    stderr: str
    return_code: int


@dataclass
class TaskConfig:
    """Parsed task.toml configuration.

    Fields:
        docker_image: Prebuilt Docker image name (e.g. ``alexgshaw/gpt2-codegolf:20251031``)
        cpus: CPU limit for the container
        memory: Memory limit (e.g. ``"4G"``)
        agent_timeout_sec: Max seconds for agent execution
        verifier_timeout_sec: Max seconds for verification
        difficulty: Task difficulty (``easy``, ``medium``, ``hard``)
        category: Task category (``software-engineering``, ``security``, etc.)
    """
    docker_image: Optional[str] = None
    cpus: int = 1
    memory: str = "2G"
    agent_timeout_sec: float = 600.0
    verifier_timeout_sec: float = 600.0
    difficulty: str = "unknown"
    category: str = "unknown"


def parse_task_config(task_dir: Path) -> TaskConfig:
    """Parse task.toml from a Terminal-Bench task directory.

    Args:
        task_dir: Path to the task directory containing task.toml.

    Returns:
        Parsed TaskConfig with Docker image, resource limits, and metadata.
    """
    toml_path = task_dir / "task.toml"
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    env_cfg = data.get("environment", {})
    metadata = data.get("metadata", {})
    agent_cfg = data.get("agent", {})
    verifier_cfg = data.get("verifier", {})

    return TaskConfig(
        docker_image=env_cfg.get("docker_image"),
        cpus=env_cfg.get("cpus", 1),
        memory=env_cfg.get("memory", "2G"),
        agent_timeout_sec=agent_cfg.get("timeout_sec", 600.0),
        verifier_timeout_sec=verifier_cfg.get("timeout_sec", 600.0),
        difficulty=metadata.get("difficulty", "unknown"),
        category=metadata.get("category", "unknown"),
    )


class TerminalBenchEnv:
    """Manages a Docker container for one Terminal-Bench task.

    Lifecycle: ``start()`` → ``exec_sync()`` (repeated) → ``verify()`` → ``stop()``.
    For MCTS, call ``reset()`` to restart the container from a clean state.

    Args:
        task_dir: Path to the cached task directory (contains task.toml, instruction.md, etc.)
        container_name: Optional custom container name. Auto-generated if not provided.

    Example::

        env = TerminalBenchEnv(Path("~/.cache/harbor/tasks/.../gpt2-codegolf"))
        env.start()
        r = env.exec_sync("ls /app")
        print(r.stdout)
        env.stop()
    """

    def __init__(self, task_dir: Path, container_name: Optional[str] = None):
        self.task_dir = Path(task_dir)
        self.config = parse_task_config(self.task_dir)
        self.task_id = self.task_dir.name
        self.container_name = container_name or f"tb-{self.task_id}-{uuid.uuid4().hex[:8]}"
        self._running = False

    def start(self) -> None:
        """Pull image (if needed) and start the container.

        The container runs ``sleep infinity`` to stay alive, waiting for
        ``exec_sync()`` calls to execute commands inside it.
        """
        if self._running:
            logger.warning("Container %s already running, skipping start", self.container_name)
            return

        image = self.config.docker_image
        if not image:
            # Fallback: build from Dockerfile (not expected for TB 2.0, all tasks use prebuilt)
            dockerfile_dir = self.task_dir / "environment"
            if not (dockerfile_dir / "Dockerfile").exists():
                raise FileNotFoundError(
                    f"No docker_image in task.toml and no Dockerfile at {dockerfile_dir}"
                )
            image = f"tb-local-{self.task_id}"
            logger.info("Building image %s from %s", image, dockerfile_dir)
            self._run_cmd(["docker", "build", "-t", image, str(dockerfile_dir)])

        logger.info("Starting container %s with image %s", self.container_name, image)
        self._run_cmd([
            "docker", "run", "-d",
            "--name", self.container_name,
            "--cpus", str(self.config.cpus),
            "--memory", self.config.memory,
            image,
            "sh", "-c", "sleep infinity",
        ])
        self._running = True

    def exec_sync(self, command: str, timeout: Optional[int] = None) -> ExecResult:
        """Execute a bash command inside the container.

        Args:
            command: Bash command string to execute.
            timeout: Optional timeout in seconds. Defaults to agent_timeout_sec from task.toml.

        Returns:
            ExecResult with stdout, stderr, and return_code.
        """
        if not self._running:
            raise RuntimeError(f"Container {self.container_name} is not running. Call start() first.")

        timeout = timeout or int(self.config.agent_timeout_sec)
        try:
            result = subprocess.run(
                ["docker", "exec", self.container_name, "bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecResult(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                return_code=-1,
            )

    def verify(self) -> float:
        """Run the task's test script and return the reward (0.0 or 1.0).

        Steps:
            1. Copy tests/ directory into the container at /tests/
            2. Create /logs/verifier/ directory
            3. Run test.sh
            4. Read reward.txt

        Returns:
            Reward value (0.0 = fail, 1.0 = pass).
        """
        tests_dir = self.task_dir / "tests"
        if not tests_dir.exists():
            raise FileNotFoundError(f"No tests directory at {tests_dir}")

        # Copy tests into container
        self._run_cmd([
            "docker", "cp",
            f"{tests_dir}/.", f"{self.container_name}:/tests/",
        ])

        # Create logs directory for reward.txt
        self.exec_sync("mkdir -p /logs/verifier")

        # Make test.sh executable and run it
        self.exec_sync("chmod +x /tests/test.sh")
        result = self.exec_sync(
            "bash /tests/test.sh",
            timeout=int(self.config.verifier_timeout_sec),
        )
        logger.debug("Verifier stdout: %s", result.stdout[-500:] if result.stdout else "")
        logger.debug("Verifier stderr: %s", result.stderr[-500:] if result.stderr else "")

        # Read reward
        reward_result = self.exec_sync("cat /logs/verifier/reward.txt")
        try:
            return float(reward_result.stdout.strip())
        except (ValueError, AttributeError):
            logger.warning(
                "Failed to parse reward.txt for %s: stdout=%r, stderr=%r",
                self.task_id, reward_result.stdout, reward_result.stderr,
            )
            return 0.0

    def reset(self) -> None:
        """Reset the container to a clean state (for MCTS replay).

        Stops and removes the current container, then starts a fresh one
        from the same image.
        """
        self.stop()
        self.start()

    def stop(self) -> None:
        """Stop and remove the container."""
        if not self._running:
            return
        logger.info("Stopping container %s", self.container_name)
        self._run_cmd(["docker", "stop", self.container_name], check=False)
        self._run_cmd(["docker", "rm", self.container_name], check=False)
        self._running = False

    def get_instruction(self) -> str:
        """Read the task instruction from instruction.md."""
        instruction_path = self.task_dir / "instruction.md"
        return instruction_path.read_text().strip()

    @staticmethod
    def _run_cmd(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
        """Run a subprocess command, logging on failure."""
        logger.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if check and result.returncode != 0:
            logger.error("Command failed: %s\nstdout: %s\nstderr: %s",
                         " ".join(cmd), result.stdout, result.stderr)
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
        return result


# ---------------------------------------------------------------------------
# Dataset loader, evaluator, and resource registration
# ---------------------------------------------------------------------------

from lits.benchmarks.registry import register_dataset, register_evaluator, register_resource


TERMINAL_BENCH_CACHE_DIR = Path.home() / ".cache" / "harbor" / "tasks"

TERMINAL_BENCH_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with solving a command-line task in a Linux "
    "Docker container. You have access to a shell tool that executes bash commands "
    "and returns stdout/stderr.\n\n"
    "Task:\n{instruction}\n\n"
    "Solve the task by executing shell commands. When you are done, provide your "
    "final answer."
)


@register_dataset("terminal_bench", task_type="tool_use")
def load_terminal_bench(
    cache_dir: Optional[str] = None,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    **kwargs,
) -> List[Dict]:
    """Load Terminal-Bench 2.0 tasks from Harbor's local cache.

    Tasks are cached at ``~/.cache/harbor/tasks/`` after running
    ``harbor run --dataset terminal-bench@2.0`` at least once.

    Args:
        cache_dir: Override cache directory. Default: ``~/.cache/harbor/tasks/``.
        category: Filter by category (e.g. ``"software-engineering"``).
        difficulty: Filter by difficulty (``"easy"``, ``"medium"``, ``"hard"``).

    Returns:
        List of example dicts, each containing:
        ``question`` (instruction text), ``answer`` (empty — eval is via verifier),
        ``task_id``, ``task_dir`` (Path), ``metadata`` (parsed task.toml fields).
    """
    cache = Path(cache_dir) if cache_dir else TERMINAL_BENCH_CACHE_DIR
    if not cache.exists():
        raise FileNotFoundError(
            f"Harbor cache not found at {cache}. "
            "Run 'harbor run --dataset terminal-bench@2.0 --agent oracle --n-tasks 1' "
            "to populate the cache."
        )

    examples = []
    for hash_dir in sorted(cache.iterdir()):
        if not hash_dir.is_dir():
            continue
        for task_dir in hash_dir.iterdir():
            if not task_dir.is_dir():
                continue
            toml_path = task_dir / "task.toml"
            instruction_path = task_dir / "instruction.md"
            if not toml_path.exists() or not instruction_path.exists():
                continue

            config = parse_task_config(task_dir)

            # Apply filters
            if category and config.category != category:
                continue
            if difficulty and config.difficulty != difficulty:
                continue

            examples.append({
                "question": instruction_path.read_text().strip(),
                "answer": None,  # eval is via verifier, not string matching
                "task_id": task_dir.name,
                "task_dir": task_dir,
                "metadata": {
                    "difficulty": config.difficulty,
                    "category": config.category,
                    "docker_image": config.docker_image,
                    "agent_timeout_sec": config.agent_timeout_sec,
                    "cpus": config.cpus,
                    "memory": config.memory,
                },
            })

    logger.info(
        "Loaded %d Terminal-Bench tasks%s%s",
        len(examples),
        f" (category={category})" if category else "",
        f" (difficulty={difficulty})" if difficulty else "",
    )
    return examples


@register_evaluator("terminal_bench")
def evaluate_terminal_bench(predicted_answer, ground_truth, **kwargs) -> float:
    """Evaluate a Terminal-Bench task by running the verifier.

    Unlike other benchmarks where evaluation compares strings, Terminal-Bench
    evaluation runs test.sh inside the Docker container and reads reward.txt.

    This evaluator expects ``env`` (a started TerminalBenchEnv) in kwargs.
    If not provided, returns 0.0 with a warning.

    Args:
        predicted_answer: Not used (agent's answer is the container state).
        ground_truth: Not used (ground truth is the test script).
        **kwargs: Must include ``env`` (TerminalBenchEnv instance).

    Returns:
        0.0 (fail) or 1.0 (pass).
    """
    env = kwargs.get("env")
    if env is None:
        logger.warning(
            "evaluate_terminal_bench called without 'env' kwarg. "
            "Cannot run verifier. Returning 0.0."
        )
        return 0.0
    return env.verify()


@register_resource("terminal_bench")
def load_terminal_bench_resource(**kwargs) -> dict:
    """Load Terminal-Bench tool-use resource: TerminalBenchEnv + ShellTool.

    This creates a TerminalBenchEnv and ShellTool for a single task.
    The env is NOT started here — caller must call ``env.start()`` before
    running the agent, and ``env.stop()`` after.

    Kwargs:
        task_dir: Path to the task directory (required).

    Returns:
        Dict with:
        - ``"tools"``: list containing one ShellTool
        - ``"tool_context"``: formatted system prompt with task instruction
        - ``"env"``: TerminalBenchEnv instance (not started)
        - ``"prepare_tool_state"``: callback ``(example) -> None`` that
          resets the env for a new example
    """
    from .terminal_bench_tools import ShellTool

    task_dir = kwargs.get("task_dir")
    if task_dir is None:
        raise ValueError("load_terminal_bench_resource requires 'task_dir' kwarg")
    task_dir = Path(task_dir)

    env = TerminalBenchEnv(task_dir)
    tool = ShellTool(env)
    instruction = env.get_instruction()

    def prepare_tool_state(example: dict) -> None:
        """Reset env for a new example (stop old container, start fresh)."""
        new_dir = example.get("task_dir")
        if new_dir and Path(new_dir) != env.task_dir:
            env.stop()
            env.__init__(Path(new_dir))
            env.start()

    return {
        "tools": [tool],
        "tool_context": TERMINAL_BENCH_SYSTEM_PROMPT.format(instruction=instruction),
        "env": env,
        "prepare_tool_state": prepare_tool_state,
    }
