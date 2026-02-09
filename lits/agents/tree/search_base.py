"""Base class for tree search algorithms.

Provides ``BaseTreeSearch`` ABC and ``SearchResult`` dataclass.
Subclasses implement ``search()`` with pure algorithm logic;
all peripheral concerns (node ID reset, root creation, checkpoint I/O,
runtime limits, terminal collection, inference logger context) are
handled by the base class.
"""

import contextlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from .node import SearchNode
from ...memory.types import TrajectoryKey
from ...lm.base import InferenceLogger
from ...log import log_metric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Base result from any tree search algorithm.

    Subclasses (``MCTSResult``, ``BFSResult``) extend this with
    algorithm-specific fields and override ``to_paths()``.
    """

    root: SearchNode
    terminal_nodes_collected: list[SearchNode] = field(default_factory=list)

    def to_paths(self) -> list[list[SearchNode]]:
        """Convert to serialisable paths.  Default: one root→leaf path per terminal node."""
        paths = []
        for node in self.terminal_nodes_collected:
            path: list[SearchNode] = []
            current = node
            while current is not None:
                path.append(current)
                current = current.parent
            path.reverse()
            paths.append(path)
        return paths


# ---------------------------------------------------------------------------
# Helper functions (module-level, private)
# ---------------------------------------------------------------------------

def _setup_checkpoint_dir(checkpoint_dir: Optional[str]) -> Optional[Path]:
    """Create ``checkpoints/`` subdirectory if *checkpoint_dir* is provided."""
    if checkpoint_dir is None:
        return None
    p = Path(checkpoint_dir)
    if p.name != "checkpoints":
        p = p / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_inference_logger(policy) -> Optional[InferenceLogger]:
    """Extract the shared InferenceLogger via policy → base_model → inference_logger.

    ``setup_inference_logging()`` attaches a single ``InferenceLogger`` instance
    to all ``LanguageModel`` objects (policy, eval, terminal).  We extract it
    from the policy because policy is always present and always wraps a
    ``LanguageModel``, whereas reward_model may be ``None`` or may not wrap
    one (e.g. ``ThinkPRM`` uses a SageMaker endpoint).

    Returns ``None`` when any hop in the accessor chain is missing.
    """
    base_model = getattr(policy, "base_model", None)
    if base_model is None:
        return None
    return getattr(base_model, "inference_logger", None)


def _optional_cuda_errors() -> list:
    """Return ``[torch.cuda.OutOfMemoryError]`` when torch is available."""
    if torch is not None:
        return [torch.cuda.OutOfMemoryError]
    return []


# ---------------------------------------------------------------------------
# BaseTreeSearch
# ---------------------------------------------------------------------------

class BaseTreeSearch(ABC):
    """Abstract base class for tree search algorithms.

    Handles peripheral boilerplate so subclasses only implement
    :meth:`search` — the pure algorithm logic.

    Class attributes:
        node_class: Node type used for the root (default ``SearchNode``).
                    Override in subclasses, e.g. ``node_class = MCTSNode``.

    Peripheral concerns handled automatically:
        1. Node ID reset
        2. TrajectoryKey + search ID creation
        3. Root node creation
        4. Checkpoint directory setup
        5. Runtime tracking + limit check
        6. Terminal node collection
        7. Error handling (OOM + runtime)
        8. InferenceLogger access + ``log_context()`` wrapping
        9. Runtime metric logging
    """

    node_class = SearchNode

    def __init__(
        self,
        config,
        world_model,
        policy,
        reward_model,
        bn_evaluator=None,
        init_state_kwargs: Optional[dict] = None,
        checkpoint_dir: Optional[str] = None,
        memory_manager=None,
    ):
        self.config = config
        self.world_model = world_model
        self.policy = policy
        self.reward_model = reward_model
        self.bn_evaluator = bn_evaluator
        self.checkpoint_dir = checkpoint_dir
        self.memory_manager = memory_manager
        self._init_kwargs: dict = init_state_kwargs or {}

        # Set during _setup(); declared here for type clarity
        self.root: Optional[SearchNode] = None
        self.inference_logger: Optional[InferenceLogger] = None
        self._checkpoint_path: Optional[Path] = None
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    def run(self, query, query_idx) -> SearchResult:
        """Template method: ``_setup`` → ``search`` → ``_teardown``.

        Not intended to be overridden by subclasses.
        """
        self._setup(query, query_idx)
        try:
            result = self.search(query, query_idx)
        except (ValueError, *_optional_cuda_errors()) as e:
            self._handle_error(e)
            result = self._fallback_result(query, query_idx)
        self._teardown()
        return result

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup(self, query, query_idx):
        """Prepare peripherals before ``search()`` is called.

        Handles: node ID reset, root creation, checkpoint dir,
        start time, inference logger extraction.
        Subclasses that override this **must** call ``super()._setup(...)``
        first.
        """
        # Peripheral 1 — node ID reset
        SearchNode.reset_id()

        # Peripheral 2 — TrajectoryKey + search ID
        search_id = f"{query_idx}_{int(time.time())}"

        # Peripheral 3 — root node creation
        self.root = self.node_class(
            state=self.world_model.init_state(**self._init_kwargs),
            action=query,
            parent=None,
            trajectory_key=TrajectoryKey(search_id=search_id, indices=()),
        )

        # Peripheral 4 — checkpoint directory
        self._checkpoint_path = _setup_checkpoint_dir(self.checkpoint_dir)

        # Peripheral 5 — runtime tracking (start)
        self._start_time = time.time()

        # Peripheral 8 — inference logger
        self.inference_logger = _get_inference_logger(self.policy)

    def _teardown(self):
        """Post-search cleanup.  Logs elapsed hours."""
        # Peripheral 9
        elapsed_hours = (time.time() - self._start_time) / 3600
        log_metric(logger, "hours_used", elapsed_hours, level="debug")

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _handle_error(self, error):
        """Handle OOM or runtime-limit errors raised during ``search()``."""
        if torch is not None and isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        logger.debug(str(error))

    def _fallback_result(self, query, query_idx) -> SearchResult:
        """Return a partial result when ``search()`` fails."""
        return SearchResult(
            root=self.root,
            terminal_nodes_collected=self.collect_terminal_nodes(),
        )

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def check_runtime_limit(self):
        """Raise ``ValueError`` if the configured runtime limit is exceeded.

        Call at the top of each iteration inside ``search()``.
        """
        limit = getattr(self.config, "runtime_limit_before_iter", None)
        if limit and time.time() - self._start_time > limit:
            raise ValueError(f"Runtime limit exceeded: {limit}")

    def collect_terminal_nodes(self) -> list[SearchNode]:
        """Walk ``self.root`` and return all terminal nodes."""
        terminals: list[SearchNode] = []

        def _walk(node: SearchNode):
            if node.is_terminal:
                terminals.append(node)
            for child in (node.children or []):
                _walk(child)

        if self.root is not None:
            _walk(self.root)
        return terminals

    def save_checkpoint(self, query_idx, iter_idx, data):
        """Serialise *data* to a checkpoint JSON file.

        No-op when checkpoint directory was not configured.
        """
        if self._checkpoint_path is None:
            return
        filepath = self._checkpoint_path / f"{query_idx}_{iter_idx}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def log_context(self, **fields):
        """Context manager attaching metadata to LLM call records.

        Delegates to ``InferenceLogger.log_context()``.
        Returns ``contextlib.nullcontext()`` when no inference logger
        is available, so callers never need a guard.
        """
        if self.inference_logger is not None:
            return self.inference_logger.log_context(**fields)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def search(self, query, query_idx) -> SearchResult:
        """Pure algorithm logic.  ``self.root`` is ready.

        Must return a ``SearchResult`` (or subclass).
        """
        ...
