"""KGQA tool wrappers for LiTS framework.

Implements 7 KG navigation tools as LiTS BaseTool subclasses for the
AgentBench Freebase KGQA benchmark. All tools share a ``KGState`` that
tracks variables (#0, #1, ...) and connects to a Freebase SPARQL endpoint.

These tools are Freebase-specific: they rely on the ``http://rdf.freebase.com/ns/``
namespace (hardcoded in AgentBench's ``SparqlExecuter``) and the AgentBench
``API`` class for symbolic program building. For other knowledge graphs
(e.g., AURIN PROV ontology), see ``kgqa_tools_for_aurin.md``.

All KG-specific logic lives here — no changes to core LiTS framework.

Usage::

    from lits_benchmark.kgqa_tools import create_kg_tools

    tools = create_kg_tools(sparql_url="http://...", entities={"first dog": "m.05t073s"})
"""

import json
import logging
import re
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from lits.tools.base import BaseTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import AgentBench KG utilities (in-workspace, not installed)
# ---------------------------------------------------------------------------

_AGENTBENCH_KG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..",  # workspace root
    "AgentBench", "src", "server", "tasks", "knowledgegraph"
)

# AgentBench's api.py uses relative imports (from .utils...), so we can't
# just add the dir to sys.path and import as top-level modules. Instead,
# register a fake package hierarchy so relative imports resolve correctly
# without triggering the real __init__.py (which pulls in agentrl).
import importlib.util
import types

def _setup_agentbench_kg_imports():
    """Register AgentBench KG modules as a fake package for relative imports.

    Why not just ``sys.path.insert(0, knowledgegraph_dir)``?

    ``api.py`` uses relative imports (``from .utils.logic_form_util import ...``).
    Relative imports require the module to belong to a package (``__package__``
    must be set).  Adding the directory to ``sys.path`` and doing
    ``from api import ...`` loads ``api.py`` as a top-level module with
    ``__package__ = None``, so the ``.`` in ``from .utils...`` has no parent
    to resolve against → ``ImportError: attempted relative import with no
    known parent package``.

    Importing as ``from knowledgegraph.api import ...`` (with the *parent*
    on ``sys.path``) would fix relative imports, but triggers
    ``knowledgegraph/__init__.py`` which pulls in ``agentrl`` and other
    uninstalled AgentBench dependencies.

    Solution: register a hollow ``knowledgegraph`` package in ``sys.modules``
    (with ``__path__`` but no ``__init__.py`` execution), then load only the
    three modules we need via ``importlib.util``.  Now ``api.py``'s
    ``from .utils...`` resolves correctly because Python sees
    ``__package__ = 'knowledgegraph'``.
    """
    if "knowledgegraph" in sys.modules:
        return  # already set up
    kg_dir = os.path.normpath(_AGENTBENCH_KG_DIR)
    utils_dir = os.path.join(kg_dir, "utils")

    # Fake package: knowledgegraph
    pkg = types.ModuleType("knowledgegraph")
    pkg.__path__ = [kg_dir]
    pkg.__package__ = "knowledgegraph"
    sys.modules["knowledgegraph"] = pkg

    # Fake subpackage: knowledgegraph.utils
    utils_pkg = types.ModuleType("knowledgegraph.utils")
    utils_pkg.__path__ = [utils_dir]
    utils_pkg.__package__ = "knowledgegraph.utils"
    sys.modules["knowledgegraph.utils"] = utils_pkg

    # Load individual modules
    for mod_name, filepath in [
        ("knowledgegraph.utils.sparql_executer", os.path.join(utils_dir, "sparql_executer.py")),
        ("knowledgegraph.utils.logic_form_util", os.path.join(utils_dir, "logic_form_util.py")),
        ("knowledgegraph.api", os.path.join(kg_dir, "api.py")),
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

_setup_agentbench_kg_imports()

from knowledgegraph.utils.sparql_executer import SparqlExecuter
from knowledgegraph.utils.logic_form_util import postprocess_raw_code, lisp_to_sparql, range_info
from knowledgegraph.api import Variable, API

# Load ontology vocab for relation/attribute filtering
_ONTOLOGY_DIR = os.path.join(_AGENTBENCH_KG_DIR, "ontology")
with open(os.path.join(_ONTOLOGY_DIR, "vocab.json")) as f:
    _vocab = json.load(f)
    _relations = set(_vocab["relations"])
    _attributes = set(_vocab["attributes"])


# ---------------------------------------------------------------------------
# KGState: shared mutable state across all 7 tools
# ---------------------------------------------------------------------------

@dataclass
class KGState:
    """Shared state for KG tools: variable tracking + SPARQL client.

    Not a LiTS abstraction — just a plain container that all 7 tools
    reference. Rebuilt from ToolUseState history via ``rebuild()``
    before each tool execution (MCTS branch isolation).
    """
    sparql: SparqlExecuter
    entities: Dict[str, str]  # entity name → Freebase ID
    api: API = field(init=False)
    variables: List[Variable] = field(default_factory=list)

    def __post_init__(self):
        self.api = API(self.sparql)

    def resolve_arg(self, value: str) -> Union[Variable, str]:
        """Three-way argument resolution (see analysis_kg.md QA).

        kgqa_tools.py::KGState.resolve_arg
        """
        if isinstance(value, str) and value.startswith("#"):
            idx = int(value[1:])
            return self.variables[idx]
        elif value in self.entities:
            return self.entities[value]
        else:
            return value

    def register_variable(self, var: Variable) -> int:
        """Append a new variable and return its index."""
        idx = len(self.variables)
        self.variables.append(var)
        return idx

    def rebuild(self, state) -> None:
        """Reconstruct variables_list by replaying tool calls from ToolUseState.

        Called via ``pre_step(state)`` before each tool execution to ensure
        correct variable state per MCTS branch.

        kgqa_tools.py::KGState.rebuild
        """
        self.variables.clear()
        if state is None:
            return
        for step in state:
            action_str = getattr(step, 'action', None)
            obs = getattr(step, 'observation', None)
            if action_str is None or obs is None:
                continue
            try:
                parsed = json.loads(str(action_str))
                tool_name = parsed.get("action", "")
                args = parsed.get("action_input", {})
            except (json.JSONDecodeError, TypeError):
                continue

            # Only tools that create variables matter for replay
            if tool_name in ("get_neighbors", "intersection", "count", "argmax", "argmin"):
                # Check if observation contains "variable #N"
                m = re.search(r'variable #(\d+)', obs)
                if m:
                    # Re-execute symbolically to get the Variable object
                    try:
                        resolved_args = []
                        if tool_name == "get_neighbors":
                            var = self.resolve_arg(args["variable"])
                            rel = args["relation"]
                            result, _ = self.api.get_neighbors(var, rel)
                        elif tool_name == "intersection":
                            v1 = self.resolve_arg(args["variable1"])
                            v2 = self.resolve_arg(args["variable2"])
                            result, _ = self.api.intersection(v1, v2)
                        elif tool_name == "count":
                            var = self.resolve_arg(args["variable"])
                            result, _ = self.api.count(var)
                        elif tool_name == "argmax":
                            var = self.resolve_arg(args["variable"])
                            attr = args["attribute"]
                            result, _ = self.api.argmax(var, attr)
                        elif tool_name == "argmin":
                            var = self.resolve_arg(args["variable"])
                            attr = args["attribute"]
                            result, _ = self.api.argmin(var, attr)
                        else:
                            continue
                        if result is not None:
                            self.variables.append(result)
                    except Exception as e:
                        logger.warning(f"KGState.rebuild: failed to replay {tool_name}: {e}")


# ---------------------------------------------------------------------------
# Pydantic input schemas for each tool
# Tool names, descriptions, and parameter descriptions are verbatim from
# AgentBench/src/server/tasks/knowledgegraph/const.py TOOLS list.
# ---------------------------------------------------------------------------

class GetRelationsInput(BaseModel):
    variable: str = Field(description="The entity or variable whose relations are to be fetched. Should be an existing entity or a result of a previous query.")

class GetNeighborsInput(BaseModel):
    variable: str = Field(description="The subject entity or variable to find neighbors for.")
    relation: str = Field(description="The relationship through which neighbors are connected.")

class IntersectionInput(BaseModel):
    variable1: str = Field(description="The first set of entities or a variable.")
    variable2: str = Field(description="The second set of entities or a variable.")

class GetAttributesInput(BaseModel):
    variable: str = Field(description="The variable whose numerical attributes are to be fetched.")

class ArgmaxInput(BaseModel):
    variable: str = Field(description="The variable or set of entities among which to find the maximum.")
    attribute: str = Field(description="The attribute to maximize over the entities.")

class ArgminInput(BaseModel):
    variable: str = Field(description="The variable or set of entities among which to find the minimum.")
    attribute: str = Field(description="The attribute to minimize over the entities.")

class CountInput(BaseModel):
    variable: str = Field(description="The variable or set of entities to count.")


# ---------------------------------------------------------------------------
# 7 Tool wrappers
# ---------------------------------------------------------------------------

class _KGToolBase(BaseTool):
    """Base for all KG tools — shares KGState and implements pre_step."""

    # SPARQL endpoint is reached over an SSH tunnel that may drop and reconnect.
    # Ride out a reconnect by re-attempting the same query before surfacing a
    # server-down error to the circuit breaker. The schedule sums to ~120s per
    # call so it can absorb slow reconnects (e.g. switching between wifi and a
    # cellular hotspot while roaming), not just short blips.
    # See ``lits/tools/utils.py::execute_tool_action`` for the retry mechanics.
    server_down_retry_delays: tuple[int, ...] = (5, 15, 40, 60)

    def __init__(self, kg_state: KGState):
        # Skip BaseTool.__init__ which expects a client arg
        object.__setattr__(self, "kg_state", kg_state)

    def pre_step(self, state) -> None:
        """Rebuild variable tracker from current ToolUseState (MCTS branch isolation)."""
        self.kg_state.rebuild(state)

    def _resolve(self, value: str):
        return self.kg_state.resolve_arg(value)

    def _register_and_format(self, result, obs: str) -> str:
        """If result is a new Variable, register it and replace ## placeholder."""
        if result is not None and "##" in obs:
            idx = self.kg_state.register_variable(result)
            obs = obs.replace("##", f"#{idx}")
        return obs


class KGGetRelationsTool(_KGToolBase):
    name: str = "get_relations"
    description: str = "Fetches all relations associated with a given entity or variable, which helps in navigating the Knowledge Base (KB) to find useful connections."
    args_schema = GetRelationsInput

    def _run(self, variable: str) -> str:
        resolved = self._resolve(variable)
        _, obs = self.kg_state.api.get_relations(resolved)
        return obs


class KGGetNeighborsTool(_KGToolBase):
    name: str = "get_neighbors"
    description: str = "Returns all entities connected to a given variable through a specified relation, can only be used after identifying valid relations with get_relations()."
    args_schema = GetNeighborsInput

    def _run(self, variable: str, relation: str) -> str:
        resolved = self._resolve(variable)
        result, obs = self.kg_state.api.get_neighbors(resolved, relation)
        return self._register_and_format(result, obs)


class KGIntersectionTool(_KGToolBase):
    name: str = "intersection"
    description: str = "Computes the intersection of two sets of entities or variables of the same type."
    args_schema = IntersectionInput

    def _run(self, variable1: str, variable2: str) -> str:
        v1 = self._resolve(variable1)
        v2 = self._resolve(variable2)
        result, obs = self.kg_state.api.intersection(v1, v2)
        return self._register_and_format(result, obs)


class KGGetAttributesTool(_KGToolBase):
    name: str = "get_attributes"
    description: str = "Retrieves all numerical attributes of a given variable, especially valuable when seeking for entities with extreme attributes (max or min)."
    args_schema = GetAttributesInput

    def _run(self, variable: str) -> str:
        resolved = self._resolve(variable)
        _, obs = self.kg_state.api.get_attributes(resolved)
        return obs


class KGArgmaxTool(_KGToolBase):
    name: str = "argmax"
    description: str = "Finds the entity with the maximum value of the specified attribute, can be used after identifying valid attributes with get_attributes()."
    args_schema = ArgmaxInput

    def _run(self, variable: str, attribute: str) -> str:
        resolved = self._resolve(variable)
        result, obs = self.kg_state.api.argmax(resolved, attribute)
        return self._register_and_format(result, obs)


class KGArgminTool(_KGToolBase):
    name: str = "argmin"
    description: str = "Finds the entity with the minimum value of the specified attribute, can be used after identifying valid attributes with get_attributes()."
    args_schema = ArgminInput

    def _run(self, variable: str, attribute: str) -> str:
        resolved = self._resolve(variable)
        result, obs = self.kg_state.api.argmin(resolved, attribute)
        return self._register_and_format(result, obs)


class KGCountTool(_KGToolBase):
    name: str = "count"
    description: str = "Counts the number of entities within a given variable."
    args_schema = CountInput

    def _run(self, variable: str) -> str:
        resolved = self._resolve(variable)
        result, obs = self.kg_state.api.count(resolved)
        return self._register_and_format(result, obs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_kg_tools(
    sparql_url: str = None,
    entities: Optional[Dict[str, str]] = None,
) -> list:
    """Create all 7 KG tools sharing a single KGState.

    Args:
        sparql_url: Freebase SPARQL endpoint URL. Supports any SPARQL 1.1
            endpoint (Virtuoso, Neptune, Jena Fuseki, etc.) as long as the
            Freebase data uses the standard ``http://rdf.freebase.com/ns/``
            namespace. If the data was loaded with a different namespace,
            queries will return empty results.

            Reads from ``FREEBASE_SPARQL_URL`` env var if not provided.
            Set this after launching the Virtuoso EC2 instance
            (see ``chore/aws/scripts/freebase/``).

            Note: ``SparqlExecuter.execute_query()`` strips the namespace
            prefix from results (``value.replace('http://rdf.freebase.com/ns/', '')``),
            so the tools and variable tracker work with short IDs like
            ``m.05t073s`` instead of full URIs. This stripping is hardcoded
            in AgentBench's ``sparql_executer.py`` and assumes the standard
            Freebase namespace.
        entities: Dict mapping entity names to Freebase IDs.

    Returns:
        List of 7 BaseTool instances.
    """
    import os
    if sparql_url is None:
        sparql_url = os.environ.get("FREEBASE_SPARQL_URL")
        if not sparql_url:
            raise ValueError(
                "sparql_url not provided and FREEBASE_SPARQL_URL env var not set. "
                "Launch Virtuoso via chore/aws/scripts/freebase/ and set the env var."
            )
    sparql = SparqlExecuter(sparql_url)
    kg_state = KGState(sparql=sparql, entities=entities or {})
    return [
        KGGetRelationsTool(kg_state),
        KGGetNeighborsTool(kg_state),
        KGIntersectionTool(kg_state),
        KGGetAttributesTool(kg_state),
        KGArgmaxTool(kg_state),
        KGArgminTool(kg_state),
        KGCountTool(kg_state),
    ]
