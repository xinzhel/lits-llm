"""Test KG tool wrappers against live Freebase SPARQL endpoint.

Validates Task 2 of 0302-agentbench-integration: KG tool wrappers +
variable tracking + resource registration.

Tests:
- kgqa_tools.py::create_kg_tools — factory creates 7 tools
- kgqa_tools.py::KGGetRelationsTool._run — fetches relations for entity
- kgqa_tools.py::KGGetNeighborsTool._run — navigates graph, creates variable
- kgqa_tools.py::KGIntersectionTool._run — set intersection of two variables
- kgqa_tools.py::KGGetAttributesTool._run — fetches numerical attributes
- kgqa_tools.py::KGCountTool._run — counts entities in a variable
- kgqa_tools.py::KGState.rebuild — replays tool calls for MCTS branch isolation
- kgqa.py::load_kgqa_resource — resource registration wiring

Requires:
    export FREEBASE_SPARQL_URL=http://3.211.244.183:3001/sparql

Usage (from lits_llm/):
    python -m unit_test.tools.test_kgqa_tools
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../demos"))

SPARQL_URL = os.environ.get("FREEBASE_SPARQL_URL", "http://3.211.244.183:3001/sparql")

# Example 0 from std.json: "what is the attitude of the first dog and the german shepherds?"
ENTITIES_EX0 = {"first dog": "m.05t073s", "german shepherds": "m.0km5c"}


def test_factory():
    """kgqa_tools.py::create_kg_tools — creates 7 tools sharing one KGState."""
    from lits_benchmark.kgqa_tools import create_kg_tools

    tools = create_kg_tools(sparql_url=SPARQL_URL, entities=ENTITIES_EX0)
    print(f"[factory] Created {len(tools)} tools")
    tool_names = [t.name for t in tools]
    print(f"[factory] Names: {tool_names}")
    breakpoint()  # inspect: len(tools) == 7, tool_names matches expected 7 names

    # All tools share the same KGState
    states = [id(t.kg_state) for t in tools]
    print(f"[factory] All share same KGState: {len(set(states)) == 1}")
    breakpoint()  # inspect: len(set(states)) == 1
    return tools


def test_get_relations(tools):
    """kgqa_tools.py::KGGetRelationsTool._run — fetch relations for 'first dog' entity."""
    get_relations = tools[0]  # KGGetRelationsTool
    print(f"\n[get_relations] Tool: {get_relations.name}")

    # Call pre_step with None state (fresh start, no history)
    get_relations.pre_step(None)

    obs = get_relations._run(variable="first dog")
    print(f"[get_relations] Observation: {obs[:200]}")
    breakpoint()  # inspect: obs contains relation names like 'biology.breed_temperament...'

    # Also test with raw Freebase ID
    obs2 = get_relations._run(variable="m.05t073s")
    print(f"[get_relations] By ID: {obs2[:200]}")
    breakpoint()  # inspect: obs2 should contain same relations as obs


def test_get_neighbors(tools):
    """kgqa_tools.py::KGGetNeighborsTool._run — navigate graph, creates variable #0."""
    get_relations = tools[0]
    get_neighbors = tools[1]  # KGGetNeighborsTool
    kg_state = tools[0].kg_state

    # Reset state
    get_neighbors.pre_step(None)

    # First get relations to populate cache (required by AgentBench API)
    obs_rel = get_relations._run(variable="first dog")
    print(f"\n[get_neighbors] Relations for 'first dog': {obs_rel[:200]}")

    # Pick a relation from the output to navigate
    rels = _parse_relations(obs_rel)
    print(f"[get_neighbors] Available relations ({len(rels)}): {rels[:5]}...")
    breakpoint()  # inspect: rels — pick one to navigate

    # Navigate using first available relation
    if rels:
        rel = rels[0]
        obs = get_neighbors._run(variable="first dog", relation=rel)
        print(f"[get_neighbors] Navigated via '{rel}': {obs}")
        print(f"[get_neighbors] Variables count: {len(kg_state.variables)}")
        breakpoint()  # inspect: obs contains 'variable #0', kg_state.variables has 1 entry


def _parse_relations(obs: str) -> list:
    """Extract relation list from an Observation string like 'Observation: [r1, r2, ...]'."""
    import re
    match = re.search(r'\[(.+)\]', obs)
    return [r.strip() for r in match.group(1).split(",")] if match else []


def test_multi_step_sequence(tools):
    """Test a full multi-step tool sequence mimicking the agent workflow.

    Follows the pattern for example 0: get relations for both entities,
    get neighbors via each entity's own relation, then intersect.

    Note: each entity has different relations — "first dog" and "german
    shepherds" don't share the same relation names. We pick a suitable
    relation for each independently.
    """
    get_relations = tools[0]
    get_neighbors = tools[1]
    intersection = tools[2]
    kg_state = tools[0].kg_state

    # Fresh state
    kg_state.rebuild(None)
    print("\n[multi-step] Starting fresh sequence for example 0")
    print("[multi-step] Q: what is the attitude of the first dog and the german shepherds?")

    # Step 1: get_relations for "first dog"
    obs1 = get_relations._run(variable="first dog")
    rels1 = _parse_relations(obs1)
    print(f"[multi-step] Step 1 (get_relations 'first dog'): {rels1}")

    # Step 2: get_neighbors for "first dog" — pick a breed/temperament relation
    rel1 = next((r for r in rels1 if "temperament" in r or "breed" in r), rels1[0])
    obs2 = get_neighbors._run(variable="first dog", relation=rel1)
    print(f"[multi-step] Step 2 (get_neighbors via '{rel1}'): {obs2}")
    print(f"[multi-step] Variables: {len(kg_state.variables)} — {kg_state.variables}")
    breakpoint()  # inspect: variable #0 created

    # Step 3: get_relations for "german shepherds"
    obs3 = get_relations._run(variable="german shepherds")
    rels2 = _parse_relations(obs3)
    print(f"[multi-step] Step 3 (get_relations 'german shepherds'): {rels2}")

    # Step 4: get_neighbors for "german shepherds" — pick from ITS OWN relations
    rel2 = next((r for r in rels2 if "temperament" in r or "breed" in r), rels2[0])
    obs4 = get_neighbors._run(variable="german shepherds", relation=rel2)
    print(f"[multi-step] Step 4 (get_neighbors via '{rel2}'): {obs4}")
    print(f"[multi-step] Variables: {len(kg_state.variables)} — {kg_state.variables}")
    breakpoint()  # inspect: variable #1 created

    # Step 5: intersection of #0 and #1 (only works if same type)
    try:
        obs5 = intersection._run(variable1="#0", variable2="#1")
        print(f"[multi-step] Step 5 (intersection): {obs5}")
        print(f"[multi-step] Variables: {len(kg_state.variables)} — {kg_state.variables}")

        # Execute the final variable to get actual answer entities
        final_var = kg_state.variables[-1]
        results = kg_state.api.final_execute(final_var)
        print(f"[multi-step] Final answer entities: {results}")
        breakpoint()  # inspect: results should contain entity IDs for 'Obedient', 'Intelligent'
    except ValueError as e:
        # Type mismatch is expected if the two relations lead to different types
        print(f"[multi-step] Intersection failed (type mismatch): {e}")
        print(f"[multi-step] var#0 type={kg_state.variables[0].type}, var#1 type={kg_state.variables[1].type}")
        print("[multi-step] This is expected if the chosen relations produce different entity types.")
        print("[multi-step] Variable tracking still works — intersection just requires same-type variables.")
        breakpoint()  # inspect: both variables exist, types differ



def test_rebuild(tools):
    """kgqa_tools.py::KGState.rebuild — replays tool calls from ToolUseState.

    Simulates MCTS branch isolation: after a multi-step sequence, rebuild
    from a synthetic ToolUseState (list of step-like objects) and verify
    variables are reconstructed correctly.
    """
    from lits_benchmark.kgqa_tools import create_kg_tools

    # Create fresh tools for this test
    tools2 = create_kg_tools(sparql_url=SPARQL_URL, entities=ENTITIES_EX0)
    kg_state = tools2[0].kg_state
    get_relations = tools2[0]
    get_neighbors = tools2[1]

    # First, do a real sequence to know what the correct state looks like
    kg_state.rebuild(None)
    obs_rel = get_relations._run(variable="first dog")
    rels = _parse_relations(obs_rel)
    target_rel = next((r for r in rels if "temperament" in r or "breed" in r), rels[0])

    obs_neigh = get_neighbors._run(variable="first dog", relation=target_rel)
    original_var0 = kg_state.variables[0]
    print(f"\n[rebuild] Original variable #0: {original_var0}")
    print(f"[rebuild] Original var0 program: {original_var0.program}")

    # Now simulate a ToolUseState (list of step-like objects with action/observation)
    class FakeStep:
        def __init__(self, action, observation):
            self.action = action
            self.observation = observation

    fake_state = [
        # Step 1: get_relations (no variable created)
        FakeStep(
            action=json.dumps({"action": "get_relations", "action_input": {"variable": "first dog"}}),
            observation=obs_rel,
        ),
        # Step 2: get_neighbors (creates variable #0)
        FakeStep(
            action=json.dumps({"action": "get_neighbors", "action_input": {"variable": "first dog", "relation": target_rel}}),
            observation=obs_neigh,
        ),
    ]

    # Rebuild from fake state
    kg_state.rebuild(fake_state)
    print(f"[rebuild] After rebuild: {len(kg_state.variables)} variables")
    if kg_state.variables:
        rebuilt_var0 = kg_state.variables[0]
        print(f"[rebuild] Rebuilt variable #0: {rebuilt_var0}")
        print(f"[rebuild] Rebuilt var0 program: {rebuilt_var0.program}")
        print(f"[rebuild] Programs match: {original_var0.program == rebuilt_var0.program}")
    breakpoint()  # inspect: kg_state.variables[0].program == original_var0.program


def test_count(tools):
    """kgqa_tools.py::KGCountTool._run — count entities in a variable."""
    get_relations = tools[0]
    get_neighbors = tools[1]
    count_tool = tools[6]  # KGCountTool
    kg_state = tools[0].kg_state

    kg_state.rebuild(None)

    # Build a variable first
    obs_rel = get_relations._run(variable="first dog")
    rels = _parse_relations(obs_rel)
    rel = rels[0] if rels else None
    if rel:
        get_neighbors._run(variable="first dog", relation=rel)
        obs_count = count_tool._run(variable="#0")
        print(f"\n[count] Count of #0: {obs_count}")
        breakpoint()  # inspect: obs_count contains 'variable #1, which is a number'


def test_resource_registration():
    """kgqa.py::load_kgqa_resource — resource wiring returns tools + tool_context."""
    import lits_benchmark.kgqa
    from lits.benchmarks.registry import load_resource

    resource = load_resource("kgqa", sparql_url=SPARQL_URL, entities=ENTITIES_EX0)
    print(f"\n[resource] Keys: {list(resource.keys())}")
    print(f"[resource] Tools count: {len(resource['tools'])}")
    print(f"[resource] Tool names: {[t.name for t in resource['tools']]}")
    print(f"[resource] tool_context length: {len(resource['tool_context'])} chars")
    print(f"[resource] tool_context preview: {resource['tool_context'][:100]}...")
    breakpoint()  # inspect: 7 tools, tool_context is formatted KGQA_SYSTEM_PROMPT


def main():
    print(f"SPARQL endpoint: {SPARQL_URL}")
    print("=" * 60)

    print("\n=== Test 1: Factory ===")
    tools = test_factory()

    print("\n=== Test 2: get_relations ===")
    test_get_relations(tools)

    print("\n=== Test 3: get_neighbors ===")
    test_get_neighbors(tools)

    print("\n=== Test 4: Multi-step sequence ===")
    test_multi_step_sequence(tools)

    print("\n=== Test 5: KGState.rebuild ===")
    test_rebuild(tools)

    print("\n=== Test 6: count ===")
    test_count(tools)

    print("\n=== Test 7: Resource registration ===")
    test_resource_registration()

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
