# tests/test_scheduler.py

"""
Tests for Phase 3A: DAGScheduler.

Validates:
- Correct topological layering
- Cycles rejected
- Deterministic ordering within layers
- Independent branches detected
- Unknown dependency rejected
- Single-node and empty-node edge cases
"""

import pytest

from ir.mission import IR_VERSION, MissionNode, MissionPlan, ExecutionMode
from execution.scheduler import DAGScheduler, CyclicDependencyError


def _make_plan(nodes_spec: list) -> MissionPlan:
    """Helper: build a MissionPlan from [(id, skill, depends_on), ...]"""
    nodes = []
    for spec in nodes_spec:
        nid, skill = spec[0], spec[1]
        deps = spec[2] if len(spec) > 2 else []
        nodes.append(MissionNode(
            id=nid,
            skill=skill,
            depends_on=deps,
            mode=ExecutionMode.foreground,
        ))
    return MissionPlan(
        id="test_plan",
        nodes=nodes,
        metadata={"ir_version": IR_VERSION},
    )


class TestTopologicalLayering:
    """Validate correct layer grouping."""

    def test_single_node(self):
        plan = _make_plan([("n_0", "fs.create")])
        layers = DAGScheduler.plan(plan)
        assert layers == [["n_0"]]

    def test_linear_chain(self):
        """A → B → C should produce 3 layers."""
        plan = _make_plan([
            ("a", "s.one"),
            ("b", "s.two", ["a"]),
            ("c", "s.three", ["b"]),
        ])
        layers = DAGScheduler.plan(plan)
        assert layers == [["a"], ["b"], ["c"]]

    def test_diamond_dag(self):
        """
        A → B, A → C, B → D, C → D
        Layer 0: [A], Layer 1: [B, C], Layer 2: [D]
        """
        plan = _make_plan([
            ("a", "s.one"),
            ("b", "s.two", ["a"]),
            ("c", "s.three", ["a"]),
            ("d", "s.four", ["b", "c"]),
        ])
        layers = DAGScheduler.plan(plan)
        assert layers == [["a"], ["b", "c"], ["d"]]

    def test_two_independent_roots(self):
        """Two disconnected nodes → same layer."""
        plan = _make_plan([
            ("x", "s.one"),
            ("y", "s.two"),
        ])
        layers = DAGScheduler.plan(plan)
        assert layers == [["x", "y"]]

    def test_wide_dag(self):
        """
        Root → A, B, C (parallel), then D depends on A and B only.
        Layer 0: [root], Layer 1: [a, b, c], Layer 2: [d]
        """
        plan = _make_plan([
            ("root", "s.init"),
            ("a", "s.a", ["root"]),
            ("b", "s.b", ["root"]),
            ("c", "s.c", ["root"]),
            ("d", "s.d", ["a", "b"]),
        ])
        layers = DAGScheduler.plan(plan)
        assert layers[0] == ["root"]
        assert layers[1] == ["a", "b", "c"]
        assert layers[2] == ["d"]


class TestDeterminism:
    """Verify ordering within layers is deterministic."""

    def test_sorted_within_layer(self):
        plan = _make_plan([
            ("z", "s.one"),
            ("a", "s.two"),
            ("m", "s.three"),
        ])
        layers = DAGScheduler.plan(plan)
        assert layers == [["a", "m", "z"]]

    def test_repeated_calls_same_result(self):
        plan = _make_plan([
            ("c", "s.one"),
            ("b", "s.two", ["c"]),
            ("a", "s.three"),
        ])
        l1 = DAGScheduler.plan(plan)
        l2 = DAGScheduler.plan(plan)
        assert l1 == l2


class TestCycleDetection:
    """Verify cycles are rejected."""

    def test_self_cycle(self):
        plan = _make_plan([("a", "s.one", ["a"])])
        with pytest.raises(CyclicDependencyError, match="cycle"):
            DAGScheduler.plan(plan)

    def test_two_node_cycle(self):
        plan = _make_plan([
            ("a", "s.one", ["b"]),
            ("b", "s.two", ["a"]),
        ])
        with pytest.raises(CyclicDependencyError, match="cycle"):
            DAGScheduler.plan(plan)

    def test_three_node_cycle(self):
        plan = _make_plan([
            ("a", "s.one", ["c"]),
            ("b", "s.two", ["a"]),
            ("c", "s.three", ["b"]),
        ])
        with pytest.raises(CyclicDependencyError):
            DAGScheduler.plan(plan)


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_unknown_dependency_raises(self):
        plan = _make_plan([
            ("a", "s.one", ["nonexistent"]),
        ])
        with pytest.raises(ValueError, match="unknown node"):
            DAGScheduler.plan(plan)

    def test_every_node_appears_once(self):
        plan = _make_plan([
            ("a", "s.one"),
            ("b", "s.two", ["a"]),
            ("c", "s.three"),
            ("d", "s.four", ["b", "c"]),
        ])
        layers = DAGScheduler.plan(plan)
        all_nodes = [nid for layer in layers for nid in layer]
        assert sorted(all_nodes) == ["a", "b", "c", "d"]
        assert len(all_nodes) == len(set(all_nodes))


class TestIndependentBranches:
    """Validate connected component detection."""

    def test_single_component(self):
        plan = _make_plan([
            ("a", "s.one"),
            ("b", "s.two", ["a"]),
        ])
        branches = DAGScheduler.independent_branches(plan)
        assert len(branches) == 1
        assert branches[0] == {"a", "b"}

    def test_two_components(self):
        plan = _make_plan([
            ("a", "s.one"),
            ("b", "s.two", ["a"]),
            ("x", "s.three"),
            ("y", "s.four", ["x"]),
        ])
        branches = DAGScheduler.independent_branches(plan)
        assert len(branches) == 2
        assert {"a", "b"} in branches
        assert {"x", "y"} in branches

    def test_three_isolated_nodes(self):
        plan = _make_plan([
            ("a", "s.one"),
            ("b", "s.two"),
            ("c", "s.three"),
        ])
        branches = DAGScheduler.independent_branches(plan)
        assert len(branches) == 3
