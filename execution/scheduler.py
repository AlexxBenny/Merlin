# execution/scheduler.py

"""
DAGScheduler — topological planning for MissionPlan execution.

Responsibilities:
- Topological sort of the DAG
- Detect independent branches (parallelizable)
- Group nodes into dependency layers
- Expose execution batches

This is PURE PLANNING. No threads, no execution, no world interaction.
Given a DAG, it produces a deterministic list of layers.

Invariants:
- Cycles rejected immediately (raise)
- Deterministic ordering within each layer (sorted by node id)
- Every node appears in exactly one layer
"""

from typing import Dict, List, Set

from ir.mission import MissionPlan, MissionNode


class CyclicDependencyError(Exception):
    """Raised when the DAG contains a cycle."""
    pass


class DAGScheduler:
    """
    Pure topological planner.

    Usage:
        scheduler = DAGScheduler()
        layers = scheduler.plan(mission_plan)
        # layers[0] = nodes with no dependencies
        # layers[1] = nodes whose deps are all in layers[0]
        # ...
    """

    @staticmethod
    def plan(mission: MissionPlan) -> List[List[str]]:
        """
        Produce an ordered list of execution layers.

        Each layer is a list of node IDs that are safe to execute
        concurrently (all their dependencies are in earlier layers).

        Within each layer, nodes are sorted by ID for determinism.

        Args:
            mission: The MissionPlan to schedule.

        Returns:
            List of layers, each layer a list of node IDs.

        Raises:
            CyclicDependencyError: if the DAG contains cycles.
        """
        # Build adjacency structures
        node_ids: Set[str] = set()
        in_degree: Dict[str, int] = {}
        dependents: Dict[str, List[str]] = {}  # node -> list of nodes that depend on it

        for node in mission.nodes:
            node_ids.add(node.id)
            in_degree.setdefault(node.id, 0)
            dependents.setdefault(node.id, [])

        for node in mission.nodes:
            for dep in node.depends_on:
                if dep not in node_ids:
                    raise ValueError(
                        f"Node '{node.id}' depends on unknown node '{dep}'"
                    )
                in_degree[node.id] = in_degree.get(node.id, 0) + 1
                dependents[dep].append(node.id)

        # Kahn's algorithm — layer-aware
        layers: List[List[str]] = []
        remaining = set(node_ids)

        # First layer: all nodes with in_degree == 0
        current_layer = sorted(
            nid for nid in remaining if in_degree[nid] == 0
        )

        while current_layer:
            layers.append(current_layer)
            remaining -= set(current_layer)

            next_layer_candidates: Set[str] = set()
            for nid in current_layer:
                for dependent in dependents[nid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_layer_candidates.add(dependent)

            current_layer = sorted(next_layer_candidates)

        # Cycle detection
        if remaining:
            raise CyclicDependencyError(
                f"DAG contains a cycle involving nodes: "
                f"{sorted(remaining)}"
            )

        return layers

    @staticmethod
    def get_node_index(mission: MissionPlan) -> Dict[str, MissionNode]:
        """Build a node ID -> MissionNode lookup dict."""
        return {node.id: node for node in mission.nodes}

    @staticmethod
    def independent_branches(mission: MissionPlan) -> List[Set[str]]:
        """
        Detect fully independent sub-DAGs (connected components).

        Returns a list of sets, where each set contains node IDs
        that form a connected component — useful for knowing which
        branches are truly independent.
        """
        node_ids = {node.id for node in mission.nodes}

        # Build undirected adjacency
        adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}
        for node in mission.nodes:
            for dep in node.depends_on:
                adj[node.id].add(dep)
                adj[dep].add(node.id)

        # BFS connected components
        visited: Set[str] = set()
        components: List[Set[str]] = []

        for nid in sorted(node_ids):  # sorted for determinism
            if nid in visited:
                continue
            component: Set[str] = set()
            queue = [nid]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)

        return components
