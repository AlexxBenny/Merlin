from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

from skills.contract import SkillContract
from skills.skill_result import SkillResult
from world.snapshot import WorldSnapshot
from world.timeline import WorldTimeline

# Import at module level for type hints — but SkillContext is optional
# at runtime for backward compatibility with existing skills.
try:
    from execution.skill_context import SkillContext
except ImportError:
    SkillContext = None  # type: ignore


class Skill(ABC):
    """
    Base class for all MERLIN skills.

    Every skill MUST define a `contract` class attribute of type SkillContract.
    The executor and cortex read guarantees from the contract, not from
    ad-hoc attributes.

    The `name`, `input_keys`, and `output_keys` properties delegate to
    the contract for backward compatibility with registry and executor.
    """
    contract: SkillContract

    # ---------------------------------------------------------------
    # Properties delegating to contract (backward-compatible accessors)
    # ---------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.contract.name

    @property
    def input_keys(self) -> Set[str]:
        """All valid input keys (required + optional)."""
        return set(self.contract.inputs.keys()) | set(
            self.contract.optional_inputs.keys()
        )

    @property
    def required_input_keys(self) -> Set[str]:
        """Required input keys only."""
        return set(self.contract.inputs.keys())

    @property
    def optional_input_keys(self) -> Set[str]:
        """Optional input keys only."""
        return set(self.contract.optional_inputs.keys())

    @property
    def output_keys(self) -> Set[str]:
        return set(self.contract.outputs.keys())

    # ---------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------

    @abstractmethod
    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
        context: Optional['SkillContext'] = None,
    ) -> SkillResult:
        """
        Execute the skill.

        Args:
            inputs: Contract-validated input parameters.
            world: Timeline for event emission (append-only).
            snapshot: Immutable upstream WorldSnapshot — read-only state.
                      Built once in handle_percept, passed down through
                      executor. Skills MUST NOT rebuild state from timeline.
                      Optional for backward compatibility; state-aware skills
                      should check for None before accessing.
            context: Optional cross-cutting execution context (user identity,
                     time). Built per-mission. Skills can access
                     context.user.name, context.time, etc.
                     None for backward compatibility.

        Returns SkillResult with:
        - outputs: contract-validated keys only
        - metadata: side-channel (entity, domain, trace IDs)

        Rules:
        - Return SkillResult — never a plain dict
        - outputs must match contract.outputs keys
        - Emit only events declared in contract.emits_events
        - When emitting events, use contract.name as the source parameter
          in timeline.emit(source=self.contract.name, ...).
          The executor filters events by source to avoid false attribution
          from concurrent background sources.
        - Never mutate world unless contract.mutates_world is True
        - Snapshot is IMMUTABLE — never mutate it
        - Raise on failure — executor handles mode-aware semantics
        """
        raise NotImplementedError
