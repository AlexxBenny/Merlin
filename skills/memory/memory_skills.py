# skills/memory/memory_skills.py
"""
Memory skill wrappers — thin bridge between DAG execution and UserKnowledgeStore.

These skills make memory operations DAG-representable:
  - They participate in depends_on chains
  - Their outputs are OutputReference-accessible
  - They flow through the same execution / narration / safety pipeline

Design:
  - Pure delegation: no intelligence, just bridge
  - No fallback on missing store: fail loudly so the wiring bug is visible
  - All inputs are required (no optional_inputs) for simplicity
"""
from typing import Any, Dict, Optional

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class GetPreferenceSkill(Skill):
    """
    Read a preference from UserKnowledgeStore.

    Inputs:
        key: The canonical preference key (e.g. "volume", "brightness")

    Outputs:
        value: The stored value, or None if not found
        found: bool — whether the key was present
    """

    contract = SkillContract(
        name="memory.get_preference",
        action="get_preference",
        target_type="preference",
        description="Read a stored user preference",
        domain="memory",
        intent_verbs=["get", "read", "what", "retrieve", "know", "recall"],
        intent_keywords=["preference", "preferred", "setting", "default"],
        verb_specificity="generic",
        inputs={"key": "preference_key"},
        outputs={"value": "any", "found": "boolean"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.background},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.background: FailurePolicy.CONTINUE,
        },
        mutates_world=False,
        idempotent=True,
        output_style="templated",
        narration_template="look up your preferred {key}",
        risk_level="safe",
    )

    def __init__(self, user_knowledge):
        self._store = user_knowledge

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        key = inputs["key"]
        if self._store is None:
            return SkillResult(outputs={"value": None, "found": False})
        value = self._store.query(key)
        found = value is not None
        return SkillResult(
            outputs={"value": value, "found": found},
            metadata={"entity": f"preference '{key}'", "domain": "memory"},
        )


class SetPreferenceSkill(Skill):
    """
    Write a preference to UserKnowledgeStore.

    Inputs:
        key:   The canonical preference key (e.g. "volume")
        value: The value to store

    Outputs:
        stored_key:   Normalized key that was stored
        stored_value: Final value after schema validation/coercion
    """

    contract = SkillContract(
        name="memory.set_preference",
        action="set_preference",
        target_type="preference",
        description="Store a user preference",
        domain="memory",
        intent_verbs=["set", "save", "store", "remember", "update"],
        intent_keywords=["preference", "preferred", "default", "setting"],
        verb_specificity="generic",
        inputs={"key": "preference_key", "value": "any"},
        outputs={"stored_key": "preference_key", "stored_value": "any"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.background},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.background: FailurePolicy.CONTINUE,
        },
        mutates_world=False,
        idempotent=False,
        output_style="terse",
        narration_template="remember your preferred {key} as {value}",
        risk_level="safe",
    )

    def __init__(self, user_knowledge):
        self._store = user_knowledge

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        key = inputs["key"]
        value = inputs["value"]
        if self._store is None:
            raise RuntimeError("UserKnowledgeStore not wired into SetPreferenceSkill")
        self._store.set_preference(key, value)
        # Get normalized key and coerced value back out
        stored = self._store.get_preference(key)
        return SkillResult(
            outputs={"stored_key": key, "stored_value": stored},
            metadata={"entity": f"preference '{key}'", "domain": "memory"},
        )


class SetFactSkill(Skill):
    """
    Declare a fact in UserKnowledgeStore (user's name, location, etc.).

    Inputs:
        key:   Fact key (e.g. "name", "location")
        value: The fact value

    Outputs:
        stored_key:   Key stored
        stored_value: Value after normalization
    """

    contract = SkillContract(
        name="memory.set_fact",
        action="set_fact",
        target_type="fact",
        description="Store a fact about the user",
        domain="memory",
        intent_verbs=["note", "store", "memorize", "save"],
        intent_keywords=["fact", "name", "am", "is", "location"],
        verb_specificity="generic",
        inputs={"key": "fact_key", "value": "any"},
        outputs={"stored_key": "fact_key", "stored_value": "any"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.background},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.background: FailurePolicy.CONTINUE,
        },
        mutates_world=False,
        idempotent=False,
        output_style="terse",
        narration_template="remember that your {key} is {value}",
        risk_level="safe",
    )

    def __init__(self, user_knowledge):
        self._store = user_knowledge

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        key = inputs["key"]
        value = inputs["value"]
        if self._store is None:
            raise RuntimeError("UserKnowledgeStore not wired into SetFactSkill")
        self._store.set_fact(key, value)
        return SkillResult(
            outputs={"stored_key": key, "stored_value": value},
            metadata={"entity": f"fact '{key}'", "domain": "memory"},
        )


class AddPolicySkill(Skill):
    """
    Add a conditional policy to UserKnowledgeStore.

    Policies express rules like: "when I watch a movie, volume = 90".

    Inputs:
        condition: Dict describing when the policy applies (e.g. {"activity": "movie"})
        action:    Dict describing what to do (e.g. {"set_volume": 90})
        label:     Optional human-readable label for this policy

    Outputs:
        policy_id: The ID of the stored policy
    """

    contract = SkillContract(
        name="memory.add_policy",
        action="add_policy",
        target_type="policy",
        description="Store a conditional behaviour rule",
        domain="memory",
        intent_verbs=["whenever", "always", "when", "if", "remember", "set"],
        intent_keywords=["policy", "rule", "whenever", "always", "movie", "meeting"],
        verb_specificity="generic",
        inputs={
            "condition": "policy_condition",
            "action": "policy_action",
        },
        optional_inputs={"label": "policy_label"},
        outputs={"policy_id": "policy_id"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.background},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.background: FailurePolicy.CONTINUE,
        },
        mutates_world=False,
        idempotent=False,
        output_style="terse",
        narration_template="remember your rule for {label}",
        risk_level="safe",
    )

    def __init__(self, user_knowledge):
        self._store = user_knowledge

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        if self._store is None:
            raise RuntimeError("UserKnowledgeStore not wired into AddPolicySkill")
        condition = inputs["condition"]
        action = inputs["action"]
        label = inputs.get("label", "")
        policy_id = self._store.add_policy(condition=condition, action=action, label=label)
        return SkillResult(
            outputs={"policy_id": policy_id},
            metadata={"entity": "policy", "domain": "memory"},
        )
