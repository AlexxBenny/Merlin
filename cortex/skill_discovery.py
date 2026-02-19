# cortex/skill_discovery.py

"""
SkillDiscovery — Abstraction seam for how skills are selected for the LLM prompt.

Today:  AllSkillsDiscovery (return everything — works at <100 skills).
Future: EmbeddingSkillDiscovery (return top-k by similarity — scales to 500+).

This seam prevents cortex from hardcoding manifest construction.
No cortex signature change required to swap discovery strategy.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from execution.registry import SkillRegistry


class SkillDiscovery(ABC):
    """How skills are selected for the LLM prompt.

    Cortex calls find_candidates() and gets back a manifest dict.
    The discovery strategy is opaque to cortex.
    """

    @abstractmethod
    def find_candidates(
        self,
        query: str,
        registry: SkillRegistry,
    ) -> Dict[str, Any]:
        """Return skill manifest dict for the LLM prompt.

        Each key is a skill name, each value has:
        - description
        - inputs: {key: semantic_type}
        - output_keys
        - allowed_modes
        """
        ...


class AllSkillsDiscovery(SkillDiscovery):
    """Return all registered skills. Works at <100 skills.

    Production scaling note:
    At 500+ skills, even 128k-token models degrade.
    Replace with EmbeddingSkillDiscovery that embeds
    skill descriptions and retrieves top-k by cosine similarity.
    """

    def find_candidates(
        self,
        query: str,
        registry: SkillRegistry,
    ) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {}
        for name in registry.all_names():
            skill = registry.get(name)
            manifest[skill.name] = {
                "description": skill.contract.description,
                "action": skill.contract.action,
                "target_type": skill.contract.target_type,
                "inputs": {
                    k: v for k, v in skill.contract.inputs.items()
                },
                "output_keys": sorted(skill.contract.outputs.keys()),
                "allowed_modes": sorted(
                    m.value for m in skill.contract.allowed_modes
                ),
            }
        return manifest

