# skills/reasoning/generate_text.py

"""
GenerateTextSkill — LLM-driven content generation within missions.

This is a SKILL, not a routing decision. It:
- Receives a prompt via contract-validated inputs
- Calls content_llm.complete() directly
- Returns generated text as a SkillResult output

Anti-recursion guarantees:
- Does NOT know about Merlin, Coordinator, or BrainCore
- Does NOT re-enter the pipeline
- Does NOT trigger SKILL_PLAN or DIRECT mode
- Identical isolation to ReportBuilder.llm.complete()

The content_llm is injected at startup via DI (inspect.signature).
If content_llm is unavailable, this skill is NOT registered (fail-fast).
"""

import logging
from typing import Any, Dict, Optional

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from world.snapshot import WorldSnapshot
from models.base import LLMClient


logger = logging.getLogger(__name__)


class GenerateTextSkill(Skill):
    """Generate text content using an LLM.

    Inputs:
        prompt: What to generate (e.g., "a short poem about AI")
        style:  Optional tone/style modifier (e.g., "formal", "casual")

    Outputs:
        text: The generated content string

    The LLM call is bounded by max_tokens in config/models.yaml
    (content_generator role). No streaming, no loops.
    """

    contract = SkillContract(
        name="reasoning.generate_text",
        action="generate_text",
        target_type="text",
        description="Generate text content",
        narration_template="generate {prompt}",
        intent_verbs=["write", "generate", "compose", "create", "tell",
                       "draft", "summarize", "describe", "explain"],
        intent_keywords=["story", "summary", "poem", "essay", "text",
                          "paragraph", "letter", "report", "description",
                          "about", "content"],
        verb_specificity="generic",
        domain="reasoning",
        requires_focus=False,
        resource_cost="medium",
        inputs={
            "prompt": "text_prompt",
        },
        optional_inputs={
            "style": "text_prompt",
        },
        outputs={
            "text": "generated_text",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=[],
        mutates_world=False,
        idempotent=True,
        data_freshness="snapshot",
        output_style="terse",
    )

    def __init__(self, content_llm: LLMClient):
        self._llm = content_llm

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
    ) -> SkillResult:
        prompt = inputs["prompt"]
        style = inputs.get("style", "")

        logger.info(
            "[TRACE] GenerateTextSkill.execute: prompt=%r, style=%r",
            prompt[:80], style,
        )

        # Build system instruction
        system = (
            "You are MERLIN, an intelligent desktop automation assistant. "
            "Generate the requested content directly. "
            "Do not explain yourself or add meta-commentary. "
            "Just produce the content."
        )
        if style:
            system += f" Write in a {style} style."

        full_prompt = f"{system}\n\nUser request: {prompt}"
        text = self._llm.complete(full_prompt)

        if not text or not text.strip():
            raise RuntimeError(
                f"Content generation returned empty response for prompt: "
                f"{prompt[:100]}"
            )

        logger.info(
            "[TRACE] GenerateTextSkill: generated %d chars",
            len(text.strip()),
        )

        return SkillResult(
            outputs={"text": text.strip()},
            metadata={"domain": "reasoning"},
        )
