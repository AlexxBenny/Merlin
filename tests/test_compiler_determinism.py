# tests/test_compiler_determinism.py

"""
Compiler Determinism Test.

Sends the same query N times at temperature=0 and verifies
that the resulting IR is byte-identical every time.

This is NOT a unit test — it requires a live Ollama instance.
Run manually:
    python -m pytest tests/test_compiler_determinism.py -v -s

Skips automatically if Ollama is not reachable.

Setup mirrors main.py exactly:
- ModelRouter reads config/models.yaml
- Skills loaded via config/skills.yaml + load_skills()
- LocationConfig from config/paths.yaml
- Temperature overridden to 0 for determinism
"""

import hashlib
import json
import logging
from pathlib import Path

import pytest
import yaml

from cortex.mission_cortex import MissionCortex
from execution.registry import SkillRegistry
from ir.mission import MissionPlan
from errors import FailureIR
from models.ollama_client import OllamaClient
from infrastructure.location_config import LocationConfig
from infrastructure.system_controller import SystemController

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────

DETERMINISM_RUNS = 10
DETERMINISM_QUERY = "unmute, set brightness to 10, and play music"
CONFIG_DIR = Path(__file__).parent.parent / "config"


# ── Fixtures ──────────────────────────────────────────────

def _load_yaml(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@pytest.fixture(scope="module")
def live_cortex():
    """Build a MissionCortex exactly as main.py does, but with temperature=0."""

    # ── LLM client from config ──
    models_config = _load_yaml("models.yaml")
    compiler_cfg = models_config.get("mission_compiler", {})

    if not compiler_cfg:
        pytest.skip("No mission_compiler config in models.yaml")

    client = OllamaClient(
        model=compiler_cfg.get("model", "mistral:7b-instruct"),
        base_url=compiler_cfg.get("base_url", "http://localhost:11434"),
        timeout=120.0,
        temperature=0,  # Override to 0 for determinism
    )

    if not client.is_available():
        pytest.skip("Ollama not available — skipping determinism test")

    # ── Registry + skills from config ──
    registry = SkillRegistry()
    skills_config = _load_yaml("skills.yaml")

    location_config = None
    paths_yaml = CONFIG_DIR / "paths.yaml"
    if paths_yaml.exists():
        location_config = LocationConfig.from_yaml(paths_yaml)

    system_controller = SystemController()
    deps = {
        "location_config": location_config,
        "system_controller": system_controller,
    }

    # Use the same loader as main.py
    from main import load_skills
    load_skills(registry, skills_config, deps=deps)

    # ── Build cortex ──
    cortex = MissionCortex(
        llm_client=client,
        registry=registry,
        location_config=location_config,
    )

    return cortex


def _ir_to_canonical_json(plan: MissionPlan) -> str:
    """Convert IR to canonical JSON for hashing.

    Strips mission_id (contains timestamp).
    Focuses on structural determinism of nodes.
    """
    canonical = {
        "nodes": [
            {
                "id": n.id,
                "skill": n.skill,
                "inputs": {
                    k: ({"$ref_node": v.node, "$ref_output": v.output}
                         if hasattr(v, "node") else v)
                    for k, v in n.inputs.items()
                },
                "outputs": {
                    k: {"name": v.name, "type": v.type}
                    for k, v in n.outputs.items()
                },
                "depends_on": n.depends_on,
                "mode": n.mode.value,
                "condition_on": (
                    {"source": n.condition_on.source, "equals": n.condition_on.equals}
                    if n.condition_on else None
                ),
            }
            for n in plan.nodes
        ],
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


# ── Test ──────────────────────────────────────────────────

@pytest.mark.slow
class TestCompilerDeterminism:
    """Same query × N runs at temperature=0 → byte-identical IR."""

    def test_ir_is_deterministic(self, live_cortex):
        """Hash IR from N runs — all must be identical."""
        cortex = live_cortex
        world_state = {
            "media": {"playing": False},
            "system": {"brightness": 50, "muted": True},
        }

        hashes = []
        canonical_jsons = []

        for i in range(DETERMINISM_RUNS):
            result = cortex._compile_once(
                user_query=DETERMINISM_QUERY,
                world_state_schema=world_state,
                temperature=0,
            )

            if isinstance(result, FailureIR):
                pytest.fail(
                    f"Run {i}: compilation failed: "
                    f"{result.error_type}: {result.error_message}"
                )

            assert isinstance(result, MissionPlan), (
                f"Run {i}: unexpected result type: {type(result)}"
            )

            canonical = _ir_to_canonical_json(result)
            h = hashlib.sha256(canonical.encode()).hexdigest()

            hashes.append(h)
            canonical_jsons.append(canonical)

            logger.info(
                "Run %d/%d: hash=%s nodes=%d",
                i + 1, DETERMINISM_RUNS, h[:16], len(result.nodes),
            )

        # All hashes must be identical
        unique_hashes = set(hashes)
        if len(unique_hashes) > 1:
            # Log first differing pair for diagnosis
            for i in range(1, len(hashes)):
                if hashes[i] != hashes[0]:
                    logger.error(
                        "DETERMINISM FAILURE:\n"
                        "Run 0: %s\n\nRun %d: %s",
                        canonical_jsons[0], i, canonical_jsons[i],
                    )
                    break

            pytest.fail(
                f"Compiler not deterministic: {len(unique_hashes)} "
                f"unique IRs from {DETERMINISM_RUNS} runs.\n"
                f"Hashes: {hashes}"
            )

        logger.info(
            "DETERMINISM PASS: %d runs, hash=%s",
            DETERMINISM_RUNS, hashes[0][:16],
        )
