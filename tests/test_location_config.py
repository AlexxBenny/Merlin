import pytest
from pathlib import Path
from unittest.mock import patch

from infrastructure.location_config import LocationConfig


# ─────────────────────────────────────────────────────────────
# Unit tests for LocationConfig
# ─────────────────────────────────────────────────────────────


class TestResolve:
    """Test anchor name → Path resolution."""

    def test_resolve_workspace_returns_cwd(self):
        lc = LocationConfig(anchors={}, cwd=Path("D:/work"))
        assert lc.resolve("WORKSPACE") == Path("D:/work")

    def test_resolve_named_anchor(self):
        lc = LocationConfig(
            anchors={"DESKTOP": Path("C:/Users/test/Desktop")},
            cwd=Path("."),
        )
        assert lc.resolve("DESKTOP") == Path("C:/Users/test/Desktop")

    def test_resolve_drive_anchor(self):
        lc = LocationConfig(
            anchors={"DRIVE_D": Path("D:/")},
            cwd=Path("."),
        )
        assert lc.resolve("DRIVE_D") == Path("D:/")

    def test_resolve_unknown_anchor_raises(self):
        lc = LocationConfig(anchors={}, cwd=Path("."))
        with pytest.raises(KeyError, match="Unknown anchor"):
            lc.resolve("NONEXISTENT")

    def test_workspace_not_in_anchors_dict(self):
        """WORKSPACE is always dynamic — never stored in _anchors."""
        lc = LocationConfig(
            anchors={"DESKTOP": Path("C:/Users/test/Desktop")},
            cwd=Path("D:/projects"),
        )
        assert "WORKSPACE" not in lc._anchors
        assert lc.resolve("WORKSPACE") == Path("D:/projects")


class TestAllAnchorNames:
    """Test anchor name listing for cortex prompt injection."""

    def test_includes_workspace(self):
        lc = LocationConfig(anchors={}, cwd=Path("."))
        assert "WORKSPACE" in lc.all_anchor_names()

    def test_includes_configured_anchors(self):
        lc = LocationConfig(
            anchors={
                "DESKTOP": Path("C:/Users/test/Desktop"),
                "DRIVE_D": Path("D:/"),
            },
            cwd=Path("."),
        )
        names = lc.all_anchor_names()
        assert "DESKTOP" in names
        assert "DRIVE_D" in names
        assert "WORKSPACE" in names

    def test_workspace_is_last(self):
        """WORKSPACE appended after sorted configured anchors."""
        lc = LocationConfig(
            anchors={"ZZZ": Path("Z:/"), "AAA": Path("A:/")},
            cwd=Path("."),
        )
        names = lc.all_anchor_names()
        assert names[-1] == "WORKSPACE"
        assert names[0] == "AAA"


class TestAllAnchors:
    """Test full name→Path mapping."""

    def test_includes_workspace_with_cwd(self):
        lc = LocationConfig(
            anchors={"DESKTOP": Path("C:/Users/test/Desktop")},
            cwd=Path("D:/work"),
        )
        all_a = lc.all_anchors()
        assert all_a["WORKSPACE"] == Path("D:/work")
        assert all_a["DESKTOP"] == Path("C:/Users/test/Desktop")


class TestFromYaml:
    """Test YAML loading and parsing."""

    def test_loads_anchors_from_yaml(self, tmp_path):
        yaml_content = """
anchors:
  DESKTOP:
    path: "{home}/Desktop"

drives:
  enabled: ["C", "D"]
"""
        config_file = tmp_path / "paths.yaml"
        config_file.write_text(yaml_content)

        lc = LocationConfig.from_yaml(config_file, cwd=Path("D:/work"))

        # Drive anchors auto-generated
        assert lc.resolve("DRIVE_C") == Path("C:/")
        assert lc.resolve("DRIVE_D") == Path("D:/")

        # Named anchor resolved (with {home} template — not env var,
        # so it stays literal in this test)
        assert "DESKTOP" in lc.all_anchor_names()

    def test_missing_yaml_uses_defaults(self, tmp_path):
        lc = LocationConfig.from_yaml(
            tmp_path / "nonexistent.yaml",
            cwd=Path("D:/work"),
        )
        # Defaults should include DESKTOP, DOCUMENTS, DOWNLOADS
        names = lc.all_anchor_names()
        assert "DESKTOP" in names
        assert "DOCUMENTS" in names
        assert "DOWNLOADS" in names

    def test_workspace_in_yaml_is_skipped(self, tmp_path):
        """WORKSPACE must never be defined in YAML."""
        yaml_content = """
anchors:
  WORKSPACE:
    path: "C:/bad"
  DESKTOP:
    path: "C:/Users/test/Desktop"
"""
        config_file = tmp_path / "paths.yaml"
        config_file.write_text(yaml_content)

        lc = LocationConfig.from_yaml(config_file, cwd=Path("D:/correct"))

        # WORKSPACE should come from cwd, not YAML
        assert lc.resolve("WORKSPACE") == Path("D:/correct")


class TestEnvVarResolution:
    """Test ${VAR} template expansion."""

    @patch.dict("os.environ", {"USERPROFILE": "C:/Users/testuser"})
    def test_resolves_env_vars(self):
        resolved = LocationConfig._resolve_template("${USERPROFILE}/Desktop")
        assert resolved == "C:/Users/testuser/Desktop"

    def test_unresolved_var_stays_literal(self):
        resolved = LocationConfig._resolve_template("${NONEXISTENT_VAR_12345}/foo")
        assert resolved == "${NONEXISTENT_VAR_12345}/foo"
