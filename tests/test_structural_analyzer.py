# tests/test_structural_analyzer.py

"""
Tests for StructuralAnalyzer feature detection.

Validates:
- Scheduling detection (_detect_scheduling)
- Multi-clause detection (_detect_multi_clause)
- Intra-query coreference derivation
- is_multi_clause + requires_context → intra_query_coreference
- Feature interaction correctness
"""

import pytest
from brain.structural_classifier import StructuralAnalyzer, QueryFeatures


@pytest.fixture
def analyzer():
    return StructuralAnalyzer()


# ─────────────────────────────────────────────────────────────
# Scheduling Detection
# ─────────────────────────────────────────────────────────────

class TestSchedulingDetection:
    """Tests for requires_scheduling feature dimension."""

    def test_pause_after_10_seconds(self, analyzer):
        """Deferred execution: 'after N seconds' must trigger scheduling."""
        f = analyzer.analyze("pause after 10 seconds")
        assert f.requires_scheduling is True
        assert f.reflex_eligible is False

    def test_mute_at_3pm_separated(self, analyzer):
        """'at' token alone is NOT in scheduling tokens. The bigram
        ('at', 'pm') requires adjacent tokens. 'at 3 pm' has a number
        between them so bigram doesn't match.
        However, 'schedule ... at 3 pm' works because 'schedule' is a token."""
        f = analyzer.analyze("schedule mute at 3 pm")
        assert f.requires_scheduling is True

    def test_mute_at_3pm_fused_token(self, analyzer):
        """Fused '3pm' now caught by _TEMPORAL_PATTERN regex."""
        f = analyzer.analyze("mute at 3pm")
        assert f.requires_scheduling is True
        assert f.reflex_eligible is False

    def test_check_battery_every_hour(self, analyzer):
        """Recurring execution: 'every N units' must trigger scheduling."""
        f = analyzer.analyze("check battery every hour")
        assert f.requires_scheduling is True

    def test_remind_me_in_5_minutes(self, analyzer):
        """Deferred execution: 'remind' + 'in N minutes'."""
        f = analyzer.analyze("remind me in 5 minutes")
        assert f.requires_scheduling is True

    def test_wait_then_pause(self, analyzer):
        """Deferred execution: 'wait' token."""
        f = analyzer.analyze("wait 5 seconds then pause")
        assert f.requires_scheduling is True

    def test_schedule_mute(self, analyzer):
        """Explicit 'schedule' token."""
        f = analyzer.analyze("schedule a mute at 3pm")
        assert f.requires_scheduling is True

    def test_set_timer(self, analyzer):
        """Timer token."""
        f = analyzer.analyze("set a timer for 10 minutes")
        assert f.requires_scheduling is True

    def test_repeat_every_day(self, analyzer):
        """Recurring via 'every day' bigram."""
        f = analyzer.analyze("repeat this every day")
        assert f.requires_scheduling is True

    def test_10_seconds_later(self, analyzer):
        """Deferred via 'seconds later' bigram."""
        f = analyzer.analyze("do it 10 seconds later")
        assert f.requires_scheduling is True

    # ── Negative cases: should NOT trigger scheduling ──

    def test_simple_pause(self, analyzer):
        """Simple reflex: no scheduling qualifier."""
        f = analyzer.analyze("pause")
        assert f.requires_scheduling is False

    def test_play_music(self, analyzer):
        """Simple reflex: no scheduling qualifier."""
        f = analyzer.analyze("play music")
        assert f.requires_scheduling is False

    def test_set_brightness_to_50(self, analyzer):
        """Parameter command: no scheduling qualifier."""
        f = analyzer.analyze("set brightness to 50")
        assert f.requires_scheduling is False

    def test_what_time_is_it(self, analyzer):
        """Temporal question — NOT scheduling."""
        f = analyzer.analyze("what time is it")
        assert f.requires_scheduling is False


# ─────────────────────────────────────────────────────────────
# Multi-Clause Detection
# ─────────────────────────────────────────────────────────────

class TestMultiClauseDetection:
    """Tests for is_multi_clause feature."""

    def test_conjunction_and(self, analyzer):
        f = analyzer.analyze("create folder and play music")
        assert f.is_multi_clause is True

    def test_conjunction_then(self, analyzer):
        f = analyzer.analyze("open chrome then mute")
        assert f.is_multi_clause is True

    def test_conjunction_also(self, analyzer):
        f = analyzer.analyze("set brightness to 50 also play music")
        assert f.is_multi_clause is True

    def test_sentence_boundary_period(self, analyzer):
        """Period-separated clauses."""
        f = analyzer.analyze("create folder alex. inside it create man")
        assert f.is_multi_clause is True

    def test_comma_separated(self, analyzer):
        """Comma-separated clauses."""
        f = analyzer.analyze("create folder alex, inside it create man")
        assert f.is_multi_clause is True

    def test_conjunction_plus(self, analyzer):
        f = analyzer.analyze("set volume to 50 plus toggle nightlight")
        assert f.is_multi_clause is True

    # ── Single clause: should NOT be multi-clause ──

    def test_single_command(self, analyzer):
        f = analyzer.analyze("pause music")
        assert f.is_multi_clause is False

    def test_single_command_delete_it(self, analyzer):
        """Pronoun without clause boundary = single clause."""
        f = analyzer.analyze("delete it")
        assert f.is_multi_clause is False

    def test_single_command_with_params(self, analyzer):
        f = analyzer.analyze("set brightness to 50")
        assert f.is_multi_clause is False

    def test_trailing_period_not_multi_clause(self, analyzer):
        """Trailing period is not a clause boundary."""
        f = analyzer.analyze("delete it.")
        assert f.is_multi_clause is False


# ─────────────────────────────────────────────────────────────
# Intra-Query Coreference
# ─────────────────────────────────────────────────────────────

class TestIntraQueryCoreference:
    """Tests for intra_query_coreference derived property."""

    def test_inside_it_multi_clause(self, analyzer):
        """Classic case: 'Inside it' in multi-clause query."""
        f = analyzer.analyze(
            "create a folder named alex. inside it create two folders"
        )
        assert f.requires_context is True
        assert f.is_multi_clause is True
        assert f.intra_query_coreference is True

    def test_set_it_with_and(self, analyzer):
        """'set it to 50' after another clause."""
        f = analyzer.analyze("set brightness to 10 and after that set it to 50")
        assert f.requires_context is True
        assert f.is_multi_clause is True
        assert f.intra_query_coreference is True

    def test_open_then_mute_it(self, analyzer):
        """'mute it' after 'open YouTube'."""
        f = analyzer.analyze("open youtube then mute it")
        assert f.requires_context is True
        assert f.is_multi_clause is True
        assert f.intra_query_coreference is True

    def test_comma_clause_inside_it(self, analyzer):
        """Comma-separated with intra-query coreference."""
        f = analyzer.analyze("create folder alex, inside it create man")
        assert f.intra_query_coreference is True

    # ── NOT intra-query: single clause with pronoun ──

    def test_delete_it_single_clause(self, analyzer):
        """'delete it' = conversational reference, not intra-query."""
        f = analyzer.analyze("delete it")
        assert f.requires_context is True
        assert f.is_multi_clause is False
        assert f.intra_query_coreference is False

    def test_play_it_single(self, analyzer):
        f = analyzer.analyze("play it again")
        assert f.requires_context is True
        assert f.intra_query_coreference is False

    # ── No context at all: not intra-query ──

    def test_no_context_no_coreference(self, analyzer):
        """No pronouns = no coreference at all."""
        f = analyzer.analyze("set brightness to 50")
        assert f.requires_context is False
        assert f.intra_query_coreference is False


# ─────────────────────────────────────────────────────────────
# Repr
# ─────────────────────────────────────────────────────────────

class TestQueryFeaturesRepr:
    def test_reflex_eligible_repr(self):
        f = QueryFeatures()
        assert repr(f) == "QueryFeatures(reflex_eligible)"

    def test_context_repr(self):
        f = QueryFeatures(requires_context=True)
        assert "context" in repr(f)

    def test_scheduling_repr(self):
        f = QueryFeatures(requires_scheduling=True)
        assert "scheduling" in repr(f)

    def test_intra_coref_repr(self):
        f = QueryFeatures(requires_context=True, is_multi_clause=True)
        assert "intra_query_coref" in repr(f)


# ─────────────────────────────────────────────────────────────
# Temporal Pattern Regex (Gap Closure)
# ─────────────────────────────────────────────────────────────

class TestTemporalPatternRegex:
    """Tests for _TEMPORAL_PATTERN regex that closes bigram detection gaps.

    These 3 queries previously slipped through all gates:
    - 'mute at 3pm' (fused time token)
    - 'mute at 3 pm' (number between 'at' and 'pm')
    - 'pause music in a bit' (vague temporal)
    """

    # ── Gap closure ──

    def test_gap1_mute_at_3pm(self, analyzer):
        """Previously: scheduling=False. Now caught by regex."""
        f = analyzer.analyze("mute at 3pm")
        assert f.requires_scheduling is True

    def test_gap2_mute_at_3_pm(self, analyzer):
        """Previously: scheduling=False. Now caught by 'at\\s+\\d'."""
        f = analyzer.analyze("mute at 3 pm")
        assert f.requires_scheduling is True

    def test_gap3_pause_in_a_bit(self, analyzer):
        """Previously: scheduling=False. Now caught by 'in a bit'."""
        f = analyzer.analyze("pause music in a bit")
        assert f.requires_scheduling is True

    # ── Additional temporal patterns (regex coverage) ──

    def test_fused_10minutes(self, analyzer):
        f = analyzer.analyze("pause after 10minutes")
        assert f.requires_scheduling is True

    def test_fused_5mins(self, analyzer):
        f = analyzer.analyze("mute in 5mins")
        assert f.requires_scheduling is True

    def test_fused_2hrs(self, analyzer):
        f = analyzer.analyze("notify after 2hrs")
        assert f.requires_scheduling is True

    def test_in_a_moment(self, analyzer):
        f = analyzer.analyze("pause in a moment")
        assert f.requires_scheduling is True

    def test_in_a_while(self, analyzer):
        f = analyzer.analyze("mute in a while")
        assert f.requires_scheduling is True

    def test_at_digit_scheduling(self, analyzer):
        """'at 3' → scheduling."""
        f = analyzer.analyze("set alarm at 3")
        assert f.requires_scheduling is True

    # ── False positives (accepted — mission handles) ──

    def test_false_positive_volume_at_30(self, analyzer):
        """'at 3' in 'volume at 30' is a false positive.
        Accepted: mission path handles correctly, just slower."""
        f = analyzer.analyze("volume at 30")
        assert f.requires_scheduling is True  # False positive, intentional

    # ── True negatives (must NOT trigger) ──

    def test_no_false_positive_brightness(self, analyzer):
        """'50' has no time unit = no trigger."""
        f = analyzer.analyze("set brightness to 50")
        assert f.requires_scheduling is False

    def test_no_false_positive_folders(self, analyzer):
        """'3' without time unit."""
        f = analyzer.analyze("create 3 folders")
        assert f.requires_scheduling is False

    def test_no_false_positive_track(self, analyzer):
        f = analyzer.analyze("play track 5")
        assert f.requires_scheduling is False

    def test_no_false_positive_bare_mute(self, analyzer):
        f = analyzer.analyze("mute")
        assert f.requires_scheduling is False


# ─────────────────────────────────────────────────────────────
# Disqualifier Token Gate
# ─────────────────────────────────────────────────────────────

class TestDisqualifierGate:
    """Tests for has_disqualifier_tokens() — BrainCore Stage 0 fast path."""

    def test_disq_mute_at_3pm(self, analyzer):
        """Was a safety gap. Now blocked by temporal regex."""
        assert analyzer.has_disqualifier_tokens("mute at 3pm") is True

    def test_disq_mute_at_3_pm(self, analyzer):
        assert analyzer.has_disqualifier_tokens("mute at 3 pm") is True

    def test_disq_pause_in_a_bit(self, analyzer):
        assert analyzer.has_disqualifier_tokens("pause music in a bit") is True

    def test_disq_simple_mute_allowed(self, analyzer):
        assert analyzer.has_disqualifier_tokens("mute") is False

    def test_disq_brightness_allowed(self, analyzer):
        assert analyzer.has_disqualifier_tokens("set brightness to 50") is False


# ─────────────────────────────────────────────────────────────
# Reflex Regex Strictness Invariant
# ─────────────────────────────────────────────────────────────

class TestReflexStrictnessInvariant:
    """Reflex templates must be exact-match patterns.

    CRITICAL SAFETY INVARIANT:
    If reflex regex is ever loosened, temporal queries like
    'pause music in a bit' could silently truncate to 'pause'.

    All reflex patterns MUST:
    - Start with ^ (anchor)
    - End with $ (anchor)
    - Reject any extra words not in the pattern
    """

    def test_all_reflex_patterns_are_anchored(self):
        import yaml
        with open("config/routing.yaml") as f:
            cfg = yaml.safe_load(f)
        templates = cfg.get("reflex_templates", [])
        for tpl in templates:
            pattern = tpl["pattern"]
            assert pattern.startswith("^"), (
                f"Reflex pattern MUST start with ^: {pattern}"
            )
            assert pattern.endswith("$"), (
                f"Reflex pattern MUST end with $: {pattern}"
            )
