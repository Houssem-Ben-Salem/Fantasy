"""
Tests for player-name resolution.

BACKGROUND
----------
Every tool taking player names used `web_name.str.contains(nm)` then `.iloc[0]`,
which fails silently in three ways — all three live in the 2026-27 data:

  1. AMBIGUITY. Two players are named `Palmer`: Cole Palmer (CHE, MID, £9.5m)
     and a goalkeeper at Ipswich (£4.0m). `.iloc[0]` picked one without saying
     which, so plan_transfers would optimise around the wrong player and return
     a squad indistinguishable from a correct answer. Silent and total: there
     is no signal to notice.
  2. SUBSTRING OVERREACH. `Raya` also matches `Rayan` (BOU), so an exact,
     correct name was ambiguous. Exact matches must win over substring.
  3. SILENT DROPS. build_squad and chip_advice appended only on a match, so a
     typo shortened the list. A 14-player squad reaching the optimiser either
     fails as infeasible or, worse, solves — answering a question nobody asked.

This is the only bug in the pipeline that was live in a real squad rather than
latent on an unwalked path, which is why it got its own test file.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fplbrain.mcp_server import (PlayerLookupError, _resolve_player,
                                 _resolve_players)


def _proj() -> pd.DataFrame:
    """Minimal projection frame: two Palmers, plus Raya/Rayan."""
    rows = [
        (1, "Palmer", "MID", "CHE", 9.5),      # Cole Palmer
        (2, "Palmer", "GK", "IPS", 4.0),       # the Ipswich keeper
        (3, "Raya", "GK", "ARS", 6.0),
        (4, "Rayan", "MID", "BOU", 5.5),
        (5, "B.Fernandes", "MID", "MUN", 12.0),
        (6, "Mateta", "FWD", "CRY", 6.5),
    ]
    # Two gameweeks per player, so the resolver must dedupe on element.
    return pd.DataFrame(
        [{"element": e, "web_name": w, "position": p, "team": t, "now_cost": c, "gw": gw}
         for e, w, p, t, c in rows for gw in (1, 2)])


# --------------------------------------------------------------- ambiguity


def test_duplicate_exact_name_is_ambiguous_not_guessed():
    """THE BUG: two players named Palmer must raise, never silently pick one."""
    with pytest.raises(PlayerLookupError) as ei:
        _resolve_player(_proj(), "Palmer")
    msg = str(ei.value)
    assert "ambiguous" in msg.lower()
    # The candidates must be listed, not just counted — an agent needs enough
    # to fix the call on its next attempt rather than guess again.
    assert "CHE" in msg and "IPS" in msg
    assert "id=1" in msg and "id=2" in msg


def test_ambiguity_message_names_the_disambiguator():
    with pytest.raises(PlayerLookupError, match="element id"):
        _resolve_player(_proj(), "Palmer")


# --------------------------------------------------------------- exact wins


def test_exact_match_beats_substring():
    """`Raya` is exact for Raya and a substring of Rayan — exact must win."""
    assert _resolve_player(_proj(), "Raya") == 3


def test_exact_match_is_case_insensitive():
    assert _resolve_player(_proj(), "rAyA") == 3


def test_unique_substring_still_resolves():
    assert _resolve_player(_proj(), "Mate") == 6


def test_name_with_regex_metacharacters():
    """'B.Fernandes' must not be treated as a regex, where '.' matches anything."""
    assert _resolve_player(_proj(), "B.Fernandes") == 5


# --------------------------------------------------------------- element ids


def test_numeric_query_resolves_as_element_id():
    assert _resolve_player(_proj(), "2") == 2       # the Ipswich Palmer, unambiguously


def test_unknown_element_id_raises():
    with pytest.raises(PlayerLookupError, match="element id"):
        _resolve_player(_proj(), "9999")


# --------------------------------------------------------------- zero matches


def test_no_match_raises_rather_than_returning_nothing():
    with pytest.raises(PlayerLookupError, match="no player matching"):
        _resolve_player(_proj(), "Haaland")


def test_typo_in_a_list_is_reported_not_dropped():
    """A dropped name silently shortens the squad handed to the optimiser."""
    with pytest.raises(PlayerLookupError) as ei:
        _resolve_players(_proj(), "Raya, Mateta, Haalnd")
    assert "Haalnd" in str(ei.value)


def test_every_problem_is_reported_not_just_the_first():
    with pytest.raises(PlayerLookupError) as ei:
        _resolve_players(_proj(), "Raya, Nobody, Palmer")
    msg = str(ei.value)
    assert "2 of 3" in msg
    assert "Nobody" in msg and "Palmer" in msg


# --------------------------------------------------------------- list guards


def test_expected_count_enforced():
    with pytest.raises(PlayerLookupError, match="expected 15"):
        _resolve_players(_proj(), "Raya, Mateta", expect=15)


def test_same_player_twice_is_rejected():
    """Duplicates are another route to a short squad."""
    with pytest.raises(PlayerLookupError, match="more than once"):
        _resolve_players(_proj(), "Raya, Mateta, Raya")


def test_clean_list_resolves_in_order():
    assert _resolve_players(_proj(), "Raya, Mateta, B.Fernandes") == [3, 6, 5]


def test_empty_list_raises():
    with pytest.raises(PlayerLookupError, match="no player names"):
        _resolve_players(_proj(), "  ,  ")
