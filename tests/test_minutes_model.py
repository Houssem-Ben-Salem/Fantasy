"""
Regression tests for models.minutes_model.

BACKGROUND
----------
minutes_model used to switch from prior-seasons-only to current-season-only on
the first current-season row, with no prior. One gameweek loaded made every
start_rate exactly 0.0 or 1.0: 92% of players, 85 provably misclassified against
later minutes — Cherki at 0.0 after 81 and 65 minutes, Senesi at 1.0 with none.

The fix blends the current-season window with the previous season as a Beta
prior worth MINUTES_PRIOR_K matches, validated out-of-sample in
analysis/validate_minutes_prior.py. These tests pin the behaviour, not the
number: if someone reintroduces the branch switch, the first test fails.

Hermetic — temp database, no network, no model fitting.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fplbrain import ingest, models

PREV, CUR = "2025-26", "2026-27"


def _db():
    tmp = tempfile.TemporaryDirectory()
    conn = ingest.connect(Path(tmp.name) / "t.db")
    ingest.init_schema(conn)
    return conn, tmp


def _player(conn, season, element, code, status="a", chance=None):
    pd.DataFrame([{"season": season, "element": element, "code": code,
                   "web_name": f"p{code}", "team_id": 1, "position": "MID",
                   "status": status, "chance_next_round": chance}]
                 ).to_sql("players", conn, if_exists="append", index=False)


def _games(conn, season, element, code, minutes_list):
    pd.DataFrame([{"season": season, "element": element, "code": code, "gw": i + 1,
                   "fixture_id": i + 1, "minutes": m}
                  for i, m in enumerate(minutes_list)]
                 ).to_sql("player_gw", conn, if_exists="append", index=False)


def _row(df, element):
    return df[df.element == element].iloc[0]


def test_one_gameweek_does_not_collapse_to_certainty():
    """THE BUG. A 27-minute cameo after a 60% season must not read as 'never starts'."""
    conn, tmp = _db()
    with tmp:
        _player(conn, PREV, 10, 500)
        _games(conn, PREV, 10, 500, [90] * 6 + [20] * 4)      # started 6 of 10
        _player(conn, CUR, 99, 500)
        _games(conn, CUR, 99, 500, [27])                        # one cameo
        conn.commit()
        p = _row(models.minutes_model(conn, CUR), 99).p_start
        assert 0.0 < p < 1.0, (
            f"p_start={p}: one current-season gameweek was read as certainty; "
            "the current-season window must be blended with the prior")


def test_one_full_game_does_not_become_a_lock_either():
    conn, tmp = _db()
    with tmp:
        _player(conn, PREV, 10, 501)
        _games(conn, PREV, 10, 501, [0] * 8 + [90] * 2)       # started 2 of 10
        _player(conn, CUR, 99, 501)
        _games(conn, CUR, 99, 501, [90])
        conn.commit()
        assert _row(models.minutes_model(conn, CUR), 99).p_start < 1.0


def test_blend_matches_the_documented_formula():
    conn, tmp = _db()
    with tmp:
        _player(conn, PREV, 10, 502)
        _games(conn, PREV, 10, 502, [90] * 3 + [0] * 7)       # prior rate 0.3
        _player(conn, CUR, 99, 502)
        _games(conn, CUR, 99, 502, [90, 90, 0])               # 2 starts in 3
        conn.commit()
        k = models.MINUTES_PRIOR_K
        expected = (2 + k * 0.3) / (3 + k)
        assert _row(models.minutes_model(conn, CUR), 99).p_start == pytest.approx(expected)


def test_no_current_games_reduces_exactly_to_the_prior():
    """Pre-season must be unchanged: with n=0 the blend is the prior rate."""
    conn, tmp = _db()
    with tmp:
        _player(conn, PREV, 10, 503)
        _games(conn, PREV, 10, 503, [90] * 4 + [0] * 6)       # 0.4
        _player(conn, CUR, 99, 503)
        conn.commit()
        assert _row(models.minutes_model(conn, CUR), 99).p_start == pytest.approx(0.4)


def test_no_history_falls_back_to_the_new_signing_guess():
    conn, tmp = _db()
    with tmp:
        _player(conn, PREV, 10, 700)                            # someone else, for a prior season
        _games(conn, PREV, 10, 700, [90])
        _player(conn, CUR, 99, 999)                             # never seen before
        conn.commit()
        assert _row(models.minutes_model(conn, CUR), 99).p_start == pytest.approx(0.35)


def test_injured_player_is_zero_regardless_of_history():
    conn, tmp = _db()
    with tmp:
        _player(conn, PREV, 10, 504)
        _games(conn, PREV, 10, 504, [90] * 10)
        _player(conn, CUR, 99, 504, status="i")
        _games(conn, CUR, 99, 504, [90])
        conn.commit()
        assert _row(models.minutes_model(conn, CUR), 99).p_start == 0.0


def test_all_null_chance_column_still_yields_float_output():
    """Pre-season FPL leaves chance_of_playing empty; it arrived as object dtype."""
    conn, tmp = _db()
    with tmp:
        _player(conn, PREV, 10, 505)
        _games(conn, PREV, 10, 505, [90] * 5)
        _player(conn, CUR, 99, 505, chance=None)
        conn.commit()
        out = models.minutes_model(conn, CUR)
        assert out["p_start"].dtype.kind == "f"
        np.log(out["p_start"].clip(1e-6, 1))                   # must not raise
