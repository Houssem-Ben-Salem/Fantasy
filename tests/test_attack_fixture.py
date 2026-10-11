"""
Attacking returns must depend on the fixture.

models.py's docstring always said returns were "scaled by opponent strength".
The code computed `exp_goals = xg90 * mins_frac` with no opponent at all, so
every attacker projected the same against the leakiest defence as against the
tightest — while the same Dixon-Coles fit was already moving the OTHER side's
clean-sheet odds for that fixture. The exponent was chosen out-of-sample in
analysis/validate_attack_fixture.py.

Hermetic: an in-memory database built from schema.sql, four clubs and a
hand-set Dixon-Coles, so the only thing that differs between two identical
forwards is who they play and where.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from fplbrain import ingest, models

SEASON = "2026-27"
XG90, MINS = 0.5, 85.0


def _db(fixtures: list[tuple[int, int, int]]) -> sqlite3.Connection:
    """fixtures: (gw, home_team_id, away_team_id). Forward 1 plays for AAA,
    forward 2 for CCC; both are identical in every input."""
    c = sqlite3.connect(":memory:")
    ingest.init_schema(c)
    for i, name in enumerate(["AAA", "BBB", "CCC", "DDD"], start=1):
        c.execute("INSERT INTO teams (season, team_id, short_name) VALUES (?,?,?)",
                  (SEASON, i, name))
    for el, team in ((1, 1), (2, 3)):
        c.execute("""INSERT INTO players (season, element, code, web_name, team_id,
                     position, now_cost, selected_by_percent) VALUES (?,?,?,?,?,?,?,?)""",
                  (SEASON, el, 100 + el, f"F{el}", team, "FWD", 7.0, 1.0))
    for fid, (gw, h, a) in enumerate(fixtures, start=1):
        c.execute("""INSERT INTO fixtures (season, fixture_id, gw, home_team, away_team,
                     finished) VALUES (?,?,?,?,?,0)""", (SEASON, fid, gw, h, a))
    c.commit()
    return c


def _goal_model() -> models.DixonColes:
    gm = models.DixonColes()
    gm.attack = {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "DDD": 0.0}
    gm.defence = {"AAA": 0.0, "BBB": 0.4, "CCC": 0.0, "DDD": -0.4}  # BBB leaky, DDD tight
    gm.home_adv, gm.rho, gm.fitted = 0.2, 0.0, True
    return gm


def _project(conn: sqlite3.Connection, gw: int) -> pd.DataFrame:
    els = [1, 2]
    minutes = pd.DataFrame({"element": els, "p_start": 0.95, "p_60": 0.9,
                            "p_any_minutes": 0.97, "exp_minutes": MINS})
    defcon = pd.DataFrame({"element": els, "p_defcon": 0.1})
    rates = pd.DataFrame({"element": els, "xg90": XG90, "xa90": 0.2, "is_pen_taker": 0})
    return models.project_gameweek(conn, SEASON, gw, _goal_model(),
                                   minutes, defcon, rates).set_index("element")


def test_attacker_projects_more_against_a_leaky_defence():
    out = _project(_db([(1, 1, 2), (1, 3, 4)]), 1)   # AAA v leaky BBB; CCC v tight DDD
    assert out.loc[1, "exp_goals"] > 1.2 * out.loc[2, "exp_goals"]
    assert out.loc[1, "exp_assists"] > out.loc[2, "exp_assists"]
    assert out.loc[1, "exp_points"] > out.loc[2, "exp_points"]


def test_attacker_projects_more_at_home():
    conn = _db([(1, 1, 2), (2, 2, 1)])                # AAA hosts BBB, then visits
    assert _project(conn, 1).loc[1, "exp_goals"] > _project(conn, 2).loc[1, "exp_goals"]


def test_a_balanced_schedule_keeps_the_season_level():
    """GUARD, not a revert test. Two exact properties, because the approximate
    one was wrong: this test first asserted the season total stays within 1% of
    xg90 * minutes, on the claim that alpha < 1 costs "well under 1%" through
    Jensen's inequality. At this test's exaggerated defence spread (+/-0.4) it
    costs 1.06%; on the real 2026-27 ratings it costs 0.39%. So:

      1. the raw multiplier averages to exactly 1 over a schedule that meets
         every opponent home and away — it moves goals between fixtures and
         creates none;
      2. the projection applies exactly m ** ATTACK_FIXTURE_ALPHA and nothing
         else, so the only level change is the measured Jensen term.
    """
    sched = [(1, 1, 2), (2, 2, 1), (3, 1, 3), (4, 3, 1), (5, 1, 4), (6, 4, 1)]
    gm, roster = _goal_model(), ["AAA", "BBB", "CCC", "DDD"]
    ms = [gm.attack_multiplier("AAA", o, h, roster)
          for o in ("BBB", "CCC", "DDD") for h in (True, False)]
    assert sum(ms) / len(ms) == pytest.approx(1.0, abs=1e-12)

    conn = _db(sched)
    total = sum(_project(conn, gw).loc[1, "exp_goals"] for gw in range(1, 7))
    expected = XG90 * MINS / 90 * sum(m ** models.ATTACK_FIXTURE_ALPHA for m in ms)
    assert total == pytest.approx(expected, rel=1e-9)
