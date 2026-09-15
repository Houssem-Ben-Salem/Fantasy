"""
Regression tests for the Dixon-Coles ridge penalty.

BACKGROUND
----------
DixonColes.fit was raw MLE bounded at +/-3. A promoted club with two matches has
a nearly flat likelihood, so the optimiser walked to the bound: in 2026-27 COV
attack and HUL defence sat at exactly -3.000. exp() of a bound is not an
estimate — it put four GBP4.0m Hull defenders in the top five projected players
and a 96% Man City clean sheet against a market 49%.

The bound disguised the failure rather than preventing it. Unbounded it would
diverge and crash; bounded it became a finite value, which the optimiser hunts.

GOAL_RIDGE_LAMBDA is validated in analysis/validate_goal_ridge.py. These pin the
behaviour. Hermetic: synthetic Poisson results with a fixed seed, no database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplbrain import models

BOUND = 3.0
ESTABLISHED = [f"T{i}" for i in range(12)]


def _season(rng, n_rounds=8):
    """A block of results among established teams, plus a thin promoted club."""
    rows, t0 = [], pd.Timestamp("2025-08-01", tz="UTC")
    day = 0
    for _ in range(n_rounds):
        order = rng.permutation(ESTABLISHED)
        for h, a in zip(order[::2], order[1::2]):
            rows.append(dict(home_team=h, away_team=a,
                             home_score=int(rng.poisson(1.5)), away_score=int(rng.poisson(1.2)),
                             kickoff=t0 + pd.Timedelta(days=day)))
            day += 1
    # Promoted club: two matches, two clean sheets, scores nothing. On raw MLE its
    # defence and attack have almost no likelihood to anchor them.
    for opp in ("T0", "T1"):
        rows.append(dict(home_team="PRO", away_team=opp, home_score=0, away_score=0,
                         kickoff=t0 + pd.Timedelta(days=day)))
        day += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def data():
    return _season(np.random.default_rng(20260915))


def _max_abs(m, team):
    return max(abs(m.attack[team]), abs(m.defence[team]))


def test_raw_mle_pins_a_thin_club_at_the_bound(data):
    """Control: without the penalty the defect reproduces, so the next test means something."""
    m = models.DixonColes(ridge=0.0).fit(data)
    assert _max_abs(m, "PRO") >= BOUND - 1e-3, (
        "the synthetic thin club did not pin under raw MLE — this fixture no "
        "longer reproduces the defect and the ridge test below proves nothing")


def test_ridge_keeps_a_thin_club_off_the_bound(data):
    """THE BUG: two matches must not produce a rating at the optimiser's bound."""
    m = models.DixonColes().fit(data)
    assert m.ridge == models.GOAL_RIDGE_LAMBDA
    assert _max_abs(m, "PRO") < BOUND - 0.5, (
        f"PRO rated at {_max_abs(m, 'PRO'):.3f}: a thin club reached the bound, "
        "which is a boundary value, not an estimate")


def test_no_rating_anywhere_at_the_bound(data):
    m = models.DixonColes().fit(data)
    vals = list(m.attack.values()) + list(m.defence.values())
    assert max(abs(v) for v in vals) < BOUND - 1e-3


def test_zero_match_team_is_rated_near_league_average_not_weakest(data):
    """GW1 symptom: a club with no results must not inherit the weakest observed ratings."""
    m = models.DixonColes().fit(data, teams=ESTABLISHED + ["PRO", "NEW"])
    assert "NEW" in m.attack and "NEW" in m.defence
    weakest_attack = min(v for t, v in m.attack.items() if t != "NEW")
    assert m.attack["NEW"] > weakest_attack, "zero-match club got the weakest attack"
    assert abs(m.attack["NEW"]) < 0.25, f"expected near league average, got {m.attack['NEW']:.3f}"


def test_extra_teams_ignored_without_ridge(data):
    """Unpenalised, a zero-match team is unidentified — so it is not rated at all."""
    m = models.DixonColes(ridge=0.0).fit(data, teams=ESTABLISHED + ["NEW"])
    assert "NEW" not in m.attack


def test_projections_for_thin_club_are_plausible(data):
    """Decision layer, in miniature: no opponent of a thin club projects near zero goals."""
    m = models.DixonColes().fit(data)
    _, away_xg = m.expected_goals("PRO", "T5")
    assert away_xg > 0.4, f"T5 projected {away_xg:.2f} goals at PRO — a phantom clean sheet"
