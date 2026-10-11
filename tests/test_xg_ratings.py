"""
Team ratings may learn from chance quality (xG), not only from goals.

Dixon-Coles rated defences on goals conceded. Over five matches that is mostly
luck: 2026-27 Hull conceded 4 from 7.40 xG against and were rated the second-
best defence in the league, which put Newcastle's expected goals against them
at 1.07 to the market's 2.02. The fit now maximises

    (1 - w) * goals log-likelihood  +  w * quasi-Poisson log-likelihood of xG

so w = 0 is the old model exactly, and w > 0 lets a lucky defence be seen as
one. w itself is chosen out-of-sample in analysis/validate_xg_ratings.py.

Hermetic: a synthetic six-club league with chosen scores and xG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fplbrain import models

TEAMS = ["LUCKY", "SOLID", "C", "D", "E", "F"]
# LUCKY concede almost nothing but allow big chances; SOLID concede a goal a
# game from very little. Everyone else is ordinary on both measures.
CONCEDED = {"LUCKY": (0, 2.2), "SOLID": (1, 0.5)}


def _league(rounds: int = 3) -> pd.DataFrame:
    rows, day = [], pd.Timestamp("2026-08-01", tz="UTC")
    for _ in range(rounds):
        for h in TEAMS:
            for a in TEAMS:
                if h == a:
                    continue
                hg, hxg = CONCEDED.get(a, (1, 1.3))     # what the AWAY side concedes
                ag, axg = CONCEDED.get(h, (1, 1.2))
                rows.append(dict(home_team=h, away_team=a, home_score=hg, away_score=ag,
                                 home_xg=hxg, away_xg=axg, kickoff=day))
                day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def _fit(w: float, matches: pd.DataFrame) -> models.DixonColes:
    return models.DixonColes(xg_weight=w).fit(matches)


def test_weight_zero_is_the_goals_only_model_exactly():
    m = _league()
    with_xg = _fit(0.0, m)
    without = models.DixonColes(xg_weight=0.0).fit(m.drop(columns=["home_xg", "away_xg"]))
    for t in TEAMS:
        assert np.isclose(with_xg.defence[t], without.defence[t], atol=1e-9)
        assert np.isclose(with_xg.attack[t], without.attack[t], atol=1e-9)


def test_xg_exposes_a_lucky_defence():
    m = _league()
    goals_only, xg_only = _fit(0.0, m), _fit(1.0, m)
    # lower defence value = fewer goals allowed
    assert goals_only.defence["LUCKY"] < goals_only.defence["SOLID"]
    assert xg_only.defence["SOLID"] < xg_only.defence["LUCKY"]


def test_missing_xg_falls_back_to_goals_for_that_match():
    m = _league()
    m.loc[m.index[:10], ["home_xg", "away_xg"]] = np.nan
    fit = _fit(0.5, m)
    assert all(np.isfinite(v) for v in fit.defence.values())
