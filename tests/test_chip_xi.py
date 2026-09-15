"""
Regression tests for optimize.evaluate_chips selecting a fieldable XI.

BACKGROUND
----------
evaluate_chips used ranked.head(11) — the top 11 by points, with no position
constraints. That is not generally a legal team. On a real squad it produced an
XI with ZERO goalkeepers in three of five gameweeks, because the outfield eleven
outscored the keeper, who was then counted on the bench. bench_boost_gain was
understated by up to 0.56 xP and xi_projected overstated by the same, on the
number compared against a ~15-point Bench Boost threshold.

Hermetic: a constructed squad, no database.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fplbrain import optimize


def _squad(gk_points=(3.0, 1.0)):
    """2 GK / 5 DEF / 5 MID / 3 FWD, built so that head(11) by points drops BOTH keepers."""
    rows, e = [], 1
    for pts in gk_points:
        rows.append(dict(element=e, web_name=f"GK{e}", position="GK", exp_points=pts, n_fixtures=1, gw=1)); e += 1
    for pts in (6.0, 5.5, 5.0, 4.8, 4.6):
        rows.append(dict(element=e, web_name=f"D{e}", position="DEF", exp_points=pts, n_fixtures=1, gw=1)); e += 1
    for pts in (7.0, 6.8, 6.5, 6.2, 6.0):
        rows.append(dict(element=e, web_name=f"M{e}", position="MID", exp_points=pts, n_fixtures=1, gw=1)); e += 1
    for pts in (5.9, 5.8, 3.5):
        rows.append(dict(element=e, web_name=f"F{e}", position="FWD", exp_points=pts, n_fixtures=1, gw=1)); e += 1
    return pd.DataFrame(rows)


def _legal(xi: pd.DataFrame) -> bool:
    c = xi.position.value_counts().to_dict()
    return (len(xi) == 11 and c.get("GK", 0) == 1 and 3 <= c.get("DEF", 0) <= 5
            and 2 <= c.get("MID", 0) <= 5 and 1 <= c.get("FWD", 0) <= 3)


def test_fixture_really_does_break_head_11():
    """Control: the naive top-11 must be illegal here, or the next tests prove nothing."""
    sq = _squad()
    assert not _legal(sq.nlargest(11, "exp_points"))


def test_best_legal_xi_is_fieldable():
    assert _legal(optimize.best_legal_xi(_squad()))


def test_chip_numbers_use_a_legal_xi():
    """THE BUG: xi_projected must be the value of a team you could actually field."""
    sq = _squad()
    ev = optimize.evaluate_chips(sq, sq.element.tolist(), gw=1)
    legal_value = optimize.best_legal_xi(sq).exp_points.sum()
    naive_value = sq.nlargest(11, "exp_points").exp_points.sum()
    assert ev["xi_projected"] == pytest.approx(legal_value)
    assert ev["xi_projected"] != pytest.approx(naive_value), (
        "xi_projected equals the top-11-by-points total, which is not a fieldable team")


def test_bench_is_the_complement_of_the_legal_xi():
    sq = _squad()
    ev = optimize.evaluate_chips(sq, sq.element.tolist(), gw=1)
    xi = optimize.best_legal_xi(sq)
    bench = sq[~sq.element.isin(xi.element)]
    assert len(bench) == 4
    assert ev["bench_boost_gain"] == pytest.approx(bench.exp_points.sum())


def test_captain_is_a_starter():
    sq = _squad()
    ev = optimize.evaluate_chips(sq, sq.element.tolist(), gw=1)
    assert ev["best_captain"] in set(optimize.best_legal_xi(sq).web_name)


def test_formation_is_chosen_not_fixed():
    """A squad strong up front should field three forwards, not a default shape."""
    sq = _squad()
    sq.loc[sq.position == "FWD", "exp_points"] = [9.0, 8.5, 8.0]
    xi = optimize.best_legal_xi(sq)
    assert (xi.position == "FWD").sum() == 3


def test_xi_projected_is_not_the_top_11_by_points():
    """
    Behavioural discriminator, using ONLY evaluate_chips (present before and after
    the fix). Every other test here touches best_legal_xi, which did not exist, so
    against the old code they fail on a missing name and prove nothing about
    behaviour. This one fails on the actual defect.
    """
    sq = _squad()
    ev = optimize.evaluate_chips(sq, sq.element.tolist(), gw=1)
    naive = sq.nlargest(11, "exp_points").exp_points.sum()
    assert ev["xi_projected"] != pytest.approx(naive), (
        f"xi_projected {ev['xi_projected']:.2f} is the top-11-by-points total — "
        "a team with no goalkeeper, which cannot be fielded")
    assert ev["xi_projected"] < naive, "a legal XI can never outscore the unconstrained top 11"
