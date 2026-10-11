"""
The optimiser must survive both ways its solver broke in September 2026.

1. PuLP 4.0 (25 Sep 2026) stopped bundling CBC. CI installed `pulp>=2.7`
   unpinned, picked 4.0 up three hours after release, and every MILP test failed
   daily from then on. requirements.txt now pins <4.
2. PuLP 3.x bundles CBC for macOS only as an x86_64 binary. On Apple Silicon
   without Rosetta (absent after the macOS 27 upgrade) exec fails with
   `OSError: [Errno 86] Bad CPU type`, so every optimiser call and every MCP
   squad tool died locally. The optimiser now falls back to HiGHS.

And a latent bug the fallback exposed. Solutions were read with
`int(var.value())`. CBC returns exact integers, so this never fired on CI.
A MIP solver is entitled to return a binary anywhere within its integrality
tolerance, e.g. 0.9999999, and `int()` truncates that to 0: a 15-man squad came
back with 14 players and status 'Optimal', and a plan that paid a -4 in its
objective reported zero hits.

Every test here goes through the public entry points and fails on the old code
for the behavioural reason (truncated squad, uncaught OSError), not because a
helper does not exist. Hermetic: synthetic projections, no data/fpl.db.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp
import pytest

from fplbrain import optimize

highspy = pytest.importorskip("highspy")

HORIZON = [1, 2, 3]


def _proj() -> pd.DataFrame:
    """36 players, 12 clubs of 3, every position deep enough to be feasible."""
    rng = np.random.default_rng(0)
    rows, e = [], 0
    for pos, n in {"GK": 4, "DEF": 12, "MID": 12, "FWD": 8}.items():
        for k in range(n):
            e += 1
            for gw in HORIZON:
                rows.append(dict(element=e, gw=gw, web_name=f"{pos}{k}", position=pos,
                                 team=f"T{e % 12}", now_cost=4.5 + (k % 4),
                                 selected_by_percent=1.0,
                                 exp_points=2.0 + 4.0 * rng.random(),
                                 p_start=0.9, sd_points=2.0))
    return pd.DataFrame(rows)


class _TolerantHiGHS(pulp.HiGHS):
    """A correct solve, then every integer variable at 1 moved to 1 - 1e-7 —
    a value any MIP solver may legally return within integrality tolerance."""

    def actualSolve(self, lp, **kw):
        status = super().actualSolve(lp, **kw)
        for v in lp.variables():
            if v.cat == pulp.LpInteger and v.varValue is not None and v.varValue > 0.5:
                v.varValue = v.varValue - 1e-7
        return status


class _ExactHiGHS(pulp.HiGHS):
    """HiGHS with integer variables rounded, i.e. what CBC hands back in practice.
    Plain HiGHS is not this: it returns binaries within tolerance."""

    def actualSolve(self, lp, **kw):
        status = super().actualSolve(lp, **kw)
        for v in lp.variables():
            if v.cat == pulp.LpInteger and v.varValue is not None:
                v.varValue = round(v.varValue)
        return status


class _CannotExecute(pulp.PULP_CBC_CMD):
    """What the bundled x86_64 CBC does on Apple Silicon without Rosetta."""

    def actualSolve(self, lp, **kw):
        raise OSError(86, "Bad CPU type in executable")


def _assert_legal(r: optimize.OptimResult) -> None:
    assert r.status == "Optimal"
    assert len(r.squad) == 15
    assert r.squad.position.value_counts().to_dict() == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert int(r.squad.in_xi.sum()) == 11
    assert int(r.squad.is_captain.sum()) == 1


def test_squad_survives_integrality_tolerance(monkeypatch):
    monkeypatch.setattr(pulp, "PULP_CBC_CMD", lambda msg=0, **kw: _TolerantHiGHS(msg=msg))
    _assert_legal(optimize.initial_squad(_proj(), HORIZON))


def test_transfer_hits_survive_integrality_tolerance(monkeypatch):
    proj = _proj()
    # Setup with exact integers, so this test can only fail on the plan itself.
    monkeypatch.setattr(pulp, "PULP_CBC_CMD", lambda msg=0, **kw: _ExactHiGHS(msg=msg))
    owned = optimize.initial_squad(proj, HORIZON).squad.element.tolist()
    # Make one unowned midfielder worth far more than a -4, and allow no free
    # transfer: the only optimal plan is exactly one transfer and one hit.
    star = proj[(proj.position == "MID") & ~proj.element.isin(owned)].element.iloc[0]
    proj.loc[proj.element == star, ["exp_points", "now_cost"]] = [30.0, 4.5]

    monkeypatch.setattr(pulp, "PULP_CBC_CMD", lambda msg=0, **kw: _TolerantHiGHS(msg=msg))
    r = optimize.transfer_plan(proj, owned, HORIZON, bank=0.0, free_transfers=0)
    _assert_legal(r)
    assert [t["in_element"] for t in r.transfers] == [int(star)]
    assert r.hits_taken == 1


def test_falls_back_to_highs_when_cbc_cannot_execute(monkeypatch):
    monkeypatch.setattr(pulp, "PULP_CBC_CMD", _CannotExecute)
    _assert_legal(optimize.initial_squad(_proj(), HORIZON))
    owned = optimize.initial_squad(_proj(), HORIZON).squad.element.tolist()
    _assert_legal(optimize.transfer_plan(_proj(), owned, HORIZON, bank=0.0, free_transfers=1))


def test_this_environment_can_solve():
    """No monkeypatching: whatever is installed here must produce a legal squad.
    On CI this is the line that names the cause when a dependency drops its solver."""
    _assert_legal(optimize.initial_squad(_proj(), HORIZON))
