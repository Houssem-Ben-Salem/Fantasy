"""
EXPERIMENT 3 — Do the standard remedies for estimation error actually help?

Experiment 2 measured three pathologies in the optimiser: instability under
noise, Michaud selection bias, and correlation blindness. This tests the two
standard fixes against realised points.

  naive       solve on the projections as they are (what fplbrain does today)
  shrunk      shrink each player's horizon xP toward its positional mean before
              solving. The classic remedy for regression-to-the-mean in the
              inputs: extreme estimates are the ones most likely to be extreme
              because of error rather than skill.
  resampled   Michaud's own remedy. Solve many times under perturbed inputs and
              select by how OFTEN a player is chosen rather than by a single
              draw. Frequency is more robust to any one draw's errors.

RECONSTRUCTION NOTE
-------------------
This file was rebuilt to the interface exp3b_power.py expects, after the
original was lost. The methodology follows the description in LIMITATIONS.md
§5, but the implementation is not byte-identical to the one that produced the
originally reported figures (+0.6 / -4.4). The numbers in §5 are therefore the
ones THIS file produces; see the header of that section.

WHY RESAMPLING IS SOLVED AS A SECOND MILP
-----------------------------------------
Taking the 15 most frequently selected players does not give a legal squad — it
ignores the budget, the 2/5/5/3 split and the 3-per-club limit. So the selection
frequencies become the objective of a final solve, which maximises total
frequency subject to the real constraints. That is the constraint-respecting
form of "pick what gets picked most".
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fplbrain import backtest, models, optimize  # noqa: E402

DB = "data/fpl.db"
SEASON, HIST = "2025-26", ["2024-25"]

SHRINK_LAMBDA = 0.30      # weight on the positional mean
PERTURB_SCALE = 0.25      # same as exp2: estimation error, not outcome variance


# ---------------------------------------------------------------- inputs


def projections_for(gw_start: int, n_gw: int) -> pd.DataFrame:
    """No-look-ahead projections for a window inside the completed 2025-26 season."""
    conn = sqlite3.connect(DB)
    frames = []
    gm = None
    for gw in range(gw_start, gw_start + n_gw):
        ko = pd.read_sql("SELECT MIN(kickoff_time) k FROM fixtures WHERE season=? AND gw=?",
                         conn, params=(SEASON, gw)).iloc[0]["k"]
        if ko is None:
            continue
        if gm is None:
            gm = backtest._goal_model_as_of(conn, SEASON, HIST, ko)
        frames.append(models.project_gameweek(
            conn, SEASON, gw, gm,
            backtest._minutes_as_of(conn, SEASON, gw),
            backtest._defcon_as_of(conn, SEASON, HIST, gw),
            backtest._rates_as_of(conn, SEASON, HIST, gw)))
    return pd.concat(frames, ignore_index=True)


def actuals(horizon: list[int]) -> pd.DataFrame:
    """
    Realised points per player per gameweek.

    Aggregated deliberately: player_gw's primary key includes fixture_id, so a
    DOUBLE GAMEWEEK stores two rows for the same (element, gw). Those points are
    both scored and must be summed — leaving them unaggregated silently produces
    duplicate index labels downstream.
    """
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT element, gw, total_points FROM player_gw WHERE season=? AND gw IN ({})".format(
            ",".join("?" * len(horizon))),
        conn, params=(SEASON, *horizon))
    return df.groupby(["element", "gw"], as_index=False)["total_points"].sum()


# ---------------------------------------------------------------- strategies


def _pack(res, proj: pd.DataFrame, label: str) -> dict:
    return {"label": label,
            "elements": [int(e) for e in res.squad.element],
            "proj": proj,
            "status": res.status}


def build_naive(proj: pd.DataFrame, horizon: list[int]) -> dict:
    return _pack(optimize.initial_squad(proj, horizon), proj, "naive")


def build_shrunk(proj: pd.DataFrame, horizon: list[int],
                 lam: float = SHRINK_LAMBDA) -> dict:
    """Pull each player's per-gameweek xP toward that position's mean."""
    p = proj.copy()
    pos_mean = p.groupby(["gw", "position"])["exp_points"].transform("mean")
    p["exp_points"] = (1 - lam) * p["exp_points"] + lam * pos_mean
    return _pack(optimize.initial_squad(p, horizon), p, "shrunk")


def build_resampled(proj: pd.DataFrame, horizon: list[int], n_resamples: int = 15,
                    seed: int = 11) -> tuple[dict, pd.Series]:
    """Michaud resampled efficiency, with a constraint-respecting final solve."""
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for _ in range(n_resamples):
        p = proj.copy()
        noise = rng.normal(0, 1, len(p)) * p["sd_points"].fillna(1.0) * PERTURB_SCALE
        p["exp_points"] = np.clip(p["exp_points"] + noise, 0, None)
        picked.extend(int(e) for e in optimize.initial_squad(p, horizon).squad.element)

    freq = pd.Series(picked).value_counts() / n_resamples

    # Objective := selection frequency, constraints unchanged. Spread the
    # per-player frequency evenly across the horizon so _prep's time-decay does
    # not reweight what is already a whole-window quantity.
    p = proj.copy()
    p["exp_points"] = p["element"].map(freq).fillna(0.0)
    res = optimize.initial_squad(p, horizon, min_bench_start=0.0)
    # Score the resulting squad with the ORIGINAL projections, since frequency
    # is a selection device, not a points estimate.
    out = _pack(res, proj, "resampled")
    return out, freq


# ---------------------------------------------------------------- evaluation


_FORMATIONS = [(d, m, f) for d, m, f in product(range(3, 6), range(2, 6), range(1, 4))
               if d + m + f == 10]


def _best_xi(sub: pd.DataFrame) -> pd.DataFrame:
    """Highest-projected legal XI from a 15, for one gameweek."""
    by_pos = {p: g.sort_values("exp_points", ascending=False)
              for p, g in sub.groupby("position")}
    gk = by_pos.get("GK")
    if gk is None or gk.empty:
        return sub.nlargest(11, "exp_points")
    best, best_pts = None, -np.inf
    for d, m, f in _FORMATIONS:
        parts = [gk.head(1)]
        ok = True
        for pos, n in (("DEF", d), ("MID", m), ("FWD", f)):
            g = by_pos.get(pos)
            if g is None or len(g) < n:
                ok = False
                break
            parts.append(g.head(n))
        if not ok:
            continue
        xi = pd.concat(parts)
        pts = xi["exp_points"].sum()
        if pts > best_pts:
            best, best_pts = xi, pts
    return best if best is not None else sub.nlargest(11, "exp_points")


def evaluate(r: dict, act: pd.DataFrame, horizon: list[int]) -> dict:
    """
    Realised points for a squad, playing it as a manager would: each gameweek
    field the highest-projected legal XI and captain the highest-projected
    starter, then score against what actually happened.

    XI and captaincy use the STRATEGY'S OWN view of the projections, because
    that is what a user of that strategy would actually see. The comparison
    therefore covers squad selection and team selection together, which is the
    end-to-end quantity that matters.
    """
    ids = r["elements"]
    proj = r["proj"]
    xi_total = cap_total = 0.0
    for gw in horizon:
        sub = proj[(proj.gw == gw) & (proj.element.isin(ids))]
        if sub.empty:
            continue
        xi = _best_xi(sub)
        real = act[act.gw == gw].set_index("element")["total_points"]
        pts = real.reindex(xi.element).fillna(0.0)
        xi_total += float(pts.sum())
        cap_el = xi.sort_values("exp_points", ascending=False).iloc[0].element
        cap_total += float(real.get(cap_el, 0.0))
    return {
        "strategy": r["label"],
        "total_xi": round(xi_total, 1),
        "captain_bonus": round(cap_total, 1),
        "total_with_captain": round(xi_total + cap_total, 1),
        "n_gameweeks": len(horizon),
    }


if __name__ == "__main__":
    START, N = 20, 6
    hz = list(range(START, START + N))
    proj = projections_for(START, N)
    act = actuals(hz)

    rows = []
    for r in (build_naive(proj, hz), build_shrunk(proj, hz),
              build_resampled(proj, hz)[0]):
        rows.append(evaluate(r, act, hz))
    print(f"GW{START}-{START + N - 1}\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nOne window proves nothing — see exp3b_power.py for the paired test.")
