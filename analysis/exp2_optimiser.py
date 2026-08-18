"""
EXPERIMENT 2 — Is the optimiser an estimation-error maximiser?

This is the deepest structural worry, and it is not specific to football. Michaud
(1989) showed that mean-variance portfolio optimisers systematically select the
assets whose expected returns are most OVERESTIMATED, because the optimiser
cannot distinguish a genuinely high estimate from a high estimation error. It
maximises error as reliably as it maximises return.

Our MILP has exactly that shape: it takes xP as a deterministic input and picks
the 15 that maximise the sum. Every one of those xP values carries error, and
the selection is biased toward the ones erring high.

Three tests:

  A. STABILITY   Perturb projections within their own stated uncertainty and
                 re-solve. If the "optimal" squad churns heavily, optimality is
                 an artefact of noise, not a property of the squad.

  B. SELECTION BIAS  Compare the realised points of selected players against
                 what was projected for them, versus the same gap for the player
                 pool at large. If the selected set's gap is systematically
                 worse, the optimiser is picking overestimates.

  C. CORRELATION The MILP treats players independently. Same-team players share
                 clean sheets and, to a degree, attacking returns. Measure the
                 real correlation and quantify the variance the optimiser cannot
                 see.
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fplbrain import backtest, models, optimize  # noqa: E402

DB = "data/fpl.db"
SEASON, HIST = "2025-26", ["2024-25"]
RNG = np.random.default_rng(7)


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


# ---------------------------------------------------------------- A. stability


def test_stability(proj: pd.DataFrame, horizon: list[int], n_trials: int = 25) -> dict:
    base = optimize.initial_squad(proj, horizon)
    base_ids = set(base.squad.element)

    overlaps, all_picked = [], []
    for _ in range(n_trials):
        p = proj.copy()
        # Perturb each projection by the model's OWN stated uncertainty, scaled
        # down to represent estimation error rather than outcome variance.
        noise = RNG.normal(0, 1, len(p)) * p["sd_points"].fillna(1.0) * 0.25
        p["exp_points"] = np.clip(p["exp_points"] + noise, 0, None)
        r = optimize.initial_squad(p, horizon)
        ids = set(r.squad.element)
        overlaps.append(len(ids & base_ids))
        all_picked.extend(ids)

    counts = pd.Series(all_picked).value_counts()
    return {
        "mean_overlap_with_base": round(float(np.mean(overlaps)), 2),
        "min_overlap": int(np.min(overlaps)),
        "distinct_players_ever_selected": int(len(counts)),
        "players_selected_every_trial": int((counts == n_trials).sum()),
        "players_selected_once_only": int((counts == 1).sum()),
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------- B. selection bias


def test_selection_bias(proj: pd.DataFrame, horizon: list[int]) -> dict:
    conn = sqlite3.connect(DB)
    actual = pd.read_sql(
        "SELECT element, gw, total_points FROM player_gw WHERE season=? AND gw IN ({})".format(
            ",".join("?" * len(horizon))),
        conn, params=(SEASON, *horizon))
    act = actual.groupby("element", as_index=False)["total_points"].sum()

    proj_tot = proj[proj.gw.isin(horizon)].groupby("element", as_index=False).agg(
        xp=("exp_points", "sum"))
    merged = proj_tot.merge(act, on="element", how="inner")
    merged["gap"] = merged["total_points"] - merged["xp"]

    res = optimize.initial_squad(proj, horizon)
    picked = set(res.squad.element)
    sel = merged[merged.element.isin(picked)]

    # Compare against the pool the optimiser could plausibly have chosen from:
    # anyone with a comparable projection, not the whole 700-player list.
    cutoff = sel["xp"].min()
    pool_contaminated = merged[merged.xp >= cutoff]

    # THE COMPARATOR MUST EXCLUDE THE SELECTED PLAYERS.
    #
    # `pool_contaminated` is a SUPERSET of `sel` — the 15 chosen players are ~20%
    # of it — so their own negative gap is averaged into the very benchmark they
    # are being measured against. That pulls the comparator down and understates
    # the bias by about a quarter (measured: -5.04 reported vs -6.33 actual).
    #
    # Both are reported below so the contaminated figure cannot be quietly
    # reintroduced by someone who thinks the two are equivalent.
    pool = pool_contaminated[~pool_contaminated.element.isin(picked)]

    return {
        "selected_n": int(len(sel)),
        "selected_mean_projected": round(float(sel.xp.mean()), 3),
        "selected_mean_actual": round(float(sel.total_points.mean()), 3),
        "selected_mean_gap": round(float(sel.gap.mean()), 3),
        "pool_excl_selected_n": int(len(pool)),
        "pool_excl_selected_mean_gap": round(float(pool.gap.mean()), 3),
        "bias_vs_pool": round(float(sel.gap.mean() - pool.gap.mean()), 3),
        "illusion_over_squad_pts": round(float((sel.gap.mean() - pool.gap.mean()) * 15), 1),
        # Retained only to document the error, never to be quoted:
        "_contaminated_pool_n": int(len(pool_contaminated)),
        "_contaminated_pool_mean_gap": round(float(pool_contaminated.gap.mean()), 3),
        "_contaminated_bias": round(
            float(sel.gap.mean() - pool_contaminated.gap.mean()), 3),
    }


# ---------------------------------------------------------------- C. correlation


def test_correlation() -> dict:
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        """SELECT g.gw, g.element, p.team_id, p.position, g.total_points, g.minutes,
                  g.clean_sheets
           FROM player_gw g JOIN players p ON p.season=g.season AND p.element=g.element
           WHERE g.season=? AND g.minutes>=60""",
        conn, params=(SEASON,))

    same_team, diff_team = [], []
    for gw, g in df.groupby("gw"):
        for team, t in g.groupby("team_id"):
            pts = t["total_points"].values
            if len(pts) < 4:
                continue
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    same_team.append((pts[i], pts[j]))
        # random cross-team pairs from the same gameweek as a control
        s = g.sample(min(len(g), 60), random_state=int(gw))
        v = s["total_points"].values
        tid = s["team_id"].values
        for i in range(0, len(v) - 1, 2):
            if tid[i] != tid[i + 1]:
                diff_team.append((v[i], v[i + 1]))

    st = np.array(same_team)
    dt = np.array(diff_team)
    r_same = float(np.corrcoef(st[:, 0], st[:, 1])[0, 1])
    r_diff = float(np.corrcoef(dt[:, 0], dt[:, 1])[0, 1])

    # Defenders specifically: they share the clean sheet outright.
    defs = df[df.position == "DEF"]
    dsame = []
    for gw, g in defs.groupby("gw"):
        for team, t in g.groupby("team_id"):
            pts = t["total_points"].values
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dsame.append((pts[i], pts[j]))
    ds = np.array(dsame)
    r_def = float(np.corrcoef(ds[:, 0], ds[:, 1])[0, 1])

    # What does ignoring correlation cost in variance terms? For 3 players from
    # one club, true var = sum(var) + 2*sum(cov). Independence assumes cov = 0.
    sd = float(df["total_points"].std())
    var_indep = 3 * sd ** 2
    var_true_def = 3 * sd ** 2 + 6 * r_def * sd ** 2
    return {
        "corr_same_team_all_positions": round(r_same, 4),
        "corr_different_team_control": round(r_diff, 4),
        "corr_same_team_defenders": round(r_def, 4),
        "sd_single_player": round(sd, 2),
        "sd_3_defenders_assuming_independence": round(float(np.sqrt(var_indep)), 2),
        "sd_3_defenders_actual": round(float(np.sqrt(var_true_def)), 2),
        "variance_understatement_pct": round(
            100 * (np.sqrt(var_true_def) / np.sqrt(var_indep) - 1), 1),
    }


if __name__ == "__main__":
    HZ = list(range(20, 26))
    proj = projections_for(20, 6)

    print("=" * 66)
    print("A. STABILITY — re-solve under projection noise")
    print("=" * 66)
    for k, v in test_stability(proj, HZ).items():
        print(f"  {k:<38} {v}")

    print()
    print("=" * 66)
    print("B. SELECTION BIAS — projected vs realised, selected vs comparable pool")
    print("=" * 66)
    for k, v in test_selection_bias(proj, HZ).items():
        label = k.lstrip("_")
        prefix = "  [wrong, kept for the record] " if k.startswith("_") else "  "
        print(f"{prefix}{label:<38} {v}" if k.startswith("_") else f"  {label:<38} {v}")

    print()
    print("=" * 66)
    print("C. CORRELATION — what the independence assumption hides")
    print("=" * 66)
    for k, v in test_correlation().items():
        print(f"  {k:<38} {v}")
