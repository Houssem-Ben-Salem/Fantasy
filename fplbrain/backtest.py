"""
Walk-forward backtest.

The whole point is to answer one question honestly: does the xP model actually
rank players better than the obvious alternatives?

STRICT NO-LOOK-AHEAD
--------------------
This is where most FPL backtests quietly cheat. Every feature here is rebuilt from
scratch at each gameweek using only rows with gw < target:

  - goal model      refit on matches played before the target gameweek
  - minutes         start rate from prior gameweeks only
  - DefCon rate     hits/starts accumulated before the target gameweek
  - xG90 / xA90     summed from prior gameweeks only

In particular we do NOT touch the `players` table's per-90 columns or `status`,
because those are an end-of-season snapshot. Using them would leak the future and
make the model look far better than it is.

BENCHMARKS
----------
A model is only good relative to what you'd otherwise do:
  fpl_ep      FPL's own published expected points for that gameweek
  form6       mean points over the previous 6 gameweeks
  ppg         season-to-date points per game
  price       current price (proxy for market consensus)

Run:  python -m fplbrain.backtest --db data/fpl.db --season 2025-26
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import models

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

DEFCON_THRESHOLD = models.DEFCON_THRESHOLD


# ---------------------------------------------------------------- as-of features


def _goal_model_as_of(conn, season: str, hist_seasons: list[str], cutoff: str,
                      xi: float = 0.0018) -> models.DixonColes:
    """Fit on every finished match that kicked off strictly before `cutoff`."""
    frames = []
    for s in hist_seasons + [season]:
        q = """
            SELECT f.kickoff_time, f.home_score, f.away_score,
                   th.short_name AS home_team, ta.short_name AS away_team
            FROM fixtures f
            JOIN teams th ON th.season=f.season AND th.team_id=f.home_team
            JOIN teams ta ON ta.season=f.season AND ta.team_id=f.away_team
            WHERE f.season = ? AND f.finished = 1 AND f.kickoff_time < ?
        """
        frames.append(pd.read_sql(q, conn, params=(s, cutoff)))
    df = pd.concat(frames, ignore_index=True)
    df["kickoff"] = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)
    return models.DixonColes(xi=xi).fit(df.dropna(subset=["kickoff"]))


def _minutes_as_of(conn, season: str, gw: int, lookback: int = 6) -> pd.DataFrame:
    """
    P(start) from the previous `lookback` gameweeks only.

    Note we deliberately skip the availability multiplier here: `players.status`
    is an end-of-season snapshot, so applying it would leak. That makes this a
    slightly pessimistic version of the live model, which is the right direction
    for an honest backtest.
    """
    recent = pd.read_sql(
        """SELECT element,
                  AVG(CASE WHEN minutes>=60 THEN 1.0 ELSE 0.0 END) AS start_rate,
                  AVG(minutes) AS avg_minutes, COUNT(*) AS apps
           FROM player_gw
           WHERE season=? AND gw < ? AND gw >= ?
           GROUP BY element""",
        conn, params=(season, gw, max(1, gw - lookback)),
    )
    base = pd.read_sql(
        "SELECT element, code, position FROM players WHERE season=?", conn, params=(season,))
    out = base.merge(recent, on="element", how="left")
    out["start_rate"] = out["start_rate"].fillna(0.15)
    out["avg_minutes"] = out["avg_minutes"].fillna(12.0)
    out["p_start"] = out["start_rate"].clip(0, 1)
    out["p_60"] = (out["p_start"] * 0.92).clip(0, 1)
    out["p_any_minutes"] = (out["p_start"] + (1 - out["p_start"]) * 0.35).clip(0, 1)
    out["exp_minutes"] = out["avg_minutes"].clip(0, 90)
    return out[["element", "code", "position", "p_start", "p_60",
                "p_any_minutes", "exp_minutes"]]


def _defcon_as_of(conn, season: str, hist_seasons: list[str], gw: int) -> pd.DataFrame:
    """
    DefCon hit rate from gameweeks strictly before `gw`.

    Only seasons that actually carry defensive stats are used. Pre-2025-26 rows
    are 100% NULL for tackles/recoveries/CBI because the rule did not exist;
    including them halves every estimate.
    """
    frames = []
    for s in hist_seasons:
        frames.append(pd.read_sql(
            """SELECT g.code, g.defensive_contribution AS dc
               FROM player_gw g
               WHERE g.season=? AND g.minutes>=60 AND g.code IS NOT NULL
                 AND g.defensive_contribution IS NOT NULL""",
            conn, params=(s,)))
    frames.append(pd.read_sql(
        """SELECT g.code, g.defensive_contribution AS dc
           FROM player_gw g
           WHERE g.season=? AND g.gw < ? AND g.minutes>=60 AND g.code IS NOT NULL
             AND g.defensive_contribution IS NOT NULL""",
        conn, params=(season, gw)))
    hist = pd.concat(frames, ignore_index=True)

    cur = pd.read_sql(
        "SELECT element, code, position FROM players WHERE season=? AND position!='GK'",
        conn, params=(season,))

    if hist.empty:
        cur["p_defcon"] = cur["position"].map({"DEF": 0.27, "MID": 0.18, "FWD": 0.01})
        return cur[["element", "code", "p_defcon"]]

    hist = hist.merge(cur[["code", "position"]], on="code", how="inner")
    hist["hit"] = (hist["dc"] >= hist["position"].map(DEFCON_THRESHOLD)).astype(int)

    agg = hist.groupby("code").agg(hits=("hit", "sum"), starts=("hit", "count")).reset_index()
    agg = agg.merge(cur, on="code", how="right")
    agg[["hits", "starts"]] = agg[["hits", "starts"]].fillna(0)
    pooled = hist.groupby("position")["hit"].mean().to_dict()
    agg["prior"] = agg["position"].map(pooled).fillna(0.15)
    agg["p_defcon"] = (agg["hits"] + agg["prior"] * 12.0) / (agg["starts"] + 12.0)
    return agg[["element", "code", "p_defcon"]]


def _rates_as_of(conn, season: str, hist_seasons: list[str], gw: int) -> pd.DataFrame:
    """xG90 / xA90 from prior seasons plus current-season gameweeks before `gw`."""
    frames = []
    for s in hist_seasons:
        frames.append(pd.read_sql(
            """SELECT code, SUM(minutes) minutes, SUM(expected_goals) xg,
                      SUM(expected_assists) xa
               FROM player_gw WHERE season=? AND code IS NOT NULL GROUP BY code""",
            conn, params=(s,)))
    frames.append(pd.read_sql(
        """SELECT code, SUM(minutes) minutes, SUM(expected_goals) xg,
                  SUM(expected_assists) xa
           FROM player_gw WHERE season=? AND gw < ? AND code IS NOT NULL GROUP BY code""",
        conn, params=(season, gw)))
    hist = pd.concat(frames).groupby("code", as_index=False).sum()

    cur = pd.read_sql(
        "SELECT element, code, position FROM players WHERE season=?", conn, params=(season,))
    df = cur.merge(hist, on="code", how="left")
    df["minutes"] = df["minutes"].fillna(0)
    df["raw_xg90"] = 90 * df["xg"] / df["minutes"].replace(0, np.nan)
    df["raw_xa90"] = 90 * df["xa"] / df["minutes"].replace(0, np.nan)

    priors = df.groupby("position")[["raw_xg90", "raw_xa90"]].median()
    df = df.join(priors, on="position", rsuffix="_prior")
    w = df["minutes"] / (df["minutes"] + models.PRIOR_STRENGTH)
    df["xg90"] = w * df["raw_xg90"].fillna(0) + (1 - w) * df["raw_xg90_prior"].fillna(0.05)
    df["xa90"] = w * df["raw_xa90"].fillna(0) + (1 - w) * df["raw_xa90_prior"].fillna(0.05)
    df["is_pen_taker"] = 0  # penalty order is a current snapshot; excluded to avoid leakage
    return df[["element", "code", "position", "xg90", "xa90", "is_pen_taker"]]


# ---------------------------------------------------------------- the loop


def run_backtest(conn, season: str, hist_seasons: list[str], start_gw: int = 8,
                 end_gw: int = 38, refit_every: int = 2,
                 min_minutes: int = 1) -> tuple[pd.DataFrame, dict]:
    """
    Returns (per-row predictions vs actuals, summary metrics).

    start_gw=8 gives the current-season features a few gameweeks to accumulate.
    min_minutes filters the evaluation set: ranking 500 players who didn't play is
    not the task. Set to 1 to score anyone who appeared.
    """
    rows = []
    gm = None
    for gw in range(start_gw, end_gw + 1):
        ko = pd.read_sql(
            "SELECT MIN(kickoff_time) k FROM fixtures WHERE season=? AND gw=?",
            conn, params=(season, gw)).iloc[0]["k"]
        if ko is None:
            continue

        if gm is None or (gw - start_gw) % refit_every == 0:
            gm = _goal_model_as_of(conn, season, hist_seasons, ko)

        mins = _minutes_as_of(conn, season, gw)
        dc = _defcon_as_of(conn, season, hist_seasons, gw)
        rates = _rates_as_of(conn, season, hist_seasons, gw)

        proj = models.project_gameweek(conn, season, gw, gm, mins, dc, rates)

        actual = pd.read_sql(
            """SELECT element, SUM(total_points) actual_points, SUM(minutes) actual_minutes,
                      MAX(clean_sheets) actual_cs, MAX(defensive_contribution) actual_dc_actions,
                      MAX(value) price, AVG(fpl_xp) fpl_ep
               FROM player_gw WHERE season=? AND gw=? GROUP BY element""",
            conn, params=(season, gw))

        # Benchmarks, all computed strictly from prior gameweeks.
        bench = pd.read_sql(
            """SELECT element,
                      AVG(CASE WHEN gw >= ? THEN total_points END) form6,
                      AVG(total_points) ppg
               FROM player_gw WHERE season=? AND gw < ? GROUP BY element""",
            conn, params=(max(1, gw - 6), season, gw))

        m = (proj[["element", "web_name", "position", "team", "exp_points",
                   "p_clean_sheet", "p_defcon", "p_start", "n_fixtures"]]
             .merge(actual, on="element", how="inner")
             .merge(bench, on="element", how="left"))
        prev = pd.read_sql(
            "SELECT element, SUM(minutes) prev_minutes FROM player_gw "
            "WHERE season=? AND gw=? GROUP BY element",
            conn, params=(season, gw - 1))
        m = m.merge(prev, on="element", how="left")
        m["prev_minutes"] = m["prev_minutes"].fillna(0)
        m["gw"] = gw
        rows.append(m)

    raw = pd.concat(rows, ignore_index=True)
    # Leakage detection needs the UNFILTERED frame: the tell is how a predictor
    # treats players who didn't play, and those are exactly the rows we filter out
    # for scoring. Run the check first, then filter.
    df = raw[raw["actual_minutes"] >= min_minutes].copy()
    df[["form6", "ppg"]] = df[["form6", "ppg"]].fillna(0)

    metrics = _evaluate(df, raw)
    return df, metrics


def detect_leakage(df: pd.DataFrame, col: str) -> dict:
    """
    Flag a benchmark that secretly knows the outcome.

    Naive version of this test ("assigns exact zeros to non-players") produces
    false positives: a backward-looking metric like points-per-game legitimately
    equals 0.0 for a player who has never returned anything.

    The sharper test restricts to players who PLAYED IN THE PREVIOUS GAMEWEEK.
    For those players every backward-looking predictor is non-zero, so if a
    predictor still assigns exactly 0.0 to the ones benched this week, it can
    only have learned that after kickoff.

    Real result: FPL's published `xP` in the community CSVs fails this badly.
    """
    need = {"actual_minutes", "prev_minutes"}
    if not need.issubset(df.columns):
        return {"contaminated": False, "reason": "insufficient columns"}

    # Only consider gameweeks where the predictor is actually populated. A column
    # that is entirely zero in a gameweek (a failed upstream scrape) tells us
    # nothing and would otherwise mask the signal.
    live_gws = [g for g, x in df.groupby("gw") if x[col].nunique() > 1]
    sub = df[df[col].notna() & (df["prev_minutes"] > 0) & df["gw"].isin(live_gws)]
    if len(sub) < 100:
        return {"contaminated": False, "reason": "too few eligible rows"}

    played = sub["actual_minutes"] > 0
    if played.all() or (~played).all():
        return {"contaminated": False, "reason": "no variation in participation"}

    zero_non = float((sub.loc[~played, col] == 0).mean())
    zero_play = float((sub.loc[played, col] == 0).mean())
    coverage = len(live_gws) / max(df["gw"].nunique(), 1)

    # Two independent grounds for rejecting a benchmark:
    #  (a) sparse coverage — scored on a different, smaller sample than the model,
    #      so the comparison is not apples-to-apples regardless of leakage;
    #  (b) participation look-ahead — assigns exact zeros to players it could not
    #      have known would be benched, at a rate far above players who featured.
    sparse = coverage < 0.5
    lookahead = zero_non > 0.10 and zero_play < 0.02 and zero_non > 5 * max(zero_play, 1e-6)
    reasons = []
    if sparse:
        reasons.append(f"only populated in {len(live_gws)}/{df['gw'].nunique()} gameweeks")
    if lookahead:
        reasons.append("assigns exact zeros to players it could not have known "
                       "would be benched")
    return {
        "contaminated": sparse or lookahead,
        "coverage": round(coverage, 3),
        "n_eligible": int(len(sub)),
        "gameweeks_populated": len(live_gws),
        "zero_rate_benched_this_gw": round(zero_non, 4),
        "zero_rate_played_this_gw": round(zero_play, 4),
        "reason": "; ".join(reasons) if reasons else "usable benchmark",
    }


def _evaluate(df: pd.DataFrame, raw: pd.DataFrame | None = None) -> dict:
    preds = {"model_xp": "exp_points", "fpl_ep": "fpl_ep",
             "form6": "form6", "ppg": "ppg", "price": "price"}
    out: dict = {"n_rows": int(len(df)), "n_gameweeks": int(df.gw.nunique()),
                 "predictors": {}}

    out["leakage_checks"] = {}
    for label, col in preds.items():
        if label == "model_xp":
            continue
        out["leakage_checks"][label] = detect_leakage(
            raw if raw is not None else df, col)

    for label, col in preds.items():
        sub = df[df[col].notna()]
        if sub.empty:
            continue
        # Rank correlation computed per gameweek then averaged — the task is
        # ranking players within a gameweek, not across the whole season.
        per_gw = []
        for _, g in sub.groupby("gw"):
            if len(g) > 10 and g[col].nunique() > 1:
                per_gw.append(spearmanr(g[col], g["actual_points"]).statistic)
        entry = {
            "spearman_mean": round(float(np.mean(per_gw)), 4) if per_gw else None,
            "spearman_std": round(float(np.std(per_gw)), 4) if per_gw else None,
            "mae": round(float(np.abs(sub[col] - sub["actual_points"]).mean()), 3),
            "rmse": round(float(np.sqrt(((sub[col] - sub["actual_points"]) ** 2).mean())), 3),
            "n_gameweeks_scored": len(per_gw),
            "contaminated": out["leakage_checks"].get(label, {}).get("contaminated", False),
        }
        # Precision at the top: mean actual points of each gameweek's top-20 by this predictor.
        tops = [g.nlargest(20, col)["actual_points"].mean() for _, g in sub.groupby("gw")
                if len(g) >= 20]
        entry["top20_mean_actual"] = round(float(np.mean(tops)), 3) if tops else None
        out["predictors"][label] = entry

    out["field_mean_actual"] = round(float(df["actual_points"].mean()), 3)

    # Calibration of the two probability outputs. Brier score: lower is better,
    # and we compare against always predicting the base rate.
    cs = df[df["position"].isin(["GK", "DEF"]) & (df["actual_minutes"] >= 60)]
    if len(cs) > 50:
        base = cs["actual_cs"].mean()
        out["clean_sheet"] = {
            "n": int(len(cs)),
            "predicted_mean": round(float(cs["p_clean_sheet"].mean()), 4),
            "actual_rate": round(float(base), 4),
            "brier": round(float(((cs["p_clean_sheet"] - cs["actual_cs"]) ** 2).mean()), 4),
            "brier_baseline": round(float(((base - cs["actual_cs"]) ** 2).mean()), 4),
        }

    dc = df[(df["position"] != "GK") & (df["actual_minutes"] >= 60)].copy()
    if len(dc) > 50 and dc["actual_dc_actions"].notna().any():
        # defensive_contribution is the position-correct action count (CBIT for
        # DEF, CBIRT for MID/FWD), NOT the points awarded. Threshold it properly.
        dc = dc[dc["actual_dc_actions"].notna()].copy()
        dc["hit"] = (dc["actual_dc_actions"] >= dc["position"].map(DEFCON_THRESHOLD)).astype(int)
        base = dc["hit"].mean()
        out["defcon"] = {
            "n": int(len(dc)),
            "predicted_mean": round(float(dc["p_defcon"].mean()), 4),
            "actual_rate": round(float(base), 4),
            "brier": round(float(((dc["p_defcon"] - dc["hit"]) ** 2).mean()), 4),
            "brier_baseline": round(float(((base - dc["hit"]) ** 2).mean()), 4),
        }

    by_pos = {}
    for pos, g in df.groupby("position"):
        per_gw = [spearmanr(x["exp_points"], x["actual_points"]).statistic
                  for _, x in g.groupby("gw") if len(x) > 10 and x["exp_points"].nunique() > 1]
        if per_gw:
            by_pos[pos] = round(float(np.mean(per_gw)), 4)
    out["spearman_by_position"] = by_pos
    return out


def report(metrics: dict) -> str:
    lines = [f"Backtest: {metrics['n_rows']} player-gameweeks over "
             f"{metrics['n_gameweeks']} gameweeks", ""]
    lines.append(f"{'predictor':<12} {'spearman':>9} {'MAE':>7} {'top20':>7} "
                 f"{'GWs':>5}  {'status':<28}")
    lines.append("-" * 72)
    for k, v in sorted(metrics["predictors"].items(),
                       key=lambda kv: -(kv[1]["spearman_mean"] or -1)):
        why = metrics.get("leakage_checks", {}).get(k, {}).get("reason", "")
        flag = f"EXCLUDED: {why[:40]}" if v.get("contaminated") else ""
        lines.append(f"{k:<12} {v['spearman_mean']:>9} {v['mae']:>7} "
                     f"{str(v['top20_mean_actual']):>7} {v.get('n_gameweeks_scored',0):>5}  {flag:<28}")
    valid = {k: v for k, v in metrics["predictors"].items() if not v.get("contaminated")}
    if valid:
        best = max(valid.items(), key=lambda kv: kv[1]["spearman_mean"] or -1)
        lines.append(f"\nbest uncontaminated predictor: {best[0]} "
                     f"(spearman {best[1]['spearman_mean']})")
    lines.append(f"\nfield mean actual points: {metrics['field_mean_actual']}")

    if "clean_sheet" in metrics:
        c = metrics["clean_sheet"]
        verdict = "better" if c["brier"] < c["brier_baseline"] else "WORSE"
        lines.append(f"\nclean sheet: predicted {c['predicted_mean']} vs actual "
                     f"{c['actual_rate']} | Brier {c['brier']} vs base "
                     f"{c['brier_baseline']} ({verdict} than base rate)")
    if "defcon" in metrics:
        d = metrics["defcon"]
        verdict = "better" if d["brier"] < d["brier_baseline"] else "WORSE"
        lines.append(f"defcon:      predicted {d['predicted_mean']} vs actual "
                     f"{d['actual_rate']} | Brier {d['brier']} vs base "
                     f"{d['brier_baseline']} ({verdict} than base rate)")
    lines.append(f"\nspearman by position: {metrics['spearman_by_position']}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/fpl.db")
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--history", nargs="*", default=["2024-25"])
    ap.add_argument("--start-gw", type=int, default=8)
    ap.add_argument("--end-gw", type=int, default=38)
    ap.add_argument("--min-minutes", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    conn = sqlite3.connect(a.db)
    t0 = datetime.now()
    df, metrics = run_backtest(conn, a.season, a.history, a.start_gw, a.end_gw,
                               min_minutes=a.min_minutes)
    print(report(metrics))
    print(f"\nelapsed {(datetime.now() - t0).total_seconds():.1f}s")
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"wrote {a.out}")
