"""
Do-nothing baselines for start prediction.

WHY THIS EXISTS
---------------
LIMITATIONS.md corrects the same error twice: Spearman read against 1.0 when the
achievable ceiling is 0.77, and Brier read against 0.5 when the reducible portion
is Var(p). Both times the mistake was quoting a score with no baseline attached.

The roadmap then recommended adding predicted lineups on the strength of a
self-reported "87% accuracy" figure with no baseline attached. Third instance of
the identical error, in the document's own top recommendation.

So the baseline lives in code rather than in prose. Persistence — "a player
starts iff they started last week" — costs nothing, needs no external source,
and is already implicit in the minutes model's rolling start rate.

THE DENOMINATOR IS THE WHOLE POINT
----------------------------------
Accuracy over all player-gameweeks is inflated by easy negatives: most of the
~700 players never start, so predicting "no" scores well and means nothing. We
therefore report several populations and always alongside the majority-class
rate, so no single number can be quoted without its baseline.

THE METRIC THAT MATTERS IS NOT OVERALL ACCURACY
-----------------------------------------------
Persistence errors are asymmetric. It is good at who KEEPS a place and bad at who
ROTATES IN. A lineup source is only worth integrating if it beats persistence on
P(come in | benched), which is where the remaining error actually lives.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

DB = "data/fpl.db"
SEASON = "2025-26"
START_MINUTES = 60          # matches models.minutes_model's definition of a start


def _frame(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT element, gw, minutes FROM player_gw WHERE season=? ORDER BY element, gw",
        conn, params=(season,))
    df["started"] = (df["minutes"] >= START_MINUTES).astype(int)
    df["prev"] = df.groupby("element")["started"].shift(1)
    df["season_starts"] = df.groupby("element")["started"].transform("sum")
    # Did they start in any of the previous three? Defines the set a manager
    # would realistically pick from, as opposed to the whole player list.
    df["prev3"] = (df.groupby("element")["started"]
                     .transform(lambda s: s.shift(1).rolling(3).sum()))
    return df.dropna(subset=["prev"]).assign(prev=lambda d: d["prev"].astype(int))


def persistence_baseline(db: str = DB, season: str = SEASON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (accuracy_by_population, churn_by_population).

    accuracy: persistence vs the majority-class rate, per population. The `lift`
    column is the only honest reading — raw accuracy is denominator-dependent.

    churn: where persistence actually fails. P(come in | benched) is the number a
    lineup source has to beat to be worth integrating.
    """
    with sqlite3.connect(db) as conn:
        d = _frame(conn, season)

    pops = {
        "all player-gameweeks": d,
        "started >=1 all season": d[d.season_starts >= 1],
        "squad-relevant (>=10 starts)": d[d.season_starts >= 10],
        "regulars (>=20 starts)": d[d.season_starts >= 20],
        "choice set (>=1 of previous 3)": d.dropna(subset=["prev3"]).query("prev3 >= 1"),
    }

    acc_rows, churn_rows = [], []
    for label, sub in pops.items():
        if len(sub) < 50:
            continue
        rate = sub["started"].mean()
        acc = (sub["started"] == sub["prev"]).mean()
        majority = max(rate, 1 - rate)
        acc_rows.append({
            "population": label, "n": len(sub),
            "start_rate": round(float(rate), 3),
            "persistence": round(float(acc), 3),
            "majority_class": round(float(majority), 3),
            "lift": round(float(acc - majority), 3),
        })
        started_prev = sub[sub.prev == 1]
        benched_prev = sub[sub.prev == 0]
        churn_rows.append({
            "population": label,
            "flip_rate": round(float((sub.started != sub.prev).mean()), 3),
            "p_drop_given_started": round(float(1 - started_prev.started.mean()), 3)
            if len(started_prev) else None,
            "p_come_in_given_benched": round(float(benched_prev.started.mean()), 3)
            if len(benched_prev) else None,
        })
    return pd.DataFrame(acc_rows), pd.DataFrame(churn_rows)


if __name__ == "__main__":
    acc, churn = persistence_baseline()
    print("PERSISTENCE BASELINE — 'starts iff started last week'")
    print(f"season {SEASON}, start = {START_MINUTES}+ minutes\n")
    print(acc.to_string(index=False))
    print("\nWHERE PERSISTENCE FAILS — the bar a lineup source must clear\n")
    print(churn.to_string(index=False))
    print("\nA source that only matches persistence on overall accuracy adds nothing.")
    print("The column that decides integration is p_come_in_given_benched.")
