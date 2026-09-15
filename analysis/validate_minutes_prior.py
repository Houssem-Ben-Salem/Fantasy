#!/usr/bin/env python3
"""
Out-of-sample test for a Beta prior in models.minutes_model.

WRITTEN BEFORE THE FIX, so the fix cannot be tuned to it.

THE QUESTION
------------
minutes_model computes start_rate from current-season rows only once any exist.
With few gameweeks that is near-binary and discards last season. Does blending
current-season evidence with the prior-season rate predict starts better?

    start_rate = (starts_this_season + k * prior_rate) / (games_this_season + k)

k = 0 reproduces today's in-season branch; k -> inf is prior-only.

METHOD — no look-ahead
----------------------
Replay 2025-26. For each target gameweek T, use ONLY 2024-25 (as the prior,
joined on the stable `code`) and 2025-26 gameweeks before T. Predict
P(60+ minutes in T). Score with log-loss and Brier against what happened.

Availability multipliers are deliberately excluded: players.status is an
end-of-season snapshot and would leak (same reason backtest._minutes_as_of
skips it). This isolates the start_rate component — the part being changed.

DENOMINATOR, STATED
-------------------
Scored on two populations, because hundreds of never-used players make any
predictor look good: all players with a row in T, and squad-relevant players
(>= 10 starts in 2024-25). LIMITATIONS.md exists because of this trap.

TWO-STAGE, SO k IS NOT CHOSEN AND GRADED ON THE SAME DATA
--------------------------------------------------------
Stage 1: sweep k on 2025-26 GW2-8. Stage 2 (confirm_2026_27): score the chosen
k on 2026-27 GW2-4 with live-API actuals — a different season, and the exact
regime that broke.
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = str(ROOT / "data" / "fpl.db")
EPS = 1e-4
NO_HISTORY_PRIOR = 0.35      # models.minutes_model's pre-season fallback
LOOKBACK = 6                 # models.minutes_model default
KS = [0, 1, 2, 3, 4, 6, 8, 12, 20]


def gw_level(conn, season: str) -> pd.DataFrame:
    """One row per (element, gw): minutes summed, started = any fixture 60+."""
    g = pd.read_sql(
        """SELECT g.element, p.code, g.gw, g.minutes
           FROM player_gw g JOIN players p ON p.season=g.season AND p.element=g.element
           WHERE g.season=?""", conn, params=(season,))
    return (g.groupby(["element", "code", "gw"], as_index=False)
             .agg(minutes=("minutes", "sum"), mx=("minutes", "max"))
             .assign(started=lambda d: (d.mx >= 60).astype(int)))


def prior_rates(prev: pd.DataFrame) -> pd.DataFrame:
    return (prev.groupby("code", as_index=False)
                .agg(prior_rate=("started", "mean"), prior_starts=("started", "sum")))


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def predict(cur_before: pd.DataFrame, target: pd.DataFrame, prior: pd.DataFrame,
            k: float, fallback_no_prior_k0: float = 0.25) -> pd.Series:
    win = cur_before.groupby("code", as_index=False).agg(
        s=("started", "sum"), n=("started", "size"))
    d = target[["code"]].merge(win, on="code", how="left").merge(prior, on="code", how="left")
    d[["s", "n"]] = d[["s", "n"]].fillna(0)
    r0 = d["prior_rate"].fillna(NO_HISTORY_PRIOR)
    if k == 0:
        # today's in-season branch: current-season rate, 0.25 if no current rows
        rate = np.where(d.n > 0, d.s / d.n.replace(0, np.nan), fallback_no_prior_k0)
    elif np.isinf(k):
        rate = r0
    else:
        rate = (d.s + k * r0) / (d.n + k)
    return pd.Series(np.asarray(rate, float), index=target.index)


def stage1(conn) -> pd.DataFrame:
    prev = gw_level(conn, "2024-25")
    cur = gw_level(conn, "2025-26")
    prior = prior_rates(prev)
    relevant = set(prior[prior.prior_starts >= 10].code)
    rows = []
    for T in range(2, 9):
        before = cur[(cur.gw < T) & (cur.gw >= T - LOOKBACK)]
        tgt = cur[cur.gw == T].reset_index(drop=True)
        y = tgt.started.values
        for k in KS + [np.inf]:
            p = predict(before, tgt, prior, k).values
            rel = tgt.code.isin(relevant).values
            rows.append(dict(T=T, k=k, ll_all=logloss(p, y), brier_all=float(np.mean((p - y) ** 2)),
                             ll_rel=logloss(p[rel], y[rel]),
                             brier_rel=float(np.mean((p[rel] - y[rel]) ** 2)),
                             binary_share=float(np.mean((p == 0) | (p == 1)))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    conn = sqlite3.connect(DB)
    r = stage1(conn)
    s = (r.groupby("k", as_index=False)
          .agg(ll_all=("ll_all", "mean"), brier_all=("brier_all", "mean"),
               ll_rel=("ll_rel", "mean"), brier_rel=("brier_rel", "mean"),
               binary=("binary_share", "mean")))
    s["k"] = s["k"].map(lambda v: "inf (prior only)" if np.isinf(v)
                        else ("0 (TODAY)" if v == 0 else str(int(v))))
    print("STAGE 1 — 2025-26, target GW2-8, mean over gameweeks (lower is better)\n")
    print(s.round(4).to_string(index=False))
    best = r[np.isfinite(r.k) & (r.k > 0)].groupby("k").ll_rel.mean().idxmin()
    base = r[r.k == 0].ll_rel.mean()
    bestll = r[r.k == best].ll_rel.mean()
    print(f"\nbest finite k on squad-relevant log-loss: k={best:g}")
    print(f"squad-relevant log-loss  today {base:.4f}  ->  k={best:g} {bestll:.4f}"
          f"   ({100 * (base - bestll) / base:+.1f}%)")
    wins = sum(r[(r.k == best) & (r["T"] == T)].ll_rel.iloc[0] <
               r[(r.k == 0) & (r["T"] == T)].ll_rel.iloc[0] for T in range(2, 9))
    print(f"beats today's branch in {wins}/7 target gameweeks")


# ------------------------------------------------------------------ stage 2

CHOSEN_K = 0.75   # frozen from stage 1 by rule: min Brier, must also beat k=0 on log-loss


def live_gw_level(conn, gws) -> pd.DataFrame:
    """2026-27 actuals from the live API — the mirror only has GW1."""
    from fplbrain import sources
    codes = pd.read_sql("SELECT element, code FROM players WHERE season='2026-27'", conn)
    rows = []
    for gw in gws:
        f = sources.live_gameweek(gw)
        if f is None:
            raise RuntimeError(f"live API unavailable for GW{gw}")
        for e in f.data["elements"]:
            m = e["stats"]["minutes"]
            rows.append(dict(element=e["id"], gw=gw, minutes=m, started=int(m >= 60)))
    return pd.DataFrame(rows).merge(codes, on="element", how="inner")


def confirm_2026_27(conn, k: float = CHOSEN_K) -> pd.DataFrame:
    prior = prior_rates(gw_level(conn, "2025-26"))
    relevant = set(prior[prior.prior_starts >= 10].code)
    cur = live_gw_level(conn, [1, 2, 3, 4])
    rows = []
    for T in (2, 3, 4):
        before = cur[cur.gw < T]
        tgt = cur[cur.gw == T].reset_index(drop=True)
        y = tgt.started.values
        m = tgt.code.isin(relevant).values
        for label, kk in (("today k=0", 0), (f"prior k={k:g}", k)):
            p = predict(before, tgt, prior, kk).values
            rows.append(dict(T=T, method=label, n_rel=int(m.sum()),
                             brier_rel=float(np.mean((p[m] - y[m]) ** 2)),
                             ll_rel=logloss(p[m], y[m]),
                             brier_all=float(np.mean((p - y) ** 2))))
    return pd.DataFrame(rows)
