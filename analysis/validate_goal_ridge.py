#!/usr/bin/env python3
"""
Out-of-sample test for shrinking Dixon-Coles team ratings.

WRITTEN BEFORE THE FIX, so the fix cannot be tuned to it.

THE DEFECT
----------
DixonColes.fit is raw MLE with parameters bounded at +/-3. A promoted club with
two matches has a nearly flat likelihood; the optimiser walks to the bound and
exp() of it is not an estimate. 2026-27: COV attack and HUL defence pinned at
-3.000, putting four GBP4.0m Hull defenders in the top five projected players.

CANDIDATE FIXES
---------------
  none          today's raw MLE
  ridge_raw     + lambda * (sum att^2 + sum dfn^2)
  ridge_centred + lambda * (sum att^2 + sum (dfn - mean dfn)^2)

The two ridge forms differ for a reason worth testing rather than asserting.
Attack is constrained to sum to zero, so shrinking it toward 0 is shrinking
toward league average. Defence is NOT centred: under this parameterisation it
carries the league's overall scoring rate. Shrinking defence toward 0 should
therefore bias every team's expected goals, not just stabilise thin clubs.

Also tested: including every current-season team in the fit even with zero
matches. Unpenalised that is unidentified; penalised, a zero-match team settles
at the prior (league average) — replacing expected_goals' fallback of the
weakest observed attack and defence, which is the GW1 symptom of the same defect.

TWO SCORES, BOTH REQUIRED
-------------------------
A. Held-out match log-likelihood (the fit).
B. The decision layer: does any team rating sit at the +/-3 bound, and do the
   promoted clubs' implied projections look like real football?
The failure was in the decision layer, so A alone is not sufficient to ship.

TWO-STAGE
---------
lambda and the form are chosen on 2025-26 early season, then confirmed with
both frozen on 2026-27 GW2-4 — including against the Pinnacle market line for
GW3, an external reference the model never saw.
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = str(ROOT / "data" / "fpl.db")
XI = 0.0018
BOUND = 3.0


def matches(conn, seasons, before=None) -> pd.DataFrame:
    fr = []
    for s in seasons:
        q = """SELECT f.season, f.gw, f.kickoff_time, f.home_score, f.away_score,
                      th.short_name h, ta.short_name a
               FROM fixtures f
               JOIN teams th ON th.season=f.season AND th.team_id=f.home_team
               JOIN teams ta ON ta.season=f.season AND ta.team_id=f.away_team
               WHERE f.season=? AND f.finished=1"""
        fr.append(pd.read_sql(q, conn, params=(s,)))
    d = pd.concat(fr, ignore_index=True)
    d["kickoff"] = pd.to_datetime(d.kickoff_time, errors="coerce", utc=True)
    return d.dropna(subset=["kickoff", "home_score", "away_score"])


def _tau(x, y, lh, la, rho):
    out = np.ones_like(lh, dtype=float)
    m00 = (x == 0) & (y == 0); m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0); m11 = (x == 1) & (y == 1)
    out[m00] = 1 - lh[m00] * la[m00] * rho
    out[m01] = 1 + lh[m01] * rho
    out[m10] = 1 + la[m10] * rho
    out[m11] = 1 - rho
    return np.clip(out, 1e-9, None)


def fit(train: pd.DataFrame, lam: float = 0.0, form: str = "none",
        extra_teams=()) -> dict:
    teams = sorted(set(train.h) | set(train.a) | set(extra_teams))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    age = (train.kickoff.max() - train.kickoff).dt.total_seconds().values / 86400
    w = np.exp(-XI * age)
    h = train.h.map(idx).values; a = train.a.map(idx).values
    hs = train.home_score.values.astype(int); as_ = train.away_score.values.astype(int)

    def nll(p):
        att = np.concatenate([p[:n - 1], [-p[:n - 1].sum()]])
        dfn = p[n - 1:2 * n - 1]
        hadv, rho = p[-2], p[-1]
        lh = np.clip(np.exp(att[h] + dfn[a] + hadv), 1e-6, 12)
        la = np.clip(np.exp(att[a] + dfn[h]), 1e-6, 12)
        ll = poisson.logpmf(hs, lh) + poisson.logpmf(as_, la) + np.log(_tau(hs, as_, lh, la, rho))
        pen = 0.0
        if form == "ridge_raw":
            pen = lam * (np.sum(att ** 2) + np.sum(dfn ** 2))
        elif form == "ridge_centred":
            pen = lam * (np.sum(att ** 2) + np.sum((dfn - dfn.mean()) ** 2))
        return -np.sum(w * ll) + pen

    x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25], [-0.05]])
    bounds = [(-BOUND, BOUND)] * (2 * n - 1) + [(-1, 1), (-0.2, 0.2)]
    r = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 800, "ftol": 1e-8})
    att = np.concatenate([r.x[:n - 1], [-r.x[:n - 1].sum()]])
    dfn = r.x[n - 1:2 * n - 1]
    return {"attack": dict(zip(teams, att)), "defence": dict(zip(teams, dfn)),
            "home_adv": r.x[-2], "rho": r.x[-1], "form": form}


def predict(m: dict, h: str, a: str) -> tuple[float, float]:
    """Unknown teams: today's rule (weakest observed) for the unpenalised fit."""
    wa, wd = min(m["attack"].values()), max(m["defence"].values())
    ah = m["attack"].get(h, wa); dh = m["defence"].get(h, wd)
    aa = m["attack"].get(a, wa); da = m["defence"].get(a, wd)
    return float(np.exp(ah + da + m["home_adv"])), float(np.exp(aa + dh))


def heldout_ll(m, test: pd.DataFrame) -> float:
    lls = []
    for r in test.itertuples():
        lh, la = predict(m, r.h, r.a)
        x, y = np.array([int(r.home_score)]), np.array([int(r.away_score)])
        ll = (poisson.logpmf(x, lh) + poisson.logpmf(y, la)
              + np.log(_tau(x, y, np.array([lh]), np.array([la]), m["rho"])))
        lls.append(float(ll[0]))
    return float(np.mean(lls))


def at_bound(m) -> int:
    vals = list(m["attack"].values()) + list(m["defence"].values())
    return int(sum(abs(v) >= BOUND - 1e-3 for v in vals))


CONFIGS = [("none", 0.0)] + [(f, l) for f in ("ridge_raw", "ridge_centred")
                             for l in (0.5, 1, 2, 4, 8, 16)]


def stage1(conn):
    hist = matches(conn, ["2024-25"])
    cur = matches(conn, ["2025-26"])
    teams_cur = set(pd.read_sql("SELECT short_name FROM teams WHERE season='2025-26'", conn).short_name)
    rows = []
    for T in range(2, 11):
        train = pd.concat([hist, cur[cur.gw < T]], ignore_index=True)
        test = cur[cur.gw == T]
        if test.empty:
            continue
        for form, lam in CONFIGS:
            extra = teams_cur if form != "none" else ()
            m = fit(train, lam, form, extra)
            rows.append(dict(T=T, form=form, lam=lam, ll=heldout_ll(m, test),
                             bound=at_bound(m),
                             mean_pred=np.mean([sum(predict(m, r.h, r.a)) for r in test.itertuples()]),
                             mean_actual=float((test.home_score + test.away_score).mean())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    conn = sqlite3.connect(DB)
    r = stage1(conn)
    s = (r.groupby(["form", "lam"], as_index=False)
          .agg(heldout_ll=("ll", "mean"), ratings_at_bound=("bound", "sum"),
               mean_pred_goals=("mean_pred", "mean"), mean_actual_goals=("mean_actual", "mean")))
    print("STAGE 1 — 2025-26, fit on 2024-25 + earlier 2025-26, score GW2-10 (higher LL is better)\n")
    print(s.round(4).to_string(index=False))
    base = s[s.form == "none"].heldout_ll.iloc[0]
    ok = s[(s.form != "none") & (s.ratings_at_bound == 0)]
    best = ok.loc[ok.heldout_ll.idxmax()]
    print(f"\nbaseline (none) held-out LL {base:.4f}; ratings pinned at bound across refits: "
          f"{int(s[s.form=='none'].ratings_at_bound.iloc[0])}")
    print(f"best config with ZERO pinned ratings: {best.form} lambda={best.lam:g}  LL {best.heldout_ll:.4f}  "
          f"({best.heldout_ll - base:+.4f} per match)")
    per = r[(r.form == best.form) & (r.lam == best.lam)].set_index("T").ll
    per0 = r[r.form == "none"].set_index("T").ll
    print(f"wins vs none: {int((per > per0).sum())}/{len(per)} gameweeks")


# ------------------------------------------------------------------ stage 2

CHOSEN = ("ridge_raw", 2.0)
# Frozen from stage 1 by rule: among configs with zero pinned ratings, most wins
# on gameweeks where the baseline was not itself pinned (a catastrophic GW2
# otherwise dominates every average), ties broken by mean held-out LL gain,
# goal-level bias within 2%. ridge_raw 1 and 2 both won 6/8; 2 had higher gain.

PINNACLE_GW3 = [("IPS", 1.13, "LIV", 2.28), ("NEW", 1.77, "BOU", 1.39),
                ("BRE", 1.82, "SUN", 0.92), ("BHA", 1.60, "LEE", 1.12),
                ("FUL", 1.48, "CRY", 1.26), ("MCI", 2.94, "COV", 0.72),
                ("NFO", 1.41, "TOT", 1.28), ("HUL", 1.04, "AVL", 1.66),
                ("EVE", 1.30, "MUN", 1.68), ("ARS", 1.88, "CHE", 1.00)]


def stage2(conn, form=CHOSEN[0], lam=CHOSEN[1]):
    hist = matches(conn, ["2024-25", "2025-26"])
    cur = matches(conn, ["2026-27"])
    teams = set(pd.read_sql("SELECT short_name FROM teams WHERE season='2026-27'", conn).short_name)
    rows, models_at = [], {}
    for T in (2, 3, 4):
        train = pd.concat([hist, cur[cur.gw < T]], ignore_index=True)
        test = cur[cur.gw == T]
        for lab, (f, l) in (("today", ("none", 0.0)), ("ridge", (form, lam))):
            m = fit(train, l, f, teams if f != "none" else ())
            models_at[(T, lab)] = m
            rows.append(dict(T=T, model=lab, heldout_ll=heldout_ll(m, test), pinned=at_bound(m)))
    return pd.DataFrame(rows), models_at


if __name__ == "__main__" and "--stage2" in sys.argv:
    conn = sqlite3.connect(DB)
    r, ms = stage2(conn)
    print(f"STAGE 2 — 2026-27, {CHOSEN[0]} lambda={CHOSEN[1]:g} FROZEN\n")
    print(r.round(4).to_string(index=False))
    p = r.pivot(index="T", columns="model", values="heldout_ll")
    print(f"\nmean held-out LL: today {p.today.mean():.4f} -> ridge {p.ridge.mean():.4f} "
          f"({p.ridge.mean() - p.today.mean():+.4f}/match)   wins {int((p.ridge > p.today).sum())}/3")

    print("\nDECISION LAYER — GW3 vs the Pinnacle market line (fit on GW1-2, as the market was)")
    for lab in ("today", "ridge"):
        m = ms[(3, lab)]
        ours, mkt, cs_o, cs_m = [], [], [], []
        for h, hx, a, ax in PINNACLE_GW3:
            lh, la = predict(m, h, a)
            ours += [lh, la]; mkt += [hx, ax]
            cs_o += [np.exp(-la), np.exp(-lh)]; cs_m += [np.exp(-ax), np.exp(-hx)]
        ours, mkt, cs_o, cs_m = map(np.array, (ours, mkt, cs_o, cs_m))
        print(f"  {lab:<6} xG corr {np.corrcoef(ours, mkt)[0,1]:.3f}   CS corr {np.corrcoef(cs_o, cs_m)[0,1]:.3f}"
              f"   CS sd ours {cs_o.std(ddof=1):.3f} vs market {cs_m.std(ddof=1):.3f}")
        for h, hx, a, ax in PINNACLE_GW3:
            if {h, a} & {"COV", "HUL"}:
                lh, la = predict(m, h, a)
                print(f"      {h} v {a}: ours {lh:.2f}-{la:.2f}  market {hx:.2f}-{ax:.2f}   "
                      f"{h} CS ours {np.exp(-la):.0%} mkt {np.exp(-ax):.0%}")
