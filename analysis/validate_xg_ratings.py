#!/usr/bin/env python3
"""
Out-of-sample test: should team ratings learn from xG as well as goals?

WRITTEN BEFORE THE CHOICE, so w cannot be tuned to the confirmation.

THE DEFECT
----------
Dixon-Coles rated teams on goals alone. Over a handful of matches goals are
mostly luck: 2026-27 Hull conceded 4 from 7.40 xG against, were rated the
second-best defence in the league (-0.305 after four matches), and the model
put Newcastle at home to them on 1.07 expected goals against Pinnacle's 2.02 —
the largest market disagreement of GW5, and again the largest of GW6.
check_promoted_prior.py found the promoted-club ratings flatter early; xG is
the obvious, less noisy evidence that might not.

THE CANDIDATE
-------------
DixonColes.fit maximises (1 - w) * goals LL + w * quasi-Poisson LL of team xG
(summed from player_gw.expected_goals). w = 0 is today's model exactly
(tests/test_xg_ratings.py pins that); ridge, decay and roster are production's.

THE SCORE — goals, not xG
-------------------------
Held-out Poisson log-likelihood of the ACTUAL goals in each match, from ratings
fitted strictly before its gameweek's first kickoff (refit every 2 gameweeks,
as the backtest does). xG is allowed to help only by predicting goals better.

TWO-STAGE
---------
Stage 1 picks w on 2025-26 GW8-38. CHOSEN_W is frozen in this file BEFORE
stage 2 runs on 2026-27 GW1-5 (`--stage2`), which reports the gain with a
bootstrap over fixtures, then the decision layer: agreement with the Pinnacle
lines for GW5 and GW6, Hull's rating, and the goal-level bias the ridge
already costs.
"""
from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fplbrain import models  # noqa: E402

DB = ROOT / "data" / "fpl.db"
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
STAGE1 = ("2025-26", ["2024-25"], range(8, 39))
STAGE2 = ("2026-27", ["2024-25", "2025-26"], range(1, 6))
REFIT_EVERY = 2
N_BOOT = 4000

# Pinnacle lines as published in the newsletters: (home, away, home xG, away xG,
# home CS, away CS). External references the model never saw.
PINNACLE = {
    5: [("BRE", "CHE", 1.62, 1.75, .174, .199), ("TOT", "AVL", 1.64, 1.15, .316, .194),
        ("BHA", "ARS", 1.03, 1.82, .162, .358), ("EVE", "IPS", 1.82, 1.07, .343, .162),
        ("NEW", "HUL", 2.02, 1.04, .355, .132), ("NFO", "COV", 1.86, 0.96, .384, .156),
        ("LEE", "CRY", 1.78, 1.07, .343, .168), ("MCI", "SUN", 2.38, 0.72, .485, .093),
        ("BOU", "LIV", 1.54, 1.84, .159, .215), ("FUL", "MUN", 1.35, 1.81, .164, .259)],
    6: [("ARS", "LEE", 2.12, 0.69, .500, .120), ("AVL", "BRE", 1.46, 1.48, .228, .233),
        ("CHE", "BOU", 2.08, 1.24, .288, .125), ("IPS", "FUL", 1.38, 1.50, .223, .252),
        ("SUN", "BHA", 1.31, 1.51, .222, .270), ("MUN", "TOT", 2.07, 1.24, .290, .126),
        ("CRY", "NFO", 1.31, 1.33, .265, .269), ("HUL", "EVE", 1.07, 1.52, .218, .343),
        ("LIV", "MCI", 1.58, 1.60, .203, .206), ("COV", "NEW", 1.30, 1.71, .181, .273)],
}


def matches(conn, seasons) -> pd.DataFrame:
    frames = [pd.read_sql(models.MATCHES_WITH_XG_SQL, conn, params=(s, s)).assign(season=s)
              for s in seasons]
    d = pd.concat(frames, ignore_index=True)
    d["kickoff"] = pd.to_datetime(d["kickoff_time"], errors="coerce", utc=True)
    return d.dropna(subset=["kickoff"])


def gw_of(conn, season) -> pd.DataFrame:
    return pd.read_sql("SELECT gw, MIN(kickoff_time) k FROM fixtures WHERE season=? AND gw IS "
                       "NOT NULL GROUP BY gw", conn, params=(season,))


def fit(train: pd.DataFrame, w: float, roster: list[str]) -> models.DixonColes:
    return models.DixonColes(xg_weight=w).fit(train, teams=roster)


def held_out(gm: models.DixonColes, test: pd.DataFrame) -> pd.Series:
    lam = [gm.expected_goals(h, a) for h, a in zip(test.home_team, test.away_team)]
    lh, la = np.array([x[0] for x in lam]), np.array([x[1] for x in lam])
    return pd.Series(poisson.logpmf(test.home_score, lh) + poisson.logpmf(test.away_score, la),
                     index=test.index)


def replay(conn, season, hist, gws, ws) -> pd.DataFrame:
    """Per-match held-out LL for each weight, refitting every REFIT_EVERY gameweeks."""
    allm = matches(conn, hist + [season])
    roster = pd.read_sql("SELECT short_name FROM teams WHERE season=?", conn,
                         params=(season,)).short_name.tolist()
    fx = pd.read_sql("SELECT gw, kickoff_time FROM fixtures WHERE season=? AND finished=1",
                     conn, params=(season,))
    cur = allm[allm.season == season].merge(
        fx.assign(kickoff=pd.to_datetime(fx.kickoff_time, utc=True))[["kickoff", "gw"]]
          .drop_duplicates("kickoff"), on="kickoff", how="left")
    starts = gw_of(conn, season).set_index("gw").k
    out = []
    gl = list(gws)
    for i in range(0, len(gl), REFIT_EVERY):
        block = gl[i:i + REFIT_EVERY]
        cutoff = pd.Timestamp(starts[block[0]], tz="UTC") if pd.Timestamp(starts[block[0]]).tzinfo is None \
            else pd.Timestamp(starts[block[0]])
        train = allm[allm.kickoff < cutoff]
        test = cur[cur.gw.isin(block)]
        if test.empty:
            continue
        row = test[["gw", "home_team", "away_team", "home_score", "away_score"]].copy()
        for w in ws:
            row[f"ll_{w}"] = held_out(fit(train, w, roster), test).values
        out.append(row)
        print(f"  {season} GW{block[0]}-{block[-1]}: {len(test)} matches", flush=True)
    return pd.concat(out, ignore_index=True)


def stage1(conn) -> None:
    season, hist, gws = STAGE1
    print(f"STAGE 1 — choose w on {season} GW{gws.start}-{gws.stop - 1}")
    d = replay(conn, season, hist, gws, WEIGHTS)
    base = d["ll_0.0"]
    print(f"\n{len(d)} matches. Held-out LL of actual goals:")
    print(f"{'w':>6} {'LL':>10} {'vs w=0':>8} {'per match':>10} {'weeks better':>13}")
    for w in WEIGHTS:
        ll = d[f"ll_{w}"]
        wk = (ll - base).groupby(d.gw).sum()
        print(f"{w:>6.2f} {ll.sum():>10.2f} {ll.sum() - base.sum():>+8.2f} "
              f"{(ll.sum() - base.sum()) / len(d):>+10.4f} {int((wk > 0).sum()):>6}/{len(wk)}")


# Frozen after stage 1, BEFORE stage 2 was run. Do not change it to fit stage 2.
# Rule, the same as validate_attack_fixture.py: maximise held-out LL.
# Stage 1 (2025-26 GW8-38, 310 matches), held-out LL of goals vs w=0:
#   0.25 +3.07 (25/31 weeks)   0.50 +4.93 (24/31)   0.75 +5.58 (21/31)   1.00 +5.03 (20/31)
# A plateau from 0.5 to 1.0 (within 0.65 LL); weekly consistency is best at low w,
# total gain at 0.75. Recorded so nobody "re-picks" 0.5 after seeing stage 2.
CHOSEN_W = 0.75


def decision_layer(conn, w: float) -> None:
    allm = matches(conn, ["2024-25", "2025-26", "2026-27"])
    roster = pd.read_sql("SELECT short_name FROM teams WHERE season='2026-27'", conn).short_name.tolist()
    starts = gw_of(conn, "2026-27").set_index("gw").k
    fin = allm[allm.season == "2026-27"]
    print("\nDECISION LAYER (models fitted as of each gameweek's first kickoff)")
    for gw, lines in PINNACLE.items():
        cutoff = pd.Timestamp(starts[gw])
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff
        res = {}
        for lab, wt in (("goals only", 0.0), (f"w={w}", w)):
            gm = fit(allm[allm.kickoff < cutoff], wt, roster)
            ours, mk, ocs, mcs = [], [], [], []
            for h, a, hx, ax, hcs, acs in lines:
                lh, la = gm.expected_goals(h, a)
                ours += [lh, la]; mk += [hx, ax]; ocs += [np.exp(-la), np.exp(-lh)]; mcs += [hcs, acs]
            ours, mk = np.array(ours), np.array(mk)
            res[lab] = (np.corrcoef(ours, mk)[0, 1], np.corrcoef(ocs, mcs)[0, 1],
                        2 * ours.mean(), gm.defence.get("HUL"),
                        dict(zip([f"{h}-{a}" for h, a, *_ in lines], ours[::2].round(2))))
        print(f"  GW{gw} vs Pinnacle (market {2 * np.mean([x for l in lines for x in l[2:4]]):.2f} goals/match):")
        for lab, (cx, cc, lvl, hul, _) in res.items():
            print(f"     {lab:<10} xG corr {cx:.3f}   CS corr {cc:.3f}   level {lvl:.2f}/match   "
                  f"HUL defence {hul:+.3f}")
    real = (fin.home_score + fin.away_score).mean()
    print(f"  realised 2026-27 goals/match so far: {real:.2f} over {len(fin)} matches")


def stage2(conn, w: float) -> None:
    season, hist, gws = STAGE2
    print(f"STAGE 2 — confirm w={w} (frozen) on {season} GW{gws.start}-{gws.stop - 1}")
    d = replay(conn, season, hist, gws, [0.0, w])
    gain = d[f"ll_{w}"] - d["ll_0.0"]
    rng = np.random.default_rng(13)
    boot = np.array([rng.choice(gain.values, len(gain)).sum() for _ in range(N_BOOT)])
    wk = gain.groupby(d.gw).sum()
    print(f"\n{len(d)} matches: held-out LL gain {gain.sum():+.2f} ({gain.mean():+.4f}/match)")
    print(f"  95% fixture bootstrap [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]"
          f"   P(gain>0)={np.mean(boot > 0):.3f}   weeks better {int((wk > 0).sum())}/{len(wk)}")
    decision_layer(conn, w)


if __name__ == "__main__":
    conn = sqlite3.connect(DB)
    if "--stage2" in sys.argv:
        if CHOSEN_W is None:
            sys.exit("Run stage 1 and freeze CHOSEN_W first.")
        stage2(conn, CHOSEN_W)
    else:
        stage1(conn)
