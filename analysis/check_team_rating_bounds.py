#!/usr/bin/env python3
"""
GUARD — not a diagnostic. Do not delete in a cleanup.

Until models.DixonColes.fit shrinks its team ratings, this check is the only
thing standing between the MILP and a squad of Hull defenders. Measured
2026-09-03, four of the top five projected players in the entire game were
GBP4.0-4.1m Hull defenders, all above Haaland, and plan_transfers wanted two -4
hits to buy three of them.

Are any Dixon-Coles team ratings pinned at the optimiser's bound?

WHY THE BOUND MADE IT WORSE, NOT BETTER
---------------------------------------
bounds=[(-3,3)] on the rating parameters looks like defensive programming. It is
the opposite here. Unbounded, a divergent parameter goes to inf and the
projection becomes inf/nan — which crashes, or at minimum cannot be selected by
a maximiser. The bound converts divergence into a FINITE, plausible-looking
number, and finite is exactly what an optimiser hunts.

LIMITATIONS.md §4 measured the MILP as an estimation-error maximiser: it selects
whatever is most overestimated, because it cannot distinguish a high estimate
from a high error. A boundary value is the limiting case of that — not an
overestimate but a non-estimate, and the most selectable object in the pool. So
the bound did not merely disguise the failure. It turned a crash into a target.

THE DEFECT
----------
models.DixonColes.fit bounds the free parameters at +/-3 and applies NO
shrinkage to team ratings. Every other component in models.py shrinks toward a
prior — attacking rates toward positional medians with PRIOR_STRENGTH=450
minutes, DefCon toward a positional rate with a 12-start beta prior — but the
team attack/defence ratings are estimated freely.

For a club with a full season of matches that is fine. For a NEWLY PROMOTED club
two gameweeks into a season it is not: the likelihood is nearly flat, the
optimiser walks to the boundary, and exp() of a boundary value produces a
projection that is not an estimate at all.

MEASURED 2026-09-03, fitting on 2024-25 + 2025-26 + 2026-27:

    COV  attack  = -3.000 (bound)   2 finished matches
    HUL  defence = -3.000 (bound)   2 finished matches

Consequences, checked against the Pinnacle market line for GW3:

    COV away at MCI   our xG 0.038  vs market 0.72
    AVL away at HUL   our xG 0.062  vs market 1.66   (HUL's -3.0 defence
                                                      annihilates AVL's attack)
    HUL home CS       ours   0.940  vs market 0.191

The distortion is not confined to the promoted clubs. It corrupts the projection
of whoever plays them, which is how Aston Villa ends up projected to score 0.06
goals. It also inflates every aggregate: including these four team-fixtures
moves our GW3 correlation with the market from 0.888 to 0.766 on xG and from
0.888 to 0.564 on clean sheets, and the clean-sheet spread from 0.096 to 0.236
against a market 0.082-0.105.

WHY THIS IS NOT FIXED HERE
--------------------------
Per the session ground rule, fplbrain/ is not modified. The fix belongs in
models.DixonColes.fit and is a modelling change (shrink team ratings toward zero
by match count, or a hierarchical prior), which LIMITATIONS.md §5 says should be
validated out-of-sample before adoption, not shipped mid-gameweek.

FIXED 2026-09-15 in 79e4712 (GOAL_RIDGE_LAMBDA=2.0, validated out-of-sample, plus
rating every current-season team so zero-match clubs sit at the prior). Both TODOs
below are therefore DONE. The guard is RETAINED, because removing the penalty or
setting ridge=0 reintroduces the defect silently.

Until then this is a GUARD: it fails loudly when any rating is at the bound, so
a projection built on one is never read as an estimate.

KNOWN GAP — THIS CATCHES ONE SYMPTOM OF TWO
-------------------------------------------
A team with ZERO finished matches never enters gm.attack/gm.defence at all, and
expected_goals falls back to min(attack)/max(defence) for it. That produces no
boundary value, so this check stays silent. It is the same root cause (no
shrinkage, no prior) presenting through a different path, and it is what the
GW1 comparison hit before the promoted clubs had played. TODO: also flag any
2026-27 team absent from the fitted ratings.

TODO, deferred past the GW3 deadline by instruction: test a ridge penalty
+ lambda*(sum att^2 + sum dfn^2) in the NLL, which is a Gaussian prior on league
average and lets match count weight itself. Score it TWICE — held-out match
log-likelihood AND whether the optimal squad under shrinkage still contains
promoted-club players at implausible rank. The second is the one that matters,
because the failure was in the decision layer, not the fit: a lambda that barely
moves likelihood could still remove four phantom players from the top five.
Ship only if it clears both, per LIMITATIONS.md §5.

EXIT CODES
  0  no rating at the bound
  1  one or more ratings pinned  -> distrust projections involving those teams
  2  could not check
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fplbrain import models  # noqa: E402

DB = str(ROOT / "data" / "fpl.db")
SEASONS = ["2024-25", "2025-26", "2026-27"]
BOUND = 3.0
TOL = 0.001
MIN_MATCHES = 10          # below this, a free estimate is not trustworthy anyway


def main() -> int:
    try:
        conn = sqlite3.connect(DB)
        gm = models.fit_goal_model(conn, SEASONS)
    except Exception as e:  # noqa: BLE001
        print(f"!! could not fit the goal model: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    frames = []
    for s in SEASONS:
        frames.append(pd.read_sql(
            """SELECT th.short_name h, ta.short_name a FROM fixtures f
               JOIN teams th ON th.season=f.season AND th.team_id=f.home_team
               JOIN teams ta ON ta.season=f.season AND ta.team_id=f.away_team
               WHERE f.season=? AND f.finished=1""", conn, params=(s,)))
    m = pd.concat(frames)
    counts = pd.concat([m.h, m.a]).value_counts()

    d = pd.DataFrame({"attack": pd.Series(gm.attack),
                      "defence": pd.Series(gm.defence),
                      "matches": counts}).sort_values("attack")
    d["matches"] = d["matches"].fillna(0).astype(int)
    d["at_bound"] = ((d.attack.abs() >= BOUND - TOL) | (d.defence.abs() >= BOUND - TOL))
    d["thin"] = d.matches < MIN_MATCHES

    print(f"fitted teams: {len(d)}   home_adv={gm.home_adv:.3f}  rho={gm.rho:.3f}")
    print(d.round(3).to_string())

    bad = d[d.at_bound]
    thin = d[d.thin & ~d.at_bound]
    if len(thin):
        print(f"\nthin but not pinned (< {MIN_MATCHES} matches): {thin.index.tolist()}")

    if bad.empty:
        print("\nno team rating is at the +/-3 bound — projections are estimates")
        return 0

    print(f"\n!! {len(bad)} team rating(s) PINNED AT THE BOUND: {bad.index.tolist()}")
    print("   These are not estimates. exp() of a boundary value produces a")
    print("   projection with no information in it.")
    affected = set(bad.index)
    nxt = pd.read_sql(
        """SELECT f.gw, th.short_name h, ta.short_name a FROM fixtures f
           JOIN teams th ON th.season=f.season AND th.team_id=f.home_team
           JOIN teams ta ON ta.season=f.season AND ta.team_id=f.away_team
           WHERE f.season='2026-27' AND f.finished=0 ORDER BY f.gw LIMIT 40""", conn)
    hit = nxt[nxt.h.isin(affected) | nxt.a.isin(affected)]
    if len(hit):
        print("\n   upcoming fixtures whose projections are therefore unusable")
        print("   (BOTH teams, not just the pinned one):")
        for r in hit.head(12).itertuples():
            print(f"     GW{r.gw}  {r.h} v {r.a}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
