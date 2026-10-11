"""
Modelling core.

Four components feed one number (xP):

  1. GOALS       Dixon-Coles team attack/defence ratings with time decay
                 -> per-fixture expected goals for/against -> P(clean sheet)
  2. MINUTES     P(start), P(60+), expected minutes from recent usage + status
  3. DEFCON      P(clearing the 10 CBIT / 12 CBIRT threshold), empirical-Bayes
                 shrunk toward positional priors
  4. RETURNS     goal/assist rates from xG90/xA90 with penalty-taker uplift,
                 scaled by the fixture: the team's Dixon-Coles expected goals in
                 this fixture relative to its average fixture, ** ATTACK_FIXTURE_ALPHA

Then FPL 2026/27 scoring converts these into expected points, plus a variance
estimate so captaincy can weigh ceiling against floor.

Deliberate choices worth knowing about:
- We regress everything toward positional priors. Small-sample per-90 rates are
  the classic way to talk yourself into a bad transfer.
- We never use FPL's own `ep_next` as a model input, only as a benchmark.
  Fitting to it would just relearn their model, including its biases.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ scoring

GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
DEFCON_POINTS = 2
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}  # DEF uses CBIT, MID/FWD use CBIRT

# Positional priors for shrinkage (per 90). Tuned from 2024-25 + 2025-26 medians.
PRIOR_STRENGTH = 450.0  # minutes of "prior evidence"; ~5 full matches


# ------------------------------------------------------------------ 1. goals


# Ridge penalty on Dixon-Coles team ratings: + lambda * (sum att^2 + sum dfn^2).
# Validated out-of-sample in analysis/validate_goal_ridge.py.
#
# Without it the fit is raw MLE bounded at +/-3. A promoted club with two matches
# has a nearly flat likelihood, the optimiser walks to the bound, and exp() of
# the bound is not an estimate — 2026-27 put COV attack and HUL defence at
# -3.000, four GBP4.0m Hull defenders in the top five projected players, and a
# 96% Man City clean sheet against a market 49%. The bound made it worse, not
# safer: unbounded it would diverge to inf and crash; bounded it becomes a
# finite value, which is exactly what the optimiser hunts.
#
# CHOSEN ON 2025-26 by a rule fixed before the choice (most wins on gameweeks
# where the baseline was not itself pinned, ties broken by mean held-out LL,
# goal-level bias within 2%): lambda=2 won 6/8 clean weeks, +0.048 LL/match, and
# removed every pinned rating. CONFIRMED on 2026-27 frozen: mean held-out LL
# +0.053/match, zero pinned ratings in every refit (the raw fit had two in every
# one), and against the Pinnacle GW3 market line clean-sheet correlation rose
# 0.563 -> 0.792 with spread 0.236 -> 0.122 (market 0.105).
#
# HONEST CAVEATS, recorded rather than buried:
#   * On 2026-27 it won only 1 of 3 gameweeks. The two losses came entirely from
#     pinned "this team scores ~0" predictions that happened to come true
#     (COV 0-1 HUL, MCI 1-0 COV, HUL 0-0 AVL); on every other match ridge was
#     neutral (-0.040, -0.002, -0.001). Four pinned matches cannot validate on
#     log-likelihood — one outcome swings it (CHE 2-2 HUL cost the raw fit -7.56).
#   * It is mildly worse on normal teams very early in a season (-0.04 LL/match
#     in 2026-27 GW2).
#   * This form shrinks defence toward 0, and under the sum-to-zero attack
#     parameterisation defence carries the league scoring rate, so it lowers
#     predicted goals as lambda grows (-1.5% at lambda=2, -5.7% at 16). A
#     centred form avoided that but won fewer weeks. Re-check the level bias if
#     lambda is ever raised.
GOAL_RIDGE_LAMBDA = 2.0

# Exponent on the fixture multiplier for attacking returns (see
# DixonColes.attack_multiplier). 0 reproduces the old fixture-blind model.
# Validated in analysis/validate_attack_fixture.py.
#
# Until October 2026 the docstring above claimed returns were "scaled by
# opponent strength" while the code computed xg90 * mins_frac with no opponent,
# so every attacker projected identically against every defence — while the
# same fit was moving the other side's clean-sheet odds for that fixture.
#
# Scored on the attacking component alone: Poisson LL of actual goals and
# assists given xg90 * ACTUAL minutes/90 * m**alpha, so minutes error cannot
# leak in. CHOSEN ON 2025-26 GW8-38 (9348 player-games, 310 fixtures): peak at
# 0.75, +13.71 LL vs 0, 21/31 weeks better; flat from 0.5 to 1.0. CONFIRMED on
# 2026-27 GW1-5 frozen: +2.64 LL over 50 fixtures, 4/5 weeks better.
#
# HONEST CAVEAT: per fixture the gain replicates (+0.044, then +0.053), but
# with a bootstrap over FIXTURES — teammates' goals in one match are not
# independent — neither stage excludes zero: stage 1 [-2.17, +29.64]
# P(>0)=0.955 (optimistic, alpha was chosen there), stage 2 [-4.58, +10.14]
# P(>0)=0.76. Shipped on that plus coherence: the model already believed the
# opponent sets the goals a team scores, for clean sheets; it now uses the same
# number for the attackers on the other side. Bonus is NOT scaled — not tested.
# Level cost: the raw multiplier averages to exactly 1 over a balanced schedule,
# but alpha < 1 costs -0.39% of attacking output via Jensen's inequality on the
# 2026-27 ratings (worst club -0.40%). Left in, because it is what was validated.
ATTACK_FIXTURE_ALPHA = 0.75


@dataclass
class DixonColes:
    """
    Dixon-Coles (1997) bivariate Poisson with low-score correlation and
    exponential time decay. Fitted on match results, gives per-fixture
    expected goals which drive clean-sheet and goals-conceded points.
    """

    xi: float = 0.0018  # decay rate per day; ~0.0018 halves weight in ~1 year
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0
    fitted: bool = False
    ridge: float = GOAL_RIDGE_LAMBDA   # see GOAL_RIDGE_LAMBDA; 0 reproduces raw MLE

    @staticmethod
    def _tau(x, y, lh, la, rho):
        out = np.ones_like(lh, dtype=float)
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        out[m00] = 1 - lh[m00] * la[m00] * rho
        out[m01] = 1 + lh[m01] * rho
        out[m10] = 1 + la[m10] * rho
        out[m11] = 1 - rho
        return np.clip(out, 1e-9, None)

    def fit(self, matches: pd.DataFrame, teams: list[str] | None = None) -> "DixonColes":
        """
        matches: home_team, away_team, home_score, away_score, kickoff (datetime)

        teams: extra teams to rate even if they have no matches yet. With a ridge
        penalty a zero-match team settles at the prior (league average) — which
        is what a promoted club with no results should get. Without this they
        never enter the ratings and expected_goals assigns them the WEAKEST
        observed attack and defence, harsher than any market prices them.
        """
        df = matches.dropna(subset=["home_score", "away_score"]).copy()
        if len(df) < 50:
            raise ValueError(f"need >=50 finished matches to fit, got {len(df)}")

        extra = set(teams or []) if self.ridge > 0 else set()
        teams = sorted(set(df.home_team) | set(df.away_team) | extra)
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        latest = df["kickoff"].max()
        age_days = (latest - df["kickoff"]).dt.total_seconds().values / 86400.0
        w = np.exp(-self.xi * age_days)

        h = df.home_team.map(idx).values
        a = df.away_team.map(idx).values
        hs = df.home_score.values.astype(int)
        as_ = df.away_score.values.astype(int)

        def nll(params):
            att = np.concatenate([params[: n - 1], [-params[: n - 1].sum()]])  # sum-to-zero
            dfn = params[n - 1 : 2 * n - 1]
            hadv, rho = params[-2], params[-1]
            lh = np.exp(att[h] + dfn[a] + hadv)
            la = np.exp(att[a] + dfn[h])
            lh = np.clip(lh, 1e-6, 12)
            la = np.clip(la, 1e-6, 12)
            ll = (
                poisson.logpmf(hs, lh)
                + poisson.logpmf(as_, la)
                + np.log(self._tau(hs, as_, lh, la, rho))
            )
            penalty = self.ridge * (np.sum(att ** 2) + np.sum(dfn ** 2)) if self.ridge else 0.0
            return -np.sum(w * ll) + penalty

        x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25], [-0.05]])
        bounds = [(-3, 3)] * (n - 1) + [(-3, 3)] * n + [(-1, 1), (-0.2, 0.2)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 800, "ftol": 1e-8})

        att = np.concatenate([res.x[: n - 1], [-res.x[: n - 1].sum()]])
        dfn = res.x[n - 1 : 2 * n - 1]
        self.attack = {t: float(att[idx[t]]) for t in teams}
        self.defence = {t: float(dfn[idx[t]]) for t in teams}
        self.home_adv = float(res.x[-2])
        self.rho = float(res.x[-1])
        self.fitted = True
        log.info("DixonColes fitted on %d matches, home_adv=%.3f rho=%.3f",
                 len(df), self.home_adv, self.rho)
        return self

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        if not self.fitted:
            raise RuntimeError("model not fitted")
        # Unknown teams (promoted clubs) get the weakest observed attack and defence.
        wa = min(self.attack.values())
        wd = max(self.defence.values())
        ah, dh = self.attack.get(home, wa), self.defence.get(home, wd)
        aa, da = self.attack.get(away, wa), self.defence.get(away, wd)
        return float(np.exp(ah + da + self.home_adv)), float(np.exp(aa + dh))

    def attack_multiplier(self, team: str, opponent: str, at_home: bool,
                          roster: list[str]) -> float:
        """
        The team's expected goals in this fixture relative to its average
        fixture against `roster` (the current season's clubs), home and away
        equally weighted. Exactly 1 averaged over a balanced schedule, so it
        moves goals between fixtures without changing a season's total.
        Under exp(att + dfn_opp + home_adv*home) the team's own attack cancels.
        """
        wd = max(self.defence.values())          # expected_goals' unknown-team fallback
        dfn = lambda t: self.defence.get(t, wd)  # noqa: E731
        others = [o for o in roster if o != team] or [opponent]
        mean_opp = float(np.mean([np.exp(dfn(o)) for o in others]))
        venue = (1.0 + np.exp(self.home_adv)) / 2.0
        this = np.exp(dfn(opponent) + (self.home_adv if at_home else 0.0))
        return float(this / (mean_opp * venue))

    def clean_sheet_prob(self, team: str, opponent: str, at_home: bool) -> float:
        lh, la = self.expected_goals(team, opponent) if at_home else self.expected_goals(opponent, team)
        conceded = la if at_home else lh
        return float(np.exp(-conceded))

    def goals_conceded_dist(self, team: str, opponent: str, at_home: bool, kmax: int = 8):
        lh, la = self.expected_goals(team, opponent) if at_home else self.expected_goals(opponent, team)
        conceded = la if at_home else lh
        return np.array([poisson.pmf(k, conceded) for k in range(kmax + 1)])


def fit_goal_model(conn: sqlite3.Connection, seasons: list[str], xi: float = 0.0018) -> DixonColes:
    """Fit on finished fixtures across seasons, joining team short_names."""
    frames = []
    for s in seasons:
        q = """
            SELECT f.kickoff_time, f.home_score, f.away_score,
                   th.short_name AS home_team, ta.short_name AS away_team
            FROM fixtures f
            JOIN teams th ON th.season=f.season AND th.team_id=f.home_team
            JOIN teams ta ON ta.season=f.season AND ta.team_id=f.away_team
            WHERE f.season = ? AND f.finished = 1
        """
        frames.append(pd.read_sql(q, conn, params=(s,)))
    df = pd.concat(frames, ignore_index=True)
    df["kickoff"] = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["kickoff"])
    # Rate every current-season team, including promoted clubs with no results
    # yet; the ridge prior places them at league average. The last season in
    # `seasons` is the current one by convention of every caller.
    current_teams = pd.read_sql(
        "SELECT short_name FROM teams WHERE season=?", conn, params=(seasons[-1],)
    )["short_name"].tolist()
    return DixonColes(xi=xi).fit(df, teams=current_teams)


def reconstruct_results_from_gw(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    """
    Fallback when fixtures.csv lacks scores: rebuild match results from
    player_gw goals_conceded, which is populated per player per fixture.
    """
    q = """
        SELECT g.fixture_id, g.gw, g.was_home, g.opponent_team,
               MAX(g.goals_conceded) AS conceded, g.kickoff_time,
               p.team_id
        FROM player_gw g JOIN players p ON p.season=g.season AND p.element=g.element
        WHERE g.season = ? AND g.minutes > 0
        GROUP BY g.fixture_id, p.team_id
    """
    return pd.read_sql(q, conn, params=(season,))


# ------------------------------------------------------------------ 2. minutes


# Strength of the prior-season start rate, in equivalent matches. Validated
# out-of-sample in analysis/validate_minutes_prior.py: chosen on 2025-26 GW2-8 by
# a rule fixed before the fine sweep (minimise Brier, must also beat no-prior on
# log-loss), then confirmed on 2026-27 GW2-4 with k frozen. Squad-relevant Brier
# improved 6.5% on the tuning season and 10.9% on the confirmation season, winning
# every target gameweek on both metrics. The optimum is flat across 0.5-1.0.
MINUTES_PRIOR_K = 0.75


def minutes_model(conn: sqlite3.Connection, season: str, as_of_gw: int | None = None,
                  lookback: int = 6) -> pd.DataFrame:
    """
    P(start) and expected minutes.

    One path for pre-season and in-season: the current-season window is blended
    with the previous season's rate as a Beta prior worth MINUTES_PRIOR_K matches,

        start_rate  = (starts_in_window + k * prior_rate)    / (games_in_window + k)
        avg_minutes = (minutes_in_window + k * prior_minutes) / (games_in_window + k)

    With no current-season games this reduces exactly to the prior. Then the
    availability multiplier from FPL's `status` / `chance_of_playing_next_round`.

    WHY NOT A BRANCH. This used to switch from prior-seasons-only to
    current-season-only on the first current-season row. With one gameweek loaded
    that made every start_rate exactly 0.0 or 1.0 and discarded all of last
    season: 92% of players at exactly 0 or 1, 85 provably misclassified against
    GW2-4 minutes (Cherki at 0.0 after 81 and 65 minutes; Senesi at 1.0 with none).
    Same defect class as an unshrunk Dixon-Coles rating. Do not reintroduce the
    switch — analysis/check_minutes_collapse.py guards against it.

    The prior is the most recent previous season with gameweek data, which is
    what was validated. Players with no history there fall back to 0.35 / 35
    minutes, the long-standing guess for new signings and promoted players.
    """
    k = MINUTES_PRIOR_K
    base = pd.read_sql(
        "SELECT element, code, position, status, chance_next_round FROM players WHERE season=?",
        conn, params=(season,),
    )

    prev = pd.read_sql(
        "SELECT DISTINCT season FROM player_gw WHERE season < ? ORDER BY season DESC LIMIT 1",
        conn, params=(season,),
    )
    if len(prev):
        prior = pd.read_sql(
            """SELECT p_cur.element,
                      AVG(CASE WHEN g.minutes>=60 THEN 1.0 ELSE 0.0 END) AS prior_rate,
                      AVG(g.minutes) AS prior_minutes
               FROM players p_cur
               JOIN players p_hist ON p_hist.code = p_cur.code AND p_hist.season = ?
               JOIN player_gw g ON g.season = p_hist.season AND g.element = p_hist.element
               WHERE p_cur.season = ?
               GROUP BY p_cur.element""",
            conn, params=(prev.iloc[0]["season"], season),
        )
    else:
        prior = pd.DataFrame(columns=["element", "prior_rate", "prior_minutes"])

    gws = pd.read_sql(
        "SELECT DISTINCT gw FROM player_gw WHERE season=? ORDER BY gw", conn, params=(season,)
    )["gw"].tolist()
    if gws:
        cutoff = as_of_gw if as_of_gw is not None else max(gws)
        window = pd.read_sql(
            """SELECT element,
                      SUM(CASE WHEN minutes>=60 THEN 1 ELSE 0 END) AS s,
                      COUNT(*) AS n,
                      SUM(minutes) AS mins
               FROM player_gw WHERE season=? AND gw > ? AND gw <= ?
               GROUP BY element""",
            conn, params=(season, cutoff - lookback, cutoff),
        )
    else:
        window = pd.DataFrame(columns=["element", "s", "n", "mins"])

    out = base.merge(prior, on="element", how="left").merge(window, on="element", how="left")
    for c in ("s", "n", "mins"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    r0 = pd.to_numeric(out["prior_rate"], errors="coerce").fillna(0.35)
    m0 = pd.to_numeric(out["prior_minutes"], errors="coerce").fillna(35.0)
    out["start_rate"] = (out["s"] + k * r0) / (out["n"] + k)
    out["avg_minutes"] = (out["mins"] + k * m0) / (out["n"] + k)
    out["apps"] = out["n"]

    # Availability multiplier. `status`: a=available, d=doubtful, i=injured,
    # s=suspended, u=unavailable, n=not in squad.
    status_mult = out["status"].map({"a": 1.0, "d": 0.5, "i": 0.0,
                                     "s": 0.0, "u": 0.0, "n": 0.0}).fillna(1.0)
    # to_numeric matters: in pre-season FPL leaves this column entirely empty, which
    # pandas reads as object dtype, and that silently turned p_start into an object
    # array. Harmless to arithmetic, fatal to anything that takes a log of it.
    chance = pd.to_numeric(out["chance_next_round"], errors="coerce")
    chance_mult = np.where(chance.notna(), chance.fillna(100) / 100.0, 1.0)
    avail = np.minimum(status_mult, chance_mult)

    out["p_start"] = np.clip(out["start_rate"] * avail, 0, 1)
    out["exp_minutes"] = np.clip(out["avg_minutes"] * avail, 0, 90)
    # P(60+) is close to p_start but allows for early substitutions.
    out["p_60"] = np.clip(out["p_start"] * 0.92, 0, 1)
    out["p_any_minutes"] = np.clip(out["p_start"] + (1 - out["p_start"]) * 0.35 * avail, 0, 1)
    out["availability"] = avail
    return out[["element", "code", "position", "p_start", "p_60", "p_any_minutes",
                "exp_minutes", "availability"]]


# ------------------------------------------------------------------ 3. defcon


def defcon_model(conn: sqlite3.Connection, hist_seasons: list[str],
                 cur_season: str) -> pd.DataFrame:
    """
    P(hitting the DefCon threshold | started).

    Empirical-Bayes: shrink each player's observed hit rate toward the prior for
    their CURRENT position. This matters because 11 players were reclassified for
    2026/27 — a converted defender inherits the midfielder threshold (12 CBIRT),
    so their old raw numbers are misleading.

    TWO THINGS THAT WILL BITE YOU IF YOU CHANGE THIS:

    1. Seasons before 2025-26 have NO defensive stats at all — tackles,
       recoveries, and clearances_blocks_interceptions are 100% NULL, because the
       rule did not exist. Including them (and filling nulls with 0) adds tens of
       thousands of guaranteed "misses" and halves every estimate. We therefore
       filter to rows where the data actually exists, rather than trusting the
       caller's season list.

    2. FPL's own `defensive_contribution` column already IS the position-correct
       metric: verified to equal CBIT for DEF and CBIRT for MID/FWD on 100% of
       2025-26 rows. Use it directly instead of recomputing.
    """
    frames = []
    for s in hist_seasons + [cur_season]:
        q = """
            SELECT g.code, g.defensive_contribution AS dc
            FROM player_gw g
            WHERE g.season=? AND g.minutes>=60 AND g.code IS NOT NULL
              AND g.defensive_contribution IS NOT NULL
        """
        frames.append(pd.read_sql(q, conn, params=(s,)))
    hist = pd.concat(frames, ignore_index=True)

    cur = pd.read_sql(
        "SELECT element, code, position FROM players WHERE season=? AND position != 'GK'",
        conn, params=(cur_season,),
    )

    if hist.empty:
        log.warning("no seasons with DefCon data available — falling back to priors")
        cur["p_defcon"] = cur["position"].map({"DEF": 0.27, "MID": 0.18, "FWD": 0.01})
        cur["defcon_sample_starts"] = 0
        return cur[["element", "code", "position", "p_defcon", "defcon_sample_starts"]]

    hist = hist.merge(cur[["code", "position"]], on="code", how="inner")
    hist["hit"] = (hist["dc"] >= hist["position"].map(DEFCON_THRESHOLD)).astype(int)

    agg = hist.groupby("code").agg(hits=("hit", "sum"), starts=("hit", "count")).reset_index()
    agg = agg.merge(cur, on="code", how="right")
    agg["hits"] = agg["hits"].fillna(0)
    agg["starts"] = agg["starts"].fillna(0)

    # Beta prior per position from that position's pooled rate.
    pooled = hist.groupby("position")["hit"].mean().to_dict()
    prior_n = 12.0  # equivalent to 12 prior starts
    agg["prior"] = agg["position"].map(pooled).fillna(0.15)
    agg["p_defcon"] = (agg["hits"] + agg["prior"] * prior_n) / (agg["starts"] + prior_n)

    agg["defcon_sample_starts"] = agg["starts"]
    return agg[["element", "code", "position", "p_defcon", "defcon_sample_starts"]]


# ------------------------------------------------------------------ 4. returns


def attacking_rates(conn: sqlite3.Connection, hist_seasons: list[str],
                    cur_season: str) -> pd.DataFrame:
    """
    Shrunk xG90 / xA90 per player, plus a penalty-taker uplift.

    Shrinkage weight = minutes / (minutes + PRIOR_STRENGTH), so a player with
    450 historical minutes sits halfway between his own rate and the positional prior.
    """
    frames = []
    for s in hist_seasons:
        q = """
            SELECT g.code, SUM(g.minutes) AS minutes,
                   SUM(g.expected_goals) AS xg, SUM(g.expected_assists) AS xa
            FROM player_gw g
            WHERE g.season=? AND g.code IS NOT NULL
            GROUP BY g.code
        """
        frames.append(pd.read_sql(q, conn, params=(s,)))
    hist = pd.concat(frames).groupby("code", as_index=False).sum()

    cur = pd.read_sql(
        """SELECT element, code, position, xg_per90, xa_per90, penalties_order,
                  corners_fk_order, direct_fk_order, minutes AS cur_minutes
           FROM players WHERE season=?""",
        conn, params=(cur_season,),
    )
    df = cur.merge(hist, on="code", how="left")
    df["minutes"] = df["minutes"].fillna(0)
    df["hist_xg90"] = 90 * df["xg"] / df["minutes"].replace(0, np.nan)
    df["hist_xa90"] = 90 * df["xa"] / df["minutes"].replace(0, np.nan)

    priors = df.groupby("position")[["hist_xg90", "hist_xa90"]].median()
    df = df.join(priors, on="position", rsuffix="_prior")

    w = df["minutes"] / (df["minutes"] + PRIOR_STRENGTH)
    df["xg90"] = (w * df["hist_xg90"].fillna(0) +
                  (1 - w) * df["hist_xg90_prior"].fillna(0.05))
    df["xa90"] = (w * df["hist_xa90"].fillna(0) +
                  (1 - w) * df["hist_xa90_prior"].fillna(0.05))

    # If the current season has meaningful minutes, blend those in too.
    live_w = np.clip(df["cur_minutes"].fillna(0) / 900.0, 0, 0.6)
    df["xg90"] = (1 - live_w) * df["xg90"] + live_w * df["xg_per90"].fillna(df["xg90"])
    df["xa90"] = (1 - live_w) * df["xa90"] + live_w * df["xa_per90"].fillna(df["xa90"])

    # First-choice penalty taker: roughly +0.10 xG per 90 at a typical PK rate.
    df["is_pen_taker"] = (df["penalties_order"] == 1).astype(int)
    df["xg90"] = df["xg90"] + 0.10 * df["is_pen_taker"]
    df["set_piece_score"] = (
        (df["corners_fk_order"] == 1).astype(int) + (df["direct_fk_order"] == 1).astype(int)
    )
    df["xa90"] = df["xa90"] * (1 + 0.08 * df["set_piece_score"])

    return df[["element", "code", "position", "xg90", "xa90",
               "is_pen_taker", "set_piece_score"]]


# ------------------------------------------------------------------ assembly


def expected_bonus(p_start: float, xgi90: float, position: str, p_cs: float) -> float:
    """
    Crude but honest expected-bonus estimate.

    2026/27 retuned BPS: CBI now scores 1 per 3 (was 1 per 2), the -1 for being
    tackled is gone, GK save rewards adjusted. Rather than pretend to model BPS
    exactly, we tie expected bonus to goal involvement and clean-sheet chance,
    which is what actually drives top-3 BPS in practice.
    """
    base = 0.35 * xgi90 + (0.25 * p_cs if position in ("GK", "DEF") else 0.10 * p_cs)
    return float(np.clip(base * p_start, 0, 1.6))


def project_gameweek(conn: sqlite3.Connection, cur_season: str, gw: int,
                     goal_model: DixonColes, minutes: pd.DataFrame,
                     defcon: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """
    Expected points for every player in one gameweek.
    Handles doubles (two fixtures) by summing, and blanks by returning 0.
    """
    fixtures = pd.read_sql(
        """SELECT f.fixture_id, f.gw, f.home_team, f.away_team,
                  th.short_name AS home_name, ta.short_name AS away_name
           FROM fixtures f
           JOIN teams th ON th.season=f.season AND th.team_id=f.home_team
           JOIN teams ta ON ta.season=f.season AND ta.team_id=f.away_team
           WHERE f.season=? AND f.gw=?""",
        conn, params=(cur_season, gw),
    )
    roster = pd.read_sql("SELECT short_name FROM teams WHERE season=?", conn,
                         params=(cur_season,))["short_name"].tolist()
    players = pd.read_sql(
        """SELECT p.element, p.code, p.web_name, p.position, p.team_id, p.now_cost,
                  p.selected_by_percent, t.short_name AS team
           FROM players p JOIN teams t ON t.season=p.season AND t.team_id=p.team_id
           WHERE p.season=?""",
        conn, params=(cur_season,),
    )

    df = (players
          .merge(minutes[["element", "p_start", "p_60", "p_any_minutes", "exp_minutes"]],
                 on="element", how="left")
          .merge(defcon[["element", "p_defcon"]], on="element", how="left")
          .merge(rates[["element", "xg90", "xa90", "is_pen_taker"]], on="element", how="left"))

    for c, d in [("p_start", 0.2), ("p_60", 0.18), ("p_any_minutes", 0.3),
                 ("exp_minutes", 20), ("p_defcon", 0.12), ("xg90", 0.05),
                 ("xa90", 0.05), ("is_pen_taker", 0)]:
        df[c] = df[c].fillna(d)

    # Goalkeepers are excluded from the DefCon rule entirely. They fall out of
    # defcon_model(), so the fillna above would otherwise hand them the default
    # 0.12 — harmless in the points maths below, but misleading in any table the
    # agent reads. Zero it explicitly.
    df.loc[df["position"] == "GK", "p_defcon"] = 0.0

    # Map each team to its fixture(s) this gameweek.
    legs = []
    for _, f in fixtures.iterrows():
        legs.append({"team_id": f.home_team, "opp": f.away_name, "home": True})
        legs.append({"team_id": f.away_team, "opp": f.home_name, "home": False})
    legs_df = pd.DataFrame(legs)
    if legs_df.empty:
        df["exp_points"] = 0.0
        df["gw"] = gw
        return df

    m = df.merge(legs_df, on="team_id", how="left")
    blank = m["opp"].isna()

    rows = []
    for _, r in m[~blank].iterrows():
        team_name = r["team"]
        p_cs = goal_model.clean_sheet_prob(team_name, r["opp"], bool(r["home"]))
        gc = goal_model.goals_conceded_dist(team_name, r["opp"], bool(r["home"]))

        mins_frac = r["exp_minutes"] / 90.0
        fixture = goal_model.attack_multiplier(
            team_name, r["opp"], bool(r["home"]), roster) ** ATTACK_FIXTURE_ALPHA
        exp_goals = r["xg90"] * mins_frac * fixture
        exp_assists = r["xa90"] * mins_frac * fixture
        pos = r["position"]

        pts = 0.0
        pts += 1.0 * r["p_any_minutes"] + 1.0 * r["p_60"]          # appearance
        pts += GOAL_POINTS.get(pos, 4) * exp_goals
        pts += ASSIST_POINTS * exp_assists
        pts += CS_POINTS.get(pos, 0) * p_cs * r["p_60"]
        if pos in ("GK", "DEF"):
            # -1 per 2 goals conceded, only when playing 60+
            exp_pen = sum(gc[k] * (k // 2) for k in range(len(gc)))
            pts -= exp_pen * r["p_60"]
        if pos == "GK":
            # ~3 saves per goal conceded faced; 1 pt per 3 saves
            exp_conceded = sum(k * gc[k] for k in range(len(gc)))
            pts += (exp_conceded * 3.0 / 3.0) * 0.33 * r["p_60"]
        if pos != "GK":
            pts += DEFCON_POINTS * r["p_defcon"] * r["p_60"]
        xgi90 = r["xg90"] + r["xa90"]
        pts += expected_bonus(r["p_start"], xgi90, pos, p_cs)
        pts -= 0.35 * r["p_start"]   # cards, misc negatives

        rows.append({
            "element": r["element"], "gw": gw, "opp": r["opp"], "home": r["home"],
            "p_start": r["p_start"], "exp_minutes": r["exp_minutes"],
            "p_clean_sheet": p_cs, "p_defcon": r["p_defcon"],
            "exp_goals": exp_goals, "exp_assists": exp_assists,
            "exp_bonus": expected_bonus(r["p_start"], xgi90, pos, p_cs),
            "exp_points": max(pts, 0.0),
        })

    leg_pts = pd.DataFrame(rows)
    if leg_pts.empty:
        df["exp_points"] = 0.0
        df["gw"] = gw
        return df

    agg = leg_pts.groupby("element", as_index=False).agg(
        p_start=("p_start", "max"), exp_minutes=("exp_minutes", "sum"),
        p_clean_sheet=("p_clean_sheet", "mean"), p_defcon=("p_defcon", "max"),
        exp_goals=("exp_goals", "sum"), exp_assists=("exp_assists", "sum"),
        exp_bonus=("exp_bonus", "sum"), exp_points=("exp_points", "sum"),
        n_fixtures=("gw", "count"),
    )

    out = df[["element", "code", "web_name", "position", "team", "now_cost",
              "selected_by_percent"]].merge(agg, on="element", how="left")
    out["exp_points"] = out["exp_points"].fillna(0.0)
    out["n_fixtures"] = out["n_fixtures"].fillna(0).astype(int)
    out["gw"] = gw

    # Variance: attacking returns are lumpy, so scale sd with goal involvement.
    out["sd_points"] = np.sqrt(
        np.clip(out["exp_points"], 0.1, None) * 1.9
        + 6.0 * np.clip(out["exp_goals"].fillna(0), 0, None)
    )
    return out


# FPL deadlines sit 90 minutes before a gameweek's first kickoff (checked on
# 2026-27 GW5, GW6 and GW7 against the API's deadline_time).
DEADLINE_LEAD = timedelta(minutes=90)


def next_gameweek(conn: sqlite3.Connection, season: str,
                  now: datetime | None = None) -> int:
    """
    The first gameweek whose deadline has not passed — the one a manager can
    still change.

    NOT "the lowest gameweek with an unfinished fixture": between a deadline and
    that gameweek's last match, every weekend, that is a locked week. On Sunday
    11 Oct 2026 it returned GW6 mid-gameweek, so the planning tools optimised a
    week nobody could change and refresh.py projected from it.
    """
    now = now or datetime.now(timezone.utc)
    firsts = pd.read_sql(
        """SELECT gw, MIN(kickoff_time) k FROM fixtures
           WHERE season=? AND gw IS NOT NULL AND kickoff_time IS NOT NULL
           GROUP BY gw ORDER BY gw""", conn, params=(season,))
    for gw, k in zip(firsts["gw"], firsts["k"]):
        if pd.Timestamp(k).to_pydatetime() - DEADLINE_LEAD > now:
            return int(gw)
    row = conn.execute("SELECT MIN(gw) FROM fixtures WHERE season=? AND finished=0 "
                       "AND gw IS NOT NULL", (season,)).fetchone()
    return int(row[0]) if row and row[0] is not None else 1


def project_horizon(conn: sqlite3.Connection, cur_season: str, start_gw: int, n_gw: int,
                    hist_seasons: list[str]) -> tuple[pd.DataFrame, dict]:
    """Project a run of gameweeks. Returns (long dataframe, diagnostics)."""
    diagnostics: dict = {}

    try:
        gm = fit_goal_model(conn, hist_seasons + [cur_season])
        diagnostics["goal_model"] = {
            "home_advantage": round(gm.home_adv, 4), "rho": round(gm.rho, 4),
            "n_teams": len(gm.attack),
        }
    except Exception as e:  # noqa: BLE001
        log.error("goal model failed: %s", e)
        raise

    mins = minutes_model(conn, cur_season)
    dc = defcon_model(conn, hist_seasons, cur_season)
    rates = attacking_rates(conn, hist_seasons, cur_season)

    diagnostics["minutes"] = {
        "players": int(len(mins)),
        "mean_p_start": round(float(mins.p_start.mean()), 3),
        "unavailable": int((mins.availability == 0).sum()),
    }
    diagnostics["defcon"] = {
        "players": int(len(dc)), "mean_p_defcon": round(float(dc.p_defcon.mean()), 3),
    }

    frames = []
    for gw in range(start_gw, start_gw + n_gw):
        frames.append(project_gameweek(conn, cur_season, gw, gm, mins, dc, rates))
    out = pd.concat(frames, ignore_index=True)
    return out, diagnostics


def _run_identity() -> dict:
    """Which code and which constants produced a run, so a run scored months later
    can be attributed to a model rather than guessed from its timestamp. CI sets
    GITHUB_SHA; locally it is read from git, with "-dirty" for uncommitted edits."""
    import os
    import subprocess
    version = os.environ.get("GITHUB_SHA", "")[:12]
    if not version:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            version = subprocess.run(["git", "-C", here, "rev-parse", "--short=12", "HEAD"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
            dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", "."],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            version = (version + "-dirty") if version and dirty else version
        except (OSError, subprocess.SubprocessError):
            version = ""
    return {"code_version": version or "unknown",
            "constants": {"ATTACK_FIXTURE_ALPHA": ATTACK_FIXTURE_ALPHA,
                          "MINUTES_PRIOR_K": MINUTES_PRIOR_K,
                          "GOAL_RIDGE_LAMBDA": GOAL_RIDGE_LAMBDA,
                          "PRIOR_STRENGTH": PRIOR_STRENGTH}}


def persist_projections(conn: sqlite3.Connection, proj: pd.DataFrame, season: str,
                        diagnostics: dict | None = None) -> str:
    run_id = "proj-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rec = proj.copy()
    rec["season"] = season
    rec["run_id"] = run_id
    cols = ["season", "element", "gw", "run_id", "p_start", "exp_minutes", "p_60",
            "exp_goals", "exp_assists", "p_clean_sheet", "p_defcon", "exp_bonus",
            "exp_points", "sd_points"]
    for c in cols:
        if c not in rec.columns:
            rec[c] = np.nan
    rec[cols].to_sql("projections", conn, if_exists="append", index=False)
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, created_at, season, horizon_start, horizon_end, notes, provenance)"
        " VALUES (?,?,?,?,?,?,?)",
        (run_id, datetime.now(timezone.utc).isoformat(), season,
         int(proj.gw.min()), int(proj.gw.max()), "projection",
         __import__("json").dumps({**(diagnostics or {}), **_run_identity()}, default=str)),
    )
    conn.commit()
    return run_id
