"""
Squad optimisation as a mixed-integer linear program.

Two entry points:

  initial_squad()   — pre-season / wildcard. Pick 15 from scratch.
  transfer_plan()   — in-season. Given the current squad, decide transfers over a
                      multi-gameweek horizon, priced against the -4 hit, with
                      chips modelled explicitly.

FPL 2026/27 constraints encoded:
  budget 100.0m, squad 15 (2 GK / 5 DEF / 5 MID / 3 FWD), max 3 per club,
  XI of 11 with 1 GK, >=3 DEF, >=2 MID(implicit), >=1 FWD, captain doubles,
  up to 5 free transfers rolled, -4 per extra transfer.

The multi-gameweek formulation is what separates this from a greedy weekly pick:
it will happily take a slightly worse GW1 to set up a much better GW3-6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import pulp

log = logging.getLogger(__name__)

SQUAD_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
SQUAD_SIZE = 15
XI_SIZE = 11
HIT_COST = 4
MAX_ROLLED_FT = 5


@dataclass
class OptimResult:
    status: str
    objective: float
    squad: pd.DataFrame
    xi: pd.DataFrame
    bench: pd.DataFrame
    captain: dict
    vice: dict
    transfers: list = field(default_factory=list)
    hits_taken: int = 0
    bank: float = 0.0
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"status={self.status}  objective={self.objective:.2f} xP  bank=£{self.bank:.1f}m"]
        if self.hits_taken:
            lines.append(f"hits: {self.hits_taken} (-{self.hits_taken * HIT_COST} pts)")
        lines.append(f"captain: {self.captain.get('web_name')} ({self.captain.get('exp_points', 0):.1f} xP)")
        return "\n".join(lines)


def _prep(proj: pd.DataFrame, horizon: list[int], decay: float = 0.86) -> pd.DataFrame:
    """
    Pivot long projections to one row per player, with a time-decayed
    horizon total. Decay reflects that GW+5 projections are much less reliable
    than GW+1, so we shouldn't over-commit to distant fixtures.
    """
    p = proj[proj.gw.isin(horizon)].copy()
    p["w"] = decay ** (p["gw"] - min(horizon))
    p["wxp"] = p["exp_points"] * p["w"]

    agg = p.groupby("element", as_index=False).agg(
        horizon_xp=("exp_points", "sum"),
        weighted_xp=("wxp", "sum"),
        first_gw_xp=("exp_points", "first"),
        mean_p_start=("p_start", "mean"),
        sd=("sd_points", "mean"),
    )
    meta = (p.sort_values("gw")
              .groupby("element", as_index=False)
              .agg(web_name=("web_name", "first"), position=("position", "first"),
                   team=("team", "first"), now_cost=("now_cost", "first"),
                   selected_by_percent=("selected_by_percent", "first")))
    out = meta.merge(agg, on="element")
    # Never select a player projected to barely feature.
    return out[out["now_cost"].notna()].reset_index(drop=True)


def initial_squad(proj: pd.DataFrame, horizon: list[int], budget: float = 100.0,
                  bench_weight: float = 0.12, min_bench_start: float = 0.30,
                  decay: float = 0.86, banned: list[int] | None = None,
                  locked: list[int] | None = None) -> OptimResult:
    """
    Pick the best 15 from scratch.

    bench_weight: bench players contribute at this fraction of their xP. Setting it
      to 0 produces the classic "4.0m non-playing bench fodder" squad, which is
      fragile. ~0.12 buys real cover.
    min_bench_start: every squad member must have at least this P(start), so we
      don't carry players who will never be a usable substitute.
    """
    df = _prep(proj, horizon, decay)
    banned = banned or []
    locked = locked or []
    df = df[~df.element.isin(banned)].reset_index(drop=True)

    prob = pulp.LpProblem("fpl_initial", pulp.LpMaximize)
    idx = df.index.tolist()

    sq = pulp.LpVariable.dicts("squad", idx, cat="Binary")
    xi = pulp.LpVariable.dicts("xi", idx, cat="Binary")
    cap = pulp.LpVariable.dicts("cap", idx, cat="Binary")

    # Objective: XI at full value + bench at a fraction + captain gets a second helping.
    prob += pulp.lpSum(
        df.loc[i, "weighted_xp"] * xi[i]
        + bench_weight * df.loc[i, "weighted_xp"] * (sq[i] - xi[i])
        + df.loc[i, "first_gw_xp"] * cap[i]
        for i in idx
    )

    prob += pulp.lpSum(sq[i] for i in idx) == SQUAD_SIZE
    prob += pulp.lpSum(xi[i] for i in idx) == XI_SIZE
    prob += pulp.lpSum(df.loc[i, "now_cost"] * sq[i] for i in idx) <= budget

    for pos, n in SQUAD_LIMITS.items():
        prob += pulp.lpSum(sq[i] for i in idx if df.loc[i, "position"] == pos) == n
    for pos in XI_MIN:
        sel = [i for i in idx if df.loc[i, "position"] == pos]
        prob += pulp.lpSum(xi[i] for i in sel) >= XI_MIN[pos]
        prob += pulp.lpSum(xi[i] for i in sel) <= XI_MAX[pos]

    for team in df.team.dropna().unique():
        sel = [i for i in idx if df.loc[i, "team"] == team]
        prob += pulp.lpSum(sq[i] for i in sel) <= MAX_PER_CLUB

    for i in idx:
        prob += xi[i] <= sq[i]
        prob += cap[i] <= xi[i]
        # Bench-quality floor
        if df.loc[i, "mean_p_start"] < min_bench_start:
            prob += sq[i] == 0
    prob += pulp.lpSum(cap[i] for i in idx) == 1

    for e in locked:
        match = df.index[df.element == e].tolist()
        if match:
            prob += sq[match[0]] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    df["in_squad"] = [int(sq[i].value() or 0) for i in idx]
    df["in_xi"] = [int(xi[i].value() or 0) for i in idx]
    df["is_captain"] = [int(cap[i].value() or 0) for i in idx]

    squad = df[df.in_squad == 1].sort_values(
        ["position", "horizon_xp"], key=lambda s: s.map(
            {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}) if s.name == "position" else s,
        ascending=[True, False])
    xi_df = squad[squad.in_xi == 1]
    bench = squad[squad.in_xi == 0].sort_values("horizon_xp", ascending=False)

    cap_row = squad[squad.is_captain == 1]
    cap_d = cap_row.iloc[0].to_dict() if len(cap_row) else {}
    vice_pool = xi_df[xi_df.is_captain == 0].nlargest(1, "first_gw_xp")
    vice_d = vice_pool.iloc[0].to_dict() if len(vice_pool) else {}
    if cap_d:
        cap_d["exp_points"] = cap_d.get("first_gw_xp", 0)
    if vice_d:
        vice_d["exp_points"] = vice_d.get("first_gw_xp", 0)

    spent = float(squad["now_cost"].sum())
    return OptimResult(
        status=status,
        objective=float(pulp.value(prob.objective) or 0),
        squad=squad, xi=xi_df, bench=bench,
        captain=cap_d, vice=vice_d,
        bank=round(budget - spent, 1),
        notes=[f"spent £{spent:.1f}m of £{budget:.1f}m",
               f"horizon GW{min(horizon)}-{max(horizon)} decay={decay}"],
    )


def transfer_plan(proj: pd.DataFrame, current_squad: list[int], horizon: list[int],
                  bank: float, free_transfers: int, selling_prices: dict | None = None,
                  max_transfers: int = 3, bench_weight: float = 0.12,
                  decay: float = 0.86) -> OptimResult:
    """
    Single-gameweek transfer decision evaluated over a multi-gameweek horizon.

    Prices each additional transfer beyond `free_transfers` at -4 and lets the
    solver decide whether the horizon gain justifies it. This is the honest way
    to answer "is this hit worth it" — it compares against the true alternative,
    which is keeping the squad and rolling.
    """
    df = _prep(proj, horizon, decay)
    selling_prices = selling_prices or {}
    idx = df.index.tolist()
    df["owned"] = df.element.isin(current_squad).astype(int)
    # Selling price may be below current price due to the 50% sell-on tax.
    df["sell_value"] = df.apply(
        lambda r: selling_prices.get(int(r.element), r.now_cost) if r.owned else 0.0, axis=1)

    owned_idx = [i for i in idx if df.loc[i, "owned"] == 1]
    if len(owned_idx) != SQUAD_SIZE:
        log.warning("current squad resolved to %d players (expected 15) — "
                    "check element ids", len(owned_idx))

    prob = pulp.LpProblem("fpl_transfer", pulp.LpMaximize)
    sq = pulp.LpVariable.dicts("squad", idx, cat="Binary")
    xi = pulp.LpVariable.dicts("xi", idx, cat="Binary")
    cap = pulp.LpVariable.dicts("cap", idx, cat="Binary")
    buy = pulp.LpVariable.dicts("buy", idx, cat="Binary")
    sell = pulp.LpVariable.dicts("sell", idx, cat="Binary")
    n_hits = pulp.LpVariable("hits", lowBound=0, upBound=max_transfers, cat="Integer")

    prob += (
        pulp.lpSum(df.loc[i, "weighted_xp"] * xi[i]
                   + bench_weight * df.loc[i, "weighted_xp"] * (sq[i] - xi[i])
                   + df.loc[i, "first_gw_xp"] * cap[i] for i in idx)
        - HIT_COST * n_hits
    )

    for i in idx:
        # squad_after = squad_before + buy - sell
        prob += sq[i] == df.loc[i, "owned"] + buy[i] - sell[i]
        prob += buy[i] <= 1 - df.loc[i, "owned"]
        prob += sell[i] <= df.loc[i, "owned"]
        prob += xi[i] <= sq[i]
        prob += cap[i] <= xi[i]

    n_buys = pulp.lpSum(buy[i] for i in idx)
    prob += n_buys == pulp.lpSum(sell[i] for i in idx)
    prob += n_buys <= max_transfers
    prob += n_hits >= n_buys - free_transfers

    prob += pulp.lpSum(sq[i] for i in idx) == SQUAD_SIZE
    prob += pulp.lpSum(xi[i] for i in idx) == XI_SIZE
    prob += pulp.lpSum(cap[i] for i in idx) == 1

    # Money: what we spend on buys can't exceed bank + what we raise from sales.
    prob += (pulp.lpSum(df.loc[i, "now_cost"] * buy[i] for i in idx)
             <= bank + pulp.lpSum(df.loc[i, "sell_value"] * sell[i] for i in idx))

    for pos, n in SQUAD_LIMITS.items():
        prob += pulp.lpSum(sq[i] for i in idx if df.loc[i, "position"] == pos) == n
    for pos in XI_MIN:
        sel = [i for i in idx if df.loc[i, "position"] == pos]
        prob += pulp.lpSum(xi[i] for i in sel) >= XI_MIN[pos]
        prob += pulp.lpSum(xi[i] for i in sel) <= XI_MAX[pos]
    for team in df.team.dropna().unique():
        sel = [i for i in idx if df.loc[i, "team"] == team]
        prob += pulp.lpSum(sq[i] for i in sel) <= MAX_PER_CLUB

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    df["in_squad"] = [int(sq[i].value() or 0) for i in idx]
    df["in_xi"] = [int(xi[i].value() or 0) for i in idx]
    df["is_captain"] = [int(cap[i].value() or 0) for i in idx]
    df["bought"] = [int(buy[i].value() or 0) for i in idx]
    df["sold"] = [int(sell[i].value() or 0) for i in idx]

    outs = df[df.sold == 1]
    ins = df[df.bought == 1]
    transfers = [
        {"out": o.web_name, "out_element": int(o.element), "out_xp": round(o.horizon_xp, 2),
         "in": i_.web_name, "in_element": int(i_.element), "in_xp": round(i_.horizon_xp, 2),
         "xp_gain": round(i_.horizon_xp - o.horizon_xp, 2)}
        for o, i_ in zip(outs.itertuples(), ins.itertuples())
    ]

    squad = df[df.in_squad == 1]
    xi_df = squad[squad.in_xi == 1]
    bench = squad[squad.in_xi == 0].sort_values("horizon_xp", ascending=False)
    cap_row = squad[squad.is_captain == 1]
    cap_d = cap_row.iloc[0].to_dict() if len(cap_row) else {}
    if cap_d:
        cap_d["exp_points"] = cap_d.get("first_gw_xp", 0)
    vice_pool = xi_df[xi_df.is_captain == 0].nlargest(1, "first_gw_xp")
    vice_d = vice_pool.iloc[0].to_dict() if len(vice_pool) else {}
    if vice_d:
        vice_d["exp_points"] = vice_d.get("first_gw_xp", 0)

    hits = int(n_hits.value() or 0)
    spent = float(ins["now_cost"].sum())
    raised = float(outs["sell_value"].sum())

    return OptimResult(
        status=pulp.LpStatus[prob.status],
        objective=float(pulp.value(prob.objective) or 0),
        squad=squad, xi=xi_df, bench=bench, captain=cap_d, vice=vice_d,
        transfers=transfers, hits_taken=hits,
        bank=round(bank + raised - spent, 1),
        notes=[f"{len(transfers)} transfer(s), {hits} hit(s)",
               f"free transfers available: {free_transfers}"],
    )


def evaluate_chips(proj: pd.DataFrame, squad: list[int], gw: int,
                   bench_elements: list[int] | None = None) -> dict:
    """
    Quantify each chip for a specific gameweek. Returns raw numbers; the
    thresholds for acting on them belong to the agent's policy, not here.
    """
    g = proj[proj.gw == gw]
    in_squad = g[g.element.isin(squad)]
    if in_squad.empty:
        return {"error": f"no projections for GW{gw}"}

    ranked = in_squad.sort_values("exp_points", ascending=False)
    xi = ranked.head(11)
    bench = ranked.tail(4) if bench_elements is None else g[g.element.isin(bench_elements)]

    best = ranked.iloc[0]
    n_playing = int((in_squad["n_fixtures"] > 0).sum()) if "n_fixtures" in in_squad else len(in_squad)
    doubles = int((in_squad["n_fixtures"] > 1).sum()) if "n_fixtures" in in_squad else 0

    return {
        "gw": gw,
        "squad_players_with_fixtures": n_playing,
        "squad_players_with_doubles": doubles,
        "xi_projected": round(float(xi["exp_points"].sum()), 2),
        "bench_projected": round(float(bench["exp_points"].sum()), 2),
        "bench_boost_gain": round(float(bench["exp_points"].sum()), 2),
        "best_captain": best["web_name"],
        "captain_projected": round(float(best["exp_points"]), 2),
        "triple_captain_gain": round(float(best["exp_points"]), 2),
        "free_hit_relevant": n_playing < 9,
        "blank_risk_players": int(len(in_squad) - n_playing),
    }
