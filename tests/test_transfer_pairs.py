"""
transfer_plan must report each transfer as a like-for-like pair.

It paired sold and bought players with a plain zip in element-id order. Ids
interleave positions in real data, so GW7 2026-27 printed "Palmer -> Calafiori;
Mykolenko -> Saka": a midfielder apparently swapped for a defender, and each
pair's xp_gain computed across positions. The squad itself was right — only the
report a person acts on was wrong.

The ids below are chosen to force that interleaving: every owned DEF has a lower
id than every owned MID, while the target MID's id is lower than the target
DEF's, so id order crosses the pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplbrain import optimize

pytest.importorskip("highspy")

HZ = [1, 2, 3]
IDS = {"GK": range(1, 5), "DEF": range(100, 112), "MID": range(200, 212), "FWD": range(300, 308)}
STAR_MID, STAR_DEF = 50, 900


def _proj() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for pos, ids in IDS.items():
        for k, e in enumerate(ids):
            for gw in HZ:
                rows.append(dict(element=e, gw=gw, web_name=f"{pos}{k}", position=pos,
                                 team=f"T{e % 12}", now_cost=4.5 + (k % 4),
                                 selected_by_percent=1.0, exp_points=2.0 + 4.0 * rng.random(),
                                 p_start=0.9, sd_points=2.0))
    for e, pos in ((STAR_MID, "MID"), (STAR_DEF, "DEF")):
        for gw in HZ:
            rows.append(dict(element=e, gw=gw, web_name=f"star{pos}", position=pos,
                             team="T_star" + pos, now_cost=4.5, selected_by_percent=1.0,
                             exp_points=30.0, p_start=0.9, sd_points=2.0))
    return pd.DataFrame(rows)


def test_transfers_are_reported_position_for_position():
    proj = _proj()
    stars = proj.element.isin([STAR_MID, STAR_DEF])
    owned = optimize.initial_squad(proj[~stars], HZ).squad.element.tolist()
    r = optimize.transfer_plan(proj, owned, HZ, bank=0.0, free_transfers=2)

    assert sorted(t["in_element"] for t in r.transfers) == [STAR_MID, STAR_DEF]
    pos = proj.drop_duplicates("element").set_index("element").position
    for t in r.transfers:
        assert pos[t["out_element"]] == pos[t["in_element"]], t
