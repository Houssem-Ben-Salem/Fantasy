"""Constraint tests. Run with: python -m pytest tests/ -q"""
import sqlite3, warnings
import pytest
warnings.filterwarnings("ignore")
from fplbrain import models, optimize

DB = "data/fpl.db"
SEASON, HIST = "2026-27", ["2024-25", "2025-26"]


@pytest.fixture(scope="module")
def proj():
    c = sqlite3.connect(DB)
    p, _ = models.project_horizon(c, SEASON, 1, 6, HIST)
    return p


@pytest.fixture(scope="module")
def squad(proj):
    return optimize.initial_squad(proj, list(range(1, 7)))


def test_squad_size(squad):        assert len(squad.squad) == 15
def test_positions(squad):         assert squad.squad.position.value_counts().to_dict() == {"DEF":5,"MID":5,"GK":2,"FWD":3}
def test_budget(squad):            assert squad.squad.now_cost.sum() <= 100.001
def test_club_limit(squad):        assert squad.squad.team.value_counts().max() <= 3
def test_xi_size(squad):           assert squad.squad.in_xi.sum() == 11
def test_one_captain(squad):       assert squad.squad.is_captain.sum() == 1
def test_optimal(squad):           assert squad.status == "Optimal"


def test_xi_formation_legal(squad):
    xi = squad.squad[squad.squad.in_xi == 1].position.value_counts().to_dict()
    assert xi.get("GK") == 1 and xi.get("DEF", 0) >= 3 and xi.get("FWD", 0) >= 1


def test_captain_is_in_xi(squad):
    cap = squad.squad[squad.squad.is_captain == 1]
    assert cap.in_xi.iloc[0] == 1


def test_no_negative_xp(proj):
    assert (proj.exp_points >= 0).all()


def test_defcon_bounded(proj):
    assert proj.p_defcon.between(0, 1).all()


def test_transfer_respects_budget(proj, squad):
    ids = squad.squad.element.tolist()
    r = optimize.transfer_plan(proj, ids, list(range(2, 7)), bank=0.0, free_transfers=1)
    assert r.status == "Optimal" and r.bank >= -0.001
    assert len(r.squad) == 15


def test_goalkeepers_never_get_defcon(proj):
    gk = proj[proj.position == "GK"]
    assert (gk.p_defcon.fillna(0) == 0).all() or gk.empty
