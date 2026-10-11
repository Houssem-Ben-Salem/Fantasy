"""
The alert logic is what stands between a scheduled run and a missed deadline
or a silent red CI, so every branch is pinned here — without GitHub, without
the network. (GW6 2026-27 was missed; CI was red for 17 days unnoticed.)
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ci_alert  # noqa: E402
import deadline_brief  # noqa: E402
import deadline_window as dw  # noqa: E402

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
DL = datetime(2026, 10, 17, 10, 0, tzinfo=timezone.utc)          # GW7, 46h away


def _issue(n, title, state="OPEN", body="", comments=()):
    return {"number": n, "title": title, "state": state, "body": body,
            "comments": [{"body": c} for c in comments]}


def _do(actions):
    return [a["do"] for a in actions]


# ------------------------------------------------------------ deadline window

def test_too_early_does_nothing():
    assert dw.decide(DL - dw.BRIEF_LEAD - timedelta(hours=1), 7, DL, []) == []


def test_brief_inside_48h_once():
    assert _do(dw.decide(NOW, 7, DL, [])) == ["brief"]
    assert dw.decide(NOW, 7, DL, [_issue(1, dw.title(7))]) == []


def test_final_call_inside_the_final_window_once():
    late = DL - dw.FINAL_LEAD + timedelta(hours=1)
    assert _do(dw.decide(late, 7, DL, [_issue(1, dw.title(7))])) == ["final"]
    done = _issue(1, dw.title(7), comments=[f"x {dw.FINAL_MARK}"])
    assert dw.decide(late, 7, DL, [done]) == []


def test_a_late_first_run_posts_a_brief_that_is_the_final_call():
    acts = dw.decide(DL - dw.FINAL_LEAD + timedelta(hours=1), 7, DL, [])
    assert _do(acts) == ["brief"] and acts[0]["final"] is True


def test_previous_briefs_close_and_nothing_posts_after_the_deadline():
    old = _issue(1, dw.title(6))
    assert _do(dw.decide(NOW, 7, DL, [old])) == ["close", "brief"]
    assert dw.decide(DL + timedelta(minutes=1), 7, DL, []) == []


def test_ci_alert_issues_are_never_closed_by_the_brief():
    ci = _issue(9, ci_alert.title("refresh-fpl-data"))
    assert "close" not in _do(dw.decide(NOW, 7, DL, [ci]))


def test_next_deadline_skips_passed_gameweeks():
    events = [{"id": 6, "deadline_time": "2026-10-10T10:00:00Z"},
              {"id": 7, "deadline_time": "2026-10-17T10:00:00Z"}]
    assert dw.next_deadline(events, NOW) == (7, DL)


# ------------------------------------------------------------ CI alert

def test_first_failure_opens_then_comments_then_recovery_closes():
    wf, url = "refresh-fpl-data", "https://x/run/1"
    assert ci_alert.plan("fail", wf, [], url, "now")[0] == "create"
    open_ = [_issue(5, ci_alert.title(wf))]
    assert ci_alert.plan("fail", wf, open_, url, "now")[:2] == ("comment", 5)
    assert ci_alert.plan("ok", wf, open_, url, "now")[:2] == ("close", 5)
    assert ci_alert.plan("ok", wf, [], url, "now") == ("none",)
    closed = [_issue(5, ci_alert.title(wf), state="CLOSED")]
    assert ci_alert.plan("fail", wf, closed, url, "now")[0] == "create"   # a new outage, a new issue


# ------------------------------------------------------------ final-call diff

def test_flag_changes_since_the_brief_are_reported():
    rows0 = [{"element": 229, "status": "a", "chance": None, "news": ""},
             {"element": 480, "status": "a", "chance": 100, "news": ""}]
    rows1 = [{"element": 229, "status": "i", "chance": 0, "news": "Hamstring - out 3 weeks"},
             {"element": 480, "status": "a", "chance": 100, "news": ""}]
    changes = deadline_brief.diff_flags(deadline_brief.flags_snapshot(rows0),
                                        deadline_brief.flags_snapshot(rows1),
                                        {"229": "Tarkowski", "480": "Gibbs-White"})
    assert len(changes) == 1 and "Tarkowski" in changes[0] and "Hamstring" in changes[0]


# ------------------------------------------------------------ preview (manual delivery test)

def test_preview_builds_the_coming_gameweek_whatever_the_window():
    acts = dw.preview(7, DL)
    assert acts == [{"do": "preview", "gw": 7, "deadline": DL.isoformat()}]


def test_a_preview_issue_is_never_mistaken_for_the_real_brief():
    """The window logic finds the real brief by exact title and closes stale
    briefs by prefix; a delivery test must match neither, or it would suppress
    Thursday's brief or be closed as stale."""
    test_issue = _issue(3, dw.preview_title(7), state="CLOSED")
    assert dw.preview_title(7) != dw.title(7)
    assert _do(dw.decide(NOW, 7, DL, [test_issue])) == ["brief"]
    assert "close" not in _do(dw.decide(NOW, 7, DL, [{**test_issue, "state": "OPEN"}]))
