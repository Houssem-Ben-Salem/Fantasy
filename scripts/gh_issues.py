"""
Thin wrapper over the `gh` CLI for the alert issues, stdlib only, so the alert
steps work even when `pip install` is what failed. GitHub emails the repository
owner when an issue is opened or commented on — that is the whole delivery
mechanism, with no accounts or API keys beyond the workflow's own token.
"""
from __future__ import annotations

import json
import subprocess

PREFIX = "[fplbrain]"


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def list_alerts() -> list[dict]:
    """Every [fplbrain] issue, open or closed, with bodies and comments."""
    out = _gh("issue", "list", "--state", "all", "--limit", "100",
              "--json", "number,title,state,body,comments")
    return [i for i in json.loads(out) if i["title"].startswith(PREFIX)]


def texts(issue: dict) -> list[str]:
    return [issue.get("body") or ""] + [c.get("body") or "" for c in issue.get("comments", [])]


def create(title: str, body: str) -> str:
    return _gh("issue", "create", "--title", title, "--body", body).strip()


def comment(number: int, body: str) -> None:
    _gh("issue", "comment", str(number), "--body", body)


def close(number: int, body: str) -> None:
    _gh("issue", "close", str(number), "--comment", body)
