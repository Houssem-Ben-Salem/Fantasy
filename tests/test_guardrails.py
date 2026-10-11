"""
The guardrails must hold against real git, not just in their own functions.

Three layers, each tested the way it is used:
  .githooks/pre-commit         nothing under data/ is committed; tests before code commits
  .githooks/pre-push           no push may rewrite or delete remote history
  .claude/hooks/guard_bash.py  an agent may not bypass the git hooks (--no-verify,
                               -c core.hooksPath=..., force pushes, add -f data/)
  .claude/hooks/session_status.py  never breaks a session, whatever is missing

The integration tests build throwaway repositories with a bare "remote", so
they need git on PATH and nothing else — no network, no data/fpl.db.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def _load(path: Path, name: str):
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


guard = _load(ROOT / ".claude/hooks/guard_bash.py", "guard_bash")
pre_push = _load(ROOT / ".githooks/pre-push", "pre_push")
pre_commit = _load(ROOT / ".githooks/pre-commit", "pre_commit")


# ------------------------------------------------------------ agent guard

@pytest.mark.parametrize("cmd, denied", [
    ("git push origin main", False),
    ("git push --force origin main", True),
    ("cd repo && git push -fu origin main", True),
    ("git push origin +main", True),
    ("FOO=1 git push --force-with-lease", True),
    ("git push -n origin main", False),                 # -n on push is a dry run
    ('git commit -m "fix -n handling"', False),          # a message is not a flag
    ('git commit -nm "x"', True),
    ("git commit --no-verify -m x", True),
    ("git -c core.hooksPath=/dev/null commit -m x", True),
    ("git config core.hooksPath .githooks", False),
    ("git config --unset core.hooksPath", True),
    ("git config core.hooksPath /tmp/none", True),
    ("git add -f data/fpl.db", True),
    ("git add -f .", True),
    ("git add -f fplbrain/models.py", False),
    ("ls -la && echo push --force", False),              # not a git command
])
def test_agent_guard(cmd, denied):
    assert (guard.verdict(cmd) is not None) == denied, guard.verdict(cmd)


# ------------------------------------------------------------ git hooks

def _git(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, capture_output=True,
                          text=True, check=check)


@pytest.fixture
def repo(tmp_path):
    """A clone with the repository's hooks installed, and a bare remote."""
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    _git(tmp_path, "init", "--bare", "-q", "-b", "main", str(remote))
    _git(tmp_path, "clone", "-q", str(remote), str(work))
    shutil.copytree(ROOT / ".githooks", work / ".githooks")
    _git(work, "config", "core.hooksPath", ".githooks")
    (work / "README").write_text("v1\n")
    _git(work, "add", "README")          # hooks installed, not committed: committing
                                         # .githooks/ would (rightly) trigger the suite
    _git(work, "commit", "-q", "-m", "init")
    _git(work, "push", "-q", "origin", "HEAD:main")
    return work


def test_a_data_file_cannot_be_committed(repo):
    (repo / "data").mkdir()
    (repo / "data" / "xmins_log.csv").write_text("gameweek\n")
    _git(repo, "add", "data/xmins_log.csv")
    r = _git(repo, "commit", "-m", "log", check=False)
    assert r.returncode != 0 and "data/" in r.stderr


def test_an_ordinary_commit_goes_through(repo):
    (repo / "README").write_text("v2\n")
    _git(repo, "add", "README")
    assert _git(repo, "commit", "-q", "-m", "v2", check=False).returncode == 0


def test_a_force_push_over_remote_history_is_refused(repo):
    (repo / "README").write_text("v2\n")
    _git(repo, "commit", "-q", "-am", "v2")
    _git(repo, "push", "-q", "origin", "HEAD:main")
    _git(repo, "reset", "-q", "--hard", "HEAD~1")         # rewrite: drop the pushed commit
    (repo / "README").write_text("v2-rewritten\n")
    _git(repo, "commit", "-q", "-am", "rewrite")
    r = _git(repo, "push", "--force", "origin", "HEAD:main", check=False)
    assert r.returncode != 0 and "rewrite" in r.stderr


def test_a_fast_forward_push_goes_through(repo):
    (repo / "README").write_text("v2\n")
    _git(repo, "commit", "-q", "-am", "v2")
    assert _git(repo, "push", "-q", "origin", "HEAD:main", check=False).returncode == 0


def test_pre_push_rules():
    z = pre_push.ZERO
    always, never = (lambda a, b: True), (lambda a, b: False)
    assert pre_push.rewrites([f"refs/heads/main {'a' * 40} refs/heads/main {'b' * 40}"], always) == []
    assert pre_push.rewrites([f"refs/heads/main {'a' * 40} refs/heads/main {'b' * 40}"], never)
    assert pre_push.rewrites([f"refs/heads/x {'a' * 40} refs/heads/x {z}"], never) == []  # new branch
    assert pre_push.rewrites([f"(delete) {z} refs/heads/main {'b' * 40}"], always)        # deletion


def test_pre_commit_rules():
    assert pre_commit.forbidden(["data/fpl.db", "fplbrain/x.py", "database.md"]) == ["data/fpl.db"]
    assert pre_commit.needs_tests(["fplbrain/x.py"]) and not pre_commit.needs_tests(["README"])


# ------------------------------------------------------------ session status

def test_session_status_survives_an_empty_checkout(tmp_path):
    """No database, no git, no network for this checkout: still valid JSON, exit 0."""
    hooks = tmp_path / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    shutil.copy(ROOT / ".claude/hooks/session_status.py", hooks)
    r = subprocess.run([sys.executable, str(hooks / "session_status.py")], input="{}",
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "database unavailable" in out["hookSpecificOutput"]["additionalContext"]
