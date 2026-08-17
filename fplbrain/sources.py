"""
Data acquisition layer.

KEY DESIGN DECISION
-------------------
`fantasy.premierleague.com` is blocked from many sandboxed agent environments
(Claude's code tool, most CI egress allowlists). `raw.githubusercontent.com` is not.

The vaastav/Fantasy-Premier-League repo mirrors the FPL API's `bootstrap-static`
elements straight to `players_raw.csv`, and the fixture list to `fixtures.csv`,
for the CURRENT season. So we can run the entire pipeline with zero access to the
live FPL API.

We therefore fetch every source through a `prefer` chain:
    live API  ->  GitHub mirror  ->  local cache
and record which path succeeded, so downstream consumers know how stale the data is.

Set FPLBRAIN_ALLOW_LIVE=0 to skip live attempts entirely (faster in sandboxes).
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

FPL_API = "https://fantasy.premierleague.com/api"
GH_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Fallback mirror; different maintainer, same FPL IDs.
GH_RAW_ALT = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"

CACHE = Path(os.environ.get("FPLBRAIN_CACHE", "./data/cache"))
CACHE.mkdir(parents=True, exist_ok=True)

ALLOW_LIVE = os.environ.get("FPLBRAIN_ALLOW_LIVE", "1") == "1"

# The live FPL API only ever serves the CURRENT season. `bootstrap-static` and
# `/fixtures/` have no season parameter and no historical mode. Requesting them
# while loading 2024-25 returns 2026-27 data, which then gets written to the
# database stamped with the wrong season.
#
# This is why every live call is gated on `_is_current(season)`. Historical
# seasons ALWAYS come from the GitHub mirror, which is season-addressable.
CURRENT_SEASON = os.environ.get("FPLBRAIN_SEASON", "2026-27")


def _is_current(season: str) -> bool:
    return season == CURRENT_SEASON

HEADERS = {
    "User-Agent": "fplbrain/1.0 (personal FPL analytics; contact via github)",
    "Accept": "application/json, text/csv",
}

# Live-API politeness. The FPL API has no published limit but 503s under load.
MIN_INTERVAL = 0.7
_last_call = [0.0]


@dataclass
class Fetched:
    """Result of a fetch, carrying provenance so the agent can reason about staleness."""

    data: object
    path: str  # "live" | "github" | "github_alt" | "cache"
    fetched_at: datetime
    url: str

    @property
    def is_live(self) -> bool:
        return self.path == "live"


def _throttle() -> None:
    delta = time.time() - _last_call[0]
    if delta < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - delta)
    _last_call[0] = time.time()


def _get(url: str, timeout: int = 30) -> requests.Response:
    _throttle()
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r


def _cache_path(key: str, ext: str) -> Path:
    return CACHE / f"{key}.{ext}"


def _write_cache(key: str, ext: str, content: bytes, path_label: str = "unknown") -> None:
    _cache_path(key, ext).write_bytes(content)
    # Record HOW this file was obtained. Without this, a cached file written by
    # the mirror gets reported as "live" on the next read, and the provenance
    # in the run log becomes fiction.
    (CACHE / f"{key}.provenance").write_text(path_label)


def _read_cache(key: str, ext: str) -> bytes | None:
    p = _cache_path(key, ext)
    return p.read_bytes() if p.exists() else None


def _cache_provenance(key: str) -> str | None:
    p = CACHE / f"{key}.provenance"
    return p.read_text().strip() if p.exists() else None


# --------------------------------------------------------------------------
# Current season: players, teams, fixtures
# --------------------------------------------------------------------------


def current_players(season: str) -> Fetched:
    """
    All players for the current season with prices, ownership, positions,
    per-90 rates, set-piece order, and defensive-contribution fields.

    Live path returns bootstrap-static['elements']; the mirror returns the
    identical rows already flattened to CSV.
    """
    now = datetime.now(timezone.utc)

    # bootstrap-static is current-season-only. Never call it for history.
    if ALLOW_LIVE and _is_current(season):
        try:
            r = _get(f"{FPL_API}/bootstrap-static/")
            payload = r.json()
            df = pd.DataFrame(payload["elements"])
            _write_cache(f"players_{season}", "csv",
                         df.to_csv(index=False).encode(), "live")
            # Cache the sibling tables too — one request gets us all three.
            _write_cache(f"teams_{season}", "csv",
                         pd.DataFrame(payload["teams"]).to_csv(index=False).encode(), "live")
            _write_cache(f"events_{season}", "csv",
                         pd.DataFrame(payload["events"]).to_csv(index=False).encode(), "live")
            return Fetched(df, "live", now, f"{FPL_API}/bootstrap-static/")
        except Exception as e:  # noqa: BLE001
            log.warning("live players fetch failed (%s); falling back to mirror", e)

    for label, base in (("github", GH_RAW), ("github_alt", GH_RAW_ALT)):
        url = f"{base}/{season}/players_raw.csv"
        try:
            r = _get(url)
            _write_cache(f"players_{season}", "csv", r.content, label)
            return Fetched(pd.read_csv(io.BytesIO(r.content)), label, now, url)
        except Exception as e:  # noqa: BLE001
            log.warning("%s players fetch failed (%s)", label, e)

    cached = _read_cache(f"players_{season}", "csv")
    if cached:
        return Fetched(pd.read_csv(io.BytesIO(cached)), "cache", now, "cache")
    raise RuntimeError(f"could not obtain players for {season} from any source")


def current_teams(season: str) -> Fetched:
    now = datetime.now(timezone.utc)

    # current_players() caches teams as a side effect of bootstrap-static, but
    # only trust that cache if it was genuinely written by the live path for
    # THIS season — otherwise we'd report mirror data as live.
    key = f"teams_{season}"
    cached = _cache_path(key, "csv")
    if _is_current(season) and cached.exists() and _cache_provenance(key) == "live":
        return Fetched(pd.read_csv(cached), "live", now, "bootstrap-static")

    for label, base in (("github", GH_RAW), ("github_alt", GH_RAW_ALT)):
        url = f"{base}/{season}/teams.csv"
        try:
            r = _get(url)
            _write_cache(key, "csv", r.content, label)
            return Fetched(pd.read_csv(io.BytesIO(r.content)), label, now, url)
        except Exception as e:  # noqa: BLE001
            log.warning("%s teams fetch failed (%s)", label, e)

    if cached.exists():
        return Fetched(pd.read_csv(cached), _cache_provenance(key) or "cache", now, "cache")
    raise RuntimeError(f"could not obtain teams for {season}")


def current_fixtures(season: str) -> Fetched:
    """Full 380-fixture list with official FDR and post-match results."""
    now = datetime.now(timezone.utc)

    # /fixtures/ is current-season-only, same trap as bootstrap-static.
    if ALLOW_LIVE and _is_current(season):
        try:
            r = _get(f"{FPL_API}/fixtures/")
            df = pd.DataFrame(r.json())
            _write_cache(f"fixtures_{season}", "csv",
                         df.to_csv(index=False).encode(), "live")
            return Fetched(df, "live", now, f"{FPL_API}/fixtures/")
        except Exception as e:  # noqa: BLE001
            log.warning("live fixtures fetch failed (%s)", e)

    for label, base in (("github", GH_RAW), ("github_alt", GH_RAW_ALT)):
        url = f"{base}/{season}/fixtures.csv"
        try:
            r = _get(url)
            _write_cache(f"fixtures_{season}", "csv", r.content, label)
            return Fetched(pd.read_csv(io.BytesIO(r.content)), label, now, url)
        except Exception as e:  # noqa: BLE001
            log.warning("%s fixtures fetch failed (%s)", label, e)
    raise RuntimeError(f"could not obtain fixtures for {season}")


# --------------------------------------------------------------------------
# Historical gameweek data
# --------------------------------------------------------------------------


def season_gameweeks(season: str) -> Fetched:
    """
    Gameweek-level rows for every player in a completed (or in-progress) season.

    Includes the fields that matter for 2026/27 scoring:
      tackles, recoveries, clearances_blocks_interceptions, defensive_contribution
    """
    now = datetime.now(timezone.utc)
    for label, base in (("github", GH_RAW), ("github_alt", GH_RAW_ALT)):
        url = f"{base}/{season}/gws/merged_gw.csv"
        try:
            r = _get(url, timeout=90)
            _write_cache(f"gw_{season}", "csv", r.content, label)
            df = pd.read_csv(io.BytesIO(r.content))
            return Fetched(df, label, now, url)
        except Exception as e:  # noqa: BLE001
            log.warning("%s gameweeks fetch failed for %s (%s)", label, season, e)

    cached = _read_cache(f"gw_{season}", "csv")
    if cached:
        return Fetched(pd.read_csv(io.BytesIO(cached)), "cache", now, "cache")
    raise RuntimeError(f"could not obtain gameweek data for {season}")


def live_gameweek(gw: int) -> Fetched | None:
    """In-progress gameweek stats. Live API only — no mirror exists."""
    if not ALLOW_LIVE:
        return None
    try:
        r = _get(f"{FPL_API}/event/{gw}/live/")
        return Fetched(r.json(), "live", datetime.now(timezone.utc), f"{FPL_API}/event/{gw}/live/")
    except Exception as e:  # noqa: BLE001
        log.warning("live gameweek %s unavailable (%s)", gw, e)
        return None


# --------------------------------------------------------------------------
# Manager's own team
# --------------------------------------------------------------------------


def manager_picks(entry_id: int, gw: int) -> Fetched | None:
    """Your actual squad for a gameweek. Public endpoint, no auth needed."""
    if not ALLOW_LIVE:
        return None
    try:
        r = _get(f"{FPL_API}/entry/{entry_id}/event/{gw}/picks/")
        return Fetched(r.json(), "live", datetime.now(timezone.utc), r.url)
    except Exception as e:  # noqa: BLE001
        log.warning("manager picks unavailable (%s) — supply squad manually", e)
        return None


def manager_history(entry_id: int) -> Fetched | None:
    if not ALLOW_LIVE:
        return None
    try:
        r = _get(f"{FPL_API}/entry/{entry_id}/history/")
        return Fetched(r.json(), "live", datetime.now(timezone.utc), r.url)
    except Exception as e:  # noqa: BLE001
        log.warning("manager history unavailable (%s)", e)
        return None


# --------------------------------------------------------------------------
# Team strength prior
# --------------------------------------------------------------------------


def club_elo(on: str | None = None) -> pd.DataFrame | None:
    """
    Club Elo ratings — a far better fixture-difficulty prior than the official FDR.
    Free, no key. Returns None if unreachable (it is not on most allowlists).
    """
    day = on or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        r = _get(f"http://api.clubelo.com/{day}")
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:  # noqa: BLE001
        log.info("Club Elo unavailable (%s) — will fall back to FPL team strength", e)
        return None


def probe_connectivity() -> dict:
    """
    Run this FIRST in any new environment. Tells you which paths are open
    so you stop guessing and design the run accordingly.
    """
    targets = {
        "fpl_api": f"{FPL_API}/bootstrap-static/",
        "github_raw": f"{GH_RAW}/2025-26/teams.csv",
        "github_alt": f"{GH_RAW_ALT}/2025-2026/teams.csv",
        "club_elo": "http://api.clubelo.com/2026-08-16",
        "understat": "https://understat.com/league/EPL",
    }
    out = {}
    for name, url in targets.items():
        try:
            r = requests.get(url, headers=HEADERS, timeout=12, stream=True)
            out[name] = {"ok": r.status_code == 200, "status": r.status_code}
            r.close()
        except Exception as e:  # noqa: BLE001
            out[name] = {"ok": False, "error": type(e).__name__}
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(probe_connectivity(), indent=2))
