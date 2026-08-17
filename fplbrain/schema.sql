-- fplbrain schema
-- Designed so an LLM agent can answer most questions with ONE readable SQL query.
-- Wide-but-tidy fact table + small dimensions + pre-built views.

PRAGMA journal_mode = WAL;

-- ---------------------------------------------------------------- dimensions

DROP TABLE IF EXISTS teams;
CREATE TABLE teams (
    season              TEXT NOT NULL,
    team_id             INTEGER NOT NULL,   -- season-scoped FPL id (1..20)
    code                INTEGER,            -- stable across seasons
    name                TEXT,
    short_name          TEXT,
    strength            INTEGER,
    str_overall_home    INTEGER,
    str_overall_away    INTEGER,
    str_attack_home     INTEGER,
    str_attack_away     INTEGER,
    str_defence_home    INTEGER,
    str_defence_away    INTEGER,
    elo                 REAL,               -- from Club Elo when reachable
    attack_rating       REAL,               -- fitted by models.goals
    defence_rating      REAL,
    PRIMARY KEY (season, team_id)
);

DROP TABLE IF EXISTS players;
CREATE TABLE players (
    season              TEXT NOT NULL,
    element             INTEGER NOT NULL,   -- season-scoped id. REASSIGNED EVERY SEASON.
    code                INTEGER NOT NULL,   -- STABLE across seasons -> join on this
    web_name            TEXT,
    first_name          TEXT,
    second_name         TEXT,
    team_id             INTEGER,
    position            TEXT,               -- GK / DEF / MID / FWD
    now_cost            REAL,               -- in £m
    status              TEXT,               -- a=available d=doubtful i=injured s=susp u=unavailable
    chance_next_round   REAL,
    news                TEXT,
    selected_by_percent REAL,
    minutes             INTEGER,
    starts              INTEGER,
    total_points        INTEGER,
    points_per_game     REAL,
    form                REAL,
    ep_next             REAL,               -- FPL's own projection: use as a benchmark, not an input
    xg_per90            REAL,
    xa_per90            REAL,
    xgi_per90           REAL,
    xgc_per90           REAL,
    defcon_per90        REAL,
    saves_per90         REAL,
    starts_per90        REAL,
    penalties_order     REAL,               -- 1 = first-choice penalty taker
    corners_fk_order    REAL,
    direct_fk_order     REAL,
    PRIMARY KEY (season, element)
);
CREATE INDEX idx_players_code ON players(code);
CREATE INDEX idx_players_team ON players(season, team_id);

DROP TABLE IF EXISTS fixtures;
CREATE TABLE fixtures (
    season          TEXT NOT NULL,
    fixture_id      INTEGER NOT NULL,
    gw              INTEGER,                -- NULL for postponed/unscheduled
    kickoff_time    TEXT,
    home_team       INTEGER,
    away_team       INTEGER,
    home_score      INTEGER,
    away_score      INTEGER,
    finished        INTEGER,
    fdr_home        INTEGER,                -- official FDR (coarse; we model our own)
    fdr_away        INTEGER,
    PRIMARY KEY (season, fixture_id)
);
CREATE INDEX idx_fixtures_gw ON fixtures(season, gw);

-- ---------------------------------------------------------------- fact table

DROP TABLE IF EXISTS player_gw;
CREATE TABLE player_gw (
    season          TEXT NOT NULL,
    element         INTEGER NOT NULL,
    code            INTEGER,                -- denormalised for cross-season joins
    gw              INTEGER NOT NULL,
    fixture_id      INTEGER,
    opponent_team   INTEGER,
    was_home        INTEGER,
    kickoff_time    TEXT,

    minutes         INTEGER,
    starts          INTEGER,
    total_points    INTEGER,
    goals_scored    INTEGER,
    assists         INTEGER,
    clean_sheets    INTEGER,
    goals_conceded  INTEGER,
    own_goals       INTEGER,
    penalties_saved INTEGER,
    penalties_missed INTEGER,
    yellow_cards    INTEGER,
    red_cards       INTEGER,
    saves           INTEGER,
    bonus           INTEGER,
    bps             INTEGER,

    -- defensive contribution inputs (the 2025/26+ scoring rule)
    tackles                         INTEGER,
    recoveries                      INTEGER,
    clearances_blocks_interceptions INTEGER,
    defensive_contribution          INTEGER,

    expected_goals              REAL,
    expected_assists            REAL,
    expected_goal_involvements  REAL,
    expected_goals_conceded     REAL,

    influence       REAL,
    creativity      REAL,
    threat          REAL,
    ict_index       REAL,

    value           REAL,   -- price at the time, £m
    selected        INTEGER,
    transfers_in    INTEGER,
    transfers_out   INTEGER,
    fpl_xp          REAL,   -- FPL's own 'xP'. Contains look-ahead; benchmark only.

    PRIMARY KEY (season, element, gw, fixture_id)
);
CREATE INDEX idx_pgw_code ON player_gw(code, season, gw);
CREATE INDEX idx_pgw_gw ON player_gw(season, gw);

-- ---------------------------------------------------------------- projections

CREATE TABLE IF NOT EXISTS projections (
    season          TEXT NOT NULL,
    element         INTEGER NOT NULL,
    gw              INTEGER NOT NULL,
    run_id          TEXT NOT NULL,          -- timestamp of the model run
    p_start         REAL,
    exp_minutes     REAL,
    p_60            REAL,
    exp_goals       REAL,
    exp_assists     REAL,
    p_clean_sheet   REAL,
    p_defcon        REAL,
    exp_bonus       REAL,
    exp_points      REAL,                   -- the headline xP
    sd_points       REAL,                   -- uncertainty; drives captaincy ceiling/floor
    PRIMARY KEY (season, element, gw, run_id)
);
CREATE INDEX IF NOT EXISTS idx_proj_run ON projections(run_id, gw);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT,
    season          TEXT,
    horizon_start   INTEGER,
    horizon_end     INTEGER,
    notes           TEXT,
    provenance      TEXT                     -- JSON: which source path each table came from
);

-- ---------------------------------------------------------------- squad state
--
-- DURABLE TABLES. Everything above is derived data that ingest fully rebuilds,
-- so dropping it is safe. Everything below is USER data or an accumulating
-- record. init_schema() runs on every ingest, so a DROP here would silently
-- destroy the squad you saved last week the first time the agent calls
-- refresh_data(). Use IF NOT EXISTS and never drop these.

CREATE TABLE IF NOT EXISTS squad_state (
    gw              INTEGER PRIMARY KEY,
    recorded_at     TEXT,
    bank            REAL,
    free_transfers  INTEGER,
    chips_remaining TEXT,                    -- JSON list
    squad           TEXT,                    -- JSON: [{element, purchase_price, selling_price}]
    notes           TEXT
);

-- ---------------------------------------------------------------- views

DROP VIEW IF EXISTS v_season_totals;
CREATE VIEW v_season_totals AS
SELECT
    g.season, g.code, g.element,
    p.web_name, p.position, t.short_name AS team,
    COUNT(*)                        AS apps,
    SUM(g.minutes)                  AS minutes,
    SUM(g.starts)                   AS starts,
    SUM(g.total_points)             AS points,
    SUM(g.goals_scored)             AS goals,
    SUM(g.assists)                  AS assists,
    SUM(g.clean_sheets)             AS clean_sheets,
    SUM(g.bonus)                    AS bonus,
    SUM(g.expected_goals)           AS xg,
    SUM(g.expected_assists)         AS xa,
    SUM(g.defensive_contribution)   AS defcon_actions,
    ROUND(90.0 * SUM(g.expected_goals)   / NULLIF(SUM(g.minutes),0), 3) AS xg_per90,
    ROUND(90.0 * SUM(g.expected_assists) / NULLIF(SUM(g.minutes),0), 3) AS xa_per90,
    ROUND(90.0 * SUM(g.defensive_contribution) / NULLIF(SUM(g.minutes),0), 2) AS defcon_per90,
    ROUND(1.0 * SUM(g.total_points) / NULLIF(COUNT(*),0), 2) AS ppg
FROM player_gw g
LEFT JOIN players p ON p.season = g.season AND p.element = g.element
LEFT JOIN teams   t ON t.season = g.season AND t.team_id  = p.team_id
GROUP BY g.season, g.element;

-- How often did each player actually clear the DefCon threshold?
-- DEF need 10 CBIT; MID/FWD need 12 CBIRT. GK excluded.
DROP VIEW IF EXISTS v_defcon_rate;
CREATE VIEW v_defcon_rate AS
SELECT
    g.season, g.code, g.element, p.web_name, p.position, t.short_name AS team,
    COUNT(*) AS apps,
    SUM(CASE WHEN g.minutes >= 60 THEN 1 ELSE 0 END) AS starts_60,
    SUM(CASE
          WHEN p.position = 'DEF' AND g.clearances_blocks_interceptions + g.tackles >= 10 THEN 1
          WHEN p.position IN ('MID','FWD')
               AND g.clearances_blocks_interceptions + g.tackles + g.recoveries >= 12 THEN 1
          ELSE 0 END) AS defcon_hits,
    ROUND(1.0 * SUM(CASE
          WHEN p.position = 'DEF' AND g.clearances_blocks_interceptions + g.tackles >= 10 THEN 1
          WHEN p.position IN ('MID','FWD')
               AND g.clearances_blocks_interceptions + g.tackles + g.recoveries >= 12 THEN 1
          ELSE 0 END) / NULLIF(SUM(CASE WHEN g.minutes >= 60 THEN 1 ELSE 0 END), 0), 3) AS defcon_rate_per_start
FROM player_gw g
JOIN players p ON p.season = g.season AND p.element = g.element
LEFT JOIN teams t ON t.season = g.season AND t.team_id = p.team_id
WHERE p.position != 'GK'
GROUP BY g.season, g.element;

-- Rolling last-6 form, computed per player.
DROP VIEW IF EXISTS v_form6;
CREATE VIEW v_form6 AS
SELECT season, element, code, gw, total_points, minutes,
       AVG(total_points) OVER (PARTITION BY season, element ORDER BY gw
                               ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS pts_avg6,
       AVG(minutes)      OVER (PARTITION BY season, element ORDER BY gw
                               ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS mins_avg6
FROM player_gw;

-- Latest projections joined to prices — the agent's main working table.
--
-- DO NOT "simplify" the run_id subquery back to
--     (SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1)
-- The `runs` table holds rows from BOTH ingest and projection runs. An ingest
-- run has no projection rows, so if it happens to be newest this view silently
-- goes empty. That is not hypothetical: mcp_server.refresh_data() calls
-- ingest.run() and nothing else, and AGENT.md tells the agent to call it at the
-- start of every session. Measured: 3,540 rows before the call, 0 after, while
-- 8,260 projection rows sat in the table unreachable.
--
-- Deriving the run from `projections` means the view can only ever resolve to a
-- run that actually has projection rows. We join rather than filter on
-- notes='projection' because that label is a string set in one place in
-- models.persist_projections() and can drift; the join cannot.
DROP VIEW IF EXISTS v_latest_projections;
CREATE VIEW v_latest_projections AS
SELECT
    pr.gw, pr.element, p.web_name, p.position, t.short_name AS team,
    p.now_cost, p.selected_by_percent, p.status, p.chance_next_round, p.news,
    pr.p_start, pr.exp_minutes, pr.p_clean_sheet, pr.p_defcon,
    pr.exp_goals, pr.exp_assists, pr.exp_bonus,
    pr.exp_points, pr.sd_points,
    ROUND(pr.exp_points / NULLIF(p.now_cost, 0), 3) AS xp_per_million
FROM projections pr
JOIN players p ON p.season = pr.season AND p.element = pr.element
LEFT JOIN teams t ON t.season = pr.season AND t.team_id = p.team_id
WHERE pr.run_id = (
    SELECT p2.run_id
    FROM projections p2
    JOIN runs r2 ON r2.run_id = p2.run_id
    ORDER BY r2.created_at DESC
    LIMIT 1
);
