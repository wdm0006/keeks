# CFB closing-line data acquisition & viability check — stage 1 report

**Repo task:** todo_jSkqSbgO (CFB profitability spike, stage 1) · **Workspace:** `/home/user/work/cfb-spike/` · **Run date:** 2026-09-08

---

## VERDICT: **NO-GO** for a two-way moneyline backtest from this source

The load-bearing question — *do the `espn_cfb_betting` files contain two-sided closing moneylines (home AND away ML)?* — is answered by direct inspection of all 23 season files:

> **The release contains no moneyline field at all — not one-sided, not two-sided, in any season, in any of the four file formats.** Two-sided closing-ML coverage of scored games is **0.0% in every season**, against a GO threshold of ≥70%. This is a structural absence, not a coverage gap.

It is worse than "ESPN gave spread and total but not ML": the betting files carry **no scores, no team names, and no dates either** — only `game_id`, `season`, `week`, and two market numbers (spread, total). Without a companion join, the file cannot even identify who played.

**Recommended path:** re-point the spike's line data at the collegefootballdata.com (CFBD) fallback, which documents per-provider moneylines (see *Fallback* below). The sportsdataverse release still earns its keep as the **schedule/score backbone**: its `espn_cfb_schedules` companion tag joins to the betting data on `game_id` at a verified 100.0% match rate with 100% final scores (details below), and `week`/`game_date` support strict chronological walk-forward splits.

No `data/analysis_ready.parquet` was staged: the GO-path deliverable is conditional on a two-sided-ML dataset, which this source cannot produce.

---

## 1. What was acquired

- **Source:** GitHub release tag [`espn_cfb_betting`](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_betting) of `sportsdataverse/sportsdataverse-data`, published 2026-06-04T02:21:58Z. No auth, no API key.
- **Asset inventory (verified via GitHub API, not assumed):** 95 assets — for each season 2004–2026, `betting_{season}.csv`, `.csv.gz`, `.parquet`, `.rds` **except 2026, which has no `.csv.gz`**. Parquet exists for all 23 seasons → **parquet is the consistent format** and is what `data/raw/` uses as primary.
- **Downloaded:** 23 parquet (2004–2026) + 24 csv.gz (2004–2025) for cross-validation → `data/raw/`. **All 47 downloads succeeded; zero failures.** The release's own `package_function.txt` names the generator: `sportsdataverse.cfb.load_cfb_betting()`.
- **Parquet vs csv.gz:** row counts match for **all 22 seasons** where both formats exist. Parquet is safe to trust.
- **Volume:** 18,722 rows total, 712–98 per season, each `game_id` unique within its file (and globally unique in practice).
- **Bonus (for the join assessment):** 23 files from the `espn_cfb_schedules` tag of the same repo → `data/raw/schedules/`.

## 2. Data dictionary — `betting_{season}.parquet` (as observed)

Nine columns, **byte-identical column sets in all 23 seasons**:

| column | dtype (parquet) | observed meaning |
|---|---|---|
| `game_id` | int64 | ESPN game identifier (e.g. `401520145`). The **only** join key the file offers. Unique per row. |
| `season` | int64 | Season year, 2004–2026. Always equals the file's season. |
| `week` | int64 | ESPN week number, 1–16. Bowl games occupy the trailing weeks. |
| `game_spread` | float64 (nullable) | Posted spread, single value per game — **no open/close pair exists in this release**. Sign convention is inconsistent across eras (see notes below). Present in 10,413 of 18,722 rows. |
| `over_under` | float64 (nullable) | Posted total, single value per game (no open/close). Present in 10,416 rows (2016 has 3 games with a total but no spread; every other season the counts match exactly). |
| `home_favorite` | bool | True when the home team is the spread favorite. Perfectly consistent with `home_team_spread` sign: home favorite ⇔ `home_team_spread < 0` in all 10,413 rows where a spread exists (0 counterexamples either way). |
| `home_team_spread` | float64 (nullable) | Spread signed from the home team's perspective (negative = home lays points). Populated exactly when `game_spread` is. |
| `game_spread_available` | bool | **Misleading name.** Observed: True ⇔ `odds_source` ∈ {`summary_pickcenter`, `core_odds_api`}; False ⇔ `odds_source` = `default` (equivalence holds for all 18,624 rows across 2004–2025). True does **not** imply a number exists — all 8,309 `summary_pickcenter` rows are True and have no numeric lines. |
| `odds_source` | string | Provenance of the row, three values: `summary_pickcenter` 8,309 rows · `default` 6,411 · `core_odds_api` 3,904. This field explains the entire coverage structure (below). |

**Spread sign conventions differ by era** (verified on the 10,413 rows with values):
- `default` rows (2004–2011 era): `game_spread` is the favorite-absolute spread; `home_team_spread` is home-signed. When home is favored, `game_spread − home_team_spread = +5.0` exactly (i.e. `game_spread = |home_team_spread|`); when away is favored they are equal.
- `core_odds_api` rows (2012–2025): `game_spread == home_team_spread` always (home-signed).
- Reconstruction rule for later stages: use `home_team_spread` (always home-signed) and drop `game_spread`.

**"Closing"?** The release provides one posted value per game and **no opening/closing pair** — the closing-line interpretation is an assumption inherited from ESPN's game-header consensus, not something these files can verify. Do not compute CLV against this release alone.

## 3. Coverage table

Per-season counts after the (verified 100%-matching) join to `cfb_schedule_{season}`. Full machine copy: `data/coverage.csv`. "ML" = two-sided closing moneyline — **structurally zero in every season** because the field does not exist.

| season | rows (games w/ odds row) | matched to schedule | with final scores | with posted spread | spread % of scored | two-sided closing ML | distinct team names |
|---|---|---|---|---|---|---|---|
| 2004 | 712 | 712 | 712 | 712 | 100.0% | 0 | 161 |
| 2005 | 728 | 728 | 728 | 728 | 100.0% | 0 | 166 |
| 2006 | 788 | 788 | 788 | 788 | 100.0% | 0 | 179 |
| 2007 | 800 | 800 | 800 | 800 | 100.0% | 0 | 186 |
| 2008 | 812 | 812 | 812 | 812 | 100.0% | 0 | 193 |
| 2009 | 810 | 810 | 810 | 810 | 100.0% | 0 | 199 |
| 2010 | 810 | 810 | 810 | 810 | 100.0% | 0 | 199 |
| 2011 | 817 | 817 | 817 | 817 | 100.0% | 0 | 205 |
| 2012 | 852 | 852 | 852 | 129 | 15.1% | 0 | 211 |
| 2013 | 861 | 861 | 861 | 123 | 14.3% | 0 | 216 |
| 2014 | 873 | 873 | 873 | 112 | 12.8% | 0 | 214 |
| 2015 | 876 | 876 | 876 | 110 | 12.6% | 0 | 216 |
| 2016 | 879 | 879 | 879 | 117 | 13.3% | 0 | 222 |
| 2017 | 890 | 890 | 890 | 107 | 12.0% | 0 | 218 |
| 2018 | 898 | 898 | 898 | 118 | 13.1% | 0 | 226 |
| 2019 | 892 | 892 | 892 | 116 | 13.0% | 0 | 223 |
| 2020 | 692 | 692 | 692 | 53 | 7.7% | 0 | 148 |
| 2021 | 895 | 895 | 895 | 122 | 13.6% | 0 | 233 |
| 2022 | 904 | 904 | 904 | 125 | 13.8% | 0 | 234 |
| 2023 | 911 | 911 | 911 | 882 | 96.8% | 0 | 229 |
| 2024 | 966 | 966 | 966 | 966 | 100.0% | 0 | 236 |
| 2025 | 958 | 958 | 958 | 958 | 100.0% | 0 | 236 |
| 2026 | 98 | 98 | 98 | 98 | 100.0% | 0 | 185 |
| **total** | **18,722** | **18,722** | **18,722** | **10,413** | **55.6%** | **0** | 293 distinct overall |

**The ugly gaps, stated plainly:**
1. **Moneylines: zero, structurally.** No ML column exists; nothing to backtest.
2. **2012–2022 spreads are near-empty.** Only 53–129 games per season (7.7–15.1%) carry numbers. Cause, verified by `odds_source`: 8,309 rows (2012–2023) come from `summary_pickcenter` and are entirely value-less; only `core_odds_api` rows (96–119/season) plus a handful of `default` rows carry lines. A spread/total backtest "over 10+ seasons" from this release alone would be 2004–2011 plus 2023–2025 with a 2012–2022 hole in the middle.
3. **2026 is a stub:** 98 rows, week 1 only, all `core_odds_api` (release built 2026-06-04).
4. **2020 is COVID-shaped:** 692 rows, unusual week structure.

## 4. Join-key assessment

- **Primary key: `game_id` (ESPN), and it is excellent.** Left-joining every betting season to `cfb_schedule_{season}` (same repo, tag `espn_cfb_schedules`, files named `cfb_schedule_{season}.parquet`) on `game_id` matched **18,722/18,722 rows — 100.0% in every season — with zero duplicate or orphaned keys**, and 100.0% of matched rows carry both final scores. Row counts of betting and schedule files are identical per season. The schedule companion adds: `season`, `week`, `season_type`, `game_date` (ISO timestamp, UTC), `neutral_site`, `conference_competition`, `home_id`/`away_id` (ESPN team ids), `home_team`/`away_team` ("USC Trojans" style), abbreviations, `home_score`/`away_score`, winner flags, `venue`, `attendance`, `status`.
- **Team-name normalization needs:** *within* ESPN, names are stable and self-disambiguating — "Miami Hurricanes" (FL) vs "Miami (OH) RedHawks" — so the brief's 'Miami' vs 'Miami (FL)' worry does not arise inside this universe (293 distinct names across all seasons). But **ESPN's own naming drifts over time**: San Jose State appears as `San Jose State Spartans` (2004–2018), `San José St Spartans` (2019), then `San José State Spartans` (2020–2026) — name-based cross-season or cross-source joins need accent/St-State normalization. For the CFBD fallback, ESPN names/ids will not match CFBD names directly; prefer id-based crosswalks (the repo's `cfb_crosswalk` tag exists but the probed asset is roster-oriented; a schedule-level ESPN↔CFBD crosswalk is not confirmed — unverified) or date+normalized-name matching.
- **Walk-forward feasibility: yes, on dates.** `game_date` + `week` are available via the schedule join. Caution: week numbers are **not** a strictly chronological boundary (2023's weeks are non-monotonic when sorted by date; every season has ≥1 week whose dates overlap the next week — 4 in 2020's COVID rebuild). Strict weekly walk-forward must cut on `game_date`, with `week` used for grouping/context, and bowl weeks (15–16, plus 2020's 16) handled explicitly. Chronological ordering is fully supported by `game_date`.

## 5. What IS available here (for the record)

- A **game universe with 100% verified scores/teams/dates** for 23 seasons (via the schedule join) — a solid backbone for any backtest's fixture list and settlement.
- A **single posted spread + total per game** (no open/close), for 10,413 games: complete 2004–2011, near-complete 2023–2025, near-zero 2012–2022. Usable for closing-line-consensus research on the two clean eras, but not for a continuous 10-season spread backtest and **not verifiable as closing lines from this release**.
- `home_favorite` + `home_team_spread` (home-signed, internally consistent) as the cleanest spread representation.

## 6. What the CFBD fallback (collegefootballdata.com) would add

- **Moneylines.** CFBD's `/lines` endpoint documents per-provider line records with `spread`/`spreadOpen`, `overUnder`/`overUnderOpen`, and **`moneylineHome`/`moneylineAway`** (per CFBD's API/GraphQL reference; cfbfastR's endpoint doc lists 21 columns including `home_moneyline`/`away_moneyline`). That is a genuine two-sided ML field, which this release lacks entirely.
- **Open/close pairs** for spread and total (enabling true CLV analysis, which the ESPN release cannot support).
- **Multi-book lines per game** (provider-level rows, enabling consensus-closing construction).
- Free tier requires only a free API key (registration); no cost gate for spike-scale pulls.
- **Unverified caveats, honestly flagged:** the CFBD docs I could inspect do **not** document moneyline-*open* fields (only close), do **not** state historical ML coverage depth (a related CFBD dataset description mentions ~11 NCAAF seasons of betting archive — unconfirmed), and ML presence per season must be re-verified the same way this report verified ESPN (per-season, not assumed). Also unverified: whether CFBD rows link to ESPN `game_id` directly (join may need date+names). **ESPN's own API was probed live as a second check and returned HTTP 403 from this environment (twice), so whether re-pulling ESPN pickcenter upstream would surface CFB moneylines remains unverified** — from training memory, ESPN pickcenter objects often carry moneyline for some sports, but treat that as unconfirmed.

## 7. Reproduction

- `scripts/download_release.py` — asset list from the GitHub API (no hardcoded names), downloads parquet + csv.gz.
- `scripts/join_and_coverage.py` — downloads schedules, joins on `game_id`, writes `data/coverage.csv`.
- Environment: Python 3.13.14 venv (`.venv/`) with pandas 3.0.5, pyarrow 25.0.1, numpy 2.5.3.
- Raw files kept under `data/raw/` (betting parquet+csv.gz, `schedules/` subfolder). Data files are deliberately excluded from the committed branch (sizes and source-replication); the report, coverage CSV, and scripts are committed.
