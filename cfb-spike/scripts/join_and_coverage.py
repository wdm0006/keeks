"""Join betting parquets to espn_cfb_schedules (cfb_schedule_{season}) on game_id
and emit the final coverage table to data/coverage.csv.

The betting release carries NO scores, teams, or dates - only game_id/season/week -
so scored-game and team-name coverage can only be measured via the schedule join.
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SCHED = RAW / "schedules"
SCHED.mkdir(exist_ok=True)


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "cfb-spike/0.1"})
    return urllib.request.urlopen(req, timeout=120).read()


def schedule_asset_map():
    d = json.loads(
        get(
            "https://api.github.com/repos/sportsdataverse/sportsdataverse-data/releases/tags/espn_cfb_schedules"
        )
    )
    return {a["name"]: a["browser_download_url"] for a in d["assets"]}


def load_schedule(season, assets):
    for suffix in ("parquet", "csv.gz"):
        name = f"cfb_schedule_{season}.{suffix}"
        if name in assets:
            dest = SCHED / name
            if not dest.exists() or dest.stat().st_size == 0:
                dest.write_bytes(get(assets[name]))
            if suffix == "parquet":
                return pd.read_parquet(dest)
            return pd.read_csv(dest, compression="gzip")
    return None


def main():
    assets = schedule_asset_map()
    rows = []
    name_problems = {}
    for season in range(2004, 2027):
        bet = pd.read_parquet(RAW / f"betting_{season}.parquet")
        sch = load_schedule(season, assets)
        if sch is None:
            rows.append(
                {
                    "season": season,
                    "betting_rows": len(bet),
                    "schedule_available": False,
                }
            )
            continue
        m = bet.merge(sch, on="game_id", how="left", suffixes=("", "_sch"))
        matched = m.home_team.notna()
        scored = matched & m.home_score.notna() & m.away_score.notna()
        spread = m.game_spread.notna()
        total = m.over_under.notna()
        teams = pd.unique(
            pd.concat([m.loc[matched, "home_team"], m.loc[matched, "away_team"]])
        )
        rows.append(
            {
                "season": season,
                "betting_rows": len(bet),
                "schedule_available": True,
                "matched_to_schedule": int(matched.sum()),
                "match_rate_pct": round(100 * matched.mean(), 1),
                "with_final_scores": int(scored.sum()),
                "with_closing_spread": int(spread.sum()),
                "spread_pct_of_scored": round(
                    100 * (spread & scored).sum() / max(scored.sum(), 1), 1
                ),
                "with_over_under": int(total.sum()),
                "two_sided_closing_ml": 0,
                "pct_two_sided_ml": 0.0,
                "distinct_team_names": len(teams),
            }
        )
        for t in teams:
            if "(" in t or t not in name_problems:
                name_problems.setdefault(t, 0)
    cov = pd.DataFrame(rows)
    cov.to_csv(ROOT / "data" / "coverage.csv", index=False)
    print(cov.to_string(index=False))
    print()
    print(
        "totals:",
        {
            c: (int(cov[c].sum()) if cov[c].dtype != object else "-")
            for c in [
                "betting_rows",
                "matched_to_schedule",
                "with_final_scores",
                "with_closing_spread",
                "two_sided_closing_ml",
            ]
        },
    )
    parens = {t: c for t, c in name_problems.items() if "(" in t}
    print("team names containing '(' (ESPN style check):", parens if parens else "none")
    nonascii = [t for t in name_problems if any(ord(ch) > 127 for ch in t)]
    print("team names with non-ASCII chars:", nonascii[:10])
    # odds_source <-> game_spread_available equivalence check
    allx = pd.concat(
        [pd.read_parquet(RAW / f"betting_{s}.parquet") for s in range(2004, 2026)],
        ignore_index=True,
    )
    ct = (
        allx.groupby(["odds_source", "game_spread_available"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    print()
    print("odds_source x game_spread_available:")
    print(ct.to_string(index=False))
    equiv = bool(
        ((allx.odds_source == "default") == (~allx.game_spread_available)).all()
    )
    print(f"equivalence (source=='default' <=> not available): {equiv}")


if __name__ == "__main__":
    main()
