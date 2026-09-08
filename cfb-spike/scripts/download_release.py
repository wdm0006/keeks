"""Download sportsdataverse espn_cfb_betting release assets into data/raw/.

Asset list is read from release.json (the GitHub API response for the tag),
so the exact asset names are verified, not assumed. Downloads both parquet
(available 2004-2026) and csv.gz (available 2004-2025) for cross-checking.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

RELEASE_JSON = Path(__file__).resolve().parents[1] / "release.json"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PATTERN = re.compile(r"^betting_(\d{4})\.(parquet|csv\.gz)$")


def main():
    assets = json.loads(RELEASE_JSON.read_text())["assets"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [a for a in assets if PATTERN.match(a["name"])]
    print(f"matched {len(wanted)} assets of {len(assets)} total")
    failures = []
    for a in wanted:
        dest = RAW_DIR / a["name"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip (exists) {a['name']}")
            continue
        try:
            req = urllib.request.Request(
                a["browser_download_url"], headers={"User-Agent": "cfb-spike/0.1"}
            )
            with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
                f.write(r.read())
            print(f"ok   {a['name']}  {dest.stat().st_size} bytes")
        except Exception as e:  # noqa: BLE001
            failures.append(a["name"])
            print(f"FAIL {a['name']}: {e}")
            dest.unlink(missing_ok=True)
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("all downloads complete")


if __name__ == "__main__":
    main()
