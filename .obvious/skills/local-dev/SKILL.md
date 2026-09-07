---
name: local-dev
description: Bring a fresh checkout of wdm0006/keeks to a working dev environment — uv venv on Python 3.12, editable install with dev extras, and the verification loop (tests, doctests, lint, example).
version: 1
---

# Local Dev — wdm0006/keeks

Recorded 2026-09-06 (UTC) from a successful onboarding run on sandbox
`cmp_6mO8dJqc` (session `ifyfz1ognuuqzzpqktrbh`, snapshot template
`5ioce1j8jp3zozzfrknr:default`).

## What this repo is

Pure Python library — bet sizing / bankroll simulation. **No services, no ports,
no Docker, no env vars, no migrations.** A healthy dev stack = an installed venv
where the checks below pass. Nothing long-running to keep alive.

## Setup (from a clean checkout)

```bash
pip install uv                      # uv 0.12.10 used in the recorded run
uv venv --python=3.12               # downloads managed CPython 3.12.14 if absent
uv pip install -e ".[dev]"          # numpy, matplotlib + pandas, pytest, ruff, tox, sphinx
```

Equivalent Makefile targets: `make setup` then `make install-dev`.

## Verification loop

```bash
uv run pytest tests                                  # 1039 passed, 2 skipped, ~97% cov
uv run pytest --doctest-modules keeks/               # doctests pass
uv run ruff check .                                  # clean
uv run ruff format --check .                         # 56 files already formatted
uv run python examples/strategy_comparison.py        # stats table + chart in examples/output/
```

Expected quickstart values (README): `KellyCriterion(payoff=1.0, loss=1.0,
transaction_cost=0.01).evaluate(probability=0.55, current_bankroll=1000)` →
`0.090009` (9.0009%, $90.01). Assert these when smoke-testing.

## Gotchas

- `pytest` addopts already include `--cov=keeks --cov-report=term-missing`; plain
  `uv run pytest` emits coverage. Use `PYTEST_ARGS` to scope: `make test PYTEST_ARGS="tests/test_bankroll.py -v"`.
- `examples/strategy_comparison.py` takes ~3 minutes (13 strategy × 1000-trial
  Monte Carlo runs) and overwrites the committed CSV/PNG in `examples/output/` —
  don't commit that churn unless the example itself changed.
- `keeks` exposes no `__version__` attribute; package version lives in
  `pyproject.toml` (0.6.0). `version.py` (0.5.0) is only read by `docs/source/conf.py`.
- Sphinx docs build needs `uv pip install -r docs/requirements.txt` on top of dev
  extras (`make docs`); docs CI uses `python -m sphinx -W --keep-going`.
- `make test-all` (tox, py310–py314) only works if those interpreters exist;
  CI covers the matrix, local single-version pytest is enough.

## Evidence from the recorded run

- pytest: `1039 passed, 2 skipped` in 4.65s, coverage 97% (689 stmts, 18 miss).
- doctests: `1 passed`. ruff check: `All checks passed!`. format: `56 files already formatted`.
- Example run produced the 13-strategy stats table (Kelly mean $1265.75, ruin 0%;
  Fixed 10% ruin 97.4%) and regenerated `examples/output/strategy_comparison.{csv,png}`.
