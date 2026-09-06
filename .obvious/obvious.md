# Keeks — Agent Guide

Repo: `wdm0006/keeks` — Python bet-sizing and bankroll-simulation library for repeated
binary outcomes (Kelly Criterion and eight other strategies). Pure library: no web app,
no services, no Docker, no required environment variables.

## Stack

| Item | Value |
|---|---|
| Language | Python 3.10–3.14 (dev venv uses CPython 3.12) |
| Package manager | `uv` (all Makefile targets wrap `uv run`) |
| Build backend | hatchling (`uv build`, `uv pip install -e ".[dev]"`) |
| Runtime deps | numpy, matplotlib |
| Dev deps | pandas, pytest, pytest-cov, ruff, tox, sphinx, flake8, wheel |
| Test framework | pytest (+ coverage via `addopts` in pyproject.toml) |
| Lint/format | ruff (`check` + `format --check` in CI) |
| Docs | Sphinx (`docs/source`, published to gh-pages) |
| Guidance docs | `CLAUDE.md` (dev commands + architecture), `README.md`, `CHANGELOG.md` |

## Commands

All commands assume a `.venv` created by `make setup` (`uv venv --python=3.12`).
First-time setup: `pip install uv && uv venv --python=3.12 && uv pip install -e ".[dev]"`.

```bash
make test           # uv run pytest (tests/ + coverage, addopts in pyproject)
make test-doctest   # uv run pytest --doctest-modules keeks/
make test-cov       # tests with html coverage report
make test-all       # tox across py310–py314
make lint           # uv run ruff check .
make format         # uv run ruff format .
make examples       # uv run python examples/strategy_comparison.py
make benchmark      # uv run python benchmarks/strategy_benchmark.py
make docs           # Sphinx html build into docs/build/html
make build          # uv build (wheel + sdist)
```

Single test: `uv run pytest tests/test_kelly_criterion.py::test_basic_functionality`.
CI runs `pytest tests --cov=keeks`, doctests, `ruff check .`, `ruff format --check .`,
and a wheel-build + clean-import check (`.github/workflows/test.yml`).

## Codebase map

See `codebase-map.md` (repo has >2 top-level dirs, map kept separate).

## Local verification

Run `make test`, `make lint`, and one example. `evaluate()` returns a bankroll
**fraction**, not a currency amount. Full summary in
`.obvious/skills/local-dev/SKILL.md`.

## Sandbox snapshot

| Field | Value |
|---|---|
| Computer | `cmp_6mO8dJqc` (repo sandbox, session `ifyfz1ognuuqzzpqktrbh`) |
| Snapshot template | `5ioce1j8jp3zozzfrknr:default` |
| Built at | 2026-09-06T20:20:54.715Z (UTC) |
| Contents | CPython 3.12.14 `.venv`, uv 0.12.10, keeks 0.6.0 editable install with dev extras |

## Notes

- No services to start; "healthy dev stack" = installed venv where tests, doctests,
  lint, and an example script all pass.
- `version.py` says 0.5.0 while `pyproject.toml`/CHANGELOG say 0.6.0; `version.py`
  is only consumed by `docs/source/conf.py`. Packaging version comes from pyproject.
- Do not edit `docs/output/`, `examples/output/`, or `benchmarks/output/` by hand —
  regenerate via `make examples` / `make benchmark`.
