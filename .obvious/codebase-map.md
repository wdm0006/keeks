# Keeks — Codebase Map

Folder-level overview (depth 2). Python library, no sub-apps.

| Path | Purpose |
|---|---|
| `keeks/` | The package. `bankroll.py` = `BankRoll` state manager (funds, history, `max_draw_down`, raises `RuinError`). `utils.py` = shared math (CRRA utility, indifference price search, etc.). |
| `keeks/binary_strategies/` | Nine strategies, all subclassing `BaseStrategy` (`base.py`) with `evaluate(probability, current_bankroll) -> fraction`. `kelly.py` = Kelly / Fractional / DrawdownAdjusted; `simple.py` = OptimalF, FixedFraction, CPPI, DynamicBankrollManagement, MertonShare, Naive. |
| `keeks/simulators/` | Trial harnesses with `evaluate_strategy(strategy, bankroll)`: `repeated_binary.py` (fixed probability), `random_binary.py` (normal-distributed probabilities), `random_uncertain_binary.py` (adds outcome uncertainty). Seeded for reproducibility. |
| `tests/` | 30 pytest files (~1041 tests) covering every strategy, simulator edge cases, bankruptcy protection, and the documented examples. CI runs these with coverage. |
| `examples/` | Runnable scripts: `strategy_comparison.py` (headless nine-strategy comparison), `st_petersburg_paradox.py`. Output committed under `examples/output/`. |
| `benchmarks/` | `strategy_benchmark.py` regenerates the published nine-strategy risk benchmark (CSV + charts into `benchmarks/output/`, docs page `strategy_benchmark.rst`). |
| `docs/` | Sphinx site (`source/` with per-module rst, `requirements.txt`, own Makefile). Published to gh-pages / keeks.mcginniscommawill.com. `conf.py` reads `version.py`. |
| `.github/` | Workflows: `test.yml` (pytest matrix 3.10–3.14 + doctests + ruff + wheel/import check), `docs.yml` (build+deploy docs on master), `publish-pypi.yml` (tag-triggered PyPI publish). Issue/PR templates. |
| Root | `pyproject.toml` (hatchling, ruff, pytest, tox config), `Makefile` (uv-wrapped targets), `CLAUDE.md` (dev guide), `README.md`, `CHANGELOG.md`, `version.py` (docs-only version, 0.5.0 — lags pyproject 0.6.0). |

Architecture in one line: strategy (`evaluate` → bet fraction) + `BankRoll` (state/risk
limits) + simulator (repeated trials, in-place bankroll updates, optional update hooks
for adaptive strategies).
