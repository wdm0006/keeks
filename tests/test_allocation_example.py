"""
Guards for the ETF allocation example: the fixture's provenance, the offline
default, and the yfinance import guard.

The example's value is that it runs offline from a committed fixture, so the
properties worth testing are the ones a reader otherwise has to take on
trust: the fixture parses and its provenance header names the tickers, the
window, and the refresh command; the module never pulls yfinance in at
runtime (yfinance is a development-only dependency); the offline main runs
end to end; and the table and chart-building steps render headless on
synthetic data, reachable without replaying the full book.
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "allocation_etfs.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("allocation_etfs", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXAMPLE = _load_example()


def test_offline_default_never_imports_yfinance():
    """The offline run — module import, fixture load, both passes, charts.

    A fresh interpreter so the guard is airtight against test ordering: the
    dev environment installs yfinance, and any other module importing it
    would otherwise poison ``sys.modules`` for this assertion.
    """
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location(\n"
        f"    'allocation_etfs', {str(EXAMPLE_PATH)!r}\n"
        ")\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "dates, returns = module.load_returns()\n"
        "assert 'yfinance' not in sys.modules, (\n"
        "    'the offline example path imports yfinance'\n"
        ")\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_no_keeks_module_imports_yfinance():
    package_root = EXAMPLE_PATH.parents[1] / "keeks"
    for path in package_root.rglob("*.py"):
        assert "yfinance" not in path.read_text(encoding="utf-8"), path


def test_yfinance_import_is_lazy_inside_download_returns():
    """The only yfinance import in the example sits inside the refresh path."""
    tree = ast.parse(EXAMPLE_PATH.read_text(encoding="utf-8"))

    def _imports_yfinance(body):
        return any(
            isinstance(node, ast.Import)
            and any(alias.name == "yfinance" for alias in node.names)
            for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        )

    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name != "yfinance" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "yfinance"
        elif isinstance(node, ast.FunctionDef):
            # Only the refresh path may import yfinance.
            assert _imports_yfinance(node.body) == (node.name == "download_returns"), (
                node.name
            )


def test_fixture_has_provenance_header():
    text = EXAMPLE.FIXTURE_PATH.read_text(encoding="utf-8")
    header = "\n".join(line for line in text.splitlines() if line.startswith("#"))
    for token in (*EXAMPLE.TICKERS, "yfinance", "--refresh", EXAMPLE.START):
        assert token in header, token


def test_load_returns_parses_the_committed_fixture():
    dates, returns = EXAMPLE.load_returns()
    assert returns.shape == (len(dates), len(EXAMPLE.TICKERS))
    assert dates[0] > EXAMPLE.START
    assert np.isfinite(returns).all()
    # Daily simple returns of liquid ETFs stay in a sane band.
    assert np.abs(returns).max() < 0.5


def test_metrics_on_a_known_history():
    history = [1000.0, 1250.0, 1000.0, 1500.0]
    assert EXAMPLE.growth(history) == pytest.approx(1.5)
    assert EXAMPLE.max_drawdown(history) == pytest.approx(0.2)
    # Per-period returns are 0.25, -0.2, 0.5.
    assert EXAMPLE.volatility(history) == pytest.approx(0.3547299442298794)
    # A wipeout is a real -100% return; only the transition out of a zero
    # bankroll is guarded to zero rather than dividing by zero.
    assert EXAMPLE.period_returns([1000.0, 0.0, 500.0]).tolist() == [-1.0, 0.0]


def test_full_offline_run_prints_tables_and_writes_charts(tmp_path, capsys):
    """The published offline path end to end, into a scratch output dir."""
    EXAMPLE.OUTPUT_DIR = tmp_path
    try:
        EXAMPLE.main([])
    finally:
        EXAMPLE.OUTPUT_DIR = (
            EXAMPLE_PATH.parent / "output"
        )  # restore the module constant
    output = capsys.readouterr().out
    assert "empirical bootstrap (scenario_model)" in output
    assert "Student-t marginals (fit_marginals_model)" in output
    assert "MeanCVaR" in output
    for name in (
        "allocation_etf_growth_paths.png",
        "allocation_etf_growth_paths_student_t.png",
        "allocation_etf_drawdown.png",
        "allocation_etf_weight_evolution.png",
    ):
        assert (tmp_path / name).stat().st_size > 0, name


@pytest.fixture
def tiny_pass(tmp_path):
    """The published code path on a small synthetic three-option book."""
    rng = np.random.default_rng(7)
    scenarios = rng.normal(0.0005, 0.01, size=(80, 3))
    model = EXAMPLE.scenario_model(scenarios)
    mean, covariance = model.moments()
    allocators = EXAMPLE.build_allocators(model.scenarios, mean, covariance)
    histories = EXAMPLE.run_pass(model, allocators, trials=30, seed=1)
    return scenarios, mean, covariance, histories, tmp_path


def test_comparison_table_shape(tiny_pass):
    _, _, _, histories, _ = tiny_pass
    assert len(histories) == 7
    table = EXAMPLE.comparison_table(histories)
    assert len(table) == 7
    assert list(table.columns) == ["final", "growth", "vol/period", "max dd"]


def test_chart_builders_render_headless(tiny_pass):
    _, _, _, histories, destination = tiny_pass
    EXAMPLE.save_growth_chart(
        histories, destination / "growth.png", "tiny growth paths"
    )
    EXAMPLE.save_drawdown_chart(
        histories["MeanVariance (lambda=1)"],
        destination / "drawdown.png",
        "tiny drawdown",
    )
    EXAMPLE.save_weight_chart(
        [[1 / 3, 1 / 3, 1 / 3]] * 10,
        destination / "weights.png",
        "tiny weight evolution",
        labels=("a", "b", "c"),
    )
    for name in ("growth.png", "drawdown.png", "weights.png"):
        assert (destination / name).stat().st_size > 0
