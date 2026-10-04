"""
Scenario-based allocation tests: the mean-CVaR family.

Layers, following the house lattice:

- the ``scenarios_to_moments`` bridge: exact moments of the scenario
  distribution, with residual probability mass read as all-cash periods and
  never renormalized;
- the LP oracle: the Rockafellar-Uryasev optimum against an independently
  formulated dual linear program on tiny and random scenario sets, plus the
  all-cash / all-in corner cases and the all-or-nothing budget behavior
  positive homogeneity forces;
- fat-tail behavior: variance-matched Student-t (nu = 2.5) marginals drop
  the gross exposure the Gaussian book takes, because the empirical tail at
  ``tail_alpha=0.01`` is deep enough to make every direction worse than cash;
- the weight contract and ``from_model`` replay under hypothesis;
- the gated-import behavior and the solver-boundary hygiene (clip and
  rescale at the house tolerance, failures surfaced).
"""

import builtins
import contextlib

import numpy as np
import pytest
import scipy.optimize
from hypothesis import given, settings
from hypothesis import strategies as st

from keeks.allocation import MeanCVaR, ScenarioModel, scenarios_to_moments
from keeks.allocation.models import marginals_model, scenario_model
from keeks.allocation.scenarios import _validate_tail_alpha
from keeks.utils import PROBABILITY_SUM_TOLERANCE

settings.register_profile("keeks", max_examples=50, deadline=None)
settings.load_profile("keeks")


@contextlib.contextmanager
def no_scipy():
    """Hide scipy the way a bare ``pip install keeks`` environment would."""
    real_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = without_scipy
    try:
        yield
    finally:
        builtins.__import__ = real_import


def _objective_oracle(weights, scenarios, tail_alpha):
    """
    Exact objective ``mu'w - CVaR(w)`` for equal-weighted scenario rows.

    The Rockafellar-Uryasev function is piecewise linear in ``alpha`` with
    its minimum at one of the loss values, so minimizing over the observed
    losses is exact - no sorted-formula shortcuts needed.
    """
    portfolio = scenarios @ np.asarray(weights, dtype=float)
    losses = -portfolio
    best = np.inf
    for alpha in losses:
        best = min(
            best,
            alpha + np.maximum(losses - alpha, 0.0).sum() / (tail_alpha * losses.size),
        )
    return portfolio.mean() - best


def _brute_force_optimum(scenarios, tail_alpha, steps=21):
    """Grid-search the budget simplex for the best objective value."""
    count = scenarios.shape[1]
    best = -np.inf
    if count == 1:
        candidates = [[1.0], [0.0]]
    else:
        candidates = []
        for first in np.linspace(0.0, 1.0, steps):
            for second in np.linspace(0.0, 1.0 - first, steps):
                candidates.append([first, second] + [0.0] * (count - 2))
    for weights in candidates:
        best = max(best, _objective_oracle(weights, scenarios, tail_alpha))
    return best


def _dual_lp_value(scenarios, tail_alpha):
    """
    The exact optimum, computed from the dual side.

    CVaR over equal-weighted empirical rows equals the largest expected loss
    under any probability vector with sum one and entries capped at
    ``1 / (tail_alpha * T)``.  The mean-CVaR optimum is therefore a zero-sum
    game between the weights and that capped probability vector, and strong
    duality pins its value as ``min t`` subject to ``t >= (mean + X'q)_i``,
    ``t >= 0``, ``sum(q) = 1`` and the cap - an independent linear program
    over ``(q, t)`` with no shared structure with the primal solve.
    """
    scenarios = np.asarray(scenarios, dtype=float)
    rows, options = scenarios.shape
    cap = 1.0 / (tail_alpha * rows)
    objective = np.zeros(rows + 1)
    objective[-1] = 1.0
    tail_constraints = np.hstack([scenarios.T, -np.ones((options, 1))])
    budget = np.zeros((1, rows + 1))
    budget[0, :rows] = 1.0
    result = scipy.optimize.linprog(
        objective,
        A_ub=tail_constraints,
        b_ub=-scenarios.mean(axis=0),
        A_eq=budget,
        b_eq=[1.0],
        bounds=[(0.0, cap)] * rows + [(0.0, None)],
        method="highs",
    )
    assert result.success
    return float(result.fun)


def _books():
    """Rectangular scenario books: fixed width per hypothesis example."""
    return st.integers(min_value=1, max_value=5).flatmap(
        lambda width: st.lists(
            st.lists(
                st.floats(min_value=-0.5, max_value=0.5, allow_nan=False),
                min_size=width,
                max_size=width,
            ),
            min_size=1,
            max_size=8,
        )
    )


class TestScenariosToMoments:
    """The scenario -> moment bridge."""

    def test_equal_weight_moments(self):
        mean, covariance = scenarios_to_moments(
            [[0.02, 0.01], [-0.01, 0.01], [0.0, 0.0]]
        )
        assert np.allclose(mean, [1.0 / 300.0, 2.0 / 300.0])
        assert covariance[0, 1] == pytest.approx(covariance[1, 0])
        assert covariance[0, 0] > 0.0
        assert covariance[1, 1] > 0.0

    def test_residual_probability_mass_is_cash(self):
        mean, covariance = scenarios_to_moments(
            [[0.02, 0.01], [-0.01, 0.01]], probabilities=[0.25, 0.25]
        )
        # The missing half of the mass is an all-cash period at zero return:
        # the mean is the probability-weighted mean of three outcomes, and
        # the covariance carries the zero row as a third observation.
        assert np.allclose(mean, [0.0025, 0.005])
        assert np.allclose(
            covariance,
            [[1.1875e-04, 1.25e-05], [1.25e-05, 2.5e-05]],
        )

    def test_probabilities_never_renormalized(self):
        scenarios = [[0.02, 0.01], [-0.01, 0.01]]
        weighted, _ = scenarios_to_moments(scenarios, probabilities=[0.25, 0.25])
        normalized = np.array([0.005, 0.01])
        assert not np.allclose(weighted, normalized)

    def test_invalid_probabilities_rejected(self):
        with pytest.raises(ValueError, match="Probabilit"):
            scenarios_to_moments([[0.02], [-0.01]], probabilities=[0.9, 0.5])
        with pytest.raises(ValueError, match="Probabilit"):
            scenarios_to_moments([[0.02], [-0.01]], probabilities=[1.2])

    def test_invalid_scenarios_rejected(self):
        with pytest.raises(ValueError):
            scenarios_to_moments([0.02, -0.01])
        with pytest.raises(ValueError):
            scenarios_to_moments([[0.02], [float("nan")]])


class TestBruteForceOracle:
    """The LP optimum against exact brute force on tiny scenario sets."""

    TINY_CASES = [
        [[0.03, 0.01], [-0.01, 0.02], [0.01, -0.01], [-0.02, -0.02]],
        [[0.02, -0.03], [-0.02, 0.04], [0.01, 0.01]],
        [[0.05, 0.01], [-0.05, 0.02], [0.02, 0.03], [-0.01, -0.01], [0.0, 0.04]],
        [[0.01, 0.02, -0.01], [-0.02, 0.01, 0.02], [0.03, -0.02, 0.01]],
    ]

    @pytest.mark.parametrize("scenarios", TINY_CASES)
    @pytest.mark.parametrize("tail_alpha", [0.05, 0.25])
    def test_lp_matches_dual_oracle(self, scenarios, tail_alpha):
        strategy = MeanCVaR(scenarios, tail_alpha=tail_alpha)
        book = np.array(scenarios)
        # The LP's value equals the independently formulated dual optimum.
        assert -strategy.objective == pytest.approx(
            _dual_lp_value(book, tail_alpha), abs=1e-9
        )
        # The LP's own weights reproduce its claimed value exactly, and no
        # point of a coarse simplex grid beats them.
        achieved = _objective_oracle(strategy.weights, book, tail_alpha)
        assert achieved == pytest.approx(-strategy.objective, abs=1e-9)
        assert _brute_force_optimum(book, tail_alpha) <= -strategy.objective + 1e-9

    @given(
        st.integers(min_value=2, max_value=3).flatmap(
            lambda width: st.lists(
                st.lists(
                    st.floats(min_value=-0.5, max_value=0.5, allow_nan=False),
                    min_size=width,
                    max_size=width,
                ),
                min_size=2,
                max_size=6,
            )
        ),
        st.floats(min_value=0.1, max_value=0.4),
    )
    def test_lp_matches_dual_oracle_random_tiny_scenarios(self, rows, tail_alpha):
        scenarios = np.array(rows)
        strategy = MeanCVaR(scenarios, tail_alpha=tail_alpha)
        # HiGHS's default feasibility tolerances are 1e-7, and on tiny
        # adversarial books the returned optimum can sit a hair outside a
        # 1e-9 duality gap - so the oracle equality is asserted at the
        # solver's own tolerance scale.  Real formulation errors (wrong
        # mean, cap, or sign) break this by O(1); the fixed cases above
        # still hold the tight 1e-9 line.
        assert -strategy.objective == pytest.approx(
            _dual_lp_value(scenarios, tail_alpha), abs=1e-7
        )
        achieved = _objective_oracle(strategy.weights, scenarios, tail_alpha)
        assert achieved == pytest.approx(-strategy.objective, abs=1e-7)
        assert (
            _brute_force_optimum(scenarios, tail_alpha, steps=11)
            <= -strategy.objective + 1e-7
        )

    def test_all_losing_book_goes_to_cash(self):
        scenarios = [[-0.01, -0.02], [-0.03, -0.01], [-0.02, -0.02]]
        strategy = MeanCVaR(scenarios)
        # Every direction loses in expectation once the tail is priced, so
        # cash - objective 0 - is the optimum.
        assert np.allclose(strategy.weights, 0.0)
        assert strategy.objective == pytest.approx(0.0)

    def test_all_winning_book_invests_fully(self):
        scenarios = [[0.01, 0.02], [0.03, 0.01], [0.02, 0.02]]
        strategy = MeanCVaR(scenarios)
        # Every direction has a negative expected tail loss - the tail is a
        # gain - so more stake is always better and the budget binds.
        assert strategy.weights.sum() == pytest.approx(1.0)

    def test_single_option_is_all_or_nothing(self):
        # Positive homogeneity makes the mean-CVaR objective linear along
        # rays: with cash available the optimum is all-cash or fully
        # invested, never a partial fraction.
        invested = MeanCVaR([[0.02], [0.01], [0.03], [0.015], [0.0]])
        assert np.allclose(invested.weights, 1.0)
        refused = MeanCVaR([[0.001], [-0.01], [0.001], [-0.01]])
        assert np.allclose(refused.weights, 0.0)


class TestFatTailShrinkage:
    """Variance-matched fat tails withdraw the exposure Gaussian books take."""

    def test_student_t_shrinks_gross_exposure_vs_gaussian(self):
        # Same mean (0.016) and same variance (sd 0.010) for both books: the
        # Student-t scale is matched so variance agrees exactly (variance =
        # scale^2 * nu / (nu - 2)).  The books are single-option because
        # independent fat-tailed options diversify the tail away (volatility
        # scales with sqrt(2) while the mean scales linearly); with one
        # option, at tail_alpha = 0.01 the t(2.5) tail is deep enough that
        # the whole book prices worse than cash, while the Gaussian book
        # still prices its tail as a near-gain and invests.  Verified stable
        # across seeds 1, 3, 7, 11 and 42 at T = 20000.
        mean, sd, nu = 0.016, 0.010, 2.5
        scale = sd * ((nu - 2.0) / nu) ** 0.5
        gaussian = MeanCVaR.from_model(
            marginals_model([("normal", mean, sd)]),
            n_samples=20_000,
            seed=7,
            tail_alpha=0.01,
        )
        student = MeanCVaR.from_model(
            marginals_model([("student_t", nu, mean, scale)]),
            n_samples=20_000,
            seed=7,
            tail_alpha=0.01,
        )
        assert float(gaussian.weights[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(student.weights[0]) == pytest.approx(0.0, abs=1e-6)
        assert student.weights.sum() < gaussian.weights.sum()
        # Both books still speak the weight contract.
        assert sum(gaussian.evaluate(1000.0)) <= 1 + PROBABILITY_SUM_TOLERANCE
        assert sum(student.evaluate(1000.0)) <= 1 + PROBABILITY_SUM_TOLERANCE


class TestResidualProbabilityHandling:
    """Residual probability mass reaches the program as all-cash rows."""

    def test_from_model_residual_mass_produces_cash_rows(self):
        # One bet with probability 0.5: the other half of the periods are
        # all-cash zero rows, and the tail (worst 5%) is all cash - so the
        # program reads a nonnegative payoff with a free tail and invests
        # the full budget.
        model = ScenarioModel([[0.9]], probabilities=[0.5])
        strategy = MeanCVaR.from_model(model, n_samples=2000, seed=3)
        zero_rows = int((strategy.scenarios == 0.0).all(axis=1).sum())
        assert 900 < zero_rows < 1100
        assert np.allclose(strategy.weights, 1.0)

    def test_bridge_residual_mass_flows_into_moments(self):
        mean, _ = scenarios_to_moments([[0.9]], probabilities=[0.5])
        assert mean[0] == pytest.approx(0.45)


class TestWeightsContract:
    """The weight contract over random scenario books."""

    @given(_books(), st.sampled_from([0.0, -5.0, 1000.0]))
    def test_evaluate_returns_valid_weight_tuple(self, rows, bankroll):
        strategy = MeanCVaR(rows)
        weights = strategy.evaluate(bankroll)
        assert isinstance(weights, tuple)
        assert len(weights) == len(rows[0])
        for weight in weights:
            assert 0.0 <= weight <= 1.0
        assert sum(weights) <= 1 + PROBABILITY_SUM_TOLERANCE

    @given(_books())
    def test_optimize_result_contract(self, rows):
        strategy = MeanCVaR(rows)
        result = strategy.optimize()
        assert result.weights.shape == (len(rows[0]),)
        assert result.converged is True
        assert isinstance(result.iterations, int)
        assert result.iterations >= 0
        assert result.volatility >= 0.0
        assert np.allclose(result.weights, strategy.evaluate(1000.0))

    def test_seed_replay_reproduces_identical_scenarios(self):
        model = scenario_model([[0.02, 0.01], [-0.01, 0.02], [0.01, -0.01]])
        first = MeanCVaR.from_model(model, n_samples=64, seed=11)
        second = MeanCVaR.from_model(model, n_samples=64, seed=11)
        assert np.array_equal(first.scenarios, second.scenarios)
        assert np.array_equal(first.weights, second.weights)

    def test_from_model_without_seed_satisfies_contract(self):
        model = scenario_model([[0.02], [-0.01], [0.01]])
        strategy = MeanCVaR.from_model(model, n_samples=16)
        assert strategy.scenarios.shape == (16, 1)
        assert sum(strategy.evaluate(1000.0)) <= 1 + PROBABILITY_SUM_TOLERANCE


class TestGatedImports:
    """scipy is an optional extra and the class says so."""

    def test_mean_cvar_requires_scipy(self):
        with (
            no_scipy(),
            pytest.raises(ImportError, match=r"keeks\[allocation\]") as excinfo,
        ):
            MeanCVaR([[0.01], [-0.01]])
        assert "Mean-CVaR" in str(excinfo.value)


class TestLpHygiene:
    """The solver boundary is cleaned into the house numeric contract."""

    def test_solver_overshoot_is_rescaled_into_contract(self):
        import scipy.optimize

        def fake_linprog(*_args, **_kwargs):
            # HiGHS-grade feasibility noise: a hair over the budget and a
            # negative weight at the clip boundary.
            class Fake:
                success = True
                status = 0
                message = "optimal"
                nit = 4
                fun = 0.0
                x = np.array([0.6, 0.5 + 1e-9, 0.0, 0.0, 0.0])

            return Fake()

        original = scipy.optimize.linprog
        scipy.optimize.linprog = fake_linprog
        try:
            strategy = MeanCVaR([[0.02, 0.01], [-0.01, 0.02]])
        finally:
            scipy.optimize.linprog = original
        weights = strategy.weights
        assert weights.min() >= 0.0
        assert weights.sum() <= 1 + PROBABILITY_SUM_TOLERANCE

    def test_solver_failure_raises(self):
        import scipy.optimize

        def failing_linprog(*_args, **_kwargs):
            class Fake:
                success = False
                status = 2
                message = "infeasible"

            return Fake()

        original = scipy.optimize.linprog
        scipy.optimize.linprog = failing_linprog
        try:
            with pytest.raises(RuntimeError, match="linear program failed"):
                MeanCVaR([[0.02, 0.01], [-0.01, 0.02]])
        finally:
            scipy.optimize.linprog = original


class TestOptimizeDiagnostics:
    """The result object reports the program's own numbers."""

    def test_diagnostics_consistency(self):
        scenarios = np.array(
            [[0.03, 0.01], [-0.01, 0.02], [0.01, -0.01], [-0.02, -0.02]]
        )
        strategy = MeanCVaR(scenarios)
        result = strategy.optimize()
        mean = scenarios.mean(axis=0)
        # cvar = objective + mu'w, the program's own identity.
        assert result.objective == pytest.approx(
            strategy.cvar - mean @ strategy.weights
        )
        # expected_growth is E[log(1 + w'R)] under the scenarios.
        portfolio = scenarios @ strategy.weights
        assert result.expected_growth == pytest.approx(np.log1p(portfolio).mean())
        # volatility reads the weights against the empirical covariance.
        _, covariance = scenarios_to_moments(scenarios)
        assert result.volatility == pytest.approx(
            float(np.sqrt(strategy.weights @ covariance @ strategy.weights))
        )

    def test_expected_growth_none_on_wipeout(self):
        # One catastrophic scenario among 499 mild gains: the program still
        # invests fully, and the optimal portfolio returns -150% in that
        # scenario, where the log is undefined.
        scenarios = np.vstack([np.full((499, 1), 0.05), [[-1.5]]])
        strategy = MeanCVaR(scenarios)
        result = strategy.optimize()
        assert np.allclose(strategy.weights, 1.0)
        assert result.expected_growth is None


class TestTailAlphaValidation:
    """The tail fraction must be a finite number strictly inside (0, 1)."""

    def test_accepts_valid_fractions(self):
        assert _validate_tail_alpha(0.05) == 0.05
        assert _validate_tail_alpha(0.5) == 0.5

    @pytest.mark.parametrize(
        "bad", [0.0, 1.0, -0.5, 1.5, float("nan"), float("inf"), "abc", None]
    )
    def test_rejects_invalid_fractions(self, bad):
        with pytest.raises(ValueError):
            _validate_tail_alpha(bad)


class TestAllCashReason:
    """The result object carries the explanation the ETF example prints."""

    def test_all_cash_optimum_carries_reason(self):
        daily = np.array(
            [
                [0.0002, 0.0001],
                [-0.0002, -0.0001],
                [0.0001, -0.0001],
                [-0.0001, 0.0002],
            ]
        )
        result = MeanCVaR(daily).optimize()

        assert result.all_cash_reason is not None
        assert "full cash" in result.all_cash_reason

    def test_risk_taking_optimum_leaves_reason_none(self):
        rng = np.random.default_rng(0)
        scenarios = rng.normal(0.02, 0.01, size=(200, 2))
        result = MeanCVaR(scenarios).optimize()

        assert not np.all(np.abs(result.weights) <= PROBABILITY_SUM_TOLERANCE)
        assert result.all_cash_reason is None
