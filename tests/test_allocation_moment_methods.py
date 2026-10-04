"""Moment-based allocator tests: oracles, contracts, and solver discipline.

The spec verification row for the moment family: ERC risk contributions
equalize on random PD matrices, the mean-variance optimum matches a
brute-force grid and converges to all cash as risk aversion grows (with a
cash option available, zero exposure minimizes variance - the classical
fully invested GMV anchor is covered by its own closed-form oracles), a
single binary bet through ``from_model`` lands on Thorp's ``f*``, maximum
Sharpe matches both its closed form and a grid, and every scipy-gated
constructor points at ``keeks[allocation]`` when the extra is absent.
Around that row sit the contract matrix: scale-free evaluation, the
zero-bankroll convention, model-door wiring (exact moments and Monte Carlo
fallback), validation gates, determinism, and the loud risk-budgeting
non-convergence failure.
"""

import builtins

import numpy as np
import pytest

from keeks.allocation import (
    BaseAllocationStrategy,
    ExponentialGradient,
    FixedWeights,
    GlobalMinimumVariance,
    MaximumDiversification,
    MaximumSharpe,
    MeanVariance,
    RiskAversionScaling,
    RiskBudgeting,
    binary_bets_model,
)
from keeks.allocation import moments as moments_module
from keeks.utils import PROBABILITY_SUM_TOLERANCE


@pytest.fixture()
def no_scipy(monkeypatch):
    """A fake ``builtins.__import__`` that fails for scipy modules."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _weight_contract(weights, option_count=2):
    """The house weight contract as assertions."""
    weights = np.asarray(weights, dtype=float)
    assert weights.shape == (option_count,)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert weights.sum() <= 1.0 + PROBABILITY_SUM_TOLERANCE
    return weights


def _random_pd_matrix(rng, size, correlation_scale=0.3):
    """A random strictly PD covariance with unit-scale variances."""
    factor = rng.normal(size=(size, size))
    matrix = factor @ factor.T + 0.5 * np.eye(size)
    np.fill_diagonal(matrix, np.diag(matrix))
    # Correlate mildly: scale off-diagonals so the matrix stays PD.
    off_diagonal = correlation_scale * (matrix - np.diag(np.diag(matrix)))
    matrix = np.diag(np.diag(matrix)) + off_diagonal + off_diagonal.T
    return matrix


class TestMeanVariance:
    def test_single_option_interior_oracle(self):
        # w = mu / (lambda * variance) = 0.02 / 0.04 = 0.5, the rest cash.
        strategy = MeanVariance([0.02], [[0.04]], risk_aversion=1.0)
        result = strategy.optimize()
        assert strategy.evaluate(1000.0) == pytest.approx((0.5,), abs=1e-8)
        assert result.converged is True
        assert result.iterations >= 1
        assert result.objective == pytest.approx(0.005, abs=1e-8)
        assert result.expected_growth == pytest.approx(0.005, abs=1e-8)
        assert result.volatility == pytest.approx(0.1, abs=1e-8)

    def test_matches_brute_force_grid(self):
        rng = np.random.default_rng(42)
        mean = np.array([0.03, 0.015, 0.022])
        covariance = _random_pd_matrix(rng, 3)

        strategy = MeanVariance(mean, covariance)
        weights = np.asarray(strategy.evaluate(1000.0))
        solved_utility = float(weights @ mean - 0.5 * weights @ covariance @ weights)

        grid = np.arange(0.0, 1.0, 0.02)
        best = -np.inf
        for first in grid:
            for second in grid:
                third = 1.0 - first - second
                if third < 0.0:
                    continue
                candidate = np.array([first, second, third])
                utility = float(
                    candidate @ mean - 0.5 * candidate @ covariance @ candidate
                )
                best = max(best, utility)

        assert solved_utility >= best - 1e-6
        _weight_contract(weights, 3)

    def test_corner_solution_binds_the_budget(self):
        # The unconstrained optimum (100000) far exceeds the budget.
        strategy = MeanVariance([10.0, 0.0], [[0.0001, 0.0], [0.0, 1.0]])
        assert strategy.evaluate(1000.0) == pytest.approx((1.0, 0.0), abs=1e-8)

    def test_no_expected_return_holds_all_cash(self):
        strategy = MeanVariance([0.0, 0.0], [[0.04, 0.0], [0.0, 0.01]])
        result = strategy.optimize()
        assert result.weights == pytest.approx([0.0, 0.0], abs=1e-8)
        assert result.objective == pytest.approx(0.0, abs=1e-10)

    def test_doubling_risk_aversion_halves_the_stake(self):
        cheap = MeanVariance([0.02], [[0.04]], risk_aversion=1.0)
        dear = MeanVariance([0.02], [[0.04]], risk_aversion=2.0)
        assert cheap.evaluate(1000.0)[0] == pytest.approx(0.5, abs=1e-8)
        assert dear.evaluate(1000.0)[0] == pytest.approx(0.25, abs=1e-8)

    @pytest.mark.parametrize("risk_aversion", [0.0, -1.0, float("inf"), float("nan")])
    def test_risk_aversion_validation(self, risk_aversion):
        with pytest.raises(ValueError, match="Risk aversion"):
            MeanVariance([0.02], [[0.04]], risk_aversion=risk_aversion)

    def test_mean_and_covariance_validation(self):
        with pytest.raises(ValueError, match="Mean"):
            MeanVariance([[0.02, 0.01]], [[0.04, 0.0], [0.0, 0.01]])
        with pytest.raises(ValueError, match="Mean"):
            MeanVariance([float("nan"), 0.01], [[0.04, 0.0], [0.0, 0.01]])
        with pytest.raises(ValueError):
            MeanVariance([0.02, 0.01], [[0.04, 0.0], [0.0, 0.01]][::-1])
        with pytest.raises(ValueError):
            MeanVariance([0.02, 0.01], [[0.04, 0.005], [0.004, 0.01]])
        with pytest.raises(ValueError):
            MeanVariance([0.02], [[0.04, 0.0], [0.0, 0.01]])

    def test_from_model_uses_exact_moments(self):
        model = binary_bets_model([(0.6, 0.1, 1.0), (0.55, 0.05, 1.0)])
        exact_mean, exact_covariance = model.moments()
        from_model = MeanVariance.from_model(model)
        explicit = MeanVariance(exact_mean, exact_covariance)

        assert np.allclose(from_model.mean, exact_mean, rtol=0, atol=0)
        assert from_model.weights == pytest.approx(explicit.weights, abs=1e-12)

    def test_from_model_estimated_fallback(self):
        class NormalSampler:
            """A thin wrapper model whose exact moments are unavailable."""

            def sample(self, n_samples, rng):
                return rng.normal(
                    loc=[0.02, 0.01], scale=[0.2, 0.1], size=(n_samples, 2)
                )

        from_model = MeanVariance.from_model(NormalSampler(), n_samples=200_000, seed=7)
        explicit = MeanVariance([0.02, 0.01], [[0.04, 0.0], [0.0, 0.01]])
        assert from_model.weights == pytest.approx(explicit.weights, abs=5e-3)

    def test_covariance_only_from_model_estimated_fallback(self):
        class NormalSampler:
            """A sampling model with no exact ``moments`` hook."""

            def sample(self, n_samples, rng):
                return rng.normal(
                    loc=[0.02, 0.01], scale=[0.2, 0.1], size=(n_samples, 2)
                )

        from_model = GlobalMinimumVariance.from_model(
            NormalSampler(), n_samples=200_000, seed=7
        )
        explicit = GlobalMinimumVariance([[0.04, 0.0], [0.0, 0.01]])
        assert from_model.weights == pytest.approx(explicit.weights, abs=5e-3)

    def test_non_converged_solve_is_loud(self, monkeypatch):
        """A failed SLSQP result raises, never ships approximate weights."""
        import scipy.optimize

        class _Failed:
            success = False
            message = "Iteration limit reached"

        monkeypatch.setattr(scipy.optimize, "minimize", lambda *_a, **_k: _Failed())
        with pytest.raises(RuntimeError, match="did not converge"):
            MeanVariance([0.02], [[0.04]])

    @pytest.mark.usefixtures("no_scipy")
    def test_gated_import(self):
        with pytest.raises(ImportError, match=r"keeks\[allocation\]"):
            MeanVariance([0.02], [[0.04]])

    def test_scale_free_evaluation(self):
        strategy = MeanVariance([0.02, 0.01], [[0.04, 0.0], [0.0, 0.01]])
        assert strategy.evaluate(1000.0) == strategy.evaluate(1e12)
        assert strategy.evaluate(0.0) == (0.0, 0.0)
        assert strategy.evaluate(-100.0) == (0.0, 0.0)

    def test_solve_is_deterministic(self):
        first = MeanVariance([0.02, 0.01], [[0.04, 0.001], [0.001, 0.01]])
        second = MeanVariance([0.02, 0.01], [[0.04, 0.001], [0.001, 0.01]])
        assert np.array_equal(first.weights, second.weights)


class TestGlobalMinimumVariance:
    def test_uncorrelated_inverse_variance_oracle(self):
        result = GlobalMinimumVariance([[0.04, 0.0], [0.0, 0.01]]).optimize()
        assert result.weights == pytest.approx([0.2, 0.8], abs=1e-6)
        assert result.objective == pytest.approx(0.008, abs=1e-8)
        assert result.volatility == pytest.approx(np.sqrt(0.008), abs=1e-8)
        assert result.expected_growth is None

    def test_matches_closed_form_on_random_pd(self):
        # A well-conditioned matrix whose closed-form solution is interior.
        covariance = np.array(
            [[0.09, 0.006, 0.004], [0.006, 0.04, 0.005], [0.004, 0.005, 0.01]]
        )
        closed_form = np.linalg.solve(covariance, np.ones(3))
        closed_form /= closed_form.sum()
        assert np.all(closed_form > 0.0)  # interior: the oracle applies

        result = GlobalMinimumVariance(covariance).optimize()
        assert result.weights == pytest.approx(closed_form, abs=1e-6)

    def test_corner_solution_on_correlated_pair(self):
        # rho = 0.9 with a dominant low-variance option: the closed form
        # wants a negative first weight, so the long-only answer is the
        # corner that holds the quiet option alone.
        covariance = np.array([[1.0, 0.09], [0.09, 0.01]])
        result = GlobalMinimumVariance(covariance).optimize()
        assert result.weights == pytest.approx([0.0, 1.0], abs=1e-6)

    def test_from_model_drops_the_mean(self):
        model = binary_bets_model([(0.6, 0.1, 1.0), (0.55, 0.05, 1.0)])
        _, exact_covariance = model.moments()
        from_model = GlobalMinimumVariance.from_model(model)
        explicit = GlobalMinimumVariance(exact_covariance)

        assert not hasattr(from_model, "mean")
        assert from_model.weights == pytest.approx(explicit.weights, abs=1e-12)

    @pytest.mark.usefixtures("no_scipy")
    def test_gated_import(self):
        with pytest.raises(ImportError, match="GlobalMinimumVariance"):
            GlobalMinimumVariance([[0.04, 0.0], [0.0, 0.01]])

    def test_invalid_covariance(self):
        with pytest.raises(ValueError):
            GlobalMinimumVariance([[0.04, 0.001], [0.002, 0.01]])
        with pytest.raises(ValueError):
            GlobalMinimumVariance([[0.04, 0.0], [0.0, -0.01]])

    def test_scale_free_evaluation(self):
        strategy = GlobalMinimumVariance([[0.04, 0.0], [0.0, 0.01]])
        assert strategy.evaluate(1000.0) == strategy.evaluate(1e12)
        assert strategy.evaluate(0.0) == (0.0, 0.0)


class TestMaximumSharpe:
    def test_matches_closed_form_tangency(self):
        mean = np.array([0.03, 0.015])
        covariance = np.array([[0.04, 0.004], [0.004, 0.01]])
        risk_free = 0.001
        excess = mean - risk_free
        closed_form = np.linalg.solve(covariance, excess)
        closed_form /= closed_form.sum()

        strategy = MaximumSharpe(mean, covariance, risk_free=risk_free)
        assert strategy.weights == pytest.approx(closed_form, abs=1e-4)

    def test_objective_reports_the_maximized_ratio(self):
        mean = np.array([0.03, 0.015])
        covariance = np.array([[0.04, 0.004], [0.004, 0.01]])
        result = MaximumSharpe(mean, covariance).optimize()
        weights = result.weights
        ratio = float(weights @ mean) / np.sqrt(float(weights @ covariance @ weights))
        assert result.objective == pytest.approx(ratio, abs=1e-8)
        growth = float(weights @ mean) - 0.5 * float(weights @ covariance @ weights)
        assert result.expected_growth == pytest.approx(growth, abs=1e-8)

    def test_matches_brute_force_grid(self):
        rng = np.random.default_rng(11)
        mean = np.array([0.04, 0.02, 0.025])
        covariance = _random_pd_matrix(rng, 3)

        strategy = MaximumSharpe(mean, covariance)
        weights = np.asarray(strategy.evaluate(1000.0))
        solved_ratio = float(weights @ mean) / np.sqrt(
            float(weights @ covariance @ weights)
        )

        grid = np.arange(0.0, 1.0, 0.02)
        best = -np.inf
        for first in grid:
            for second in grid:
                third = 1.0 - first - second
                if third <= 0.0:
                    continue
                candidate = np.array([first, second, third])
                ratio = float(candidate @ mean) / np.sqrt(
                    float(candidate @ covariance @ candidate)
                )
                best = max(best, ratio)

        assert solved_ratio >= best - 1e-9
        _weight_contract(weights, 3)

    def test_single_option_goes_all_in(self):
        # A winning bet (positive expected return) puts the whole budget on
        # the option: the Sharpe ratio is scale-free, so fully invest.
        # Net odds need payoff > 1 for the win to profit: (0.9, 1.2, 1.0)
        # wins +0.2 and loses -1.0 per unit staked.
        strategy = MaximumSharpe.from_model(binary_bets_model([(0.9, 1.2, 1.0)]))
        assert strategy.evaluate(1000.0) == pytest.approx((1.0,), abs=1e-8)

    def test_requires_excess_return(self):
        with pytest.raises(ValueError, match="risk-free"):
            MaximumSharpe([0.01, 0.02], [[0.04, 0.0], [0.0, 0.01]], risk_free=0.05)

    def test_risk_free_validation(self):
        with pytest.raises(ValueError):
            MaximumSharpe(
                [0.03, 0.02], [[0.04, 0.0], [0.0, 0.01]], risk_free=float("nan")
            )

    @pytest.mark.usefixtures("no_scipy")
    def test_gated_import(self):
        with pytest.raises(ImportError, match="MaximumSharpe"):
            MaximumSharpe([0.03, 0.02], [[0.04, 0.0], [0.0, 0.01]])

    def test_scale_free_evaluation(self):
        strategy = MaximumSharpe(
            [0.03, 0.02], [[0.04, 0.0], [0.0, 0.01]], risk_free=0.0
        )
        assert strategy.evaluate(1000.0) == strategy.evaluate(1e12)
        assert strategy.evaluate(0.0) == (0.0, 0.0)


class TestMaximumDiversification:
    def test_diagonal_inverse_volatility_oracle(self):
        result = MaximumDiversification([[0.04, 0.0], [0.0, 0.01]]).optimize()
        assert result.weights == pytest.approx([1.0 / 3.0, 2.0 / 3.0], abs=1e-6)

    def test_matches_closed_form_and_reads_the_ratio(self):
        covariance = np.array([[0.09, 0.006], [0.006, 0.04]])
        volatility = np.sqrt(np.diag(covariance))
        closed_form = np.linalg.solve(covariance, volatility)
        closed_form /= closed_form.sum()

        result = MaximumDiversification(covariance).optimize()
        assert result.weights == pytest.approx(closed_form, abs=1e-4)

        weights = result.weights
        ratio = float(weights @ volatility) / np.sqrt(
            float(weights @ covariance @ weights)
        )
        assert result.objective == pytest.approx(ratio, abs=1e-8)

    def test_beats_equal_weights_on_the_ratio(self):
        covariance = np.array([[0.09, 0.006], [0.006, 0.04]])
        result = MaximumDiversification(covariance).optimize()
        weights = result.weights
        volatility = np.sqrt(np.diag(covariance))

        def ratio(candidate):
            return float(candidate @ volatility) / np.sqrt(
                float(candidate @ covariance @ candidate)
            )

        assert ratio(weights) > ratio(np.array([0.5, 0.5]))

    def test_zero_variance_rejected(self):
        with pytest.raises(ValueError):
            MaximumDiversification([[0.04, 0.0], [0.0, 0.0]])

    @pytest.mark.usefixtures("no_scipy")
    def test_gated_import(self):
        with pytest.raises(ImportError, match="MaximumDiversification"):
            MaximumDiversification([[0.04, 0.0], [0.0, 0.01]])

    def test_from_model_uses_exact_covariance(self):
        model = binary_bets_model([(0.6, 0.1, 1.0), (0.55, 0.05, 1.0)])
        _, exact_covariance = model.moments()
        from_model = MaximumDiversification.from_model(model)
        explicit = MaximumDiversification(exact_covariance)
        assert from_model.weights == pytest.approx(explicit.weights, abs=1e-12)

    def test_scale_free_evaluation(self):
        strategy = MaximumDiversification([[0.04, 0.0], [0.0, 0.01]])
        assert strategy.evaluate(1000.0) == strategy.evaluate(1e12)
        assert strategy.evaluate(-1.0) == (0.0, 0.0)


class TestRiskBudgeting:
    def test_equal_risk_contributions_on_random_pd_matrices(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            size = int(rng.integers(2, 6))
            covariance = _random_pd_matrix(rng, size)
            result = RiskBudgeting(covariance).optimize()

            weights = _weight_contract(result.weights, size)
            assert weights.sum() == pytest.approx(1.0, abs=1e-9)
            contributions = weights * (covariance @ weights)
            assert contributions == pytest.approx(contributions.mean(), rel=1e-6)

    def test_custom_budgets_are_contribution_shares(self):
        rng = np.random.default_rng(101)
        covariance = _random_pd_matrix(rng, 3)
        budgets = np.array([0.7, 0.2, 0.1])
        result = RiskBudgeting(covariance, risk_budgets=budgets).optimize()

        weights = _weight_contract(result.weights, 3)
        contributions = weights * (covariance @ weights)
        shares = contributions / contributions.sum()
        assert shares == pytest.approx(budgets, abs=1e-6)

    def test_budgets_are_scale_free(self):
        covariance = [[0.04, 0.001], [0.001, 0.01]]
        doubled = RiskBudgeting(covariance, risk_budgets=[2.0, 1.0])
        plain = RiskBudgeting(covariance, risk_budgets=[1.0, 0.5])
        assert doubled.weights == pytest.approx(plain.weights, abs=1e-9)

    def test_diagonal_closed_form_oracle(self):
        result = RiskBudgeting([[0.04, 0.0], [0.0, 0.01]]).optimize()
        assert result.weights == pytest.approx([1.0 / 3.0, 2.0 / 3.0], abs=1e-8)
        budgeted = RiskBudgeting([[0.04, 0.0], [0.0, 0.01]], risk_budgets=[0.75, 0.25])
        closed_form = np.sqrt([0.75 / 0.04, 0.25 / 0.01])
        closed_form /= closed_form.sum()
        assert budgeted.weights == pytest.approx(closed_form, abs=1e-8)

    def test_optimize_diagnostics(self):
        strategy = RiskBudgeting([[0.04, 0.001], [0.001, 0.01]])
        result = strategy.optimize()
        assert result.converged is True
        assert result.iterations >= 1
        assert result.objective is not None
        assert result.expected_growth is None
        assert result.volatility == pytest.approx(
            np.sqrt(float(strategy.weights @ strategy.covariance @ strategy.weights)),
            abs=1e-12,
        )

    @pytest.mark.parametrize(
        "budgets",
        [
            "abc",
            [None, 1.0],
            [-0.1, 1.1],
            [0.0, 1.0],
            [float("nan"), 1.0],
            [float("inf"), 0.0],
            [0.5],
            [0.5, 0.25, 0.25],
            [[0.5], [0.5]],
        ],
    )
    def test_budget_validation(self, budgets):
        with pytest.raises(ValueError):
            RiskBudgeting([[0.04, 0.001], [0.001, 0.01]], risk_budgets=budgets)

    def test_zero_variance_rejected(self):
        with pytest.raises(ValueError, match="strictly positive variances"):
            RiskBudgeting([[0.04, 0.0], [0.0, 0.0]])

    def test_non_convergence_is_loud(self, monkeypatch):
        monkeypatch.setattr(moments_module, "_CCD_MAX_SWEEPS", 1)
        with pytest.raises(RuntimeError, match="did not converge"):
            RiskBudgeting([[0.04, 0.01], [0.01, 0.01]])

    @pytest.mark.usefixtures("no_scipy")
    def test_numpy_only_construction(self):
        strategy = RiskBudgeting([[0.04, 0.0], [0.0, 0.01]])
        assert strategy.evaluate(1000.0) == pytest.approx(
            (1.0 / 3.0, 2.0 / 3.0), abs=1e-8
        )

    def test_from_model_uses_exact_covariance(self):
        model = binary_bets_model([(0.6, 0.1, 1.0), (0.55, 0.05, 1.0)])
        _, exact_covariance = model.moments()
        from_model = RiskBudgeting.from_model(model)
        explicit = RiskBudgeting(exact_covariance)
        assert not hasattr(from_model, "mean")
        assert from_model.weights == pytest.approx(explicit.weights, abs=1e-12)

    def test_scale_free_evaluation(self):
        strategy = RiskBudgeting([[0.04, 0.0], [0.0, 0.01]])
        assert strategy.evaluate(1000.0) == strategy.evaluate(1e12)
        assert strategy.evaluate(0.0) == (0.0, 0.0)

    def test_solve_is_deterministic(self):
        covariance = [[0.04, 0.005], [0.005, 0.01]]
        first = RiskBudgeting(covariance)
        second = RiskBudgeting(covariance)
        assert np.array_equal(first.weights, second.weights)


class TestRiskAversionScaling:
    def test_scales_weights_toward_cash(self):
        inner = FixedWeights([0.2, 0.3, 0.4])
        wrapper = RiskAversionScaling(inner, factor=0.5)
        assert wrapper.evaluate(1000.0) == (0.1, 0.15, 0.2)

    def test_factor_one_is_the_identity(self):
        inner = FixedWeights([0.2, 0.3, 0.4])
        assert RiskAversionScaling(inner, factor=1.0).evaluate(1000.0) == (
            0.2,
            0.3,
            0.4,
        )

    @pytest.mark.parametrize("factor", [-0.1, 1.5, float("inf"), float("nan")])
    def test_factor_validation(self, factor):
        with pytest.raises(ValueError, match="factor"):
            RiskAversionScaling(FixedWeights([0.5]), factor=factor)

    def test_inner_type_validation(self):
        with pytest.raises(ValueError, match="BaseAllocationStrategy"):
            RiskAversionScaling(FixedWeights([0.5]).evaluate)
        with pytest.raises(ValueError, match="BaseAllocationStrategy"):
            RiskAversionScaling(0.5)

    def test_optimize_scales_a_static_inner(self):
        inner = RiskBudgeting([[0.04, 0.001], [0.001, 0.01]])
        wrapper = RiskAversionScaling(inner, factor=0.25)
        result = wrapper.optimize()

        assert result.weights == pytest.approx(inner.weights * 0.25, abs=1e-12)
        assert result.volatility == pytest.approx(
            inner.optimize().volatility * 0.25, abs=1e-12
        )
        assert result.converged == inner.optimize().converged
        assert result.iterations == inner.optimize().iterations
        assert result.objective is None

    def test_optimize_with_an_online_inner_has_no_diagnostics(self):
        wrapper = RiskAversionScaling(ExponentialGradient(option_count=2), factor=0.5)
        result = wrapper.optimize()
        assert result.objective is None
        assert result.converged is None
        assert result.iterations is None

    def test_evaluate_passes_the_bankroll_through(self):
        inner = FixedWeights([0.25, 0.75])
        wrapper = RiskAversionScaling(inner, factor=0.5)
        assert wrapper.evaluate(0.0) == (0.0, 0.0)
        assert wrapper.evaluate(-100.0) == (0.0, 0.0)
        assert wrapper.evaluate(1000.0) == wrapper.evaluate(1e12)

    @pytest.mark.usefixtures("no_scipy")
    def test_numpy_only_wrapper(self):
        inner = RiskBudgeting([[0.04, 0.0], [0.0, 0.01]])
        wrapper = RiskAversionScaling(inner, factor=0.5)
        assert wrapper.evaluate(1000.0) == pytest.approx(
            (1.0 / 6.0, 1.0 / 3.0), abs=1e-8
        )


class TestExports:
    def test_family_is_importable_from_the_subpackage(self):
        for allocator in (
            MeanVariance,
            GlobalMinimumVariance,
            MaximumSharpe,
            MaximumDiversification,
            RiskBudgeting,
            RiskAversionScaling,
        ):
            assert issubclass(allocator, BaseAllocationStrategy)
