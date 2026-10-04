"""
Tests for the joint-return input models (``keeks.allocation.models``).

The layers under test, per the spec's models verification row:

- the binary-bets adapter's stream keying (a port of the portfolio
  simulator's seeding guarantees at model level) and its settlement
  arithmetic matching :class:`keeks.multi_outcome.PortfolioSimulator`;
- marginal built-ins drawing with the right shape, sample moments within
  tolerance for the finite-variance families, and ``None`` moments for
  ``nu <= 2``;
- closed-form ``moments()`` for the binary and scenario adapters;
- :func:`fit_marginals_model` recovering known parameters on synthetic
  draws and pointing at ``keeks[allocation]`` when scipy is absent;
- the Gaussian-copula gated import;
- the callable escape hatch and :func:`estimate_moments` consistency.
"""

import builtins
import re

import numpy as np
import pytest
import scipy.stats

from keeks.allocation.models import (
    BinaryBetsModel,
    JointReturnModel,
    MarginalModel,
    ModelInputMixin,
    ScenarioModel,
    _bet_spawn_key,
    _validate_dependence,
    _validate_draws,
    _validate_n_samples,
    binary_bets_model,
    estimate_moments,
    fit_marginals_model,
    marginals_model,
    scenario_model,
)
from keeks.bankroll import BankRoll
from keeks.multi_outcome.simulators import PortfolioSimulator, _portfolio_rngs


def _deterministic_sampler(mean=0.01, sd=0.02):
    """Build a duck-typed sampler: any object with ``sample(n, rng)``."""

    class Sampler:
        def sample(self, n_samples, rng):
            return rng.normal(mean, sd, (n_samples, 2))

    return Sampler()


class TestSampleCountValidation:
    """The draw-count gate every sampler shares."""

    def test_rejects_non_integer_counts(self):
        with pytest.raises(ValueError, match="Sample count must be a positive integer"):
            _validate_n_samples("12")

    def test_rejects_nonpositive_counts(self):
        with pytest.raises(ValueError, match="Sample count must be a positive integer"):
            _validate_n_samples(0)

    def test_accepts_positive_integer(self):
        assert _validate_n_samples(5) == 5


class TestDrawsValidation:
    """The (n_samples, N) contract every model's output must satisfy."""

    def test_accepts_valid_draws(self):
        draws = _validate_draws(np.ones((3, 2)), 3)
        assert draws.shape == (3, 2)

    def test_rejects_non_2d(self):
        with pytest.raises(
            ValueError, match="Joint-return draws must be two-dimensional"
        ):
            _validate_draws([0.1, 0.2], 1)

    def test_rejects_row_mismatch(self):
        with pytest.raises(
            ValueError,
            match=re.escape("Joint-return draws must have shape (2, N)"),
        ):
            _validate_draws(np.ones((3, 2)), 2)

    def test_rejects_empty_options(self):
        with pytest.raises(
            ValueError,
            match=re.escape("Joint-return draws must have shape (2, N)"),
        ):
            _validate_draws(np.ones((2, 0)), 2)

    def test_rejects_non_finite(self):
        with pytest.raises(
            ValueError, match="Joint-return draws must contain only finite values"
        ):
            _validate_draws(np.array([[0.1], [np.inf]]), 2)

    def test_rejects_ragged_input(self):
        with pytest.raises(
            ValueError, match="Joint-return draws must be a finite sequence"
        ):
            _validate_draws([[0.1, 0.2], [0.3]], 2)


class TestScenarioModel:
    """Empirical rows: iid bootstrap, weighted draws, cash residual."""

    def test_shape_and_determinism(self):
        scenarios = [[0.01, 0.02], [-0.01, 0.0], [0.03, -0.02]]
        model = scenario_model(scenarios)
        assert isinstance(model, ScenarioModel)
        assert isinstance(model, JointReturnModel)
        first = model.sample(8, np.random.default_rng(5))
        second = model.sample(8, np.random.default_rng(5))
        assert first.shape == (8, 2)
        assert np.array_equal(first, second)

    def test_equal_weight_resampling_draws_only_observed_rows(self):
        scenarios = np.array([[0.05, 0.05], [-0.03, -0.03]])
        model = scenario_model(scenarios)
        draws = model.sample(500, np.random.default_rng(1))
        for row in draws:
            assert any(np.allclose(row, observed) for observed in scenarios)

    def test_weighted_draws_follow_probabilities(self):
        # One heavy row and one light row: the empirical frequency of each
        # tracks its probability.
        model = scenario_model([[1.0, 1.0], [2.0, 2.0]], probabilities=[0.9, 0.1])
        draws = model.sample(2000, np.random.default_rng(2))
        fraction_heavy = float(np.mean(draws[:, 0] == 1.0))
        assert fraction_heavy == pytest.approx(0.9, abs=0.05)

    def test_residual_probability_is_all_cash(self):
        # Residual mass below one is an all-cash zero row, never renormalized.
        model = scenario_model([[0.1, 0.2]], probabilities=[0.4])
        draws = model.sample(4000, np.random.default_rng(3))
        cash_rows = draws[np.all(draws == 0.0, axis=1)]
        non_cash = draws[~np.all(draws == 0.0, axis=1)]
        assert cash_rows.shape[0] / 4000 == pytest.approx(0.6, abs=0.03)
        assert np.allclose(non_cash, [0.1, 0.2])

    def test_exact_moments_equal_weight(self):
        scenarios = np.array([[0.02, 0.0], [-0.02, 0.0]])
        mean, covariance = scenario_model(scenarios).moments()
        assert np.allclose(mean, [0.0, 0.0])
        assert np.allclose(covariance, np.diag([0.0004, 0.0]))

    def test_exact_moments_with_residual_cash(self):
        # The zero row carries the residual mass in the moment sums too.
        mean, _ = scenario_model([[0.2, 0.2]], probabilities=[0.5]).moments()
        assert np.allclose(mean, [0.1, 0.1])

    def test_validates_inputs(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            scenario_model([0.1, 0.2])
        with pytest.raises(ValueError, match="one entry per scenario row"):
            scenario_model([[0.1], [0.2]], probabilities=[0.5])


class TestBinaryBetsModel:
    """Keeks-native bets: settlement arithmetic and stream keying."""

    def test_is_a_joint_return_model(self):
        assert isinstance(binary_bets_model([(0.5, 2.0, 1.0)]), JointReturnModel)

    def test_shape_and_determinism(self):
        model = binary_bets_model([(0.5, 2.0, 1.0), (0.25, 3.0, 1.0)])
        first = model.sample(16, np.random.default_rng(7))
        second = model.sample(16, np.random.default_rng(7))
        assert first.shape == (16, 2)
        assert np.array_equal(first, second)

    def test_stream_family_matches_the_portfolio_simulator(self):
        # The model derives each bet's stream from SeedSequence(child,
        # spawn_key=BLAKE2b(bet) + occurrence) -- exactly the portfolio
        # simulator's keying family, with the simulator seed replaced by
        # the per-call child entropy.
        bet = (0.6, 0.5, 0.2)
        portfolio_rng = _portfolio_rngs(0, [bet])[0]
        model_rng = np.random.default_rng(
            np.random.SeedSequence(0, spawn_key=_bet_spawn_key(bet, {}))
        )
        assert np.array_equal(portfolio_rng.random(10), model_rng.random(10))

    def test_settlement_arithmetic_matches_portfolio_simulator(self):
        # The model's per-period returns ARE the portfolio simulator's
        # settlement per unit staked: win (payoff - 1) - tc, loss -(loss + tc)
        # -- the simulator's flat fee on a unit stake is exactly the per-unit
        # transaction cost.
        probability, payoff, loss = 0.6, 0.5, 0.2
        rate = 0.02
        model = BinaryBetsModel([(probability, payoff, loss)], rate)
        draws = model.sample(300, np.random.default_rng(0))

        # Reproduce the model's stream with the documented keying family.
        rng = np.random.default_rng(0)
        child_entropy = int.from_bytes(rng.bytes(8), "big")
        stream = np.random.default_rng(
            np.random.SeedSequence(
                child_entropy, spawn_key=_bet_spawn_key((probability, payoff, loss), {})
            )
        )
        uniforms = stream.random(300)

        expected = np.where(
            uniforms < probability,
            (payoff - 1.0) - rate,
            -(loss + rate),
        )
        assert np.array_equal(draws[:, 0], expected)

        # And the same uniform maps to the PortfolioSimulator's settlement
        # on a unit stake with the flat fee set to the per-unit cost.
        fee = rate  # flat fee on a stake of 1.0
        portfolio_net = np.where(
            uniforms < probability,
            (payoff - 1.0) * 1.0 - fee,
            -((loss * 1.0) + fee),
        )
        assert np.allclose(draws[:, 0], portfolio_net)

    def test_bankroll_replay_matches_portfolio_settlement(self):
        # Full loop, byte-identical streams: the model's child entropy for
        # default_rng(0) is the simulator seed that reproduces the exact
        # same per-bet stream, so staking 100% on one fee-less bet replays
        # the model's returns as the bankroll history itself. Net odds need
        # payoff > 1 for the win to profit: (0.6, 1.5, 0.2) wins +0.5 and
        # loses -0.2 per unit staked, matching the gross-era economics.
        bet = (0.6, 1.5, 0.2)
        trials = 50
        model = binary_bets_model([bet])
        child_entropy = int.from_bytes(np.random.default_rng(0).bytes(8), "big")
        draws = model.sample(trials, np.random.default_rng(0))[:, 0]

        class AllInStrategy:
            def evaluate(self, probabilities, current_bankroll):  # noqa: ARG002
                return (1.0,)

        bankroll = BankRoll(initial_funds=1000.0, max_transaction_loss=None)
        simulator = PortfolioSimulator(
            bets=[bet], fee_per_bet=0.0, trials=trials, seed=child_entropy
        )
        simulator.evaluate_strategy(AllInStrategy(), bankroll)
        assert len(bankroll.history) == trials + 1
        replayed = 1000.0 * np.cumprod(1.0 + draws)
        assert np.allclose(bankroll.history[1:], replayed)

    def test_stream_keying_survives_insertion(self):
        # A bet keeps its stream when another bet is inserted before it --
        # the portfolio simulator's documented stream family, at model level.
        bet = (0.55, 2.0, 1.0)
        base = binary_bets_model([bet])
        inserted = binary_bets_model([(0.9, 1.2, 0.3), bet])
        base_draws = base.sample(40, np.random.default_rng(9))
        inserted_draws = inserted.sample(40, np.random.default_rng(9))
        assert np.array_equal(base_draws[:, 0], inserted_draws[:, 1])

    def test_stream_keying_survives_removal_and_reordering(self):
        bets = [(0.55, 2.0, 1.0), (0.3, 1.5, 0.5), (0.7, 1.1, 0.2)]
        full = binary_bets_model(bets)
        pruned = binary_bets_model([bets[0], bets[2]])
        full_draws = full.sample(40, np.random.default_rng(4))
        pruned_draws = pruned.sample(40, np.random.default_rng(4))
        assert np.array_equal(full_draws[:, 0], pruned_draws[:, 0])
        assert np.array_equal(full_draws[:, 2], pruned_draws[:, 1])

    def test_identical_bets_get_independent_streams(self):
        # Identical tuples disambiguate by occurrence ordinal: two copies
        # draw different, but still deterministic, streams.
        duplicated = binary_bets_model([(0.5, 2.0, 1.0), (0.5, 2.0, 1.0)])
        draws = duplicated.sample(200, np.random.default_rng(8))
        assert not np.array_equal(draws[:, 0], draws[:, 1])
        again = duplicated.sample(200, np.random.default_rng(8))
        assert np.array_equal(draws, again)

    def test_streams_differ_across_seeds(self):
        model = binary_bets_model([(0.5, 2.0, 1.0)])
        assert not np.array_equal(
            model.sample(32, np.random.default_rng(1)),
            model.sample(32, np.random.default_rng(2)),
        )

    def test_closed_form_moments_with_costs(self):
        probability, payoff, loss = 0.55, 2.0, 1.0
        tc = 0.01
        model = BinaryBetsModel([(probability, payoff, loss)], transaction_cost_rate=tc)
        mean, covariance = model.moments()
        win_return = (payoff - 1.0) - tc
        loss_return = loss + tc
        expected_mean = probability * win_return - (1.0 - probability) * loss_return
        expected_second = (
            probability * win_return**2 + (1.0 - probability) * loss_return**2
        )
        assert np.allclose(mean, [expected_mean])
        assert np.allclose(covariance, [[expected_second - expected_mean**2]])

    def test_closed_form_moments_diagonal_for_independent_bets(self):
        mean, covariance = BinaryBetsModel(
            [(0.5, 2.0, 1.0), (0.25, 3.0, 1.0)]
        ).moments()
        assert np.allclose(mean, [0.0, -0.25])
        assert np.allclose(covariance[0, 1], 0.0)
        assert np.allclose(covariance[1, 0], 0.0)

    def test_sample_moments_converge_to_closed_form(self):
        model = binary_bets_model([(0.55, 2.0, 1.0)])
        draws = model.sample(200_000, np.random.default_rng(42))
        mean, covariance = model.moments()
        assert np.allclose(draws.mean(axis=0), mean, atol=0.01)
        assert np.allclose(np.diag(covariance), np.var(draws, axis=0), atol=0.02)

    def test_validates_inputs(self):
        with pytest.raises(
            ValueError, match="Bet 0 probability must be between 0 and 1"
        ):
            binary_bets_model([(1.5, 2.0, 1.0)])
        with pytest.raises(
            ValueError, match="Transaction cost rate must be non-negative"
        ):
            BinaryBetsModel([(0.5, 2.0, 1.0)], transaction_cost_rate=-0.01)


class TestMarginalModel:
    """Per-option parametric marginals: independence and the Gaussian copula."""

    def test_is_a_joint_return_model(self):
        assert isinstance(marginals_model([("normal", 0.0, 1.0)]), JointReturnModel)

    @pytest.mark.parametrize(
        "spec",
        [
            ("normal", 0.01, 0.02),
            ("student_t", 5.0),
            ("student_t", 5.0, 0.01, 0.02),
            ("laplace", 0.0, 0.01),
            ("lognormal", 1.0, 0.05),
            ("binary", 0.5, 2.0, 1.0),
        ],
    )
    def test_builtin_families_draw_the_right_shape(self, spec):
        model = marginals_model([spec, spec])
        draws = model.sample(11, np.random.default_rng(3))
        assert draws.shape == (11, 2)
        assert np.all(np.isfinite(draws))

    @pytest.mark.parametrize(
        "spec",
        [
            ("normal", 0.01, 0.02),
            ("laplace", 0.0, 0.01),
            ("lognormal", 1.0, 0.05),
            ("binary", 0.5, 2.0, 1.0),
        ],
    )
    def test_finite_variance_families_sample_moments_within_tolerance(self, spec):
        model = marginals_model([spec])
        mean, covariance = model.moments()
        draws = model.sample(400_000, np.random.default_rng(13))
        assert np.allclose(draws.mean(axis=0), mean, atol=0.005)
        assert np.allclose(np.var(draws, axis=0), covariance[0, 0], atol=0.005)

    def test_student_t_sample_moments_within_tolerance(self):
        # nu = 5: finite variance nu / (nu - 2) around the location.
        model = marginals_model([("student_t", 5.0, 0.01, 0.02)])
        draws = model.sample(400_000, np.random.default_rng(13))
        assert draws.mean(axis=0)[0] == pytest.approx(0.01, abs=0.002)
        assert np.var(draws[:, 0]) == pytest.approx(0.02**2 * 5.0 / 3.0, rel=0.05)

    def test_student_t_moments_none_for_infinite_variance(self):
        assert marginals_model([("student_t", 2.0)]).moments() is None
        assert marginals_model([("student_t", 1.5)]).moments() is None

    def test_student_t_moments_above_two_finite_variance(self):
        # nu = 5: variance is scale^2 * nu / (nu - 2).
        mean, covariance = marginals_model([("student_t", 5.0, 0.01, 0.02)]).moments()
        assert np.allclose(mean, [0.01])
        assert np.allclose(covariance, [[0.02**2 * 5.0 / 3.0]])

    def test_mixed_marginals_moments_none_when_any_is_fat(self):
        model = marginals_model([("normal", 0.0, 1.0), ("student_t", 1.5)])
        assert model.moments() is None

    def test_mixed_marginals_moments_exact_when_all_finite(self):
        mean, covariance = marginals_model(
            [("normal", 0.01, 0.02), ("binary", 0.5, 2.0, 1.0)]
        ).moments()
        assert np.allclose(mean, [0.01, 0.5])
        assert np.allclose(np.diag(covariance), [0.0004, 2.25])
        assert np.allclose(covariance[0, 1], 0.0)

    def test_determinism_given_a_generator(self):
        model = marginals_model([("normal", 0.0, 1.0), ("laplace", 0.0, 1.0)])
        first = model.sample(50, np.random.default_rng(17))
        second = model.sample(50, np.random.default_rng(17))
        assert np.array_equal(first, second)

    def test_gaussian_copula_recovers_correlation(self):
        # For jointly normal marginals the Pearson correlation is the
        # copula parameter itself.
        dependence = [[1.0, 0.7], [0.7, 1.0]]
        model = MarginalModel(
            [("normal", 0.0, 1.0), ("normal", 0.0, 1.0)], dependence=dependence
        )
        draws = model.sample(200_000, np.random.default_rng(6))
        correlation = np.corrcoef(draws, rowvar=False)[0, 1]
        assert correlation == pytest.approx(0.7, abs=0.02)

    def test_gaussian_copula_preserves_marginals(self):
        dependence = [[1.0, 0.4], [0.4, 1.0]]
        model = MarginalModel(
            [("normal", 0.01, 0.02), ("binary", 0.5, 2.0, 1.0)],
            dependence=dependence,
        )
        draws = model.sample(200_000, np.random.default_rng(6))
        assert draws[:, 0].mean() == pytest.approx(0.01, abs=0.001)
        assert draws[:, 0].std() == pytest.approx(0.02, abs=0.001)
        assert float(np.mean(draws[:, 1] == 2.0)) == pytest.approx(0.5, abs=0.01)

    def test_gaussian_copula_moments_are_none(self):
        # Pearson covariances have no closed form under a copula.
        model = MarginalModel(
            [("normal", 0.0, 1.0), ("normal", 0.0, 1.0)],
            dependence=[[1.0, 0.3], [0.3, 1.0]],
        )
        assert model.moments() is None

    @pytest.mark.parametrize(
        "spec",
        [
            ("student_t", 5.0, 0.01, 0.02),
            ("laplace", 0.0, 0.01),
            ("lognormal", 1.0, 0.05),
        ],
    )
    def test_gaussian_copula_samples_non_normal_marginals(self, spec):
        # The copula path ppf-warps each family's uniforms (scipy-gated).
        model = MarginalModel([spec, spec], dependence=[[1.0, 0.3], [0.3, 1.0]])
        draws = model.sample(64, np.random.default_rng(12))
        assert draws.shape == (64, 2)
        assert np.all(np.isfinite(draws))
        assert model.moments() is None

    def test_dependence_validation(self):
        with pytest.raises(ValueError, match="Dependence must be a valid correlation"):
            marginals_model([("normal", 0.0, 1.0)], dependence=[[1.0, 0.5], [0.4, 1.0]])
        with pytest.raises(ValueError, match="unit diagonal"):
            _validate_dependence([[2.0]])
        with pytest.raises(ValueError, match="unit diagonal"):
            _validate_dependence([[0.5]])

    def test_marginal_spec_validation(self):
        empty = "Marginals must be a non-empty sequence"
        with pytest.raises(ValueError, match=empty):
            marginals_model([])
        with pytest.raises(ValueError, match=empty):
            marginals_model(42)
        with pytest.raises(ValueError, match="Marginal 0 must be a"):
            marginals_model(["normal"])
        with pytest.raises(ValueError, match="unknown family 'invented'"):
            marginals_model([("invented", 1.0)])
        with pytest.raises(ValueError, match="Marginal 0 sd must be non-negative"):
            marginals_model([("normal", 0.0, -1.0)])
        with pytest.raises(ValueError, match="Marginal 0 nu must be positive"):
            marginals_model([("student_t", 0.0)])
        with pytest.raises(ValueError, match="Marginal 0 scale must be non-negative"):
            marginals_model([("student_t", 5.0, 0.0, -1.0)])
        with pytest.raises(ValueError, match="Marginal 0 scale must be non-negative"):
            marginals_model([("laplace", 0.0, -1.0)])
        with pytest.raises(ValueError, match="Marginal 0 mean must be positive"):
            marginals_model([("lognormal", 0.0, 0.1)])
        with pytest.raises(ValueError, match="Marginal 0 sd must be non-negative"):
            marginals_model([("lognormal", 1.0, -0.1)])
        with pytest.raises(ValueError, match="Marginal 0 probability must be between"):
            marginals_model([("binary", 1.5, 2.0, 1.0)])
        with pytest.raises(ValueError, match="Marginal 0 payoff must be greater"):
            marginals_model([("binary", 0.5, 0.0, 1.0)])
        with pytest.raises(ValueError, match="Marginal 0 loss must be non-negative"):
            marginals_model([("binary", 0.5, 2.0, -1.0)])
        with pytest.raises(
            ValueError, match=re.escape("must be a ('normal', mean, sd) tuple")
        ):
            marginals_model([("normal", 0.0)])
        with pytest.raises(
            ValueError, match=re.escape("must be a ('student_t', nu) or")
        ):
            marginals_model([("student_t", 5.0, 0.0)])
        with pytest.raises(
            ValueError, match=re.escape("must be a ('laplace', loc, scale) tuple")
        ):
            marginals_model([("laplace", 0.0)])
        with pytest.raises(
            ValueError, match=re.escape("must be a ('lognormal', mean, sd) tuple")
        ):
            marginals_model([("lognormal", 1.0)])
        with pytest.raises(
            ValueError,
            match=re.escape("must be a ('binary', probability, payoff, loss) tuple"),
        ):
            marginals_model([("binary", 0.5, 2.0)])


class TestFitMarginalsModel:
    """Fitting parametric marginals to a returns series."""

    def test_normal_fit_recovers_parameters(self):
        rng = np.random.default_rng(21)
        returns = rng.normal(0.01, 0.02, (200_000, 2))
        model = fit_marginals_model(returns, family="normal")
        assert isinstance(model, MarginalModel)
        assert [m.family for m in model.marginals] == ["normal", "normal"]
        mean, covariance = model.moments()
        assert np.allclose(mean, [0.01, 0.01], atol=1e-4)
        assert np.allclose(np.diag(covariance), [0.0004, 0.0004], atol=1e-6)

    def test_laplace_fit_moment_matches(self):
        # Moment-matching: the Laplace scale is sd / sqrt(2), so the fitted
        # model's exact variance equals the sample variance.
        rng = np.random.default_rng(22)
        returns = rng.laplace(0.0, 0.01, (100_000, 1))
        model = fit_marginals_model(returns, family="laplace")
        _, covariance = model.moments()
        assert covariance[0, 0] == pytest.approx(float(np.var(returns[:, 0])), rel=0.01)

    def test_lognormal_fit_on_positive_returns(self):
        rng = np.random.default_rng(23)
        returns = rng.lognormal(-0.05, 0.1, (100_000, 1))
        model = fit_marginals_model(returns, family="lognormal")
        assert model.marginals[0].family == "lognormal"
        mean, _ = model.moments()
        assert mean[0] == pytest.approx(float(returns[:, 0].mean()), rel=1e-6)

    def test_lognormal_fit_rejects_nonpositive_returns(self):
        with pytest.raises(
            ValueError, match="Lognormal marginals require strictly positive returns"
        ):
            fit_marginals_model([[0.1], [-0.1]], family="lognormal")

    def test_student_t_fit_recovers_nu(self):
        # Fit a t-distributed series, then refit a t to the fitted model's
        # own draws: the recovered shape round-trips near the truth.
        rng = np.random.default_rng(24)
        returns = rng.standard_t(6.0, (50_000, 1))
        model = fit_marginals_model(returns, family="student_t")
        assert [spec.family for spec in model.marginals] == ["student_t"]
        draws = model.sample(50_000, np.random.default_rng(2))
        fitted_nu, _, _ = scipy.stats.t.fit(draws[:, 0])
        assert fitted_nu == pytest.approx(6.0, rel=0.6)

    def test_unknown_family_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown family 'garch'"):
            fit_marginals_model([[0.01], [0.02]], family="garch")

    def test_validates_returns(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            fit_marginals_model([0.01, 0.02])


class TestEstimateMoments:
    """Monte Carlo estimation over the sampling contract."""

    def test_matches_exact_moments_for_a_known_model(self):
        model = binary_bets_model([(0.55, 2.0, 1.0)])
        mean, covariance = estimate_moments(model, 200_000, seed=3)
        exact_mean, exact_covariance = model.moments()
        assert np.allclose(mean, exact_mean, atol=0.01)
        assert np.allclose(covariance, exact_covariance, atol=0.02)

    def test_seeded_estimation_is_reproducible(self):
        model = scenario_model([[0.01, 0.02], [-0.01, 0.0]])
        first = estimate_moments(model, 256, seed=9)
        second = estimate_moments(model, 256, seed=9)
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])

    def test_unseeded_estimation_runs(self):
        model = _deterministic_sampler()
        mean, covariance = estimate_moments(model, 1000)
        assert mean.shape == (2,)
        assert covariance.shape == (2, 2)

    def test_rejects_too_few_samples(self):
        model = scenario_model([[0.01, 0.02]])
        with pytest.raises(
            ValueError, match="Estimating a covariance needs at least two samples"
        ):
            estimate_moments(model, 1)

    def test_rejects_invalid_seed(self):
        model = scenario_model([[0.01, 0.02]])
        with pytest.raises(ValueError, match="Seed must be a nonnegative integer"):
            estimate_moments(model, 10, seed=-1)

    def test_rejects_bad_sampler_output(self):
        class BadSampler:
            def sample(self, n_samples, rng):  # noqa: ARG002
                return np.ones((n_samples, 2))[:, 0]

        with pytest.raises(
            ValueError, match="Joint-return draws must be two-dimensional"
        ):
            estimate_moments(BadSampler(), 4)


class TestModelInputMixin:
    """The from_model constructor: exact moments win, draws fill the rest."""

    def test_prefers_exact_moments(self):
        # A binary model knows its moments exactly, so from_model needs no
        # Monte Carlo at all -- the constructed inputs are the closed form.
        model = binary_bets_model([(0.5, 2.0, 1.0), (0.25, 3.0, 1.0)])

        class Allocator(ModelInputMixin):
            def __init__(self, mean, covariance):
                self.mean = mean
                self.covariance = covariance

        allocator = Allocator.from_model(model)
        exact_mean, exact_covariance = model.moments()
        assert np.array_equal(allocator.mean, exact_mean)
        assert np.array_equal(allocator.covariance, exact_covariance)

    def test_falls_back_to_estimation(self):
        # A sampler without a moments() hook is estimated from draws.
        model = _deterministic_sampler(mean=0.02, sd=0.01)

        class Allocator(ModelInputMixin):
            def __init__(self, mean, covariance):
                self.mean = mean
                self.covariance = covariance

        allocator = Allocator.from_model(model, n_samples=100_000, seed=7)
        assert allocator.mean.shape == (2,)
        assert np.allclose(allocator.mean, [0.02, 0.02], atol=0.001)

    def test_kwargs_pass_through_to_the_allocator(self):
        model = binary_bets_model([(0.5, 2.0, 1.0)])

        class Allocator(ModelInputMixin):
            def __init__(self, mean, covariance, risk_aversion=1.0):
                self.mean = mean
                self.covariance = covariance
                self.risk_aversion = risk_aversion

        allocator = Allocator.from_model(model, risk_aversion=3.0)
        assert allocator.risk_aversion == 3.0

    def test_models_without_moments_use_the_fallback(self):
        # An object exposing only the sampling contract (no moments hook,
        # not even a JointReturnModel) still works: the escape hatch holds
        # all the way in.
        sampler = _deterministic_sampler()

        class NotAModel:
            def sample(self, n_samples, rng):
                return sampler.sample(n_samples, rng)

        class Allocator(ModelInputMixin):
            def __init__(self, mean, covariance):  # noqa: ARG002
                self.mean = mean

        allocator = Allocator.from_model(NotAModel(), n_samples=10, seed=1)
        assert allocator.mean.shape == (2,)

    def test_base_moments_default_is_none(self):
        # A JointReturnModel that does not override moments() reports None,
        # so from_model routes it to Monte Carlo estimation.
        class Bare(JointReturnModel):
            def sample(self, n_samples, rng):  # noqa: ARG002
                return np.zeros((n_samples, 1))

        assert Bare().moments() is None


class TestGatedImports:
    """The scipy-gated features point at keeks[allocation] when it is absent."""

    @pytest.fixture()
    def no_scipy(self, monkeypatch):
        """A fake ``builtins.__import__`` that fails for scipy modules."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("scipy"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    @pytest.mark.usefixtures("no_scipy")
    def test_marginal_model_points_at_the_extra(self):
        with pytest.raises(ImportError, match=r"keeks\[allocation\]"):
            MarginalModel(
                [("normal", 0.0, 1.0), ("normal", 0.0, 1.0)],
                dependence=[[1.0, 0.2], [0.2, 1.0]],
            )

    @pytest.mark.usefixtures("no_scipy")
    def test_fit_student_t_points_at_the_extra(self):
        with pytest.raises(ImportError, match=r"keeks\[allocation\]"):
            fit_marginals_model([[0.01], [0.02]], family="student_t")

    @pytest.mark.usefixtures("no_scipy")
    def test_independence_stays_numpy_only(self):
        # Without scipy, independent marginals still sample fine.
        model = marginals_model([("normal", 0.0, 1.0), ("binary", 0.5, 2.0, 1.0)])
        assert model.sample(5, np.random.default_rng(1)).shape == (5, 2)
