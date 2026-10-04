"""Estimator tests: shrinkage endpoints, Ledoit-Wolf intensity, BL identities.

The spec verification row for the estimators: ``alpha = 0`` recovers the
sample covariance, ``alpha = 1`` hits the structured target, the
Black-Litterman posterior equals the prior with no views, the shrunk
covariance is positive semidefinite, and shapes are checked. Around that row
sit the intensity oracle (the closed-form Ledoit-Wolf ratio against a
brute-force sum of rank-one scatter norms), the confidence behavior of the
posterior (a confident view is satisfied, an uninformative one is ignored),
and the full validation matrix for both estimators.
"""

import numpy as np
import pytest

from keeks.allocation import black_litterman_mean, shrink_covariance
from keeks.allocation.estimators import (
    _validate_samples,
    _validate_shrinkage_intensity,
    _validate_views,
)

# Deterministic two-option covariance with a positive off-diagonal, shared
# by the Black-Litterman tests so their numbers stay comparable.
COVARIANCE = [[0.02, 0.004], [0.004, 0.03]]
MARKET_WEIGHTS = [0.6, 0.4]
RISK_AVERSION = 2.5
TAU = 1.0


def _centered(samples):
    """The samples with their column means removed, as a float array."""
    samples = np.asarray(samples, dtype=float)
    return samples - samples.mean(axis=0)


def _sample_covariance(samples):
    """The unbiased sample covariance, the estimator's alpha = 0 endpoint."""
    centered = _centered(samples)
    return centered.T @ centered / (centered.shape[0] - 1)


def _structured_target(samples):
    """The scaled identity the estimator shrinks toward."""
    covariance = _sample_covariance(samples)
    mean_variance = float(np.trace(covariance)) / covariance.shape[0]
    return mean_variance * np.eye(covariance.shape[0])


def _brute_force_intensity(samples):
    """The Ledoit-Wolf intensity straight from its defining sum.

    No closed-form shortcuts: the noise numerator is the plain sum of the
    squared Frobenius norms of the rank-one scatter matrices around the
    sample covariance, divided by the squared distance to the target. The
    closed form in ``_ledoit_wolf_intensity`` must agree with this.
    """
    centered = _centered(samples)
    n_samples, n_options = centered.shape
    covariance = _sample_covariance(samples)
    deviation = covariance - _structured_target(samples)
    distance_squared = float(np.sum(deviation**2)) / n_options
    if distance_squared <= 0.0:
        return 0.0
    noise_total = sum(
        float(np.sum((np.outer(row, row) - covariance) ** 2)) for row in centered
    )
    noise = noise_total / (n_samples**2 * n_options)
    return max(0.0, min(1.0, noise / distance_squared))


class TestShrinkageEndpoints:
    """The alpha = 0 and alpha = 1 endpoints are exact."""

    def test_alpha_zero_recovers_sample_covariance(self):
        samples = [[0.01, -0.01], [0.03, 0.01], [-0.02, 0.0], [0.0, 0.02]]
        shrunk, alpha = shrink_covariance(samples, alpha=0.0)
        assert alpha == 0.0
        assert np.array_equal(shrunk, _sample_covariance(samples))
        assert np.allclose(shrunk, np.cov(np.asarray(samples), rowvar=False))

    def test_alpha_one_hits_structured_target(self):
        samples = [[0.01, -0.01], [0.03, 0.01], [-0.02, 0.0], [0.0, 0.02]]
        shrunk, alpha = shrink_covariance(samples, alpha=1.0)
        assert alpha == 1.0
        assert np.array_equal(shrunk, _structured_target(samples))

    def test_explicit_alpha_blends_linearly(self):
        samples = [[0.01, -0.01], [0.03, 0.01], [-0.02, 0.0], [0.0, 0.02]]
        shrunk, alpha = shrink_covariance(samples, alpha=0.25)
        expected = 0.75 * _sample_covariance(samples) + 0.25 * _structured_target(
            samples
        )
        assert alpha == 0.25
        assert np.allclose(shrunk, expected)

    def test_estimate_is_array_like_input(self):
        shrunk, _ = shrink_covariance([[0.01, 0.02], [-0.01, 0.0]])
        assert shrunk.shape == (2, 2)


class TestLedoitWolfIntensity:
    """The estimated intensity matches its defining sum and stays in [0, 1]."""

    def test_matches_brute_force_oracle(self):
        for seed in (0, 1, 7, 42):
            rng = np.random.default_rng(seed)
            samples = rng.normal(scale=0.02, size=(40, 3))
            _, alpha = shrink_covariance(samples)
            assert alpha == pytest.approx(_brute_force_intensity(samples), abs=1e-12)

    def test_intensity_within_unit_interval(self):
        # Heteroscedastic options: the population covariance is far from the
        # scaled-identity target, so the estimated intensity stays interior.
        # (For spherical samples the target IS the population covariance and
        # Ledoit-Wolf correctly shrinks fully.)
        rng = np.random.default_rng(0)
        samples = rng.normal(scale=np.array([0.005, 0.02, 0.05]), size=(128, 3))
        _, alpha = shrink_covariance(samples)
        assert 0.0 < alpha < 1.0

    def test_intensity_caps_at_one(self):
        # Isotropic dispersion with one slightly fatter option: the noise
        # estimate dwarfs the distance to the target, so the intensity caps.
        samples = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.1], [0.0, -1.1]]
        shrunk, alpha = shrink_covariance(samples)
        assert alpha == 1.0
        assert np.array_equal(shrunk, _structured_target(samples))

    def test_single_option_has_zero_intensity(self):
        # With one option the sample covariance is its own target, so the
        # distance to the target is identically zero and nothing is shrunk.
        samples = [[0.01], [0.03], [-0.02]]
        shrunk, alpha = shrink_covariance(samples)
        assert alpha == 0.0
        assert np.array_equal(shrunk, _sample_covariance(samples))

    def test_identical_samples_have_zero_dispersion(self):
        samples = [[0.01, 0.02]] * 5
        shrunk, alpha = shrink_covariance(samples)
        assert alpha == 0.0
        assert np.array_equal(shrunk, np.zeros((2, 2)))

    def test_intensity_is_scale_invariant(self):
        rng = np.random.default_rng(3)
        samples = rng.normal(scale=0.02, size=(60, 3))
        _, plain = shrink_covariance(samples)
        _, scaled = shrink_covariance(samples * 10.0)
        assert plain == pytest.approx(scaled)


class TestShrunkCovarianceIsPSD:
    """Every shrunk output is positive semidefinite."""

    def test_all_modes_positive_semidefinite(self):
        rng = np.random.default_rng(11)
        samples = rng.normal(scale=0.02, size=(64, 3))
        for mode in (None, 0.0, 0.25, 1.0):
            shrunk, _ = shrink_covariance(samples, alpha=mode)
            eigenvalues = np.linalg.eigvalsh(shrunk)
            assert float(np.min(eigenvalues)) >= -1e-12


class TestShrinkageValidation:
    """Sample and intensity gates reject malformed inputs."""

    @pytest.mark.parametrize(
        "samples",
        [
            None,
            object(),
            [0.01, 0.02],
            np.empty((0, 2)),
            [[0.01, 0.02]],
            [[0.01, 0.02], [0.03, np.inf]],
            [[0.01, 0.02], [float("nan"), 0.0]],
            [[[0.01, 0.02], [0.03, 0.04]]],
        ],
    )
    def test_rejects_bad_samples(self, samples):
        with pytest.raises(ValueError):
            shrink_covariance(samples)

    @pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf"), object()])
    def test_rejects_bad_alpha(self, alpha):
        samples = [[0.01, -0.01], [0.03, 0.01], [-0.02, 0.0], [0.0, 0.02]]
        with pytest.raises(ValueError):
            shrink_covariance(samples, alpha=alpha)

    def test_intensity_validator_helpers(self):
        assert _validate_shrinkage_intensity(0.25) == 0.25
        with pytest.raises(ValueError):
            _validate_shrinkage_intensity("not a number")

    def test_samples_validator_helper(self):
        assert _validate_samples([[0.01, -0.01], [0.03, 0.01]]).shape == (2, 2)
        with pytest.raises(ValueError, match="two-dimensional"):
            _validate_samples([0.01, 0.02])


class TestBlackLittermanPrior:
    """With no views the posterior is exactly the equilibrium prior."""

    def test_no_views_returns_prior(self):
        prior = black_litterman_mean(
            COVARIANCE,
            MARKET_WEIGHTS,
            risk_aversion=RISK_AVERSION,
            views=None,
            omega=None,
            tau=TAU,
        )
        expected = RISK_AVERSION * (np.asarray(COVARIANCE) @ np.asarray(MARKET_WEIGHTS))
        assert prior.shape == (2,)
        assert np.array_equal(prior, expected)

    def test_empty_views_return_prior(self):
        views = (np.empty((0, 2)), np.empty(0))
        prior = black_litterman_mean(
            COVARIANCE,
            MARKET_WEIGHTS,
            risk_aversion=RISK_AVERSION,
            views=views,
            omega=np.empty((0, 0)),
            tau=TAU,
        )
        expected = RISK_AVERSION * (np.asarray(COVARIANCE) @ np.asarray(MARKET_WEIGHTS))
        assert np.array_equal(prior, expected)

    def test_partial_market_weights_are_accepted(self):
        # The long-only budget contract allows a cash remainder.
        posterior = black_litterman_mean(
            COVARIANCE,
            [0.3, 0.3],
            risk_aversion=RISK_AVERSION,
            views=None,
            omega=None,
            tau=TAU,
        )
        assert posterior.shape == (2,)


class TestBlackLittermanPosterior:
    """Views move the posterior the way the precision weighting says."""

    def test_matches_manual_posterior_computation(self):
        views = (np.array([[1.0, -1.0], [0.0, 1.0]]), np.array([0.01, 0.005]))
        omega = np.array([[0.001, 0.0], [0.0, 0.002]])
        posterior = black_litterman_mean(
            COVARIANCE,
            MARKET_WEIGHTS,
            risk_aversion=RISK_AVERSION,
            views=views,
            omega=omega,
            tau=TAU,
        )

        covariance = np.asarray(COVARIANCE, dtype=float)
        prior = RISK_AVERSION * (covariance @ np.asarray(MARKET_WEIGHTS))
        prior_precision = np.linalg.inv(TAU * covariance)
        omega_inverse = np.linalg.inv(omega)
        expected = np.linalg.solve(
            prior_precision + views[0].T @ omega_inverse @ views[0],
            prior_precision @ prior + views[0].T @ omega_inverse @ views[1],
        )
        assert np.allclose(posterior, expected)

    def test_confident_view_is_satisfied(self):
        # As the view uncertainty vanishes the posterior must satisfy the
        # view: P @ mu -> q.
        views = (np.array([[1.0, -1.0]]), np.array([0.01]))
        posterior = black_litterman_mean(
            COVARIANCE,
            MARKET_WEIGHTS,
            risk_aversion=RISK_AVERSION,
            views=views,
            omega=np.array([[1e-6]]),
            tau=TAU,
        )
        assert np.allclose(views[0] @ posterior, views[1], atol=1e-6)

    def test_uninformative_view_leaves_the_prior(self):
        # A huge view uncertainty contributes almost no precision, so the
        # posterior stays at the prior.
        views = (np.array([[1.0, -1.0]]), np.array([0.01]))
        prior = black_litterman_mean(
            COVARIANCE,
            MARKET_WEIGHTS,
            risk_aversion=RISK_AVERSION,
            views=None,
            omega=None,
            tau=TAU,
        )
        posterior = black_litterman_mean(
            COVARIANCE,
            MARKET_WEIGHTS,
            risk_aversion=RISK_AVERSION,
            views=views,
            omega=np.array([[1e8]]),
            tau=TAU,
        )
        assert np.allclose(posterior, prior, atol=1e-10)


class TestBlackLittermanValidation:
    """Every input gate rejects malformed inputs with a pointed error."""

    def test_views_without_omega_is_rejected(self):
        views = (np.array([[1.0, -1.0]]), np.array([0.01]))
        with pytest.raises(ValueError, match="together"):
            black_litterman_mean(
                COVARIANCE,
                MARKET_WEIGHTS,
                risk_aversion=RISK_AVERSION,
                views=views,
                omega=None,
                tau=TAU,
            )

    def test_omega_without_views_is_rejected(self):
        with pytest.raises(ValueError, match="together"):
            black_litterman_mean(
                COVARIANCE,
                MARKET_WEIGHTS,
                risk_aversion=RISK_AVERSION,
                views=None,
                omega=np.array([[0.001]]),
                tau=TAU,
            )

    def test_singular_covariance_is_rejected(self):
        # The prior-precision inverse only runs when views are supplied -
        # with no views the prior needs no inversion of the covariance.
        views = (np.array([[1.0, -1.0]]), np.array([0.01]))
        with pytest.raises(ValueError, match="invertible"):
            black_litterman_mean(
                [[1.0, 1.0], [1.0, 1.0]],
                MARKET_WEIGHTS,
                risk_aversion=RISK_AVERSION,
                views=views,
                omega=np.array([[0.001]]),
                tau=TAU,
            )

    def test_singular_omega_is_rejected(self):
        views = (np.array([[1.0, -1.0], [1.0, 1.0]]), np.array([0.01, 0.02]))
        with pytest.raises(ValueError, match="invertible"):
            black_litterman_mean(
                COVARIANCE,
                MARKET_WEIGHTS,
                risk_aversion=RISK_AVERSION,
                views=views,
                omega=[[1.0, 1.0], [1.0, 1.0]],
                tau=TAU,
            )

    def test_singular_posterior_precision_is_rejected(self, monkeypatch):
        def raise_singular(*_args, **_kwargs):
            raise np.linalg.LinAlgError("Singular matrix")

        monkeypatch.setattr(np.linalg, "solve", raise_singular)
        views = (np.array([[1.0, -1.0]]), np.array([0.01]))
        with pytest.raises(ValueError, match="posterior precision"):
            black_litterman_mean(
                COVARIANCE,
                MARKET_WEIGHTS,
                risk_aversion=RISK_AVERSION,
                views=views,
                omega=np.array([[0.001]]),
                tau=TAU,
            )

    @pytest.mark.parametrize(
        "market_weights",
        [[1.2, 0.0], [0.6, 0.6], [0.6], [[0.6, 0.4]], [-0.1, 0.5]],
    )
    def test_market_weights_follow_the_weight_contract(self, market_weights):
        with pytest.raises(ValueError):
            black_litterman_mean(
                COVARIANCE,
                market_weights,
                risk_aversion=RISK_AVERSION,
                views=None,
                omega=None,
                tau=TAU,
            )

    @pytest.mark.parametrize("risk_aversion", [0.0, -1.0, float("nan"), object()])
    def test_risk_aversion_must_be_positive_finite(self, risk_aversion):
        with pytest.raises(ValueError):
            black_litterman_mean(
                COVARIANCE,
                MARKET_WEIGHTS,
                risk_aversion=risk_aversion,
                views=None,
                omega=None,
                tau=TAU,
            )

    @pytest.mark.parametrize("tau", [0.0, -1.0, float("nan"), object()])
    def test_tau_must_be_positive_finite(self, tau):
        with pytest.raises(ValueError):
            black_litterman_mean(
                COVARIANCE,
                MARKET_WEIGHTS,
                risk_aversion=RISK_AVERSION,
                views=None,
                omega=None,
                tau=tau,
            )

    def test_indefinite_covariance_is_rejected(self):
        with pytest.raises(ValueError, match="semidefinite"):
            black_litterman_mean(
                [[0.01, 0.02], [0.02, 0.01]],
                MARKET_WEIGHTS,
                risk_aversion=RISK_AVERSION,
                views=None,
                omega=None,
                tau=TAU,
            )

    def test_indefinite_omega_is_rejected(self):
        views = (np.array([[1.0, -1.0], [1.0, 1.0]]), np.array([0.01, 0.02]))
        with pytest.raises(ValueError, match="semidefinite"):
            black_litterman_mean(
                COVARIANCE,
                MARKET_WEIGHTS,
                risk_aversion=RISK_AVERSION,
                views=views,
                omega=[[0.001, 0.01], [0.01, 0.001]],
                tau=TAU,
            )


class TestViewValidation:
    """The (P, q) view pair is shape- and finiteness-checked."""

    def test_validator_helper_round_trip(self):
        matrix, returns = _validate_views(([[1.0, -1.0]], [0.01]), 2)
        assert matrix.shape == (1, 2)
        assert returns.shape == (1,)

    @pytest.mark.parametrize(
        "views",
        [
            5,
            (1, 2, 3),
            (object(), [0.01]),
            ([[1.0, -1.0]], [object()]),
            ([1.0, -1.0], [0.01]),
            ([[float("nan"), -1.0]], [0.01]),
            ([[1.0, -1.0, 0.0]], [0.01]),
            ([[1.0, -1.0]], [[0.01]]),
            ([[1.0, -1.0]], [0.01, 0.02]),
            ([[1.0, -1.0]], [float("inf")]),
        ],
    )
    def test_rejects_bad_view_pairs(self, views):
        with pytest.raises(ValueError):
            _validate_views(views, 2)

    def test_view_matrix_must_match_option_count(self):
        with pytest.raises(ValueError, match="columns"):
            _validate_views(([[1.0, -1.0, 0.5]], [0.01]), 2)

    def test_empty_view_matrix_passes(self):
        matrix, returns = _validate_views((np.empty((0, 2)), np.empty(0)), 2)
        assert matrix.shape == (0, 2)
        assert returns.shape == (0,)
