"""
Golden, property, and integration tests for the multi-outcome Kelly criterion.

The golden contract pinned by the phase plan: at two legs, with equivalent
binary inputs, :class:`MultiOutcomeKellyCriterion` returns exactly what the
existing :class:`keeks.binary_strategies.KellyCriterion` returns per leg.
"""

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from keeks.bankroll import BankRoll
from keeks.binary_strategies import KellyCriterion
from keeks.multi_outcome import (
    MultiOutcomeKellyCriterion,
    RepeatedMultiOutcomeSimulator,
)
from keeks.multi_outcome.kelly import (
    _delegation_is_joint_optimum,
    _multipliers_within_floor,
    _optimal_pairwise_transfer,
    _slice_slope,
    _solve_log_growth_allocation,
)
from keeks.utils import PROBABILITY_SUM_TOLERANCE

settings.register_profile("keeks", max_examples=50, deadline=None)
settings.load_profile("keeks")

# A two-leg market whose single priced leg is a plain binary Kelly bet: the
# delegation must reproduce the binary strategy bit for bit.
GOLDEN_CASES = [
    ((3.0, 1.5), [0.5, 0.5], 1.0, 0.0, 0.5, 1000.0),
    ((2.5, 1.2), [0.6, 0.4], 1.0, 0.0, 0.5, 1000.0),
    ((3.0, 1.5), [0.6, 0.4], 1.0, 0.01, 0.5, 500.0),
    ((10.0, 1.1), [0.45, 0.55], 1.0, 0.0, 0.3, 100.0),
]


def _expected_binary_legs(
    payoffs, probabilities, loss, transaction_cost, min_probability, bankroll
):
    """The per-leg binary Kelly stakes the two-leg fallback must reproduce."""
    legs = []
    for payoff, probability in zip(payoffs, probabilities, strict=True):
        if payoff > 1.0:
            scorer = KellyCriterion(
                payoff=payoff - 1.0,
                loss=loss,
                transaction_cost=transaction_cost,
                min_probability=min_probability,
            )
            legs.append(scorer.evaluate(probability, bankroll))
        else:
            legs.append(0.0)
    return tuple(legs)


def _objective(stakes, probabilities, payoffs, loss, transaction_cost):
    """
    The expected log growth the strategy maximizes.

    Multiplier hits zero on a realized outcome is unconditional ruin, so the
    objective is minus infinity there; a zero-probability leg contributes
    nothing regardless of its multiplier.
    """
    total = float(sum(stakes))
    value = 0.0
    for stake, probability, payoff in zip(stakes, probabilities, payoffs, strict=True):
        if probability == 0.0:
            continue
        multiplier = (
            1.0
            + (payoff - 1.0 - transaction_cost) * stake
            - (loss + transaction_cost) * (total - stake)
        )
        if multiplier <= 0.0:
            return -math.inf
        value += probability * math.log(multiplier)
    return value


# ---------------------------------------------------------------------------
# Golden: the two-leg fallback is exact binary Kelly.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    (
        "payoffs",
        "probabilities",
        "loss",
        "transaction_cost",
        "min_probability",
        "bankroll",
    ),
    GOLDEN_CASES,
)
def test_two_leg_matches_binary_kelly_exactly(
    payoffs, probabilities, loss, transaction_cost, min_probability, bankroll
):
    strategy = MultiOutcomeKellyCriterion(
        payoffs=payoffs,
        loss=loss,
        transaction_cost=transaction_cost,
        min_probability=min_probability,
    )
    assert strategy.evaluate(probabilities, bankroll) == _expected_binary_legs(
        payoffs, probabilities, loss, transaction_cost, min_probability, bankroll
    )


def test_two_leg_golden_pinned_fractions():
    """Pin the literal floats so drift in either strategy cannot hide."""
    strategy = MultiOutcomeKellyCriterion(payoffs=(3.0, 1.5), loss=1.0)
    assert strategy.evaluate([0.5, 0.5], 1000.0) == (0.25, 0.0)
    tc_strategy = MultiOutcomeKellyCriterion(
        payoffs=(3.0, 1.5), loss=1.0, transaction_cost=0.01
    )
    assert tc_strategy.evaluate([0.6, 0.4], 500.0) == (0.3930543808149659, 0.0)


def test_two_leg_coupled_edges_solve_jointly():
    """
    When both legs' net gains multiply past the loss charge squared, the
    delegated binary point is not the joint optimum: the coupled marginal
    value of the other leg is positive there, and the growth-optimal book
    stakes BOTH legs. Payoffs (2.0, 2.5) at (0.45, 0.55) with loss 1: the
    cap-face optimum is the probability vector itself, worth ~3x the binary
    fallback's expected log growth (0.128 vs 0.046).
    """
    strategy = MultiOutcomeKellyCriterion(payoffs=(2.0, 2.5), loss=1.0)
    stakes = strategy.evaluate([0.45, 0.55], 1000.0)
    assert stakes == pytest.approx((0.45, 0.55), abs=1e-9)
    fallback = _expected_binary_legs((2.0, 2.5), [0.45, 0.55], 1.0, 0.0, 0.5, 1000.0)
    assert _objective(stakes, [0.45, 0.55], (2.0, 2.5), 1.0, 0.0) > _objective(
        fallback, [0.45, 0.55], (2.0, 2.5), 1.0, 0.0
    )


def test_two_leg_low_loss_cap_binds_jointly():
    """
    With loss 0.5 the aggregate cap (1.0) binds below the binary Kelly
    stake (0.959), and the coupled edge pulls the second leg in: the joint
    optimum balances the realized multipliers on the cap face instead of
    all-inning one leg.
    """
    strategy = MultiOutcomeKellyCriterion(payoffs=(4.2, 1.3), loss=0.5)
    stakes = strategy.evaluate([0.55, 0.45], 1.0)
    assert stakes == pytest.approx((0.8329391892, 0.1670608108), abs=1e-8)
    fallback = _expected_binary_legs((4.2, 1.3), [0.55, 0.45], 0.5, 0.0, 0.5, 1.0)
    assert _objective(stakes, [0.55, 0.45], (4.2, 1.3), 0.5, 0.0) > _objective(
        fallback, [0.55, 0.45], (4.2, 1.3), 0.5, 0.0
    )


# ---------------------------------------------------------------------------
# Boundary branches of the numeric helpers.
# ---------------------------------------------------------------------------
def test_slice_slope_at_zero_own_multiplier_is_positive_infinity():
    """A zero own multiplier at t = 0 means the slice grows without bound."""
    slope = _slice_slope(0.0, 0.5, 1.0, 0.0, np.array([]), np.array([]), 0.0)
    assert slope == math.inf


def test_pairwise_transfer_backs_off_the_donor_log_boundary():
    """
    When the bisection lands the donor's multiplier on its log-domain
    boundary, the returned transfer backs off by whole ulps so the recomputed
    multiplier stays strictly positive.
    """
    delta = _optimal_pairwise_transfer(
        prob_to=0.5,
        spread_to=1.0,
        multiplier_to=1.0,
        prob_from=1e-300,
        spread_from=1.0,
        multiplier_from=1.0,
        available=1.0,
    )
    assert delta > 0.0
    assert 1.0 - 1.0 * delta > 0.0


def test_solver_lands_on_the_multiplier_floor_frontier():
    """
    A huge-edge book with a subnormal-probability no-edge leg wants to stake
    the aggregate cap, but the no-edge leg's realized multiplier (1 - cost *
    F) is floor-capped: the solve stops at the frontier instead of ruining
    the tiny leg on realization.
    """
    stakes = _solve_log_growth_allocation(
        net_gains=[99.0, 0.0, 99.0],
        cost=1.0,
        probabilities=[0.3, 1e-13, 0.7],
        max_total=1.0,
    )
    assert stakes == pytest.approx((0.3, 0.0, 0.7), abs=1e-9)
    total = sum(stakes)
    assert 1.0 - total >= 1e-12


def test_delegation_gate_rejects_zero_multiplier_candidates():
    """
    A delegated candidate that zeros a positive-probability leg's realized
    multiplier is infeasible: the gate answers False instead of dividing by
    zero on its gradient. (The strategy checks the multiplier floor first,
    but the gate must stay safe standing alone.)
    """
    assert (
        _delegation_is_joint_optimum([0.0, 1.0], [1.0, 1.0], 1.0, [0.5, 0.5], 1.0)
        is False
    )


def test_two_leg_both_legs_priced_positive_solves_jointly():
    """
    Both legs pricing a positive binary stake is an arbitrage book the
    per-leg fallback cannot size: the joint solve beats it at its own game.
    """
    strategy = MultiOutcomeKellyCriterion(payoffs=(3.0, 3.0), loss=1.0)
    stakes = strategy.evaluate([0.5, 0.5], 1000.0)
    assert stakes == (0.5, 0.5)
    objective = _objective(stakes, [0.5, 0.5], (3.0, 3.0), 1.0, 0.0)
    fallback = _expected_binary_legs((3.0, 3.0), [0.5, 0.5], 1.0, 0.0, 0.5, 1000.0)
    assert objective > _objective(fallback, [0.5, 0.5], (3.0, 3.0), 1.0, 0.0)


# ---------------------------------------------------------------------------
# Golden: the general N-leg solver against hand-derived optima.
# ---------------------------------------------------------------------------
def test_three_leg_single_priced_leg_matches_closed_form():
    """
    One priced leg and two no-edge legs carrying mass: the optimum is the
    1-D root of p0*w0/(1 + w0*f) = cost*(p1 + p2)/(1 - cost*f), and the
    no-edge legs keep their multipliers strictly positive (staking to the
    aggregate cap would zero them out and ruin the bankroll on realization).
    """
    strategy = MultiOutcomeKellyCriterion(payoffs=(11.0, 1.2, 1.3), loss=2.0)
    stakes = strategy.evaluate([0.9, 0.05, 0.05], 1000.0)
    # Solve the root directly instead of trusting a rearranged formula.
    lo, hi = 0.0, 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.9 * 10.0 / (1.0 + 10.0 * mid) > 2.0 * 0.1 / (1.0 - 2.0 * mid):
            lo = mid
        else:
            hi = mid
    expected = 0.5 * (lo + hi)
    assert stakes[0] == pytest.approx(expected, abs=1e-9)
    assert stakes[1] == 0.0
    assert stakes[2] == 0.0
    multipliers = [
        1.0 + (payoff - 1.0) * stake - 2.0 * (sum(stakes) - stake)
        for payoff, stake in zip((11.0, 1.2, 1.3), stakes, strict=True)
    ]
    assert min(multipliers) > 0.0


def test_three_leg_research_market_matches_kkt_point():
    """
    The phase-plan market (void mass 0.03): two legs price positive and the
    KKT conditions pin the unique optimum - equal marginal growth across the
    staked legs, nonpositive marginal on the benched leg.
    """
    payoffs = (3.2, 3.4, 2.4)
    probabilities = [0.42, 0.27, 0.28]
    strategy = MultiOutcomeKellyCriterion(payoffs=payoffs, loss=1.0)
    stakes = strategy.evaluate(probabilities, 1000.0)
    assert stakes == pytest.approx(
        (0.20368050871952964, 0.0625301088736872, 0.0), abs=1e-12
    )
    multipliers = [
        1.0 + (payoff - 1.0) * stake - (sum(stakes) - stake)
        for payoff, stake in zip(payoffs, stakes, strict=True)
    ]
    marginal = [
        probability * (payoff - 1.0 + 1.0) / multiplier
        - sum(p / m for p, m in zip(probabilities, multipliers, strict=True))
        for probability, payoff, multiplier in zip(
            probabilities, payoffs, multipliers, strict=True
        )
    ]
    assert marginal[0] == pytest.approx(marginal[1], abs=1e-9)
    assert marginal[2] < 0.0


def test_three_leg_capped_face_splits_by_equal_marginals():
    """
    Two edge legs on a market whose optimum spends the whole aggregate cap:
    single-leg moves cannot trade stake, so the pairwise transfers must find
    the equal-marginal split f2 = (5/3) * f1.
    """
    strategy = MultiOutcomeKellyCriterion(payoffs=(2.0, 4.0, 3.0), loss=1.0)
    stakes = strategy.evaluate([0.0, 0.3, 0.5], 1000.0)
    assert stakes == (0.0, 0.375, 0.625)


def test_single_leg_market_stakes_the_cap():
    """One leg, no other outcome to lose to: the whole safe cap goes in."""
    strategy = MultiOutcomeKellyCriterion(payoffs=(3.0,), loss=1.0)
    assert strategy.evaluate([0.7], 1000.0) == (1.0,)


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------
def test_all_zero_edge_market_stakes_nothing():
    below = MultiOutcomeKellyCriterion(payoffs=(0.5, 0.9), loss=1.0)
    assert below.evaluate([0.4, 0.4], 1000.0) == (0.0, 0.0)
    at_cost = MultiOutcomeKellyCriterion(payoffs=(1.0, 1.0), loss=1.0)
    assert at_cost.evaluate([0.5, 0.5], 1000.0) == (0.0, 0.0)


def test_payoff_below_total_cost_never_staked():
    strategy = MultiOutcomeKellyCriterion(
        payoffs=(3.0, 1.02, 1.02), loss=1.0, transaction_cost=0.02
    )
    stakes = strategy.evaluate([0.5, 0.25, 0.25], 1000.0)
    assert stakes[1] == 0.0
    assert stakes[2] == 0.0
    assert stakes[0] > 0.0


def test_zero_probability_leg_gets_no_stake():
    # Void mass is refunded, never lost: with the only other leg at zero
    # probability the priced leg cannot lose, so all-in is the optimum and
    # the binary-style fraction would understake.
    two_leg = MultiOutcomeKellyCriterion(payoffs=(3.0, 2.0), loss=1.0)
    assert two_leg.evaluate([0.0, 0.9], 1000.0) == (0.0, 1.0)
    three_leg = MultiOutcomeKellyCriterion(payoffs=(2.0, 4.0, 3.0), loss=1.0)
    assert three_leg.evaluate([0.0, 0.3, 0.5], 1000.0) == (0.0, 0.375, 0.625)


@pytest.mark.parametrize("bankroll", [0.0, -5.0])
def test_nonpositive_bankroll_stakes_nothing(bankroll):
    strategy = MultiOutcomeKellyCriterion(payoffs=(3.0, 1.5), loss=1.0)
    assert strategy.evaluate([0.6, 0.4], bankroll) == (0.0, 0.0)


def test_probability_length_mismatch_raises():
    strategy = MultiOutcomeKellyCriterion(payoffs=(3.0, 1.5, 1.2), loss=1.0)
    with pytest.raises(ValueError, match="must match the number of payoffs"):
        strategy.evaluate([0.5, 0.5], 1000.0)


@pytest.mark.parametrize("min_probability", [-0.1, 1.5])
def test_min_probability_must_be_in_unit_interval(min_probability):
    with pytest.raises(ValueError, match="Minimum probability must be between 0 and 1"):
        MultiOutcomeKellyCriterion(
            payoffs=(3.0, 1.5), loss=1.0, min_probability=min_probability
        )


def test_constructor_rejects_invalid_payoffs():
    with pytest.raises(ValueError, match="Payoffs must be greater than 0"):
        MultiOutcomeKellyCriterion(payoffs=(3.0, 0.0), loss=1.0)


# ---------------------------------------------------------------------------
# Property tests.
# ---------------------------------------------------------------------------
@st.composite
def _markets(draw):
    legs = draw(st.integers(min_value=1, max_value=6))
    weights = draw(
        st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=legs, max_size=legs)
    )
    total = sum(weights)
    if total <= 0:
        weights, total = [1.0] * legs, float(legs)
    scale = min(1.0, total)
    probabilities = [weight / total * scale for weight in weights]
    payoffs = draw(
        st.lists(st.floats(1.01, 50.0, allow_nan=False), min_size=legs, max_size=legs)
    )
    loss = draw(st.floats(0.0, 2.0, allow_nan=False))
    transaction_cost = draw(st.floats(0.0, 0.05, allow_nan=False))
    assume(loss + transaction_cost > 0.0)
    bankroll = draw(st.floats(1.0, 1e6, allow_nan=False))
    return payoffs, probabilities, loss, transaction_cost, bankroll


@given(market=_markets())
def test_stakes_satisfy_the_vector_contract(market):
    payoffs, probabilities, loss, transaction_cost, bankroll = market
    strategy = MultiOutcomeKellyCriterion(
        payoffs=payoffs, loss=loss, transaction_cost=transaction_cost
    )
    stakes = strategy.evaluate(probabilities, bankroll)
    assert strategy.evaluate(probabilities, bankroll) == stakes  # deterministic
    assert len(stakes) == len(probabilities)
    for stake in stakes:
        assert math.isfinite(stake)
        assert 0.0 <= stake <= 1.0
    assert sum(stakes) <= 1.0 + PROBABILITY_SUM_TOLERANCE


@given(market=_markets())
def test_no_positive_probability_leg_risks_ruin(market):
    """Every multiplier a realized outcome can produce must stay positive."""
    payoffs, probabilities, loss, transaction_cost, bankroll = market
    strategy = MultiOutcomeKellyCriterion(
        payoffs=payoffs, loss=loss, transaction_cost=transaction_cost
    )
    stakes = strategy.evaluate(probabilities, bankroll)
    total = sum(stakes)
    for probability, payoff, stake in zip(probabilities, payoffs, stakes, strict=True):
        if probability == 0.0:
            continue
        multiplier = (
            1.0
            + (payoff - 1.0 - transaction_cost) * stake
            - (loss + transaction_cost) * (total - stake)
        )
        assert multiplier > 0.0


@st.composite
def _markets_with_zero_edge_leg(draw):
    legs = draw(st.integers(min_value=2, max_value=6))
    weights = draw(
        st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=legs, max_size=legs)
    )
    total = sum(weights)
    if total <= 0:
        weights, total = [1.0] * legs, float(legs)
    scale = min(1.0, total)
    probabilities = [weight / total * scale for weight in weights]
    transaction_cost = draw(st.floats(0.0, 0.05, allow_nan=False))
    zero_edge_payoff = draw(st.floats(1.0, 1.0 + transaction_cost, allow_nan=False))
    payoffs = [zero_edge_payoff] + draw(
        st.lists(
            st.floats(1.01, 50.0, allow_nan=False), min_size=legs - 1, max_size=legs - 1
        )
    )
    loss = draw(st.floats(0.0, 2.0, allow_nan=False))
    assume(loss + transaction_cost > 0.0)
    bankroll = draw(st.floats(1.0, 1e6, allow_nan=False))
    return payoffs, probabilities, loss, transaction_cost, bankroll


@given(market=_markets_with_zero_edge_leg())
def test_zero_edge_leg_never_staked(market):
    payoffs, probabilities, loss, transaction_cost, bankroll = market
    strategy = MultiOutcomeKellyCriterion(
        payoffs=payoffs, loss=loss, transaction_cost=transaction_cost
    )
    stakes = strategy.evaluate(probabilities, bankroll)
    assert stakes[0] == 0.0


@st.composite
def _markets_with_zero_probability_leg(draw):
    legs = draw(st.integers(min_value=2, max_value=6))
    weights = [0.0] + draw(
        st.lists(
            st.floats(0.01, 1.0, allow_nan=False), min_size=legs - 1, max_size=legs - 1
        )
    )
    total = sum(weights)
    scale = min(1.0, total)
    probabilities = [weight / total * scale for weight in weights]
    payoffs = draw(
        st.lists(st.floats(1.01, 50.0, allow_nan=False), min_size=legs, max_size=legs)
    )
    loss = draw(st.floats(0.0, 2.0, allow_nan=False))
    transaction_cost = draw(st.floats(0.0, 0.05, allow_nan=False))
    assume(loss + transaction_cost > 0.0)
    bankroll = draw(st.floats(1.0, 1e6, allow_nan=False))
    return payoffs, probabilities, loss, transaction_cost, bankroll


@given(market=_markets_with_zero_probability_leg())
def test_zero_probability_leg_never_staked(market):
    payoffs, probabilities, loss, transaction_cost, bankroll = market
    strategy = MultiOutcomeKellyCriterion(
        payoffs=payoffs, loss=loss, transaction_cost=transaction_cost
    )
    stakes = strategy.evaluate(probabilities, bankroll)
    assert stakes[0] == 0.0


@st.composite
def _fully_priced_two_leg_markets(draw):
    """Two-leg books whose probabilities sum to one by construction."""
    p = draw(st.floats(0.0, 1.0, allow_nan=False))
    probabilities = [p, 1.0 - p]
    payoffs = draw(
        st.lists(st.floats(1.01, 50.0, allow_nan=False), min_size=2, max_size=2)
    )
    loss = draw(st.floats(0.0, 2.0, allow_nan=False))
    transaction_cost = draw(st.floats(0.0, 0.05, allow_nan=False))
    assume(loss + transaction_cost > 0.0)
    bankroll = draw(st.floats(1.0, 1e6, allow_nan=False))
    return payoffs, probabilities, loss, transaction_cost, bankroll


@given(market=_fully_priced_two_leg_markets())
def test_two_leg_falls_back_to_binary_kelly(market):
    """For fully priced books with one priced leg, output is exact binary Kelly."""
    payoffs, probabilities, loss, transaction_cost, bankroll = market
    min_probability = 0.5
    strategy = MultiOutcomeKellyCriterion(
        payoffs=payoffs, loss=loss, transaction_cost=transaction_cost
    )
    stakes = strategy.evaluate(probabilities, bankroll)
    expected = _expected_binary_legs(
        payoffs, probabilities, loss, transaction_cost, min_probability, bankroll
    )
    # Mirror the delegation's full gate: at most one positive binary stake,
    # every relevant multiplier above the log-domain floor, and no other leg
    # with an improving direction at that point.
    net_gains = [payoff - 1.0 - transaction_cost for payoff in payoffs]
    total_cost = loss + transaction_cost
    assume(sum(stake > 0.0 for stake in expected) <= 1)
    assume(
        _multipliers_within_floor(list(expected), net_gains, total_cost, probabilities)
        is not None
    )
    assume(
        _delegation_is_joint_optimum(
            list(expected),
            net_gains,
            total_cost,
            probabilities,
            min(1.0, 1.0 / total_cost),
        )
    )
    assert stakes == expected


def test_two_leg_void_mass_routes_to_the_joint_solver():
    """
    With void mass, binary Kelly would count the refund branch as a loss:
    probabilities (0.0, 0.5) on equal 2.0 payoffs is a fair bet per leg, but
    the zero-probability leg can never take the stake, so the joint optimum
    is all-in on the priced leg. The delegation must not fire here.
    """
    strategy = MultiOutcomeKellyCriterion(payoffs=(2.0, 2.0), loss=1.0)
    assert strategy.evaluate([0.0, 0.5], 1000.0) == (0.0, 1.0)


@given(market=_markets())
def test_objective_beats_naive_allocations(market):
    """The optimizer must not lose to zero, all-on-one-leg, or equal splits."""
    payoffs, probabilities, loss, transaction_cost, bankroll = market
    strategy = MultiOutcomeKellyCriterion(
        payoffs=payoffs, loss=loss, transaction_cost=transaction_cost
    )
    stakes = strategy.evaluate(probabilities, bankroll)
    best = _objective(stakes, probabilities, payoffs, loss, transaction_cost)
    cap = min(1.0, 1.0 / (loss + transaction_cost))
    candidates = [(0.0,) * len(payoffs)]
    candidates.extend(
        tuple(cap if leg == hero else 0.0 for leg in range(len(payoffs)))
        for hero in range(len(payoffs))
    )
    legs = len(payoffs)
    candidates.append(tuple(cap / legs for _ in range(legs)))
    for candidate in candidates:
        assert (
            best
            >= _objective(candidate, probabilities, payoffs, loss, transaction_cost)
            - 1e-12
        )


# ---------------------------------------------------------------------------
# Integration with the P3 simulator.
# ---------------------------------------------------------------------------
def test_simulator_integration():
    simulator = RepeatedMultiOutcomeSimulator(
        payoffs=(3.2, 3.4, 2.4),
        loss=1.0,
        transaction_costs=0.0,
        probabilities=[0.42, 0.27, 0.28],
        trials=100,
        seed=42,
    )
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(
        MultiOutcomeKellyCriterion(payoffs=(3.2, 3.4, 2.4), loss=1.0), bankroll
    )
    assert bankroll.total_funds > 0.0
