"""Property-based tests for the pure numerical core (Hypothesis).

The six families follow the coverage-baseline research (art_XLYfJoRd §5.5):
validator round-trips, CRRA utility laws, gamble normalization laws,
indifference-price contracts, cross-strategy invariants, and seeded-simulator
determinism. Everything here is hermetic: simulators are always seeded so the
process-global RNG is never touched.

One baseline suggestion is deliberately absent: CRRA "continuity as γ→1".
``crra_utility`` implements ``w^(1-γ) / (1-γ)`` without the ``-1`` term of the
standard CRRA form, so the power branch diverges as γ→1 by construction; the
γ==1 log branch is pinned instead.
"""

import contextlib
import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from keeks.bankroll import BankRoll
from keeks.binary_strategies import (
    CPPIStrategy,
    DrawdownAdjustedKelly,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    FractionalKellyCriterion,
    KellyCriterion,
    MertonShare,
    NaiveStrategy,
    OptimalF,
)
from keeks.simulators import (
    RandomBinarySimulator,
    RandomUncertainBinarySimulator,
    RepeatedBinarySimulator,
)
from keeks.utils import (
    PROBABILITY_SUM_TOLERANCE,
    RuinError,
    _normalize_gamble,
    _require_finite,
    _validate_probability,
    _validate_simulator_probability,
    _validate_simulator_seed,
    _validate_simulator_stdev,
    _validate_stake_fraction,
    _validate_strategy_odds,
    crra_utility,
    expected_utility,
    find_indifference_price,
)

settings.register_profile("keeks", max_examples=50, deadline=None)
settings.load_profile("keeks")

finite_floats = st.floats(allow_nan=False, allow_infinity=False)
positive_floats = st.floats(min_value=0.01, max_value=1e6, allow_nan=False)
probabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

NAMES = ("Payoff", "Loss", "Bankroll", "Wealth")


# ---------------------------------------------------------------------------
# Family 1: validator round-trips — valid values pass through unchanged,
# invalid ones raise the documented message without partial mutation.
# ---------------------------------------------------------------------------
@given(name=st.sampled_from(NAMES), value=finite_floats)
def test_require_finite_round_trips_finite_values(name, value):
    assert _require_finite(value, name) == float(value)


def _unparseable_as_float(text):
    try:
        float(text)
    except ValueError:
        return True
    return False


@given(
    name=st.sampled_from(NAMES),
    value=st.one_of(
        st.none(),
        st.text().filter(_unparseable_as_float),
        st.sampled_from([float("nan"), float("inf"), float("-inf")]),
    ),
)
def test_require_finite_rejects_non_finite_and_non_numeric(name, value):
    with pytest.raises(ValueError, match=rf"^{name} must be a finite number$"):
        _require_finite(value, name)


@given(name=st.sampled_from(NAMES), digits=st.integers(0, 10**6))
def test_require_finite_coerces_numeric_strings(name, digits):
    """Numeric strings are coerced, matching the documented float() behavior."""
    assert _require_finite(str(digits), name) == float(digits)


@given(value=probabilities)
def test_validate_probability_round_trips_unit_interval(value):
    assert _validate_probability(value) == value


@given(
    value=st.one_of(
        st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0000000001, allow_nan=False, allow_infinity=False),
    )
)
def test_validate_probability_rejects_outside_unit_interval(value):
    with pytest.raises(ValueError, match=r"^Probability must be between 0 and 1$"):
        _validate_probability(value)


@given(value=probabilities)
def test_validate_stake_fraction_round_trips_unit_interval(value):
    assert _validate_stake_fraction(value) == value


@given(
    value=st.one_of(
        st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0000000001, allow_nan=False, allow_infinity=False),
    )
)
def test_validate_stake_fraction_rejects_outside_unit_interval(value):
    with pytest.raises(
        ValueError, match=r"^Strategy stake fraction must be between 0 and 1$"
    ):
        _validate_stake_fraction(value)


@given(value=probabilities)
def test_simulator_probability_validator_round_trips(value):
    assert _validate_simulator_probability(value, "Probability") == value


@given(value=st.floats(min_value=0.0, max_value=1e6, allow_nan=False))
def test_simulator_stdev_validator_round_trips_nonnegative(value):
    assert _validate_simulator_stdev(value, "Standard deviation") == value


@given(value=st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False))
def test_simulator_stdev_validator_rejects_negative(value):
    with pytest.raises(ValueError, match=r"^Standard deviation must be non-negative$"):
        _validate_simulator_stdev(value, "Standard deviation")


@given(seed=st.integers(min_value=0, max_value=2**31))
def test_simulator_seed_validator_round_trips_nonnegative_integers(seed):
    assert _validate_simulator_seed(seed) == seed


@given(
    seed=st.one_of(
        st.integers(max_value=-1),
        st.floats(min_value=0, max_value=100, allow_nan=False),
    )
)
def test_simulator_seed_validator_rejects_negative_and_non_integer(seed):
    with pytest.raises(
        ValueError, match=r"^Seed must be a nonnegative integer or None$"
    ):
        _validate_simulator_seed(seed)


# ---------------------------------------------------------------------------
# Family 2: CRRA utility laws — scalar/array equivalence, the -inf arms,
# monotonicity in wealth, and the γ==1 log identity.
# ---------------------------------------------------------------------------
@given(wealth=positive_floats, risk_aversion=st.floats(0.1, 5.0, allow_nan=False))
def test_crra_scalar_equals_one_element_array(wealth, risk_aversion):
    scalar = crra_utility(wealth, risk_aversion)
    array = crra_utility(np.array([wealth]), risk_aversion)

    assert array.shape == (1,)
    assert array[0] == pytest.approx(scalar, rel=1e-12)


@given(
    wealth=st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False),
    risk_aversion=st.floats(0.1, 5.0, allow_nan=False),
)
def test_crra_nonpositive_wealth_is_negative_infinity_everywhere(wealth, risk_aversion):
    assert crra_utility(wealth, risk_aversion) == -math.inf

    masked = crra_utility(np.array([wealth, 1.0]), risk_aversion)

    assert masked[0] == -math.inf
    assert math.isfinite(masked[1])


@given(
    small=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
    growth=st.floats(min_value=1.01, max_value=10.0, allow_nan=False),
    risk_aversion=st.one_of(
        st.floats(0.1, 0.9, allow_nan=False), st.floats(1.1, 5.0, allow_nan=False)
    ),
)
def test_crra_is_monotonically_increasing_in_wealth(small, growth, risk_aversion):
    wealth = small * growth

    assert crra_utility(wealth, risk_aversion) > crra_utility(small, risk_aversion)


@given(wealth=positive_floats)
def test_crra_log_identity_at_gamma_one(wealth):
    assert crra_utility(wealth, 1.0) == pytest.approx(math.log(wealth), rel=1e-12)


# ---------------------------------------------------------------------------
# Family 3: gamble normalization laws — the implicit zero-payout leg, the
# tolerance window, rejection bounds, and permutation invariance of EU.
# ---------------------------------------------------------------------------
@st.composite
def unit_gambles(draw):
    """A valid (outcomes, probabilities) pair whose mass never exceeds one."""
    n = draw(st.integers(min_value=1, max_value=5))
    weights = draw(
        st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=n, max_size=n)
    )
    outcomes = draw(
        st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=n, max_size=n)
    )
    total = sum(weights)
    if total <= 0:
        weights = [1.0] * n
        total = float(n)
    scale = min(1.0, total)
    probabilities = [w / total * scale for w in weights]

    return outcomes, probabilities


@given(gamble=unit_gambles())
def test_normalize_gamble_output_sums_to_one(gamble):
    outcomes, probabilities = gamble
    norm_outcomes, norm_probabilities = _normalize_gamble(outcomes, probabilities)

    assert norm_probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    assert len(norm_outcomes) == len(norm_probabilities)


@given(gamble=unit_gambles())
def test_normalize_gamble_deficit_becomes_single_zero_payout_leg(gamble):
    """A mass deficit appends exactly one zero-payout outcome carrying it."""
    outcomes, probabilities = gamble
    norm_outcomes, norm_probabilities = _normalize_gamble(outcomes, probabilities)

    if len(norm_outcomes) == len(outcomes):
        assert list(norm_outcomes) == outcomes
    else:
        assert len(norm_outcomes) == len(outcomes) + 1
        assert norm_outcomes[-1] == 0.0
        deficit = 1.0 - sum(probabilities)
        assert norm_probabilities[-1] == pytest.approx(deficit, abs=1e-12)
        assert norm_probabilities[-1] >= 0.0


@given(gamble=unit_gambles(), excess=st.floats(1e-11, 1.0, allow_nan=False))
def test_normalize_gamble_rejects_mass_above_tolerance_window(gamble, excess):
    outcomes, probabilities = gamble
    complete = probabilities + [max(0.0, 1.0 - sum(probabilities))]
    scaled = [p * (1.0 + excess) for p in complete]
    assert sum(scaled) > 1 + PROBABILITY_SUM_TOLERANCE

    with pytest.raises(
        ValueError, match=r"^Probabilities must sum to no more than one$"
    ):
        _normalize_gamble(outcomes + [0.0], scaled)


@given(gamble=unit_gambles())
def test_normalize_gamble_accepts_inside_tolerance_window(gamble):
    """A sum inside 1 + PROBABILITY_SUM_TOLERANCE is accepted and clamped."""
    outcomes, probabilities = gamble
    scaled = [p * (1.0 + PROBABILITY_SUM_TOLERANCE / 2) for p in probabilities]

    norm_outcomes, norm_probabilities = _normalize_gamble(outcomes, scaled)

    assert norm_probabilities.sum() == pytest.approx(1.0, abs=1e-12)


@given(gamble=unit_gambles())
def test_normalize_gamble_rejects_negative_legs(gamble):
    outcomes, probabilities = gamble

    with pytest.raises(ValueError, match=r"^Probabilities must be nonnegative$"):
        _normalize_gamble(outcomes, [-0.1] + probabilities[1:])


@given(
    gamble=unit_gambles(),
    data=st.data(),
    risk_aversion=st.floats(0.5, 3.0, allow_nan=False),
)
def test_expected_utility_is_invariant_under_joint_permutation(
    gamble, data, risk_aversion
):
    outcomes, probabilities = gamble
    n = len(outcomes)
    order = data.draw(st.permutations(range(n)))
    permuted_outcomes = [outcomes[i] for i in order]
    permuted_probabilities = [probabilities[i] for i in order]

    base = expected_utility(outcomes, probabilities, 1000.0, 0.0, risk_aversion)
    permuted = expected_utility(
        permuted_outcomes, permuted_probabilities, 1000.0, 0.0, risk_aversion
    )

    assert base == pytest.approx(permuted, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# Family 4: indifference-price contracts — bracket, indifference accuracy,
# saturation iff the bound is worth paying, and zero willingness.
# ---------------------------------------------------------------------------
@st.composite
def priced_gambles(draw):
    """A gamble, wealth, and gamma where the indifference price is interior."""
    outcomes, probabilities = draw(unit_gambles())
    wealth = draw(st.floats(10.0, 10_000.0, allow_nan=False))
    risk_aversion = draw(st.floats(0.5, 3.0, allow_nan=False))
    tolerance = draw(st.floats(1e-4, 0.01, allow_nan=False))

    return outcomes, probabilities, wealth, risk_aversion, tolerance


@given(priced=priced_gambles())
def test_indifference_price_stays_inside_search_bracket(priced):
    outcomes, probabilities, wealth, risk_aversion, tolerance = priced

    price = find_indifference_price(
        outcomes, probabilities, wealth, risk_aversion, tolerance=tolerance
    )

    assert 0.0 <= price <= wealth * 0.5 + tolerance


@given(priced=priced_gambles())
def test_indifference_price_matches_utility_indifference_point(priced):
    """For interior prices |EU(price) - U(wealth)| <= a small multiple of tol."""
    outcomes, probabilities, wealth, risk_aversion, tolerance = priced
    bound = wealth * 0.5

    worth_zero = expected_utility(
        outcomes, probabilities, wealth, 0.0, risk_aversion
    ) > crra_utility(wealth, risk_aversion)
    below_bound = expected_utility(
        outcomes, probabilities, wealth, bound, risk_aversion
    ) <= crra_utility(wealth, risk_aversion)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        price = find_indifference_price(
            outcomes, probabilities, wealth, risk_aversion, tolerance=tolerance
        )

    if not (worth_zero and below_bound):
        return

    # The solved price is the largest payment that still beats sitting out:
    # EU(price) must land on U(wealth) within a small multiple of tol.
    gamble_utility = expected_utility(
        outcomes, probabilities, wealth, price, risk_aversion
    )
    outside_utility = crra_utility(wealth, risk_aversion)

    assert gamble_utility == pytest.approx(outside_utility, abs=50 * tolerance)


@given(priced=priced_gambles())
def test_indifference_price_saturates_iff_bound_is_worth_paying(priced):
    """A bound worth paying saturates with a warning; worthlessness pins 0."""
    outcomes, probabilities, wealth, risk_aversion, tolerance = priced
    bound = wealth * 0.5

    # The search saturates when the gamble is still worth buying at the
    # bound, i.e. EU(bound) beats not participating at all (U(wealth)).
    gamble_utility_at_bound = expected_utility(
        outcomes, probabilities, wealth, bound, risk_aversion
    )
    outside_utility = crra_utility(wealth, risk_aversion)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        price = find_indifference_price(
            outcomes, probabilities, wealth, risk_aversion, tolerance=tolerance
        )

    saturated = any("saturated" in str(w.message) for w in caught)

    if gamble_utility_at_bound > outside_utility:
        assert saturated
        assert price == pytest.approx(bound, abs=tolerance)
    elif expected_utility(
        outcomes, probabilities, wealth, 0.0, risk_aversion
    ) <= crra_utility(wealth, risk_aversion):
        # Not worth paying anything: the search converges toward the low end.
        assert price <= 2 * tolerance


@given(priced=priced_gambles())
def test_indifference_price_default_tolerance_is_accurate(priced):
    """The default tolerance=0.01 search lands on the participation point."""
    outcomes, probabilities, wealth, risk_aversion, _ = priced
    bound = wealth * 0.5

    worth_zero = expected_utility(
        outcomes, probabilities, wealth, 0.0, risk_aversion
    ) > crra_utility(wealth, risk_aversion)
    below_bound = expected_utility(
        outcomes, probabilities, wealth, bound, risk_aversion
    ) <= crra_utility(wealth, risk_aversion)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        price = find_indifference_price(outcomes, probabilities, wealth, risk_aversion)

    if not (worth_zero and below_bound):
        return

    gamble_utility = expected_utility(
        outcomes, probabilities, wealth, price, risk_aversion
    )
    outside_utility = crra_utility(wealth, risk_aversion)

    assert gamble_utility == pytest.approx(outside_utility, abs=0.5)


# ---------------------------------------------------------------------------
# Family 5: cross-strategy invariants over all nine shipped strategies.
# ---------------------------------------------------------------------------
def build_strategy(strategy_cls, draw):
    """Construct any shipped strategy from drawn economic controls."""
    payoff = draw(st.floats(0.5, 5.0, allow_nan=False))
    loss = draw(st.floats(0.5, 5.0, allow_nan=False))
    transaction_cost = draw(st.floats(0.0, 0.05, allow_nan=False))
    common = {"payoff": payoff, "loss": loss, "transaction_cost": transaction_cost}

    if strategy_cls is KellyCriterion:
        return KellyCriterion(**common)
    if strategy_cls is FractionalKellyCriterion:
        return FractionalKellyCriterion(fraction=draw(probabilities), **common)
    if strategy_cls is DrawdownAdjustedKelly:
        return DrawdownAdjustedKelly(
            max_acceptable_drawdown=draw(st.floats(0.01, 0.99, allow_nan=False)),
            **common,
        )
    if strategy_cls is NaiveStrategy:
        return NaiveStrategy(**common)
    if strategy_cls is FixedFractionStrategy:
        return FixedFractionStrategy(fraction=draw(probabilities), **common)
    if strategy_cls is CPPIStrategy:
        return CPPIStrategy(
            floor_fraction=draw(st.floats(0.05, 0.9, allow_nan=False)),
            multiplier=draw(st.floats(1.0, 8.0, allow_nan=False)),
            initial_bankroll=1000.0,
            **common,
        )
    if strategy_cls is DynamicBankrollManagement:
        base_fraction = draw(st.floats(0.01, 0.5, allow_nan=False))
        return DynamicBankrollManagement(
            base_fraction=base_fraction,
            min_fraction=0.0,
            max_fraction=1.0,
            **common,
        )
    if strategy_cls is OptimalF:
        return OptimalF(win_rate=draw(st.floats(0.01, 0.99, allow_nan=False)), **common)
    return MertonShare(
        risk_aversion=draw(st.floats(0.5, 5.0, allow_nan=False)),
        min_probability=draw(probabilities),
        max_fraction=draw(st.floats(0.001, 1.0, allow_nan=False)),
        **common,
    )


STRATEGY_CLASSES = (
    KellyCriterion,
    FractionalKellyCriterion,
    DrawdownAdjustedKelly,
    NaiveStrategy,
    FixedFractionStrategy,
    CPPIStrategy,
    DynamicBankrollManagement,
    OptimalF,
    MertonShare,
)


@given(
    strategy_cls=st.sampled_from(STRATEGY_CLASSES),
    data=st.data(),
    probability=probabilities,
    bankroll=st.floats(0.01, 1e5, allow_nan=False),
)
def test_all_strategies_stay_within_max_safe_bet(
    strategy_cls, data, probability, bankroll
):
    """0 <= evaluate(p, b) <= get_max_safe_bet(b) for every valid input."""
    strategy = build_strategy(strategy_cls, data.draw)

    proportion = strategy.evaluate(probability, bankroll)

    assert 0.0 <= proportion <= strategy.get_max_safe_bet(bankroll)


@given(
    strategy_cls=st.sampled_from(STRATEGY_CLASSES),
    data=st.data(),
    bankroll=st.floats(-1e4, 0.0, allow_nan=False),
)
def test_all_strategies_give_zero_max_safe_bet_on_empty_bankroll(
    strategy_cls, data, bankroll
):
    strategy = build_strategy(strategy_cls, data.draw)

    assert strategy.get_max_safe_bet(bankroll) == 0.0


@given(
    strategy_cls=st.sampled_from(STRATEGY_CLASSES),
    data=st.data(),
    probability=st.floats(0.0, 0.49, allow_nan=False),
    bankroll=st.floats(1.0, 1e5, allow_nan=False),
)
def test_strategies_do_not_bet_below_their_probability_gate(
    strategy_cls, data, probability, bankroll
):
    """Strategies exposing min_probability refuse to size below it."""
    strategy = build_strategy(strategy_cls, data.draw)
    minimum = getattr(strategy, "min_probability", None)

    if minimum is None or probability >= minimum:
        return

    assert strategy.evaluate(probability, bankroll) == 0.0


@given(
    fraction=st.floats(0.01, 1.0, allow_nan=False),
    data=st.data(),
)
def test_fractional_kelly_entry_price_scales_kelly_price(fraction, data):
    """FractionalKelly's price is exactly fraction x Kelly's price."""
    outcomes, probabilities = data.draw(unit_gambles())
    wealth = data.draw(st.floats(10.0, 10_000.0, allow_nan=False))

    kelly = KellyCriterion(payoff=2.0, loss=1.0, transaction_cost=0.0)
    fractional = FractionalKellyCriterion(
        payoff=2.0, loss=1.0, transaction_cost=0.0, fraction=fraction
    )
    tolerance = 1e-4

    kelly_price = kelly.calculate_max_entry_price(
        outcomes, probabilities, wealth, tolerance=tolerance
    )
    fractional_price = fractional.calculate_max_entry_price(
        outcomes, probabilities, wealth, tolerance=tolerance
    )

    assert fractional_price == pytest.approx(fraction * kelly_price, abs=1e-6)


@given(
    drawdown=st.floats(0.01, 0.99, allow_nan=False),
    data=st.data(),
)
def test_drawdown_adjusted_kelly_entry_price_scales_kelly_price(drawdown, data):
    """DrawdownAdjusted's price is factor x Kelly's price, factor = mdd/0.5."""
    outcomes, probabilities = data.draw(unit_gambles())
    wealth = data.draw(st.floats(10.0, 10_000.0, allow_nan=False))
    factor = min(1.0, drawdown / 0.5)

    kelly = KellyCriterion(payoff=2.0, loss=1.0, transaction_cost=0.0)
    adjusted = DrawdownAdjustedKelly(
        payoff=2.0, loss=1.0, transaction_cost=0.0, max_acceptable_drawdown=drawdown
    )
    tolerance = 1e-4

    kelly_price = kelly.calculate_max_entry_price(
        outcomes, probabilities, wealth, tolerance=tolerance
    )
    adjusted_price = adjusted.calculate_max_entry_price(
        outcomes, probabilities, wealth, tolerance=tolerance
    )

    assert adjusted_price == pytest.approx(factor * kelly_price, abs=1e-6)


# ---------------------------------------------------------------------------
# Family 6: seeded-simulator determinism and settlement invariants over
# arbitrary seeds and controls. Simulators are always seeded.
# ---------------------------------------------------------------------------
SIMULATOR_CONFIGS = st.fixed_dictionaries(
    {
        "payoff": st.floats(0.5, 3.0, allow_nan=False),
        "loss": st.floats(0.5, 2.0, allow_nan=False),
        "transaction_costs": st.floats(0.0, 0.05, allow_nan=False),
        "trials": st.integers(1, 8),
        "seed": st.integers(0, 2**31),
    }
)


@given(config=SIMULATOR_CONFIGS, probability=probabilities)
def test_repeated_simulator_is_deterministic_over_arbitrary_seeds(config, probability):
    def run():
        simulator = RepeatedBinarySimulator(probability=probability, **config)
        bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
        simulator.evaluate_strategy(
            FixedFractionStrategy(
                payoff=config["payoff"], loss=config["loss"], fraction=0.02
            ),
            bankroll,
        )
        return bankroll.history

    assert run() == run()


@given(config=SIMULATOR_CONFIGS, stdev=st.floats(0.0, 0.2, allow_nan=False))
def test_random_simulator_is_deterministic_over_arbitrary_seeds(config, stdev):
    def run():
        simulator = RandomBinarySimulator(stdev=stdev, **config)
        bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
        simulator.evaluate_strategy(
            FixedFractionStrategy(
                payoff=config["payoff"], loss=config["loss"], fraction=0.02
            ),
            bankroll,
        )
        return bankroll.history

    assert run() == run()


@given(config=SIMULATOR_CONFIGS, stdev=st.floats(0.0, 0.2, allow_nan=False))
def test_uncertain_simulator_is_deterministic_over_arbitrary_seeds(config, stdev):
    def run():
        simulator = RandomUncertainBinarySimulator(stdev=stdev, **config)
        bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
        simulator.evaluate_strategy(
            FixedFractionStrategy(
                payoff=config["payoff"], loss=config["loss"], fraction=0.02
            ),
            bankroll,
        )
        return bankroll.history

    assert run() == run()


class OverbettingStrategy:
    """Duck-typed strategy that always requests an invalid 2x stake."""

    payoff = 2.0
    loss = 1.0

    def evaluate(self, _probability, _current_bankroll):
        return 2.0


@given(config=SIMULATOR_CONFIGS, probability=probabilities)
def test_invalid_stake_fraction_never_mutates_bankroll(config, probability):
    simulator = RepeatedBinarySimulator(probability=probability, **config)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    with pytest.raises(ValueError, match=r"^Strategy stake fraction must be"):
        simulator.evaluate_strategy(OverbettingStrategy(), bankroll)

    assert bankroll.total_funds == 1000.0
    assert bankroll.history == [1000.0]


@given(config=SIMULATOR_CONFIGS, probability=probabilities)
def test_settlement_never_leaves_funds_negative(config, probability):
    """Every settled run keeps funds >= 0 and records every trial."""
    simulator = RepeatedBinarySimulator(probability=probability, **config)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    with contextlib.suppress(RuinError):
        simulator.evaluate_strategy(
            FixedFractionStrategy(
                payoff=config["payoff"], loss=config["loss"], fraction=0.02
            ),
            bankroll,
        )

    assert bankroll.total_funds >= 0.0
    assert len(bankroll.history) >= 1


@given(
    payoff=st.floats(0.5, 3.0, allow_nan=False),
    delta=st.floats(0.5, 2.0, allow_nan=False),
)
def test_odds_mismatch_rejects_concrete_strategy(payoff, delta):
    """A concrete strategy must never be settled at odds it did not size for."""
    strategy = KellyCriterion(payoff=payoff, loss=1.0, transaction_cost=0.0)

    with pytest.raises(ValueError, match=r"^Strategy payoff \("):
        _validate_strategy_odds(strategy, payoff + delta, 1.0)
