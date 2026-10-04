"""
Allocation simulator semantics: settlement, gating, and the trial flow.

``AllocationSimulator`` is the settlement heart of the allocation layer -
per staked trial it validates the weights before any draw is consumed,
draws one joint realization, settles the batch net through ``BankRoll``
(exactly one deposit or withdrawal per period), refuses-and-stops on a
``RuinError``, and reports the realized joint returns through the
``record_settlement`` hook. These tests pin that flow end to end with
scripted deterministic models: batch-net arithmetic, one-transaction-per-
period accounting, cash periods from residual probability mass, the
descriptor-equality gate, and every constructor gate.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from keeks import BankRoll
from keeks.allocation import AllocationSimulator, scenario_model
from keeks.utils import PROBABILITY_SUM_TOLERANCE

settings.register_profile("keeks", max_examples=50, deadline=None)
settings.load_profile("keeks")

# Four deterministic joint realizations over three options, small enough
# that no settlement in these tests threatens a 1000.0 bankroll.
MATRIX = [
    [0.02, -0.01, 0.015],
    [-0.005, 0.012, -0.008],
    [0.018, 0.004, 0.011],
    [-0.012, -0.006, 0.009],
]


class _ScriptedModel:
    """
    Deterministic opaque joint-return model: tiles the given rows.

    It carries no ``scenarios``/``bets``/``marginals`` descriptors and no
    ``moments()`` - exactly what a user's raw callable looks like to the
    simulator - and records every sampling call so tests can assert when
    draws happen.
    """

    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=float)
        self.sample_calls = []

    def sample(self, n_samples, _rng):
        self.sample_calls.append(n_samples)
        repeats = -(-n_samples // len(self.rows))
        return np.tile(self.rows, (repeats, 1))[:n_samples]


class _RecordingAllocator:
    """
    Fixed weights with recorded hooks.

    ``raw`` (when set) is returned unvalidated by ``evaluate`` - the door
    through which tests feed invalid weight vectors to the simulator's
    gates.
    """

    def __init__(self, weights=None, raw=None, scenarios=None):
        self.weights = tuple(weights) if weights is not None else None
        self.raw = raw
        self.scenarios = scenarios
        self.events = []
        self.settlements = []

    def evaluate(self, current_bankroll):
        self.events.append(("evaluate", current_bankroll))
        if self.raw is not None:
            return self.raw
        return self.weights

    def update_bankroll(self, total_funds):
        self.events.append(("hook", total_funds))

    def record_settlement(self, realized_returns):
        self.settlements.append(realized_returns)


class _StaticAllocator:
    """Fixed weights with no hooks at all - the plainest possible allocator."""

    def __init__(self, weights):
        self.weights = tuple(weights)

    def evaluate(self, _current_bankroll):
        return self.weights


class _RecordingBankRoll(BankRoll):
    """BankRoll that records each deposit and withdrawal amount."""

    def __init__(self, *args, **kwargs):
        self.transactions = []
        super().__init__(*args, **kwargs)

    def deposit(self, amount):
        self.transactions.append(("deposit", amount))
        super().deposit(amount)

    def withdraw(self, amount):
        self.transactions.append(("withdraw", amount))
        super().withdraw(amount)


def test_constructor_rejects_models_without_the_sampling_contract():
    with pytest.raises(ValueError, match="sampling contract"):
        AllocationSimulator([[0.01, 0.02]])


def test_constructor_rejects_invalid_probabilities():
    model = _ScriptedModel(MATRIX)
    with pytest.raises(ValueError, match="sum"):
        AllocationSimulator(model, probabilities=[0.6, 0.6])
    with pytest.raises(ValueError):
        AllocationSimulator(model, probabilities=[0.5, np.nan])
    with pytest.raises(ValueError):
        AllocationSimulator(model, probabilities=[-0.1, 0.2])


def test_constructor_rejects_non_finite_or_negative_transaction_costs():
    model = _ScriptedModel(MATRIX)
    with pytest.raises(ValueError, match="non-negative"):
        AllocationSimulator(model, transaction_costs=-0.01)
    with pytest.raises(ValueError):
        AllocationSimulator(model, transaction_costs=np.nan)


def test_constructor_rejects_non_integer_or_negative_trials():
    model = _ScriptedModel(MATRIX)
    with pytest.raises(ValueError, match="nonnegative integer"):
        AllocationSimulator(model, trials=3.5)
    with pytest.raises(ValueError, match="nonnegative integer"):
        AllocationSimulator(model, trials=-1)


def test_constructor_rejects_invalid_seeds():
    with pytest.raises(ValueError):
        AllocationSimulator(_ScriptedModel(MATRIX), seed=-1)


def test_one_transaction_per_settled_period():
    model = _ScriptedModel(MATRIX)
    bankroll = _RecordingBankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=4, seed=1)
    simulator.evaluate_strategy(_RecordingAllocator([0.5, 0.25, 0.25]), bankroll)
    assert len(bankroll.transactions) == 4
    assert len(bankroll.history) == 5


def test_settlement_amount_is_batch_net_minus_the_flat_fee():
    # Mirror the simulator's arithmetic exactly: stakes from the bettable
    # funds as they stood when the trial began, per-option amounts, one net.
    weights = (0.5, 0.25, 0.25)
    model = _ScriptedModel(MATRIX)
    bankroll = _RecordingBankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, transaction_costs=0.25, trials=1, seed=1)
    simulator.evaluate_strategy(_RecordingAllocator(weights), bankroll)
    amounts = [1000.0 * w * r for w, r in zip(weights, MATRIX[0], strict=True)]
    expected_net = sum(amounts) - 0.25
    assert len(bankroll.transactions) == 1
    method, amount = bankroll.transactions[0]
    assert method == ("deposit" if expected_net >= 0 else "withdraw")
    assert amount == abs(expected_net)


def test_losing_period_withdraws_the_net_loss():
    weights = (0.5, 0.25, 0.25)
    model = _ScriptedModel([[-0.01, 0.005, -0.02]])
    bankroll = _RecordingBankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=1, seed=2)
    simulator.evaluate_strategy(_RecordingAllocator(weights), bankroll)
    amounts = [
        1000.0 * w * r for w, r in zip(weights, (-0.01, 0.005, -0.02), strict=True)
    ]
    expected_net = sum(amounts)
    assert bankroll.transactions == [("withdraw", -expected_net)]


def test_invalid_weights_rejected_before_any_draw():
    model = _ScriptedModel(MATRIX)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=5, seed=3)
    allocation = _RecordingAllocator(raw=(0.6, 0.6, 0.0))
    with pytest.raises(ValueError, match="sum to no more than one"):
        simulator.evaluate_strategy(allocation, bankroll)
    assert bankroll.history == [1000.0]
    assert model.sample_calls == []


def test_wrong_weight_count_rejected_before_any_draw():
    # A scenario model states its option count without sampling, so the
    # length gate runs before the draw too.
    model = scenario_model(MATRIX)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=5, seed=3)
    with pytest.raises(ValueError, match="exactly 3 weights"):
        simulator.evaluate_strategy(_RecordingAllocator(raw=(0.5, 0.5)), bankroll)
    assert bankroll.history == [1000.0]


def test_opaque_model_length_gate_lands_at_the_first_settlement():
    # An opaque model reveals its option count only by sampling; the length
    # gate then fires right after the draw and before anything settles.
    model = _ScriptedModel(MATRIX)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=5, seed=3)
    allocation = _RecordingAllocator(raw=(0.5, 0.5))
    with pytest.raises(ValueError, match="exactly 3"):
        simulator.evaluate_strategy(allocation, bankroll)
    assert model.sample_calls == [5]
    assert bankroll.history == [1000.0]
    assert allocation.settlements == []


def test_zero_weight_trials_skip_draws_fees_and_settlements():
    model = _ScriptedModel(MATRIX)
    bankroll = _RecordingBankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, transaction_costs=0.5, trials=3, seed=2)
    allocation = _RecordingAllocator([0.0, 0.0, 0.0])
    simulator.evaluate_strategy(allocation, bankroll)
    assert model.sample_calls == []
    assert bankroll.transactions == []
    assert bankroll.history == [1000.0]
    assert allocation.settlements == []
    # The bankroll hook still fires on every trial.
    assert len([event for event in allocation.events if event[0] == "hook"]) == 3


def test_update_bankroll_fires_before_each_evaluation():
    model = _ScriptedModel(MATRIX)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=2, seed=4)
    allocation = _RecordingAllocator([0.5, 0.25, 0.25])
    simulator.evaluate_strategy(allocation, bankroll)
    assert allocation.events == [
        ("hook", 1000.0),
        ("evaluate", 1000.0),
        # Trial 1 sees the bankroll after trial 0's settlement: the hook is
        # the adaptive feedback path, so it reports the current total funds.
        ("hook", 1011.25),
        ("evaluate", 1011.25),
    ]


def test_record_settlement_receives_the_realized_vector_in_trial_order():
    # No probabilities: the model's draws are the trial sequence - row t
    # settles trial t.
    model = _ScriptedModel(MATRIX[:2])
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=3, seed=4)
    allocation = _RecordingAllocator([0.5, 0.25, 0.25])
    simulator.evaluate_strategy(allocation, bankroll)
    assert allocation.settlements == [
        (0.02, -0.01, 0.015),
        (-0.005, 0.012, -0.008),
        (0.02, -0.01, 0.015),
    ]


def test_static_allocators_run_without_hooks():
    model = _ScriptedModel(MATRIX)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=2, seed=5)
    simulator.evaluate_strategy(_StaticAllocator([0.2, 0.2, 0.2]), bankroll)
    assert len(bankroll.history) == 3


def test_residual_probability_mass_is_an_all_cash_period():
    # probabilities=[0.25] leaves 0.75 of every trial's mass in cash: some
    # trials stake the single state, some are all-cash periods.
    model = _ScriptedModel(MATRIX[:1])
    bankroll = _RecordingBankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(
        model, probabilities=[0.25], transaction_costs=0.5, trials=40, seed=11
    )
    allocation = _RecordingAllocator([0.5, 0.25, 0.25])
    simulator.evaluate_strategy(allocation, bankroll)
    stake_settlement = (0.02, -0.01, 0.015)
    cash_settlement = (0.0, 0.0, 0.0)
    assert set(allocation.settlements) == {stake_settlement, cash_settlement}
    assert cash_settlement in allocation.settlements
    # Cash periods cost nothing: one transaction per staked settlement only.
    staked = sum(1 for s in allocation.settlements if s != cash_settlement)
    assert len(bankroll.transactions) == staked


def test_drawdown_refusal_refuses_then_stops():
    model = _ScriptedModel([[-0.03, -0.03, -0.03]])
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=0.01)
    simulator = AllocationSimulator(model, trials=5, seed=5)
    allocation = _RecordingAllocator([0.4, 0.3, 0.3])
    simulator.evaluate_strategy(allocation, bankroll)
    # The refused period left the bankroll unchanged and the simulation
    # stopped after it - but the market's realization was still reported.
    assert bankroll.history == [1000.0]
    assert len([e for e in allocation.events if e[0] == "hook"]) == 1
    assert len(allocation.settlements) == 1
    assert allocation.settlements[0] == (-0.03, -0.03, -0.03)


def test_bankruptcy_stops_the_simulation():
    model = _ScriptedModel([[-1.0, -1.0, -1.0]])
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=5, seed=6)
    allocation = _RecordingAllocator([0.4, 0.3, 0.3])
    simulator.evaluate_strategy(allocation, bankroll)
    assert bankroll.history[-1] == 0.0
    # The next trial's depletion check broke the loop.
    assert len([e for e in allocation.events if e[0] == "hook"]) == 1


def test_scenario_bound_allocator_gate_runs_before_any_draw():
    model = scenario_model(MATRIX)
    other = [[r[0], r[1], r[2]] for r in MATRIX]
    other[0] = [0.5, -0.5, 0.1]
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=2, seed=7)
    with pytest.raises(ValueError, match="do not match the"):
        simulator.evaluate_strategy(
            _RecordingAllocator([0.5, 0.25, 0.25], scenarios=other), bankroll
        )
    assert bankroll.history == [1000.0]


def test_scenario_gate_fires_even_with_zero_trials():
    # The gate precedes the loop: a mismatch is a construction-level
    # incompatibility, not a per-trial accident.
    model = scenario_model(MATRIX)
    other = [[0.5, -0.5, 0.1]] + MATRIX[1:]
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=0, seed=7)
    with pytest.raises(ValueError, match="do not match the"):
        simulator.evaluate_strategy(
            _RecordingAllocator([0.5, 0.25, 0.25], scenarios=other), bankroll
        )


def test_scenario_gate_passes_matched_and_unbound_allocators():
    model = scenario_model(MATRIX)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=1, seed=7)
    matched = _RecordingAllocator([0.5, 0.25, 0.25], scenarios=MATRIX)
    simulator.evaluate_strategy(matched, bankroll)
    assert len(matched.settlements) == 1
    unbound = _StaticAllocator([0.5, 0.25, 0.25])
    simulator.evaluate_strategy(unbound, bankroll)
    assert len(bankroll.history) == 3


def test_zero_trials_settle_nothing():
    model = _ScriptedModel(MATRIX)
    bankroll = _RecordingBankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=0, seed=7)
    allocation = _RecordingAllocator([0.5, 0.25, 0.25])
    simulator.evaluate_strategy(allocation, bankroll)
    assert bankroll.transactions == []
    assert allocation.settlements == []
    assert model.sample_calls == []


def test_depleted_bankroll_breaks_before_the_first_hook():
    model = _ScriptedModel(MATRIX)
    bankroll = BankRoll(initial_funds=0.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=3, seed=7)
    allocation = _RecordingAllocator([0.5, 0.25, 0.25])
    simulator.evaluate_strategy(allocation, bankroll)
    assert allocation.events == []
    assert model.sample_calls == []


class _WrongWidthModel:
    """Claims two options through moments(), draws three columns."""

    def moments(self):
        return np.array([0.0, 0.0]), np.eye(2)

    def sample(self, n_samples, _rng):
        return np.zeros((n_samples, 3))


def test_model_draw_width_mismatch_is_a_contract_violation():
    # Weights matching the probe (2) but draws carrying 3 columns: the
    # draw-time reconciliation catches the inconsistent model.
    model = _WrongWidthModel()
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=2, seed=8)
    with pytest.raises(ValueError, match="option count is fixed"):
        simulator.evaluate_strategy(_RecordingAllocator(raw=(0.5, 0.5)), bankroll)
    assert bankroll.history == [1000.0]


class _MalformedMomentsModel:
    """moments() returns a scalar mean - not a (mean, covariance) pair."""

    def moments(self):
        return np.float64(0.5), np.eye(1)

    def sample(self, n_samples, _rng):
        return np.zeros((n_samples, 1))


def test_model_with_malformed_moments_is_rejected_at_construction():
    with pytest.raises(ValueError, match="one-dimensional mean"):
        AllocationSimulator(_MalformedMomentsModel())


def test_unseeded_runs_use_a_fresh_generator():
    model = _ScriptedModel(MATRIX)
    simulator = AllocationSimulator(model, trials=3)
    assert simulator.seed is None
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(_StaticAllocator([0.2, 0.2, 0.2]), bankroll)
    assert model.sample_calls == [3]
    assert len(bankroll.history) == 4


def test_probability_validation_uses_the_shared_tolerance():
    # A sum exactly at the tolerance boundary is a valid probability vector.
    model = _ScriptedModel(MATRIX[:1])
    simulator = AllocationSimulator(
        model, probabilities=[0.5, 0.5 + PROBABILITY_SUM_TOLERANCE], trials=1, seed=1
    )
    assert simulator.probabilities is not None


@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=30),
    st.integers(min_value=0, max_value=2**32 - 1),
)
def test_random_runs_preserve_the_bankroll_and_hook_contract(
    option_count, trials, seed
):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.02, size=(12, option_count))
    weights = (rng.dirichlet(np.ones(option_count)) * 0.8).tolist()
    model = scenario_model(matrix)
    allocation = _RecordingAllocator(weights)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(model, trials=trials, seed=seed)
    simulator.evaluate_strategy(allocation, bankroll)
    assert bankroll.history[0] == 1000.0
    assert all(np.isfinite(bankroll.history))
    assert all(fund >= 0 for fund in bankroll.history)
    assert len(allocation.settlements) <= trials
    assert all(
        len(settlement) == option_count and all(np.isfinite(settlement))
        for settlement in allocation.settlements
    )
    # Exactly-once settlement accounting: every settled period writes one
    # history entry, so the history never outgrows the settlements.
    assert len(bankroll.history) - 1 <= len(allocation.settlements)


@given(
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=0, max_value=2**32 - 1),
)
def test_random_cash_weights_keep_the_residual_mass_in_cash(option_count, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.02, size=(6, option_count))
    model = scenario_model(matrix)
    allocation = _RecordingAllocator([1.0 / option_count] * option_count)
    bankroll = _RecordingBankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator = AllocationSimulator(
        model, probabilities=[0.1] * option_count, trials=25, seed=seed
    )
    simulator.evaluate_strategy(allocation, bankroll)
    # Every settlement is either the single staked row or the zero vector,
    # and only staked settlements touch the bankroll.
    rows = {tuple(row) for row in matrix}
    cash = (0.0,) * option_count
    assert all(
        settlement in rows or settlement == cash
        for settlement in allocation.settlements
    )
    staked = sum(1 for settlement in allocation.settlements if settlement != cash)
    assert len(bankroll.transactions) == staked
