"""
Binary-bets bridge: the allocation simulator settles keeks bets exactly.

A book of keeks binary bets enters the allocation layer through
``binary_bets_model`` - the adapter turns (probability, payoff, loss)
triples into per-period joint simple returns using the settlement
arithmetic and per-bet BLAKE2b-keyed streams the multi-outcome
simulator already defines. Under matched seeding and matched weights,
``AllocationSimulator`` must therefore reproduce ``PortfolioSimulator``
outcome streams bit-for-bit: identical bankroll histories and identical
won/lost sequences. This is the proof that keeks' existing binary bets
genuinely enter the allocation context rather than a parallel universe
with its own arithmetic.

The two sides are seeded through the model's documented consumption chain:
the binary-bets adapter keys each bet's stream on a child entropy drawn
from the simulator's single seeded stream, so ``PortfolioSimulator`` is
seeded with exactly that child entropy - the same derivation the
model-level bridge test in ``test_allocation_models.py`` uses.
"""

import numpy as np
import pytest

from keeks import BankRoll
from keeks.allocation import AllocationSimulator, binary_bets_model
from keeks.multi_outcome import PortfolioSimulator
from keeks.multi_outcome.base import _validate_stake_fractions

BETS = [(0.55, 2.0, 1.0), (0.45, 3.0, 1.0), (0.30, 2.4, 1.0)]
FRACTIONS = (0.4, 0.35, 0.25)


class _RecordingWeights:
    """Fixed weights that record each realized joint-return vector."""

    def __init__(self, weights):
        self.weights = tuple(weights)
        self.realized = []

    def evaluate(self, _current_bankroll):
        return self.weights

    def record_settlement(self, won, realized_returns):
        del won  # hooks must accept the outcome flags; only returns matter
        self.realized.append(realized_returns)


class _SameFractions:
    """Duck-typed strategy staking fixed fractions of the bettable funds."""

    def __init__(self, fractions):
        self.fractions = _validate_stake_fractions(tuple(fractions))
        self.won_flags = []

    def evaluate(self, _probabilities, _current_bankroll):
        return self.fractions

    def update_bankroll(self, _current_bankroll):
        pass

    def record_settlement(self, won, _realized_returns):
        self.won_flags.append(won)


def run_portfolio_simulation(seed, trials=40):
    simulator = PortfolioSimulator(bets=BETS, fee_per_bet=0.0, trials=trials, seed=seed)
    bankroll = BankRoll(initial_funds=1000.0, max_transaction_loss=None)
    strategy = _SameFractions(FRACTIONS)
    simulator.evaluate_strategy(strategy, bankroll)
    return bankroll.history, strategy.won_flags


def portfolio_seed_matching(seed):
    # The binary-bets model derives its per-bet stream entropy from the
    # first 8 bytes of the simulator's single seeded stream; seeding
    # PortfolioSimulator with that exact value keys both sides' per-bet
    # streams identically (same BLAKE2b spawn-key family).
    parent = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    return int.from_bytes(parent.bytes(8), "big")


def run_allocation_simulation(seed, trials=40):
    model = binary_bets_model(BETS)
    simulator = AllocationSimulator(model, fee_per_bet=0.0, trials=trials, seed=seed)
    bankroll = BankRoll(initial_funds=1000.0, max_transaction_loss=None)
    strategy = _RecordingWeights(FRACTIONS)
    simulator.evaluate_strategy(strategy, bankroll)
    return bankroll.history, strategy.realized


@pytest.mark.parametrize("seed", [0, 1, 42, 20260803])
def test_binary_bets_replay_portfolio_streams_under_matched_seeding(seed):
    ps_history, ps_won = run_portfolio_simulation(portfolio_seed_matching(seed))
    allocation_history, realized = run_allocation_simulation(seed)
    assert allocation_history == ps_history
    assert len(realized) == len(ps_won)
    for joint_returns, won_flags in zip(realized, ps_won, strict=True):
        # A binary bet's simple return is +payoff on a win and -loss on a
        # loss, so the sign of the realized return is the won flag.
        assert tuple(r > 0 for r in joint_returns) == tuple(bool(w) for w in won_flags)


def test_bridge_history_is_bit_identical_not_approximate():
    # Guard against a regression to approximate equality: the float bits of
    # every history entry must match exactly.
    ps_history, _ps_won = run_portfolio_simulation(portfolio_seed_matching(42))
    allocation_history, _realized = run_allocation_simulation(seed=42)
    assert [v.hex() for v in allocation_history] == [v.hex() for v in ps_history]
