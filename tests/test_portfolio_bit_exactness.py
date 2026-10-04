"""Bit-exactness oracle for the portfolio simulator.

Freezes the exact public outputs of the simulator — the float bits of
``BankRoll.history`` — for every strategy x scenario x seed x trial-count
combination. It is the portfolio analogue of
``tests/test_multi_outcome_bit_exactness.py`` and, like it, the acceptance
oracle for output-identical performance work on this class: any change that
alters a single settlement bit fails this module.

Regenerate on purpose only (then format)::

    uv run python tests/test_portfolio_bit_exactness.py
    ruff format tests/test_portfolio_bit_exactness.py

which rewrites the GOLDEN block between the markers below from the current
code.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from keeks.bankroll import BankRoll
from keeks.multi_outcome import PortfolioSimulator
from keeks.multi_outcome.base import BaseMultiOutcomeStrategy, _validate_stake_fractions

INITIAL_FUNDS = 1000.0

# Scenario axes: a standard three-bet portfolio with a small fee, a
# longshot-heavy portfolio (most bets lose; the stateful strategy's growth
# path saturates), and a fee-free portfolio under a tight drawdown limit so
# the RuinError refuse-full-batch-then-stop path triggers.
SCENARIOS = {
    "standard": {
        "bets": ((0.55, 2.0, 1.0), (0.45, 3.0, 1.0), (0.30, 2.4, 1.0)),
        "fee": 0.01,
        "max_transaction_loss": None,
    },
    "longshot-heavy": {
        "bets": ((0.08, 10.0, 1.0), (0.12, 8.0, 1.0), (0.06, 12.0, 1.0)),
        "fee": 0.0,
        "max_transaction_loss": None,
    },
    "drawdown-tight": {
        "bets": ((0.60, 1.5, 1.0), (0.50, 1.8, 1.0), (0.55, 1.6, 1.0)),
        "fee": 0.0,
        "max_transaction_loss": 0.05,
    },
}

# Mirrors the other oracles' discipline: stateful strategies must be rebuilt
# per run because they carry state across evaluate() calls. The losing-batch
# strategy exercises the update_bankroll and record_settlement hooks and the
# aggregate safe-stake cap.
STRATEGY_FACTORIES = {
    "Flat stakes": lambda s: _FlatStakesStrategy(
        payoffs=_payoffs(s), loss=_loss(s), fraction=0.02, transaction_cost_rate=0.0
    ),
    "Probability weighted": lambda s: _ProbabilityWeightedStrategy(
        payoffs=_payoffs(s), loss=_loss(s), aggregate=0.9, transaction_cost_rate=0.0
    ),
    "Losing-batch tracking": lambda s: _LosingBatchTrackingStrategy(
        payoffs=_payoffs(s), loss=_loss(s), base_total=0.3, transaction_cost_rate=0.0
    ),
}

# Grid of seeds and trial counts. Short runs catch early stops and the
# no-bet path; the 400-trial runs compound.
SEEDS = (0, 1, 42, 20260803)
TRIAL_COUNTS = (7, 53, 400)


def _payoffs(scenario):
    return tuple(bet[1] for bet in scenario["bets"])


def _loss(scenario):
    losses = {bet[2] for bet in scenario["bets"]}
    assert len(losses) == 1
    return losses.pop()


class _FlatStakesStrategy(BaseMultiOutcomeStrategy):
    """The same fraction on every bet, every trial."""

    def __init__(self, payoffs, loss, fraction, transaction_cost_rate=0):
        super().__init__(payoffs, loss, transaction_cost_rate)
        self._fraction = fraction

    def evaluate(self, probabilities, _current_bankroll):
        return _validate_stake_fractions([self._fraction] * len(probabilities))


class _ProbabilityWeightedStrategy(BaseMultiOutcomeStrategy):
    """Stakes proportional to each bet's probability under a fixed aggregate."""

    def __init__(self, payoffs, loss, aggregate, transaction_cost_rate=0):
        super().__init__(payoffs, loss, transaction_cost_rate)
        self._aggregate = aggregate

    def evaluate(self, probabilities, _current_bankroll):
        probabilities = np.asarray(probabilities, dtype=float)
        weights = probabilities / probabilities.sum()
        return _validate_stake_fractions((self._aggregate * weights).tolist())


class _LosingBatchTrackingStrategy(BaseMultiOutcomeStrategy):
    """Grows the aggregate stake after all-losing batches, under the cap."""

    def __init__(self, payoffs, loss, base_total, transaction_cost_rate=0):
        super().__init__(payoffs, loss, transaction_cost_rate)
        self._base_total = base_total
        self._losing_batches = 0

    def update_bankroll(self, _current_bankroll):
        pass

    def record_settlement(self, won, _realized_returns):
        staked = [flag for flag in won if flag is not None]
        if staked and not any(staked):
            self._losing_batches += 1

    def evaluate(self, probabilities, current_bankroll):
        grown = min(1.0, self._base_total * (1.0 + 0.1 * self._losing_batches))
        grown = min(grown, self.get_max_safe_total_bet(current_bankroll))
        return _validate_stake_fractions(
            [grown / len(probabilities)] * len(probabilities)
        )


def run_case(strategy_name, scenario_name, seed, trials):
    """Run one grid cell and return the full bankroll history."""
    scenario = SCENARIOS[scenario_name]
    strategy = STRATEGY_FACTORIES[strategy_name](scenario)
    bankroll = BankRoll(
        initial_funds=INITIAL_FUNDS,
        max_transaction_loss=scenario["max_transaction_loss"],
    )
    simulator = PortfolioSimulator(
        bets=scenario["bets"],
        fee_per_bet=scenario["fee"],
        trials=trials,
        seed=seed,
    )
    simulator.evaluate_strategy(strategy, bankroll)
    return bankroll.history


def _digest(history):
    """SHA-256 over the hex repr of every history float (exact bits)."""
    joined = " ".join(value.hex() for value in history)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def generate_golden():
    """Digest every cell of the grid against the current code."""
    golden = {}
    for strategy_name in STRATEGY_FACTORIES:
        for scenario_name in SCENARIOS:
            for seed in SEEDS:
                for trials in TRIAL_COUNTS:
                    key = (strategy_name, scenario_name, seed, trials)
                    golden[key] = _digest(run_case(*key))
    return golden


_BEGIN_MARKER = "# === GOLDEN FIXTURE BEGIN (regenerate: uv run python tests/test_portfolio_bit_exactness.py) ===\n"  # noqa: E501
_END_MARKER = "# === GOLDEN FIXTURE END ===\n"

# === GOLDEN FIXTURE BEGIN (regenerate: uv run python tests/test_portfolio_bit_exactness.py) ===

GOLDEN = {
    (
        "Flat stakes",
        "drawdown-tight",
        0,
        7,
    ): "c7dc1c8a51b3848ec71fb35a88726008df32e37d3f22e7bcc929839d34777f07",
    (
        "Flat stakes",
        "drawdown-tight",
        0,
        53,
    ): "c7dc1c8a51b3848ec71fb35a88726008df32e37d3f22e7bcc929839d34777f07",
    (
        "Flat stakes",
        "drawdown-tight",
        0,
        400,
    ): "c7dc1c8a51b3848ec71fb35a88726008df32e37d3f22e7bcc929839d34777f07",
    (
        "Flat stakes",
        "drawdown-tight",
        1,
        7,
    ): "684dbed16604a2133a4298423a8bd654b3a5427016ff114fd9f9fac12ab639b0",
    (
        "Flat stakes",
        "drawdown-tight",
        1,
        53,
    ): "684dbed16604a2133a4298423a8bd654b3a5427016ff114fd9f9fac12ab639b0",
    (
        "Flat stakes",
        "drawdown-tight",
        1,
        400,
    ): "684dbed16604a2133a4298423a8bd654b3a5427016ff114fd9f9fac12ab639b0",
    (
        "Flat stakes",
        "drawdown-tight",
        42,
        7,
    ): "0b7ca6deb0e7a83fa6066ece90ac93926364b43c053dfdfcbda42fd166a16e02",
    (
        "Flat stakes",
        "drawdown-tight",
        42,
        53,
    ): "dde41f7708f0a544520851d871cf2e67eaca6ae9af3c76072a3cf985bff4d6f1",
    (
        "Flat stakes",
        "drawdown-tight",
        42,
        400,
    ): "dde41f7708f0a544520851d871cf2e67eaca6ae9af3c76072a3cf985bff4d6f1",
    (
        "Flat stakes",
        "drawdown-tight",
        20260803,
        7,
    ): "f17dfb2f5c78e72d08e1f77ee0bcbda6eddfc883fa5a186ce057a1cf58ff0c3f",
    (
        "Flat stakes",
        "drawdown-tight",
        20260803,
        53,
    ): "f17dfb2f5c78e72d08e1f77ee0bcbda6eddfc883fa5a186ce057a1cf58ff0c3f",
    (
        "Flat stakes",
        "drawdown-tight",
        20260803,
        400,
    ): "f17dfb2f5c78e72d08e1f77ee0bcbda6eddfc883fa5a186ce057a1cf58ff0c3f",
    (
        "Flat stakes",
        "longshot-heavy",
        0,
        7,
    ): "d782bb1ff22c1f1deb6aa54ee878811b9e468770a8180d4fcf31dfdc36ec9251",
    (
        "Flat stakes",
        "longshot-heavy",
        0,
        53,
    ): "7df4a0cbbbea18477ca3058620d45b5604bae0fb3f86197cb0d7bd94f973aba0",
    (
        "Flat stakes",
        "longshot-heavy",
        0,
        400,
    ): "ae8b876cdb869777716a8cc4c87afb7c86c6e759909700d4869b03698840bf6c",
    (
        "Flat stakes",
        "longshot-heavy",
        1,
        7,
    ): "5831ab29d4e95ec230d800ff159ef51874ec758e178881874f2f0b9df9fbb359",
    (
        "Flat stakes",
        "longshot-heavy",
        1,
        53,
    ): "7833825dd411fdf409afe8a1e5478b1f278b59c320dc41517eb589c356c4b60f",
    (
        "Flat stakes",
        "longshot-heavy",
        1,
        400,
    ): "740552c26a34f0c6d4893c168ce6aafd91001521d2a3f022ee66dedae0727cfb",
    (
        "Flat stakes",
        "longshot-heavy",
        42,
        7,
    ): "2b13f09403b9a4b11576ffe649f042e9d113c4f3549257d5187fc678eadf0df3",
    (
        "Flat stakes",
        "longshot-heavy",
        42,
        53,
    ): "eeebdf0a8aafd48acd8ce3cd64fbcade57fc1ee3e133fbd28d823e1761f3609f",
    (
        "Flat stakes",
        "longshot-heavy",
        42,
        400,
    ): "34ee9897676b3c2e082f6b37ce552eaf529a6816e9b45508a5d40a7c6830e405",
    (
        "Flat stakes",
        "longshot-heavy",
        20260803,
        7,
    ): "04f3c9ff3b28719e0de8cb2515d0ad187d362085d573d594e091dd014cac7107",
    (
        "Flat stakes",
        "longshot-heavy",
        20260803,
        53,
    ): "702831a949a48bc997a1ff219a2e9b085dab84553a62d1cf9fa2bc8639e1cf94",
    (
        "Flat stakes",
        "longshot-heavy",
        20260803,
        400,
    ): "265b3b5092b8b26d139c61c3a0412d6c0436a355795a7661bc7d27069eb4bbbb",
    (
        "Flat stakes",
        "standard",
        0,
        7,
    ): "b00055e978d2aad6e98e69900dfb4dd75d7646e403036f0873fe2660f429fe15",
    (
        "Flat stakes",
        "standard",
        0,
        53,
    ): "f8aa9ffb31a2b0737da46d283c56326aa413a02ee3b83cf5a24bf73296709408",
    (
        "Flat stakes",
        "standard",
        0,
        400,
    ): "ffc341a702a8f65ad2e24c7faf816cd40eefc981fca06971214b8acc45475460",
    (
        "Flat stakes",
        "standard",
        1,
        7,
    ): "dbd08110dd9c26c9a032cb16c9d08b8c6179db7ec3844e13b03c8fda493a602e",
    (
        "Flat stakes",
        "standard",
        1,
        53,
    ): "27187b1264084c39d4f19048e627b476cd2aa417ef3fa94a336d3379b60a9479",
    (
        "Flat stakes",
        "standard",
        1,
        400,
    ): "4b449d28765eece419d1dacd5ecb3e4f1fc9c79e8f7e0e9038eff48cfcc9a188",
    (
        "Flat stakes",
        "standard",
        42,
        7,
    ): "c0c88cd0d135134b808b67d98a421475f4725e7944897eec236aa9ecf0c1bf31",
    (
        "Flat stakes",
        "standard",
        42,
        53,
    ): "cceafc7cabd38e2de398261f1ee93ed823d79f665cea01301e1349910e205213",
    (
        "Flat stakes",
        "standard",
        42,
        400,
    ): "66d747ed23b169d031eb4f9ec770cdc9150c7a8c0855acaf32d8b039407ffcf5",
    (
        "Flat stakes",
        "standard",
        20260803,
        7,
    ): "4a064884c30b127b4f547989337872fb5e33f91291cb02d77f6135319b2b3c17",
    (
        "Flat stakes",
        "standard",
        20260803,
        53,
    ): "625cb69acfd0965c7f80d5f660e6ffc8ccd5bc8191aab87ba8067d27ddeec481",
    (
        "Flat stakes",
        "standard",
        20260803,
        400,
    ): "5eb84284c4196246df4986b2b8cde53d6a9709ce41b8d97f27dbaa6722ea30ca",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        0,
        7,
    ): "31ab210e135e7dfdde2488843e436e4484655eaebb84a5d195bd8d7cb1333fb7",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        0,
        53,
    ): "31ab210e135e7dfdde2488843e436e4484655eaebb84a5d195bd8d7cb1333fb7",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        0,
        400,
    ): "31ab210e135e7dfdde2488843e436e4484655eaebb84a5d195bd8d7cb1333fb7",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        1,
        7,
    ): "bdeb8a3d5910cd4784dc18d5f357d7e82c66f9537cb9eb69cd8ee277c923e834",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        1,
        53,
    ): "bdeb8a3d5910cd4784dc18d5f357d7e82c66f9537cb9eb69cd8ee277c923e834",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        1,
        400,
    ): "bdeb8a3d5910cd4784dc18d5f357d7e82c66f9537cb9eb69cd8ee277c923e834",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        42,
        7,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        42,
        53,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        42,
        400,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        20260803,
        7,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        20260803,
        53,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        20260803,
        400,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        0,
        7,
    ): "9935e69d7a9cb69ebd05ab7a2e79dd07be8742851f5c5bbc42beebd2a097b896",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        0,
        53,
    ): "ddce576687fb56f6ca5745197fa04b5e8c47cd6dc08653d6229aeeb4cb522778",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        0,
        400,
    ): "ddce576687fb56f6ca5745197fa04b5e8c47cd6dc08653d6229aeeb4cb522778",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        1,
        7,
    ): "84caa8d8f5af2bb10f62df1021a4b51c251bea80f41af4907217cfb9395e19d9",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        1,
        53,
    ): "fc34c22f0d90fd5e9f93db60c5ebf7bac6826ef4fbc15948d5e0d75a156f81dc",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        1,
        400,
    ): "fc34c22f0d90fd5e9f93db60c5ebf7bac6826ef4fbc15948d5e0d75a156f81dc",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        42,
        7,
    ): "58a7be73ddd8904d5cd199362aada6af9103d6c4ebfafb5967e586aba50e2f7c",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        42,
        53,
    ): "19f1342c9aff14ee26735ed483d610ec2c9989925cc1006fb3825df3be3c1b87",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        42,
        400,
    ): "19f1342c9aff14ee26735ed483d610ec2c9989925cc1006fb3825df3be3c1b87",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        20260803,
        7,
    ): "23b186ed09b8ba312945648ab1df29d1b51f05f484d999418d6e219689e04ec7",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        20260803,
        53,
    ): "e363abc737657227fcde3006d02aef699c933d50c7f1b21c2e4b1e91b73e5710",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        20260803,
        400,
    ): "e363abc737657227fcde3006d02aef699c933d50c7f1b21c2e4b1e91b73e5710",
    (
        "Losing-batch tracking",
        "standard",
        0,
        7,
    ): "f589bf2f81cd704ae50b105eac94d9c4fff7eef88b494d0f844c2d314326e407",
    (
        "Losing-batch tracking",
        "standard",
        0,
        53,
    ): "10a79c5457920c9fc4fc78d802e1a6bc397ef19720ea0699b6aa36c2ea49d27b",
    (
        "Losing-batch tracking",
        "standard",
        0,
        400,
    ): "f8aca840b72c60afd584b2400c08493f6f8fbafc7df0a1ade0cf5c6c2be3e5f2",
    (
        "Losing-batch tracking",
        "standard",
        1,
        7,
    ): "f09f72620c382665410d033881e2e9ad4087e4841b5136acc793484477173200",
    (
        "Losing-batch tracking",
        "standard",
        1,
        53,
    ): "6664aa5c89ff58df2715751049eb70fdcefc2e579c1c441ccd11fad8fce20268",
    (
        "Losing-batch tracking",
        "standard",
        1,
        400,
    ): "47819c7169d04aa7b01ca406c21d070f318a101026835fb591a26cc1b61942a6",
    (
        "Losing-batch tracking",
        "standard",
        42,
        7,
    ): "d137e3765a2c9faefd88487a8ec2ee90c6782acc0810bcf0e0ed3a165dfee981",
    (
        "Losing-batch tracking",
        "standard",
        42,
        53,
    ): "154033fcd572e057fa215cd35ae320c15c407f5af89712aa05eaae05c0d09f3a",
    (
        "Losing-batch tracking",
        "standard",
        42,
        400,
    ): "4a8f3ae48f8f337bea146c019e74d26b2c7da91b31132d4707d6d6da3dabc1f0",
    (
        "Losing-batch tracking",
        "standard",
        20260803,
        7,
    ): "9c10c0e6c4ca3e1d83f325a99ac7d6203682be9a466b65b0c00f6b0953c1153c",
    (
        "Losing-batch tracking",
        "standard",
        20260803,
        53,
    ): "a4b79987a60e06cd82dca296db5e885c4ee308a98b73de9e15a9fdd0d287e1f4",
    (
        "Losing-batch tracking",
        "standard",
        20260803,
        400,
    ): "2033ac799647648a97e217faa00ffd4d5aaa37bed0dbc56ac1b353d917b80a76",
    (
        "Probability weighted",
        "drawdown-tight",
        0,
        7,
    ): "31f6c325f88e5ce4188e2e391ebdddf53d70e1ba5606b10a8b79fb2a84d4a44b",
    (
        "Probability weighted",
        "drawdown-tight",
        0,
        53,
    ): "31f6c325f88e5ce4188e2e391ebdddf53d70e1ba5606b10a8b79fb2a84d4a44b",
    (
        "Probability weighted",
        "drawdown-tight",
        0,
        400,
    ): "31f6c325f88e5ce4188e2e391ebdddf53d70e1ba5606b10a8b79fb2a84d4a44b",
    (
        "Probability weighted",
        "drawdown-tight",
        1,
        7,
    ): "b899231e80cf74c1faa7ec62b621fd9e70f55f843892e5dfb4e2d7936c8ad9a2",
    (
        "Probability weighted",
        "drawdown-tight",
        1,
        53,
    ): "b899231e80cf74c1faa7ec62b621fd9e70f55f843892e5dfb4e2d7936c8ad9a2",
    (
        "Probability weighted",
        "drawdown-tight",
        1,
        400,
    ): "b899231e80cf74c1faa7ec62b621fd9e70f55f843892e5dfb4e2d7936c8ad9a2",
    (
        "Probability weighted",
        "drawdown-tight",
        42,
        7,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Probability weighted",
        "drawdown-tight",
        42,
        53,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Probability weighted",
        "drawdown-tight",
        42,
        400,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Probability weighted",
        "drawdown-tight",
        20260803,
        7,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Probability weighted",
        "drawdown-tight",
        20260803,
        53,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Probability weighted",
        "drawdown-tight",
        20260803,
        400,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Probability weighted",
        "longshot-heavy",
        0,
        7,
    ): "0c5522cd5c3b735f80444f7b9c7b183b22e55c744acb1e92b6e671e0ddf4e492",
    (
        "Probability weighted",
        "longshot-heavy",
        0,
        53,
    ): "0c5522cd5c3b735f80444f7b9c7b183b22e55c744acb1e92b6e671e0ddf4e492",
    (
        "Probability weighted",
        "longshot-heavy",
        0,
        400,
    ): "0c5522cd5c3b735f80444f7b9c7b183b22e55c744acb1e92b6e671e0ddf4e492",
    (
        "Probability weighted",
        "longshot-heavy",
        1,
        7,
    ): "688788cf5ef75445ce1e4ca833f4467d3ac2c74eb008a69ed389cb4652a845db",
    (
        "Probability weighted",
        "longshot-heavy",
        1,
        53,
    ): "9587e7e2d30bb47a78f6c5c13ddf33f6de6bbd21ea5e94bf234fee3c6d9fa995",
    (
        "Probability weighted",
        "longshot-heavy",
        1,
        400,
    ): "9587e7e2d30bb47a78f6c5c13ddf33f6de6bbd21ea5e94bf234fee3c6d9fa995",
    (
        "Probability weighted",
        "longshot-heavy",
        42,
        7,
    ): "3de0019bf9c48f4f16ce1b0b261eb649f293f1f3e65ccf16b077adf8dd83ba8c",
    (
        "Probability weighted",
        "longshot-heavy",
        42,
        53,
    ): "3de0019bf9c48f4f16ce1b0b261eb649f293f1f3e65ccf16b077adf8dd83ba8c",
    (
        "Probability weighted",
        "longshot-heavy",
        42,
        400,
    ): "3de0019bf9c48f4f16ce1b0b261eb649f293f1f3e65ccf16b077adf8dd83ba8c",
    (
        "Probability weighted",
        "longshot-heavy",
        20260803,
        7,
    ): "03eaf719c33bb6e80b0faafadde4a6d8696312912dcee2ca65eb4043dd0ba081",
    (
        "Probability weighted",
        "longshot-heavy",
        20260803,
        53,
    ): "03eaf719c33bb6e80b0faafadde4a6d8696312912dcee2ca65eb4043dd0ba081",
    (
        "Probability weighted",
        "longshot-heavy",
        20260803,
        400,
    ): "03eaf719c33bb6e80b0faafadde4a6d8696312912dcee2ca65eb4043dd0ba081",
    (
        "Probability weighted",
        "standard",
        0,
        7,
    ): "b95ca62191e58a03c2a31bdeccbf7e1947b9e8692505b93c117bae98d2a20197",
    (
        "Probability weighted",
        "standard",
        0,
        53,
    ): "8b755cc51dc1ab8374640f7391b54e91ad3830ce3c83a552e2bc2c9e2bcdf70f",
    (
        "Probability weighted",
        "standard",
        0,
        400,
    ): "8b755cc51dc1ab8374640f7391b54e91ad3830ce3c83a552e2bc2c9e2bcdf70f",
    (
        "Probability weighted",
        "standard",
        1,
        7,
    ): "b6701beede35be1e21376c4b14c337d6c74dd41e2a165f045c524fe377636907",
    (
        "Probability weighted",
        "standard",
        1,
        53,
    ): "1a37881da19a850085076e3573e38c440d66f8f5ed3e0590c926a92db944bd25",
    (
        "Probability weighted",
        "standard",
        1,
        400,
    ): "1a37881da19a850085076e3573e38c440d66f8f5ed3e0590c926a92db944bd25",
    (
        "Probability weighted",
        "standard",
        42,
        7,
    ): "3b50cad7371ce9c8241ab1ffa8109ec5f96608fd8aa68fcb753dceb947f1b954",
    (
        "Probability weighted",
        "standard",
        42,
        53,
    ): "dba9e43b291766f6d5ef7cbe444795d6da6f3ba778db312353f2091d91067607",
    (
        "Probability weighted",
        "standard",
        42,
        400,
    ): "dba9e43b291766f6d5ef7cbe444795d6da6f3ba778db312353f2091d91067607",
    (
        "Probability weighted",
        "standard",
        20260803,
        7,
    ): "e4c61d6fa401c28f2629cf791b91e2884ecd83c9dfb95d646fd7afd706f1ffbd",
    (
        "Probability weighted",
        "standard",
        20260803,
        53,
    ): "5c536424801f950bffcfea2f42021d683ae1d84285bf4eadf6c27f34588c2bf3",
    (
        "Probability weighted",
        "standard",
        20260803,
        400,
    ): "5c536424801f950bffcfea2f42021d683ae1d84285bf4eadf6c27f34588c2bf3",
}
# === GOLDEN FIXTURE END ===

_CASES = sorted(GOLDEN)


@pytest.mark.parametrize(
    ("strategy_name", "scenario_name", "seed", "trials"),
    _CASES,
    ids=[f"{case[0]}|{case[1]}|{case[2]}|{case[3]}" for case in _CASES],
)
def test_seeded_outputs_are_bit_exact(strategy_name, scenario_name, seed, trials):
    history = run_case(strategy_name, scenario_name, seed, trials)
    expected = GOLDEN[(strategy_name, scenario_name, seed, trials)]
    actual = _digest(history)
    assert actual == expected, (
        f"Output bits changed for portfolio/{strategy_name}/{scenario_name} "
        f"seed={seed} trials={trials}: final funds {history[-1]!r}, "
        f"history length {len(history)}"
    )


if __name__ == "__main__":
    _path = Path(__file__)
    _text = _path.read_text()
    _start = _text.index(_BEGIN_MARKER) + len(_BEGIN_MARKER)
    _end = _text.index(_END_MARKER)
    _entries = "\n".join(
        f"    {key!r}: {value!r}," for key, value in sorted(generate_golden().items())
    )
    _path.write_text(
        _text[:_start] + "\nGOLDEN = {\n" + _entries + "\n}\n" + _text[_end:]
    )
    print(f"Regenerated {len(_entries.splitlines())} golden entries")
