"""Bit-exactness oracle for the repeated multi-outcome simulator.

Freezes the exact public outputs of the simulator — the float bits of
``BankRoll.history`` — for every strategy x scenario x seed x trial-count
combination. It is the multi-outcome analogue of ``tests/test_bit_exactness.py``
and, like it, the acceptance oracle for output-identical performance work on
this class: any change that alters a single settlement bit fails this module.

Regenerate on purpose only (then format)::

    uv run python tests/test_multi_outcome_bit_exactness.py
    ruff format tests/test_multi_outcome_bit_exactness.py

which rewrites the GOLDEN block between the markers below from the current
code. The binary simulators' oracle in ``tests/test_bit_exactness.py`` is
separate and must stay untouched.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from keeks.bankroll import BankRoll
from keeks.multi_outcome import RepeatedMultiOutcomeSimulator
from keeks.multi_outcome.base import BaseMultiOutcomeStrategy, _validate_stake_fractions

INITIAL_FUNDS = 1000.0

# Scenario axes: a standard near-complete market with a small fee, a market
# where most of the mass is residual (voids dominate and the stateful
# strategy's growth path saturates), and a fee-dominated market under a tight
# drawdown limit so the RuinError settle-full-batch-then-stop path triggers.
SCENARIOS = {
    "standard": {
        "payoffs": (2.0, 3.0, 2.4),
        "loss": 1.0,
        "fee": 0.01,
        "probabilities": (0.42, 0.27, 0.28),
        "max_transaction_loss": None,
    },
    "residual-heavy": {
        "payoffs": (2.5, 2.2, 3.0),
        "loss": 1.0,
        "fee": 0.0,
        "probabilities": (0.2, 0.15, 0.1),
        "max_transaction_loss": None,
    },
    "fee-heavy": {
        "payoffs": (1.2, 1.1, 1.3),
        "loss": 0.7,
        "fee": 0.5,
        "probabilities": (0.45, 0.3, 0.2),
        "max_transaction_loss": 0.05,
    },
}

# Mirrors the binary oracle's discipline: stateful strategies must be rebuilt
# per run because they carry state across evaluate() calls. The void-tracking
# strategy exercises the update_bankroll and record_settlement hooks and the
# aggregate safe-stake cap.
STRATEGY_FACTORIES = {
    "Flat stakes": lambda s: _FlatStakesStrategy(
        payoffs=s["payoffs"], loss=s["loss"], fraction=0.02, transaction_cost_rate=0.0
    ),
    "Probability weighted": lambda s: _ProbabilityWeightedStrategy(
        payoffs=s["payoffs"], loss=s["loss"], aggregate=0.9, transaction_cost_rate=0.0
    ),
    "Void tracking": lambda s: _VoidTrackingStrategy(
        payoffs=s["payoffs"], loss=s["loss"], base_total=0.3, transaction_cost_rate=0.0
    ),
}

# Grid of seeds and trial counts. Short runs catch early stops and the no-bet
# path; the 400-trial runs compound.
SEEDS = (0, 1, 42, 20260803)
TRIAL_COUNTS = (7, 53, 400)


class _FlatStakesStrategy(BaseMultiOutcomeStrategy):
    """The same fraction on every leg, every trial."""

    def __init__(self, payoffs, loss, fraction, transaction_cost_rate=0):
        super().__init__(payoffs, loss, transaction_cost_rate)
        self._fraction = fraction

    def evaluate(self, probabilities, _current_bankroll):
        return _validate_stake_fractions([self._fraction] * len(probabilities))


class _ProbabilityWeightedStrategy(BaseMultiOutcomeStrategy):
    """Stakes proportional to each leg's probability under a fixed aggregate."""

    def __init__(self, payoffs, loss, aggregate, transaction_cost_rate=0):
        super().__init__(payoffs, loss, transaction_cost_rate)
        self._aggregate = aggregate

    def evaluate(self, probabilities, _current_bankroll):
        probabilities = np.asarray(probabilities, dtype=float)
        weights = probabilities / probabilities.sum()
        return _validate_stake_fractions((self._aggregate * weights).tolist())


class _VoidTrackingStrategy(BaseMultiOutcomeStrategy):
    """Grows the aggregate stake after voids, under the safe-stake cap."""

    def __init__(self, payoffs, loss, base_total, transaction_cost_rate=0):
        super().__init__(payoffs, loss, transaction_cost_rate)
        self._base_total = base_total
        self._voids = 0

    def update_bankroll(self, _current_bankroll):
        pass

    def record_settlement(self, won, _realized_returns):
        if all(flag is None for flag in won):
            self._voids += 1

    def evaluate(self, probabilities, current_bankroll):
        grown = min(1.0, self._base_total * (1.0 + 0.1 * self._voids))
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
    simulator = RepeatedMultiOutcomeSimulator(
        payoffs=scenario["payoffs"],
        loss=scenario["loss"],
        fee_per_bet=scenario["fee"],
        probabilities=scenario["probabilities"],
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


_BEGIN_MARKER = "# === GOLDEN FIXTURE BEGIN (regenerate: uv run python tests/test_multi_outcome_bit_exactness.py) ===\n"  # noqa: E501
_END_MARKER = "# === GOLDEN FIXTURE END ===\n"

# === GOLDEN FIXTURE BEGIN (regenerate: uv run python tests/test_multi_outcome_bit_exactness.py) ===

GOLDEN = {
    (
        "Flat stakes",
        "fee-heavy",
        0,
        7,
    ): "2cf225f7d2c6fddf2268a79dbfe42c1ce8b0aa5674fa26e699587d8da50c2bbe",
    (
        "Flat stakes",
        "fee-heavy",
        0,
        53,
    ): "6d8f0fa41d6cde1ad7e43fee73c1d7200c79160417a03fd092b13fdbb443a447",
    (
        "Flat stakes",
        "fee-heavy",
        0,
        400,
    ): "5d894d97e8d1ea03097397324f4fc2f4730b4f07966b1850c78dc14849625bd4",
    (
        "Flat stakes",
        "fee-heavy",
        1,
        7,
    ): "46c1e6b3cc73884523ed27a633629e08443fac3be07efcbccc212a92bce2bed2",
    (
        "Flat stakes",
        "fee-heavy",
        1,
        53,
    ): "60123e9d4ef7a235db8fdd525f2b8c69042755e2b59e1fb36d126ad0ebb97fbc",
    (
        "Flat stakes",
        "fee-heavy",
        1,
        400,
    ): "141df2a6b1c02e2b303a3b39616aef6a7a9074de7a694eb27d7d7a207c16d090",
    (
        "Flat stakes",
        "fee-heavy",
        42,
        7,
    ): "7c517c5f0a360cbd7ef85923f751459389e58e669b951fa60a5dc34055cac222",
    (
        "Flat stakes",
        "fee-heavy",
        42,
        53,
    ): "c61d6abe58c45e3aaa0df71c5bc02a5e92fb947c8b8b54511119a4185ccb98bd",
    (
        "Flat stakes",
        "fee-heavy",
        42,
        400,
    ): "ff13c1bbd1af083954408ca29e879dc7675478d8687e18ad2896d6a2b0bf3d6a",
    (
        "Flat stakes",
        "fee-heavy",
        20260803,
        7,
    ): "4bef9655a70c9280f92cef176f455790b28351cb58d1f54e5c20c1cec541bb31",
    (
        "Flat stakes",
        "fee-heavy",
        20260803,
        53,
    ): "d8e44a4ce3fe48ee2500ffaadfa8df898c6ed3d12d3bf9721c42e8772349517d",
    (
        "Flat stakes",
        "fee-heavy",
        20260803,
        400,
    ): "4a3c51e72cef6e7bf128dcc89e810687609bb2011b059b5bf9b5e3c512cddf9a",
    (
        "Flat stakes",
        "residual-heavy",
        0,
        7,
    ): "d8a079b1a22390b39d13fbc439141b5cc22aa0e0efd33190e02f2a836198319f",
    (
        "Flat stakes",
        "residual-heavy",
        0,
        53,
    ): "cd49efb1765416f9cdd01a1b87a2a5c51f713db9081b17cd8d14fe57bb68b41f",
    (
        "Flat stakes",
        "residual-heavy",
        0,
        400,
    ): "c4a9974dc63107da668d49d472e8cacb8d750506cc5d5538a6039ddfde54be2d",
    (
        "Flat stakes",
        "residual-heavy",
        1,
        7,
    ): "b3f5ebca3c53422e0597c2dccaf8d4aa81ecbecaccc85d853889057cbe63097b",
    (
        "Flat stakes",
        "residual-heavy",
        1,
        53,
    ): "6437a3b184353a1628b118230c7c37e3ff28347431e45f72cc8d1332ba1de1ce",
    (
        "Flat stakes",
        "residual-heavy",
        1,
        400,
    ): "c113cbd7b951e08d30dcfe19d193b6d2febfa342085b8c848c42f19935fabb4a",
    (
        "Flat stakes",
        "residual-heavy",
        42,
        7,
    ): "7e63ca880cda8116f928f9695921805bc92427281973d4a1266eeecd7840987d",
    (
        "Flat stakes",
        "residual-heavy",
        42,
        53,
    ): "1bf3ca72fbcc6acc251ccf62a6e13217a60cbbacfa362cf1f752d89c24e6b10f",
    (
        "Flat stakes",
        "residual-heavy",
        42,
        400,
    ): "4ce49332cb737355dd926db3137b8b3f419ef8c42090e8a7f46256126a91ed18",
    (
        "Flat stakes",
        "residual-heavy",
        20260803,
        7,
    ): "bb2a79f35733959812ee91dd1608ffa09c644af47c94d0d1939a21a184c91a9c",
    (
        "Flat stakes",
        "residual-heavy",
        20260803,
        53,
    ): "758a10abf4f401a39ee75b0419fed5a22d4d4113a90d87fbb4169f6a2fc4a150",
    (
        "Flat stakes",
        "residual-heavy",
        20260803,
        400,
    ): "fe7a8c4abea07aa22063726a49dee70ed787ac4dedd6f3d68eb60d048bb92c8b",
    (
        "Flat stakes",
        "standard",
        0,
        7,
    ): "0538b4f069524447e01272584fc4b0de84b5069541d3685e029be725a33a981d",
    (
        "Flat stakes",
        "standard",
        0,
        53,
    ): "c10be1555c0586515eddd3bca456c583904b8354baa15fc81aa2b488c67d9d19",
    (
        "Flat stakes",
        "standard",
        0,
        400,
    ): "e6f01064b6c0ed88e42b6ee149c5945472461f5120445daae2e6049c75ed406a",
    (
        "Flat stakes",
        "standard",
        1,
        7,
    ): "31cf358b61b7f3fb0efe084d44d68ff229794f6df29bb8df35090336b812bbff",
    (
        "Flat stakes",
        "standard",
        1,
        53,
    ): "7da779634733b98dfe360e6d5adc455735d77efa54525c26b7d8b5a9aa76fe6e",
    (
        "Flat stakes",
        "standard",
        1,
        400,
    ): "d1f09ccdf6e15ecf21a04ed9f15c830b20aa25de2333c6b0df8743567067ea7a",
    (
        "Flat stakes",
        "standard",
        42,
        7,
    ): "3d89b8f55bd4c1ccae18717ccfb3bab8704dd29bcea0c3fcfe7f2556f1636836",
    (
        "Flat stakes",
        "standard",
        42,
        53,
    ): "2776997e7eebba9aa9804eeb3f265952c6c388308061465251733d53a4070b0c",
    (
        "Flat stakes",
        "standard",
        42,
        400,
    ): "d5bc057eb5aace68df9bdb1367b19727ca4ab2e0d4cff769bd474f9147c49031",
    (
        "Flat stakes",
        "standard",
        20260803,
        7,
    ): "b9aa7b1c0b21d61bfea9f33528469ac3f663abfd4435cf39a514d25f389d7881",
    (
        "Flat stakes",
        "standard",
        20260803,
        53,
    ): "dea04c57f6a7b99b7ef4e3a567455f341bf31d798f6c4f76e4cbb7cfeb43f81d",
    (
        "Flat stakes",
        "standard",
        20260803,
        400,
    ): "b75d7b436b86be748248bcd29bdee971e8b35949752a2470d9a0833d69dc71d3",
    (
        "Probability weighted",
        "fee-heavy",
        0,
        7,
    ): "bc6db7e025ca96bb7aad4ebdeeceeaa6623b95b96384dd0d095bc37e66dfc2f7",
    (
        "Probability weighted",
        "fee-heavy",
        0,
        53,
    ): "bc6db7e025ca96bb7aad4ebdeeceeaa6623b95b96384dd0d095bc37e66dfc2f7",
    (
        "Probability weighted",
        "fee-heavy",
        0,
        400,
    ): "bc6db7e025ca96bb7aad4ebdeeceeaa6623b95b96384dd0d095bc37e66dfc2f7",
    (
        "Probability weighted",
        "fee-heavy",
        1,
        7,
    ): "7b881fccd1b2d4a6bec51b4a324c6dcbd74f7fe5a0b0ebb6ff046232607509fe",
    (
        "Probability weighted",
        "fee-heavy",
        1,
        53,
    ): "7b881fccd1b2d4a6bec51b4a324c6dcbd74f7fe5a0b0ebb6ff046232607509fe",
    (
        "Probability weighted",
        "fee-heavy",
        1,
        400,
    ): "7b881fccd1b2d4a6bec51b4a324c6dcbd74f7fe5a0b0ebb6ff046232607509fe",
    (
        "Probability weighted",
        "fee-heavy",
        42,
        7,
    ): "bc6db7e025ca96bb7aad4ebdeeceeaa6623b95b96384dd0d095bc37e66dfc2f7",
    (
        "Probability weighted",
        "fee-heavy",
        42,
        53,
    ): "bc6db7e025ca96bb7aad4ebdeeceeaa6623b95b96384dd0d095bc37e66dfc2f7",
    (
        "Probability weighted",
        "fee-heavy",
        42,
        400,
    ): "bc6db7e025ca96bb7aad4ebdeeceeaa6623b95b96384dd0d095bc37e66dfc2f7",
    (
        "Probability weighted",
        "fee-heavy",
        20260803,
        7,
    ): "c4d99b1a5017b13e08276054daea4d1f940f49be22e20babdfaacb114bdaccf8",
    (
        "Probability weighted",
        "fee-heavy",
        20260803,
        53,
    ): "c4d99b1a5017b13e08276054daea4d1f940f49be22e20babdfaacb114bdaccf8",
    (
        "Probability weighted",
        "fee-heavy",
        20260803,
        400,
    ): "c4d99b1a5017b13e08276054daea4d1f940f49be22e20babdfaacb114bdaccf8",
    (
        "Probability weighted",
        "residual-heavy",
        0,
        7,
    ): "37965f895be4e1efc1bf4d75a58305ab05d1708b3dfee94fcab67eb7014a7954",
    (
        "Probability weighted",
        "residual-heavy",
        0,
        53,
    ): "a32253a2b44980bd968acb67410228e3f5ea5a57c66e36ae225e6499c132e31e",
    (
        "Probability weighted",
        "residual-heavy",
        0,
        400,
    ): "6756f1d4c6da130a8082c60e9dabab54a8f7d2320ed5a22328ad1563eb7a9ad7",
    (
        "Probability weighted",
        "residual-heavy",
        1,
        7,
    ): "68d036bb5628ef1b899822f84c04e1016274811a8105cd629361379ce04a6195",
    (
        "Probability weighted",
        "residual-heavy",
        1,
        53,
    ): "d969389c0c2f079935c052345c2c9130ce373253bb1b25bd067b905a46a8f76d",
    (
        "Probability weighted",
        "residual-heavy",
        1,
        400,
    ): "2d038cce3ee52189a24ab9ddb15d4f3a8dd19f78005491441a206bd1dd25a0d0",
    (
        "Probability weighted",
        "residual-heavy",
        42,
        7,
    ): "7bd1548ea64dbe89dd7f621d6894339861e63cb186ec249335919a19916aaae9",
    (
        "Probability weighted",
        "residual-heavy",
        42,
        53,
    ): "87460db89f672b6886485006d9f9206d458776ccdc34433942538bac3cd1d236",
    (
        "Probability weighted",
        "residual-heavy",
        42,
        400,
    ): "e92611b5e1116a94bc0eb5ec9a111510fda54b490587d9db69f1fb6893ea5b8f",
    (
        "Probability weighted",
        "residual-heavy",
        20260803,
        7,
    ): "c25e656f2ba9dad443ade683e11b883d60e1c9b1447d775a702a1d2127569b8b",
    (
        "Probability weighted",
        "residual-heavy",
        20260803,
        53,
    ): "e452d6cb3b05da7f616569fe588b3461f53b4f9604afc51f307e735c6a9b9dea",
    (
        "Probability weighted",
        "residual-heavy",
        20260803,
        400,
    ): "fb50d9a497645add223dfe2b5066d507b84955d9eae50f9791a667def137641a",
    (
        "Probability weighted",
        "standard",
        0,
        7,
    ): "70d5adedb71b935c4363ac4642b7611b71f7dc14131df741654b119094976f76",
    (
        "Probability weighted",
        "standard",
        0,
        53,
    ): "7d6ebecf12856afa83b6dc31f8de5dc01bb64dcf39e9201f4fcba63c1279eab8",
    (
        "Probability weighted",
        "standard",
        0,
        400,
    ): "7d6ebecf12856afa83b6dc31f8de5dc01bb64dcf39e9201f4fcba63c1279eab8",
    (
        "Probability weighted",
        "standard",
        1,
        7,
    ): "c2c56cf6c0ada471d87da17ff5cb2ab8158af90112d8c3a5a0be57d73c1bf1c7",
    (
        "Probability weighted",
        "standard",
        1,
        53,
    ): "a39bcf8903b35fb246cf162b7a011c6449e08dd1f168d76d9c6792efd9032d96",
    (
        "Probability weighted",
        "standard",
        1,
        400,
    ): "a39bcf8903b35fb246cf162b7a011c6449e08dd1f168d76d9c6792efd9032d96",
    (
        "Probability weighted",
        "standard",
        42,
        7,
    ): "7af0eb01daf9754cc80ce9f85a68998561415087c7dceef71ea3aac358cab1fe",
    (
        "Probability weighted",
        "standard",
        42,
        53,
    ): "392e32d35c9535879bb21fdc02804c45f32e62fc722eca21285a9d8a7a85fed8",
    (
        "Probability weighted",
        "standard",
        42,
        400,
    ): "392e32d35c9535879bb21fdc02804c45f32e62fc722eca21285a9d8a7a85fed8",
    (
        "Probability weighted",
        "standard",
        20260803,
        7,
    ): "330983d9edbcd4019f88492d78ce362361a6b10d6afdec4ab7c6566790f1d6bf",
    (
        "Probability weighted",
        "standard",
        20260803,
        53,
    ): "10eb9616fa57db895b1e18594ca590bb8992a6c761fc569f7970afbf16f185ce",
    (
        "Probability weighted",
        "standard",
        20260803,
        400,
    ): "10eb9616fa57db895b1e18594ca590bb8992a6c761fc569f7970afbf16f185ce",
    (
        "Void tracking",
        "fee-heavy",
        0,
        7,
    ): "167d60080d8338a7203f3a8f8221e06385f715e1f55fd72a1eaa2f60e64f7825",
    (
        "Void tracking",
        "fee-heavy",
        0,
        53,
    ): "167d60080d8338a7203f3a8f8221e06385f715e1f55fd72a1eaa2f60e64f7825",
    (
        "Void tracking",
        "fee-heavy",
        0,
        400,
    ): "167d60080d8338a7203f3a8f8221e06385f715e1f55fd72a1eaa2f60e64f7825",
    (
        "Void tracking",
        "fee-heavy",
        1,
        7,
    ): "ce370f7164661fa2be6797c302c32e28c6118c1751fc58131f767fac7867fa8c",
    (
        "Void tracking",
        "fee-heavy",
        1,
        53,
    ): "ce370f7164661fa2be6797c302c32e28c6118c1751fc58131f767fac7867fa8c",
    (
        "Void tracking",
        "fee-heavy",
        1,
        400,
    ): "ce370f7164661fa2be6797c302c32e28c6118c1751fc58131f767fac7867fa8c",
    (
        "Void tracking",
        "fee-heavy",
        42,
        7,
    ): "167d60080d8338a7203f3a8f8221e06385f715e1f55fd72a1eaa2f60e64f7825",
    (
        "Void tracking",
        "fee-heavy",
        42,
        53,
    ): "167d60080d8338a7203f3a8f8221e06385f715e1f55fd72a1eaa2f60e64f7825",
    (
        "Void tracking",
        "fee-heavy",
        42,
        400,
    ): "167d60080d8338a7203f3a8f8221e06385f715e1f55fd72a1eaa2f60e64f7825",
    (
        "Void tracking",
        "fee-heavy",
        20260803,
        7,
    ): "4207e0038e9903098bd0c73f333ace44ab207666dbc7618d62054a896d7e4c08",
    (
        "Void tracking",
        "fee-heavy",
        20260803,
        53,
    ): "4207e0038e9903098bd0c73f333ace44ab207666dbc7618d62054a896d7e4c08",
    (
        "Void tracking",
        "fee-heavy",
        20260803,
        400,
    ): "4207e0038e9903098bd0c73f333ace44ab207666dbc7618d62054a896d7e4c08",
    (
        "Void tracking",
        "residual-heavy",
        0,
        7,
    ): "b68e642a1423434f3b29c7bbd84c5fb3fb6a48fb43bce7a00a3845249c89a7d3",
    (
        "Void tracking",
        "residual-heavy",
        0,
        53,
    ): "efa8115f383fef7960663372afd943f1d21ab742f39a15920e7dfb423f87f032",
    (
        "Void tracking",
        "residual-heavy",
        0,
        400,
    ): "f84b433369a913025436b46519cca7514e6d513e42f2f29bc81b6e3b8ab94f42",
    (
        "Void tracking",
        "residual-heavy",
        1,
        7,
    ): "adb77f1419a8b26868c3f783ba82b1398fc00d83df349f7d766079419d286f0d",
    (
        "Void tracking",
        "residual-heavy",
        1,
        53,
    ): "0c61d1ddb9a61aa0697bd4d440780f52553df6772a99d8b8b2c81bb7b133e824",
    (
        "Void tracking",
        "residual-heavy",
        1,
        400,
    ): "872cd81beb6a5f029bda9eb652c0d89414e30fe7d3a740ce39f2c49e735f1e43",
    (
        "Void tracking",
        "residual-heavy",
        42,
        7,
    ): "1245dab43ac85bcde3d067df15177679ebe63a2db327bdc604cf54a7c40a08c0",
    (
        "Void tracking",
        "residual-heavy",
        42,
        53,
    ): "3b88eec6a8f757fe66fa48fe9aba05fb12b703b23eb9395e735ee11f661356d3",
    (
        "Void tracking",
        "residual-heavy",
        42,
        400,
    ): "003022909c3dc28b6f70fd08c260e978b27f830c36186e7c094c633cd8b949a5",
    (
        "Void tracking",
        "residual-heavy",
        20260803,
        7,
    ): "3afecb2c278485c67cb718ab1433531188842c418c5fd36800c1c4419f777dcd",
    (
        "Void tracking",
        "residual-heavy",
        20260803,
        53,
    ): "9c87277a2a5687361e3387af4f275cb1bc84ffc717dadfa0044edeaba4344e14",
    (
        "Void tracking",
        "residual-heavy",
        20260803,
        400,
    ): "8c3d3bb6bdd2acd0c580128611a7afa0f86d887d87d65fd9e8c72e26ba378d35",
    (
        "Void tracking",
        "standard",
        0,
        7,
    ): "8e55fc5e8bee9e23777544039e711fa8d6fd116d3c97bb766a0f13be50a23cb4",
    (
        "Void tracking",
        "standard",
        0,
        53,
    ): "f94b252574af898c9de235035355016c00f1345aca3ac3da9f07a0233ad484aa",
    (
        "Void tracking",
        "standard",
        0,
        400,
    ): "d9214c5b0d62e2b67f700f21731f737f49f164e7f5f3c8ee5a7b1d0d0df9fc86",
    (
        "Void tracking",
        "standard",
        1,
        7,
    ): "299a23f826acd0fb784986a83121f232d058319fb61acc0b56c002011f654a4e",
    (
        "Void tracking",
        "standard",
        1,
        53,
    ): "135087ae566d8c41fe436433f4a4a30ef9f26f5b5a16028667e27aa21adcdbeb",
    (
        "Void tracking",
        "standard",
        1,
        400,
    ): "28efbc2a123d7da37be6a36969c07e38e79b70b30eddc8c081def8da5d89eed1",
    (
        "Void tracking",
        "standard",
        42,
        7,
    ): "a2e9a7aec4cd2c31b970073dc8a2d60b6fe751a59e9314ae83b776ffd36a0e4e",
    (
        "Void tracking",
        "standard",
        42,
        53,
    ): "cccfc6b9e33c05c7db941884519953f09fb7a889a07429f0279c08e16a9088c3",
    (
        "Void tracking",
        "standard",
        42,
        400,
    ): "f034100f85e75a47dbe3acee41602a2946ca54545ed0556931fc13a5ee9ce83a",
    (
        "Void tracking",
        "standard",
        20260803,
        7,
    ): "b568680cd62845958bddc3c867d6f5db80121f1e9a3ede45b0136cadc7d73528",
    (
        "Void tracking",
        "standard",
        20260803,
        53,
    ): "8a166828fdea97f3cf506ed45cf42215e858b423672dceae2697cccd579b453b",
    (
        "Void tracking",
        "standard",
        20260803,
        400,
    ): "cfb24e83f66ab7d6ddef09214959aa3a1b4d884fe7f70184879c5a240a2d655c",
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
        f"Output bits changed for multi-outcome/{strategy_name}/{scenario_name} "
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
