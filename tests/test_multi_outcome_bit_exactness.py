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
        "max_draw_down": None,
    },
    "residual-heavy": {
        "payoffs": (2.5, 2.2, 3.0),
        "loss": 1.0,
        "fee": 0.0,
        "probabilities": (0.2, 0.15, 0.1),
        "max_draw_down": None,
    },
    "fee-heavy": {
        "payoffs": (1.2, 1.1, 1.3),
        "loss": 0.7,
        "fee": 0.5,
        "probabilities": (0.45, 0.3, 0.2),
        "max_draw_down": 0.05,
    },
}

# Mirrors the binary oracle's discipline: stateful strategies must be rebuilt
# per run because they carry state across evaluate() calls. The void-tracking
# strategy exercises the update_bankroll and record_settlement hooks and the
# aggregate safe-stake cap.
STRATEGY_FACTORIES = {
    "Flat stakes": lambda s: _FlatStakesStrategy(
        payoffs=s["payoffs"], loss=s["loss"], fraction=0.02, transaction_cost=0.0
    ),
    "Probability weighted": lambda s: _ProbabilityWeightedStrategy(
        payoffs=s["payoffs"], loss=s["loss"], aggregate=0.9, transaction_cost=0.0
    ),
    "Void tracking": lambda s: _VoidTrackingStrategy(
        payoffs=s["payoffs"], loss=s["loss"], base_total=0.3, transaction_cost=0.0
    ),
}

# Grid of seeds and trial counts. Short runs catch early stops and the no-bet
# path; the 400-trial runs compound.
SEEDS = (0, 1, 42, 20260803)
TRIAL_COUNTS = (7, 53, 400)


class _FlatStakesStrategy(BaseMultiOutcomeStrategy):
    """The same fraction on every leg, every trial."""

    def __init__(self, payoffs, loss, fraction, transaction_cost=0):
        super().__init__(payoffs, loss, transaction_cost)
        self._fraction = fraction

    def evaluate(self, probabilities, _current_bankroll):
        return _validate_stake_fractions([self._fraction] * len(probabilities))


class _ProbabilityWeightedStrategy(BaseMultiOutcomeStrategy):
    """Stakes proportional to each leg's probability under a fixed aggregate."""

    def __init__(self, payoffs, loss, aggregate, transaction_cost=0):
        super().__init__(payoffs, loss, transaction_cost)
        self._aggregate = aggregate

    def evaluate(self, probabilities, _current_bankroll):
        probabilities = np.asarray(probabilities, dtype=float)
        weights = probabilities / probabilities.sum()
        return _validate_stake_fractions((self._aggregate * weights).tolist())


class _VoidTrackingStrategy(BaseMultiOutcomeStrategy):
    """Grows the aggregate stake after voids, under the safe-stake cap."""

    def __init__(self, payoffs, loss, base_total, transaction_cost=0):
        super().__init__(payoffs, loss, transaction_cost)
        self._base_total = base_total
        self._voids = 0

    def update_bankroll(self, _current_bankroll):
        pass

    def record_settlement(self, won_leg, _return_pcts):
        if won_leg is None:
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
        initial_funds=INITIAL_FUNDS, max_draw_down=scenario["max_draw_down"]
    )
    simulator = RepeatedMultiOutcomeSimulator(
        payoffs=scenario["payoffs"],
        loss=scenario["loss"],
        transaction_costs=scenario["fee"],
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
    ): "0f0545174fb353533a4854e72b1f0a5871ea83bd5d9ea0e5c2c748e54522e406",
    (
        "Flat stakes",
        "fee-heavy",
        0,
        53,
    ): "ebd67d75c2c200388fd5fd63b9899ac6f57b3ec810aed51f04c27ac9fda4350f",
    (
        "Flat stakes",
        "fee-heavy",
        0,
        400,
    ): "918355aadd7e14560d61741b58769629c1d2f623b3c1d4a1546a00080733140b",
    (
        "Flat stakes",
        "fee-heavy",
        1,
        7,
    ): "c8e79c50429c9ddb5da1217cb5a3a6fae3ba396279052e0e31eb2be6fe2a944d",
    (
        "Flat stakes",
        "fee-heavy",
        1,
        53,
    ): "e4bd3b0b4ca33a5268ccce2596fa663fe5e39437439070e4aadaee60c6fbc023",
    (
        "Flat stakes",
        "fee-heavy",
        1,
        400,
    ): "4c78d76ab940371a244bf3532e412e427be712b5636233a25d6ec33a67f29c30",
    (
        "Flat stakes",
        "fee-heavy",
        42,
        7,
    ): "7e8d7daba6f076b3477dca67d9d3183f8557c38283cf1962a460e5eb3c0b1727",
    (
        "Flat stakes",
        "fee-heavy",
        42,
        53,
    ): "680fe5deddb7e0862111859280125e3abc8aa7aa5be002ac662fe2d87c0a756a",
    (
        "Flat stakes",
        "fee-heavy",
        42,
        400,
    ): "f286510b9fa1ca082a190ad1326ada36cd114a67df892b4c6a6962dfacff4d9c",
    (
        "Flat stakes",
        "fee-heavy",
        20260803,
        7,
    ): "1e73a30ed700059414ecf3e8336b0d859ab200dcc116ad7286f1c77068ad6f8c",
    (
        "Flat stakes",
        "fee-heavy",
        20260803,
        53,
    ): "5fc8e41202fdb85c5a81e70c11c1c874037547961d4906a70dfa32240d865c5d",
    (
        "Flat stakes",
        "fee-heavy",
        20260803,
        400,
    ): "1839021363551b3ae7a48c2d1f45b8ef96e16c7797041b0e0bb2904035a5417a",
    (
        "Flat stakes",
        "residual-heavy",
        0,
        7,
    ): "bf10efac29bed077816f6dbe9dab7b6402758f1c9fd7cee968c1e3b968e6c303",
    (
        "Flat stakes",
        "residual-heavy",
        0,
        53,
    ): "9836c9c3825b028aae9e06b7e5852f2139ed85db836cb34b4ec35ec194762f66",
    (
        "Flat stakes",
        "residual-heavy",
        0,
        400,
    ): "d0247091ea47b062813fa411b6a65fc8841bcb0fd594336cf190168a4ad52c53",
    (
        "Flat stakes",
        "residual-heavy",
        1,
        7,
    ): "6d2878b84fc58c3d7be3eff3c30b05d4dbc97b404ef8e9292b54e321cc111c49",
    (
        "Flat stakes",
        "residual-heavy",
        1,
        53,
    ): "c73ae27242ef87c24c9f31e5328306769746ebb763c2c38feec5e11377697248",
    (
        "Flat stakes",
        "residual-heavy",
        1,
        400,
    ): "98731315e2855c0af0ed216c958069193c43b2c8164ea8c409cb9a524977860f",
    (
        "Flat stakes",
        "residual-heavy",
        42,
        7,
    ): "f233b5195db0d1ff40bc608746c02ce9fb49ca4c6f159ee551bff2db7aaedf21",
    (
        "Flat stakes",
        "residual-heavy",
        42,
        53,
    ): "fcfd4068745c1e6c26363a1f61b03803f9fe78fe1edf4a4037b8f587abd99a3f",
    (
        "Flat stakes",
        "residual-heavy",
        42,
        400,
    ): "40516e8f43563b3b814a89fbd3f954c67a7d5ac4c405cd16f0c75248fcd34f5c",
    (
        "Flat stakes",
        "residual-heavy",
        20260803,
        7,
    ): "a5fd54ff20a00258319b8d23b5d5374a9da6b7b85fff274a4bdf1c57ab7e83c6",
    (
        "Flat stakes",
        "residual-heavy",
        20260803,
        53,
    ): "822f17c5026a8e90c41c287f1afa7c4443b810d55cd23b5bc0ef1c717f4258a4",
    (
        "Flat stakes",
        "residual-heavy",
        20260803,
        400,
    ): "2b35d4724eafaacbeb7e7d65ad9b7f23312dfffcc719f9afb66824300fa1400f",
    (
        "Flat stakes",
        "standard",
        0,
        7,
    ): "417f5bfb4fb47518c3c8efc81396aa1d26486958b9add2f2718e4d167bb0021d",
    (
        "Flat stakes",
        "standard",
        0,
        53,
    ): "7d062b9ee29613cc2292ad4a7c9465ab788d207f59a29a607c9c1c9cef46cc60",
    (
        "Flat stakes",
        "standard",
        0,
        400,
    ): "597588a193ec98479f79ff1b45b06ac9a0a89c5182abe38f1ce40fa886d76026",
    (
        "Flat stakes",
        "standard",
        1,
        7,
    ): "d93620cb1199db4e1bb65f65ed9beb3f4aacd30cb578a627251fa1de01b7f855",
    (
        "Flat stakes",
        "standard",
        1,
        53,
    ): "4084dddee4fb820f1f01e889c3f3d44dad217b54cf79f5be4726d25c5ec0596c",
    (
        "Flat stakes",
        "standard",
        1,
        400,
    ): "6960fb992ec382728d776d5bcf6e344c98d8f24ce454602ea1594997a4008cf2",
    (
        "Flat stakes",
        "standard",
        42,
        7,
    ): "58f81b4c7000444b7c1a2015e352fd93d2454ce5b523e00635bfd89d85e0b7cf",
    (
        "Flat stakes",
        "standard",
        42,
        53,
    ): "3f1c9e9d540fcafad23839f5bfd236c64f18539bd4c5538b55b92b500b039947",
    (
        "Flat stakes",
        "standard",
        42,
        400,
    ): "56b83fb702db4e102a1a2e92c78d38be23ff6a16b1dc74cf99df23d4360146b9",
    (
        "Flat stakes",
        "standard",
        20260803,
        7,
    ): "18a354fbf12fb5757e578bd581e61a6a389d3cc96e9d60d33041aeb783521ae1",
    (
        "Flat stakes",
        "standard",
        20260803,
        53,
    ): "2c269d1b549ce52c6fd0b1e432693a1d4cd955541a2cb3c00e4432dc3358b311",
    (
        "Flat stakes",
        "standard",
        20260803,
        400,
    ): "142e888cf3ae9e71c663de8c77c796fe715b389cd27efe1f8e43cec4e9f4e4db",
    (
        "Probability weighted",
        "fee-heavy",
        0,
        7,
    ): "df5ed795f6d766122a91adbea1382251b830efd099868a072b400219b1e4eaf8",
    (
        "Probability weighted",
        "fee-heavy",
        0,
        53,
    ): "df5ed795f6d766122a91adbea1382251b830efd099868a072b400219b1e4eaf8",
    (
        "Probability weighted",
        "fee-heavy",
        0,
        400,
    ): "df5ed795f6d766122a91adbea1382251b830efd099868a072b400219b1e4eaf8",
    (
        "Probability weighted",
        "fee-heavy",
        1,
        7,
    ): "cd1a1bbcab8bcead3128dd6531bd64067050aec133624fd02020e01c925768a7",
    (
        "Probability weighted",
        "fee-heavy",
        1,
        53,
    ): "cd1a1bbcab8bcead3128dd6531bd64067050aec133624fd02020e01c925768a7",
    (
        "Probability weighted",
        "fee-heavy",
        1,
        400,
    ): "cd1a1bbcab8bcead3128dd6531bd64067050aec133624fd02020e01c925768a7",
    (
        "Probability weighted",
        "fee-heavy",
        42,
        7,
    ): "df5ed795f6d766122a91adbea1382251b830efd099868a072b400219b1e4eaf8",
    (
        "Probability weighted",
        "fee-heavy",
        42,
        53,
    ): "df5ed795f6d766122a91adbea1382251b830efd099868a072b400219b1e4eaf8",
    (
        "Probability weighted",
        "fee-heavy",
        42,
        400,
    ): "df5ed795f6d766122a91adbea1382251b830efd099868a072b400219b1e4eaf8",
    (
        "Probability weighted",
        "fee-heavy",
        20260803,
        7,
    ): "974108c79cba5374ba73d244bd004c1745c42e30aaf00536920664d4c7722450",
    (
        "Probability weighted",
        "fee-heavy",
        20260803,
        53,
    ): "974108c79cba5374ba73d244bd004c1745c42e30aaf00536920664d4c7722450",
    (
        "Probability weighted",
        "fee-heavy",
        20260803,
        400,
    ): "974108c79cba5374ba73d244bd004c1745c42e30aaf00536920664d4c7722450",
    (
        "Probability weighted",
        "residual-heavy",
        0,
        7,
    ): "acde7a504b891f73f6b53919130414f77c0fb18a6048a83132355426e8492e57",
    (
        "Probability weighted",
        "residual-heavy",
        0,
        53,
    ): "78f52147de1881020a9b6923167ae76968eeaf69a5ef9815c76302323dca1e22",
    (
        "Probability weighted",
        "residual-heavy",
        0,
        400,
    ): "4d2730eee4549b470b15e0627b11e529127a8a2a02337b329004c55913329f79",
    (
        "Probability weighted",
        "residual-heavy",
        1,
        7,
    ): "de2d86e28e0bfd8e063770d7f0c45123f51443984bee341e7dc159f55cdbae7a",
    (
        "Probability weighted",
        "residual-heavy",
        1,
        53,
    ): "7a1ecee3f79edf826caf36ccc0b84a9b7b8e3c6bc3cb8c941764a12ee582ccbb",
    (
        "Probability weighted",
        "residual-heavy",
        1,
        400,
    ): "45a054f3c45e8507677f86bbc4d8cfa76bd4b54aeb7b3152faa7fc036b570c96",
    (
        "Probability weighted",
        "residual-heavy",
        42,
        7,
    ): "47b6c73a7380400924319a6a2127faefc5bca9d82da3dcc5383dc66cfd887851",
    (
        "Probability weighted",
        "residual-heavy",
        42,
        53,
    ): "53e3dd01ab312251fbca8d67718d1e8007ce6ec68bf7616e31454b5a518b18fe",
    (
        "Probability weighted",
        "residual-heavy",
        42,
        400,
    ): "6371eb91c0c82416e97fee88dbb43451cdd688ca84f2dd0d992061b804e8544a",
    (
        "Probability weighted",
        "residual-heavy",
        20260803,
        7,
    ): "b4f17ede78627c8dcf82df83e64a68031bafb1b3c6c66458f8d1b22d510e7ead",
    (
        "Probability weighted",
        "residual-heavy",
        20260803,
        53,
    ): "81989f5d544504c8d906353040767ee0d9a65b4be71d0e31dcffac1d5dac980a",
    (
        "Probability weighted",
        "residual-heavy",
        20260803,
        400,
    ): "ac7387903317734f7244f4c983c51dd8e87bd56d304f0bc442a82536fc3fdb47",
    (
        "Probability weighted",
        "standard",
        0,
        7,
    ): "efb5f6c8607663a9c31141d09e6136e36e66df59c947d1eb869a6d0a204f227e",
    (
        "Probability weighted",
        "standard",
        0,
        53,
    ): "46c030336c14f5985e9730d32614ac71d6618288ad2d9e1f3293bf6f4f827195",
    (
        "Probability weighted",
        "standard",
        0,
        400,
    ): "4131f59f47288508499f9948009de9d1779ae90f4c49f6a91d12a2ed015438ba",
    (
        "Probability weighted",
        "standard",
        1,
        7,
    ): "14a40d3dfbcb2b7db8ad0257ac2816c682634affcc82cb9e55a73f09c847c001",
    (
        "Probability weighted",
        "standard",
        1,
        53,
    ): "f39c2b9f829d52c82597ef0851deeadcb0e5dcafba08d5211791a59c550c7f3d",
    (
        "Probability weighted",
        "standard",
        1,
        400,
    ): "d664f78ad2f5664ae232d586f681681688342d0a23084c4085905090e93eaa06",
    (
        "Probability weighted",
        "standard",
        42,
        7,
    ): "6692d1bd6af59be3fcaafd102188c3d942a4b005d196edad1ff299d54cf67efb",
    (
        "Probability weighted",
        "standard",
        42,
        53,
    ): "92dbedb00c47e7993216b1ad416404571b2d825bfbd7e317521e189d315d1586",
    (
        "Probability weighted",
        "standard",
        42,
        400,
    ): "a7bdd698b17f83f9b17f71f27f73f8b0e64b028e982ce679b212d1824b156c4e",
    (
        "Probability weighted",
        "standard",
        20260803,
        7,
    ): "9c373968e2c6362961170a3b947336bf4ba917cd02c2ecf7823e2f02dc8f03a9",
    (
        "Probability weighted",
        "standard",
        20260803,
        53,
    ): "19b8915368c6e4f6471e7277c5226c312f91bf60699ccd42747a25f9886fddb4",
    (
        "Probability weighted",
        "standard",
        20260803,
        400,
    ): "6bbfbf6c7e76dd0b709255b3cb05a102f33ee123a07962b39c9aa103bce16891",
    (
        "Void tracking",
        "fee-heavy",
        0,
        7,
    ): "90423aa537cb40b0cba34fb7e5129ef24bc6601b8c2464e5560c0a32792c07e8",
    (
        "Void tracking",
        "fee-heavy",
        0,
        53,
    ): "90423aa537cb40b0cba34fb7e5129ef24bc6601b8c2464e5560c0a32792c07e8",
    (
        "Void tracking",
        "fee-heavy",
        0,
        400,
    ): "90423aa537cb40b0cba34fb7e5129ef24bc6601b8c2464e5560c0a32792c07e8",
    (
        "Void tracking",
        "fee-heavy",
        1,
        7,
    ): "0465046989a3610fb8db7098c69b323bee7d0bd1ebb3f7b8e44a2f5c2e48366f",
    (
        "Void tracking",
        "fee-heavy",
        1,
        53,
    ): "0465046989a3610fb8db7098c69b323bee7d0bd1ebb3f7b8e44a2f5c2e48366f",
    (
        "Void tracking",
        "fee-heavy",
        1,
        400,
    ): "0465046989a3610fb8db7098c69b323bee7d0bd1ebb3f7b8e44a2f5c2e48366f",
    (
        "Void tracking",
        "fee-heavy",
        42,
        7,
    ): "90423aa537cb40b0cba34fb7e5129ef24bc6601b8c2464e5560c0a32792c07e8",
    (
        "Void tracking",
        "fee-heavy",
        42,
        53,
    ): "90423aa537cb40b0cba34fb7e5129ef24bc6601b8c2464e5560c0a32792c07e8",
    (
        "Void tracking",
        "fee-heavy",
        42,
        400,
    ): "90423aa537cb40b0cba34fb7e5129ef24bc6601b8c2464e5560c0a32792c07e8",
    (
        "Void tracking",
        "fee-heavy",
        20260803,
        7,
    ): "477874d1047f2b0a631c3fbafee4f37ae596b8858bfeb700c9c526eedf61b5b3",
    (
        "Void tracking",
        "fee-heavy",
        20260803,
        53,
    ): "477874d1047f2b0a631c3fbafee4f37ae596b8858bfeb700c9c526eedf61b5b3",
    (
        "Void tracking",
        "fee-heavy",
        20260803,
        400,
    ): "477874d1047f2b0a631c3fbafee4f37ae596b8858bfeb700c9c526eedf61b5b3",
    (
        "Void tracking",
        "residual-heavy",
        0,
        7,
    ): "19890b1fc5af3fc2098f99e4a9da9b402ce2a536dfd8a2d80cfdeb21048670ff",
    (
        "Void tracking",
        "residual-heavy",
        0,
        53,
    ): "b6426064a35422e87d140d46798a9bee578427755553183c92ecf3c347cbc245",
    (
        "Void tracking",
        "residual-heavy",
        0,
        400,
    ): "f9dce1457b84836cc6d94c6e016aea5f99cd5361de6baecd9fbcd10e2835a84c",
    (
        "Void tracking",
        "residual-heavy",
        1,
        7,
    ): "0a8e2257a4a39bc5e9304a84c2e7baed6e3159e5bd5c35e774a5b99bc7fda80f",
    (
        "Void tracking",
        "residual-heavy",
        1,
        53,
    ): "5a7d88f537ac202edd9bd514d5eba3af372bb76166787282801c7a4827d4c611",
    (
        "Void tracking",
        "residual-heavy",
        1,
        400,
    ): "0cd3f1b350e3134470b66c492b8907bf5fe6767a9d1e3afdfde8803586dc8949",
    (
        "Void tracking",
        "residual-heavy",
        42,
        7,
    ): "bc80c54049c920d502913f222854d6bf198a5bae30578a7d1ba4c5b93eef3800",
    (
        "Void tracking",
        "residual-heavy",
        42,
        53,
    ): "1f649980e7bb0d56021742906e4a75b39dedc9df55565f3286430f50554c9d09",
    (
        "Void tracking",
        "residual-heavy",
        42,
        400,
    ): "37d75ff377cf4da66536e205f39986879e5d449c700166cf4d38f47060ae4c9b",
    (
        "Void tracking",
        "residual-heavy",
        20260803,
        7,
    ): "6115667946fb9b63058b47922a24b6949dd91f9d537605b0e27e79895d657018",
    (
        "Void tracking",
        "residual-heavy",
        20260803,
        53,
    ): "b1e83d974476717d0b26afe0a794e41d69dd1a26a0b6d4c7e53f3e2d2ee4c55f",
    (
        "Void tracking",
        "residual-heavy",
        20260803,
        400,
    ): "f28026be9edbb7cc47e4ae9ece331a3dcc3acff93a2a44755c5634672b940a14",
    (
        "Void tracking",
        "standard",
        0,
        7,
    ): "69599e56064c4c774a8adce80f8f00118692801efa1f57b87b3d5c6186b92ddb",
    (
        "Void tracking",
        "standard",
        0,
        53,
    ): "f4d1041a98485007f3568228817e3ffe7f08b04093595c83605f3270de52ca8e",
    (
        "Void tracking",
        "standard",
        0,
        400,
    ): "b2740d15cb9da9aa4a2a754103a23c5a31096fc4ed9b45a6409d7415eec93997",
    (
        "Void tracking",
        "standard",
        1,
        7,
    ): "4b162983d99174fa37dc2fd12fbfc18ed228220a699dd3e652be4273125d96ac",
    (
        "Void tracking",
        "standard",
        1,
        53,
    ): "6a8b65b35845fa6a3fdfa66bf78e829b67dcba420bd3a9a89f6820242aebd2cd",
    (
        "Void tracking",
        "standard",
        1,
        400,
    ): "b7e2c6ae5246942afc569f5d8b7f6d005419ccdb45a0d0a974777ae20f44142b",
    (
        "Void tracking",
        "standard",
        42,
        7,
    ): "ef49c668ee6bf0e6e9aa11dfb81727801823bc8e46fe04efb0e6f55995135cb1",
    (
        "Void tracking",
        "standard",
        42,
        53,
    ): "9a48d40f991b8d8333997bded6f528c86c906fd58714a3978e19a5355e8c2f6b",
    (
        "Void tracking",
        "standard",
        42,
        400,
    ): "81b6a6bda848d0386a5e299d4e0e3645f81d236eea3d75032b0a0e1e9010d1a0",
    (
        "Void tracking",
        "standard",
        20260803,
        7,
    ): "00a2b43202cc61044ca08dcbb815c6d1d7bd114d1b4f5d914ed6b093137ef20d",
    (
        "Void tracking",
        "standard",
        20260803,
        53,
    ): "7d2a32da171193dccee1bf46ec123eda737c2afc894b6e4d7ab32191a8efcd3e",
    (
        "Void tracking",
        "standard",
        20260803,
        400,
    ): "1230462e63ff96aa7f84d63aaefb48f875a798eff7049db9e2682705a497b31e",
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
