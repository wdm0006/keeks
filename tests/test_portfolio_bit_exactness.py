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
        "max_draw_down": None,
    },
    "longshot-heavy": {
        "bets": ((0.08, 10.0, 1.0), (0.12, 8.0, 1.0), (0.06, 12.0, 1.0)),
        "fee": 0.0,
        "max_draw_down": None,
    },
    "drawdown-tight": {
        "bets": ((0.60, 1.5, 1.0), (0.50, 1.8, 1.0), (0.55, 1.6, 1.0)),
        "fee": 0.0,
        "max_draw_down": 0.05,
    },
}

# Mirrors the other oracles' discipline: stateful strategies must be rebuilt
# per run because they carry state across evaluate() calls. The losing-batch
# strategy exercises the update_bankroll and record_settlement hooks and the
# aggregate safe-stake cap.
STRATEGY_FACTORIES = {
    "Flat stakes": lambda s: _FlatStakesStrategy(
        payoffs=_payoffs(s), loss=_loss(s), fraction=0.02, transaction_cost=0.0
    ),
    "Probability weighted": lambda s: _ProbabilityWeightedStrategy(
        payoffs=_payoffs(s), loss=_loss(s), aggregate=0.9, transaction_cost=0.0
    ),
    "Losing-batch tracking": lambda s: _LosingBatchTrackingStrategy(
        payoffs=_payoffs(s), loss=_loss(s), base_total=0.3, transaction_cost=0.0
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

    def __init__(self, payoffs, loss, fraction, transaction_cost=0):
        super().__init__(payoffs, loss, transaction_cost)
        self._fraction = fraction

    def evaluate(self, probabilities, _current_bankroll):
        return _validate_stake_fractions([self._fraction] * len(probabilities))


class _ProbabilityWeightedStrategy(BaseMultiOutcomeStrategy):
    """Stakes proportional to each bet's probability under a fixed aggregate."""

    def __init__(self, payoffs, loss, aggregate, transaction_cost=0):
        super().__init__(payoffs, loss, transaction_cost)
        self._aggregate = aggregate

    def evaluate(self, probabilities, _current_bankroll):
        probabilities = np.asarray(probabilities, dtype=float)
        weights = probabilities / probabilities.sum()
        return _validate_stake_fractions((self._aggregate * weights).tolist())


class _LosingBatchTrackingStrategy(BaseMultiOutcomeStrategy):
    """Grows the aggregate stake after all-losing batches, under the cap."""

    def __init__(self, payoffs, loss, base_total, transaction_cost=0):
        super().__init__(payoffs, loss, transaction_cost)
        self._base_total = base_total
        self._losing_batches = 0

    def update_bankroll(self, _current_bankroll):
        pass

    def record_settlement(self, won_bets, _return_pcts):
        staked = [won for won in won_bets if won is not None]
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
        initial_funds=INITIAL_FUNDS, max_draw_down=scenario["max_draw_down"]
    )
    simulator = PortfolioSimulator(
        bets=scenario["bets"],
        transaction_costs=scenario["fee"],
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
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Flat stakes",
        "drawdown-tight",
        0,
        53,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Flat stakes",
        "drawdown-tight",
        0,
        400,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Flat stakes",
        "drawdown-tight",
        1,
        7,
    ): "6f75aaa04902c0dec1787539c223f4ca86161100fc19e55981e63f76df217059",
    (
        "Flat stakes",
        "drawdown-tight",
        1,
        53,
    ): "5c30566710231d771eeee24e332cd2d70e3b8061e27cbe9c08d0adb2ff4ce31d",
    (
        "Flat stakes",
        "drawdown-tight",
        1,
        400,
    ): "5c30566710231d771eeee24e332cd2d70e3b8061e27cbe9c08d0adb2ff4ce31d",
    (
        "Flat stakes",
        "drawdown-tight",
        42,
        7,
    ): "df0314404bd8159e3dbb5d45800db119ec62704ac64a05fa807f7d1369345703",
    (
        "Flat stakes",
        "drawdown-tight",
        42,
        53,
    ): "df0314404bd8159e3dbb5d45800db119ec62704ac64a05fa807f7d1369345703",
    (
        "Flat stakes",
        "drawdown-tight",
        42,
        400,
    ): "df0314404bd8159e3dbb5d45800db119ec62704ac64a05fa807f7d1369345703",
    (
        "Flat stakes",
        "drawdown-tight",
        20260803,
        7,
    ): "c678a7ef8bf6c50e7f2fa07851a08474f6c0c6d0cbee09024707fd639462943f",
    (
        "Flat stakes",
        "drawdown-tight",
        20260803,
        53,
    ): "63b39e5ccd722f4259c0f5aeb7202e872460d44833af772cbacd026d3f872692",
    (
        "Flat stakes",
        "drawdown-tight",
        20260803,
        400,
    ): "63b39e5ccd722f4259c0f5aeb7202e872460d44833af772cbacd026d3f872692",
    (
        "Flat stakes",
        "longshot-heavy",
        0,
        7,
    ): "39cc3b654e721f1f44fcec5448a59e4904097e4817e84068973667e4b667bf8e",
    (
        "Flat stakes",
        "longshot-heavy",
        0,
        53,
    ): "3be2b52de5dc39c867c7e163d229ea8ef0d37ecffe99af340c38dda38932aa9e",
    (
        "Flat stakes",
        "longshot-heavy",
        0,
        400,
    ): "3d8df4ec1df691a2f20486d4d504ef3397258e5b8664347381c6e6e064151c43",
    (
        "Flat stakes",
        "longshot-heavy",
        1,
        7,
    ): "97c33199988a36465240a8e3ab452fdaf26904b68a334495f435adadbcd7d0c4",
    (
        "Flat stakes",
        "longshot-heavy",
        1,
        53,
    ): "1fa013a1e0542be50d5e6039704ca2322c7578c35c45f4fc0a0115d6b454741e",
    (
        "Flat stakes",
        "longshot-heavy",
        1,
        400,
    ): "f63c92b0b76fe307d3a94410e5b7b81716c546f9ac5586d6dbd4241c9661962c",
    (
        "Flat stakes",
        "longshot-heavy",
        42,
        7,
    ): "81245dcecf843f8f554090b70650f85148e3bf87a6a5462b073bca5d703aa754",
    (
        "Flat stakes",
        "longshot-heavy",
        42,
        53,
    ): "2eb7fab3a17d00e1e6dfd82b1f094ee4e7c677c252bbb9cf21884a1304c3e007",
    (
        "Flat stakes",
        "longshot-heavy",
        42,
        400,
    ): "500beeebcaf70fbd0f43749c8adfad3b28fabd36876dbc68859fe61778f7e2df",
    (
        "Flat stakes",
        "longshot-heavy",
        20260803,
        7,
    ): "dfb9d924a5b6aea1eea06694c4ba020f6a78221f4445420922f08ddd3766d904",
    (
        "Flat stakes",
        "longshot-heavy",
        20260803,
        53,
    ): "5242ec0b8c05ed0b46785afc089fc2d79506044adfb75b7704f31bfd78315de2",
    (
        "Flat stakes",
        "longshot-heavy",
        20260803,
        400,
    ): "1dd035c2684d70624730bb352bd8c463f3a994f5b22a97c89458622dacd719e6",
    (
        "Flat stakes",
        "standard",
        0,
        7,
    ): "09aefebb066646c7131b429b2677b3c05194919da9be9746ccc4ffeebc58291d",
    (
        "Flat stakes",
        "standard",
        0,
        53,
    ): "bb5022c71a88f54a29a97d967d6a8709eb2d2adb1e33cfa247621575e0e86a85",
    (
        "Flat stakes",
        "standard",
        0,
        400,
    ): "11efa6eebf297cc7dc552222ec6d34329eff78646a7bd69d789ca71bdc7ff60a",
    (
        "Flat stakes",
        "standard",
        1,
        7,
    ): "16583da721251b76282c5a259fce5dc37e5d55cc180201095546f1ee44821046",
    (
        "Flat stakes",
        "standard",
        1,
        53,
    ): "c805a8f2e37eb87489157ebde3f8f2d3ed8b95f3d74051099b114965fd88569d",
    (
        "Flat stakes",
        "standard",
        1,
        400,
    ): "6582927272a5b327ef4f4c779bb59371fcb028f10b696297f998461d1648d102",
    (
        "Flat stakes",
        "standard",
        42,
        7,
    ): "dde946a648da064d2104420371301f4edc2847aed8dfd04ca742b924b932e340",
    (
        "Flat stakes",
        "standard",
        42,
        53,
    ): "b0b5bb61fd3fefd8e4ef1907c51e12d67908b4d4be34f8696424ac28c59388f0",
    (
        "Flat stakes",
        "standard",
        42,
        400,
    ): "f9a88eb95b0a6cc483b333505e50ad66da67bb3cb72ac60290d8fca18b08aaf2",
    (
        "Flat stakes",
        "standard",
        20260803,
        7,
    ): "8f975bf045054197053042d5119c2ec659602d2c3cf1d8bdf970cf5b74c2558d",
    (
        "Flat stakes",
        "standard",
        20260803,
        53,
    ): "41733509754b424fc1fe267fa24e1d62a801613309b1c3a46626c48b851d7279",
    (
        "Flat stakes",
        "standard",
        20260803,
        400,
    ): "49146f966816b75dc47310d5990322c4b1b8dd7d9b9eec4c32ee6bc15328fcdc",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        0,
        7,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        0,
        53,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        0,
        400,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        1,
        7,
    ): "12827492c548366730a1ea711528261b6a84b217bdfe59883ca66e0af452b421",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        1,
        53,
    ): "07bdc64a869a4defaa616aaba028dd22bf0903edc01655a8d37d6583de7ca4d4",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        1,
        400,
    ): "07bdc64a869a4defaa616aaba028dd22bf0903edc01655a8d37d6583de7ca4d4",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        42,
        7,
    ): "a86deaeb7d89ef118dce84b86938a9efd8279474eb06c2e59736b30d63f35380",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        42,
        53,
    ): "a86deaeb7d89ef118dce84b86938a9efd8279474eb06c2e59736b30d63f35380",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        42,
        400,
    ): "a86deaeb7d89ef118dce84b86938a9efd8279474eb06c2e59736b30d63f35380",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        20260803,
        7,
    ): "0fc7eab7ebfdcda0996f41e42ab72de1ed6df4ee89f19fdde64ca64261390902",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        20260803,
        53,
    ): "93b2b9e6e84ed817a26e0d240281c90e74ca9d59fb462579e043d16d4cac48d5",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        20260803,
        400,
    ): "93b2b9e6e84ed817a26e0d240281c90e74ca9d59fb462579e043d16d4cac48d5",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        0,
        7,
    ): "7aa604c963a17b63170eb81d7321829d9585c4f884490cb164212e0413ede9af",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        0,
        53,
    ): "53d80f3c5beae1ba6d838e90ef162ddee72e7d832e3a258aa9a4921504e22703",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        0,
        400,
    ): "53d80f3c5beae1ba6d838e90ef162ddee72e7d832e3a258aa9a4921504e22703",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        1,
        7,
    ): "29c825206517a5d57200471dbceb6e1619c6da72395f6838f7201e130641cc1a",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        1,
        53,
    ): "901fd7ddd205a8038d868fa57001dd04a4ade16de36c2f13015f037bfcbe0867",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        1,
        400,
    ): "901fd7ddd205a8038d868fa57001dd04a4ade16de36c2f13015f037bfcbe0867",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        42,
        7,
    ): "9b0f9e1e394c26dc4d5119f884864bca2d6de91569cd67992b0a57116f6ead4b",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        42,
        53,
    ): "c9d6ee62319959bf58130ce954213ff2323723ebe18535ce3c72ad923584cf22",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        42,
        400,
    ): "c9d6ee62319959bf58130ce954213ff2323723ebe18535ce3c72ad923584cf22",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        20260803,
        7,
    ): "392d63b852e1c73a3c8337f3ef10965b998d4a8883752dac42fc26f225dd73b9",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        20260803,
        53,
    ): "fc73735c87069a7f71b36c6465dd7bd8613a4145f2e1102ee1bf6b2c4cee99c1",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        20260803,
        400,
    ): "fc73735c87069a7f71b36c6465dd7bd8613a4145f2e1102ee1bf6b2c4cee99c1",
    (
        "Losing-batch tracking",
        "standard",
        0,
        7,
    ): "c3b6081a756f6c75da80e25e6a4cab7e76fd6dd0cb5ef63a910a2f5e10bd4649",
    (
        "Losing-batch tracking",
        "standard",
        0,
        53,
    ): "75bc23cf1162f5bbcdc805dbea742ac1c858b952f4da1474b6552dae6cdbdded",
    (
        "Losing-batch tracking",
        "standard",
        0,
        400,
    ): "088b94787746a550deee71f8c749f557aee3a85f3dd19c1b3630c03794be95ed",
    (
        "Losing-batch tracking",
        "standard",
        1,
        7,
    ): "b50ca0e3b2903591ca6b110aff103b96388058b69ea8b7dea90bfc587f5d8005",
    (
        "Losing-batch tracking",
        "standard",
        1,
        53,
    ): "49211bd7c358813bcb74a7df6dda3c8ee56aa4a191c09060b0cc8150a3c1e578",
    (
        "Losing-batch tracking",
        "standard",
        1,
        400,
    ): "bca20d7b3826fd3e6a8a91ac82e0f70403ded5439ff0f4ebc698f8100be84a39",
    (
        "Losing-batch tracking",
        "standard",
        42,
        7,
    ): "0f39a1168c0ced96ea2a4a92f307d3226ec9c61aefb20bdb54b579a16ecb3c9b",
    (
        "Losing-batch tracking",
        "standard",
        42,
        53,
    ): "1fb1c8ac07007382fcaefabe39348d0b38e57c421174997d8a2ffa9b5d037d7a",
    (
        "Losing-batch tracking",
        "standard",
        42,
        400,
    ): "a699fc40d1e2648e41bb635476005eff36071d3e7adce8e95f165568167f69b4",
    (
        "Losing-batch tracking",
        "standard",
        20260803,
        7,
    ): "78ed8c26ac2a57d28b1d60fb84d491d38014a663d86b6515b0b560174f7fb5a6",
    (
        "Losing-batch tracking",
        "standard",
        20260803,
        53,
    ): "975d05175ec0c09690a29b96f9645cc6b11a8ea13bf53d60f72c85d8810b59b0",
    (
        "Losing-batch tracking",
        "standard",
        20260803,
        400,
    ): "a4b4618fb2ca4dde314f8795f8da1409294651964e5e44281a6134fe4a4e4445",
    (
        "Probability weighted",
        "drawdown-tight",
        0,
        7,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Probability weighted",
        "drawdown-tight",
        0,
        53,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Probability weighted",
        "drawdown-tight",
        0,
        400,
    ): "0ea68d787463f3ae61be487d44d89ee1089343b138ec38b5cd1cb7a746b9c87e",
    (
        "Probability weighted",
        "drawdown-tight",
        1,
        7,
    ): "96edb8c1a9bbc6bf0c15aec491f232340433787726063cecc9ffa16fcb948a36",
    (
        "Probability weighted",
        "drawdown-tight",
        1,
        53,
    ): "96edb8c1a9bbc6bf0c15aec491f232340433787726063cecc9ffa16fcb948a36",
    (
        "Probability weighted",
        "drawdown-tight",
        1,
        400,
    ): "96edb8c1a9bbc6bf0c15aec491f232340433787726063cecc9ffa16fcb948a36",
    (
        "Probability weighted",
        "drawdown-tight",
        42,
        7,
    ): "382a506c05b0e5803a6a017a803aad773ccf4e040ce97e232da70730b5b1426e",
    (
        "Probability weighted",
        "drawdown-tight",
        42,
        53,
    ): "382a506c05b0e5803a6a017a803aad773ccf4e040ce97e232da70730b5b1426e",
    (
        "Probability weighted",
        "drawdown-tight",
        42,
        400,
    ): "382a506c05b0e5803a6a017a803aad773ccf4e040ce97e232da70730b5b1426e",
    (
        "Probability weighted",
        "drawdown-tight",
        20260803,
        7,
    ): "8d4558039a2f9e6461df65e82085cea76d62372a5fc56e5ce470e540a23bf9c1",
    (
        "Probability weighted",
        "drawdown-tight",
        20260803,
        53,
    ): "8d4558039a2f9e6461df65e82085cea76d62372a5fc56e5ce470e540a23bf9c1",
    (
        "Probability weighted",
        "drawdown-tight",
        20260803,
        400,
    ): "8d4558039a2f9e6461df65e82085cea76d62372a5fc56e5ce470e540a23bf9c1",
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
    ): "e48a80b8925bf8181d9a6569a0493e66e0728ff62a321aa05eff54fde0821e8c",
    (
        "Probability weighted",
        "longshot-heavy",
        1,
        53,
    ): "e48a80b8925bf8181d9a6569a0493e66e0728ff62a321aa05eff54fde0821e8c",
    (
        "Probability weighted",
        "longshot-heavy",
        1,
        400,
    ): "e48a80b8925bf8181d9a6569a0493e66e0728ff62a321aa05eff54fde0821e8c",
    (
        "Probability weighted",
        "longshot-heavy",
        42,
        7,
    ): "d94fc961acc2d4c32e34c03f268e4b53b99ee7f240cf95e77eaf84bd83c1ebcb",
    (
        "Probability weighted",
        "longshot-heavy",
        42,
        53,
    ): "508539a862bf4e983b31c5160c1db4ad73733dd93fcb1b7aefe2be1976c006e5",
    (
        "Probability weighted",
        "longshot-heavy",
        42,
        400,
    ): "508539a862bf4e983b31c5160c1db4ad73733dd93fcb1b7aefe2be1976c006e5",
    (
        "Probability weighted",
        "longshot-heavy",
        20260803,
        7,
    ): "1de4bbc905bd078aa5b6aa7bda96956282b5f8825ae69ad864ad1c805de1f7f1",
    (
        "Probability weighted",
        "longshot-heavy",
        20260803,
        53,
    ): "f8068ded9ba1aff23c3d4d96ae99c5d4c19ed8895e7c09d535e4078626747936",
    (
        "Probability weighted",
        "longshot-heavy",
        20260803,
        400,
    ): "f8068ded9ba1aff23c3d4d96ae99c5d4c19ed8895e7c09d535e4078626747936",
    (
        "Probability weighted",
        "standard",
        0,
        7,
    ): "afde41bf23b21ed57b168119da73aa6a412ce7280997b3571c370feb776c77c5",
    (
        "Probability weighted",
        "standard",
        0,
        53,
    ): "d791ae0dae8f5bc05b3c9445e07d0f44f2ceef61212e4b17c82a551c515c7267",
    (
        "Probability weighted",
        "standard",
        0,
        400,
    ): "d791ae0dae8f5bc05b3c9445e07d0f44f2ceef61212e4b17c82a551c515c7267",
    (
        "Probability weighted",
        "standard",
        1,
        7,
    ): "5b312be2fa38d56145decbf56ea544175c4337bc97429a72fb886b678bfda769",
    (
        "Probability weighted",
        "standard",
        1,
        53,
    ): "e8adc2f112fb7186521f5b07e251f3ccad21d994bc444dbe5b6d606c6aedff48",
    (
        "Probability weighted",
        "standard",
        1,
        400,
    ): "d38080216ae24568373a2ac5468b70d06e9a6a01cf095f5af18de0f95f57d354",
    (
        "Probability weighted",
        "standard",
        42,
        7,
    ): "ac57c4c91169e49996e29a0d4dc2cb82fc9909e0e69f5ce5c570dc7b10ffd83f",
    (
        "Probability weighted",
        "standard",
        42,
        53,
    ): "764a50b0585eb23489266d7a1f1980c792f4848b03ac2774adef99862f5a5d4d",
    (
        "Probability weighted",
        "standard",
        42,
        400,
    ): "764a50b0585eb23489266d7a1f1980c792f4848b03ac2774adef99862f5a5d4d",
    (
        "Probability weighted",
        "standard",
        20260803,
        7,
    ): "cd51e13009ba713e4ce1fe3be893ef8154008c010e74f14b6c3e063fb73fd585",
    (
        "Probability weighted",
        "standard",
        20260803,
        53,
    ): "9fde8b5b70e1eb0ca82db9e7c541cce7dbadc2b6fc2d83a5f4248d32ffb03e2e",
    (
        "Probability weighted",
        "standard",
        20260803,
        400,
    ): "d8ddec7ba81f329bf4a679b37b74f4b89fcbdf3845993a1b7845573945d808eb",
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
