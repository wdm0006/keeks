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
    ): "f990012c595af1e7e39346927afc91366cb5ab7165c7b78141778d6b4f6a39d0",
    (
        "Flat stakes",
        "drawdown-tight",
        0,
        53,
    ): "f990012c595af1e7e39346927afc91366cb5ab7165c7b78141778d6b4f6a39d0",
    (
        "Flat stakes",
        "drawdown-tight",
        0,
        400,
    ): "f990012c595af1e7e39346927afc91366cb5ab7165c7b78141778d6b4f6a39d0",
    (
        "Flat stakes",
        "drawdown-tight",
        1,
        7,
    ): "ae8e1d803306b1bf13fe124aca0108d18e0de09e86684ed49ca5851c8db4eeee",
    (
        "Flat stakes",
        "drawdown-tight",
        1,
        53,
    ): "ae8e1d803306b1bf13fe124aca0108d18e0de09e86684ed49ca5851c8db4eeee",
    (
        "Flat stakes",
        "drawdown-tight",
        1,
        400,
    ): "ae8e1d803306b1bf13fe124aca0108d18e0de09e86684ed49ca5851c8db4eeee",
    (
        "Flat stakes",
        "drawdown-tight",
        42,
        7,
    ): "bc11e1224b3eba1d85e74e87e517f98b73b3cab00d6ed967dc3007e38f3b83d7",
    (
        "Flat stakes",
        "drawdown-tight",
        42,
        53,
    ): "23efe862da7b2114dc3348d5a7b9506d15b6a35514877decc072a63a11a9ce52",
    (
        "Flat stakes",
        "drawdown-tight",
        42,
        400,
    ): "23efe862da7b2114dc3348d5a7b9506d15b6a35514877decc072a63a11a9ce52",
    (
        "Flat stakes",
        "drawdown-tight",
        20260803,
        7,
    ): "65d88dcb02bd746c5eef4a6ad21abd23acacdacb63f9270eb5b2ee0de315b9e8",
    (
        "Flat stakes",
        "drawdown-tight",
        20260803,
        53,
    ): "65d88dcb02bd746c5eef4a6ad21abd23acacdacb63f9270eb5b2ee0de315b9e8",
    (
        "Flat stakes",
        "drawdown-tight",
        20260803,
        400,
    ): "65d88dcb02bd746c5eef4a6ad21abd23acacdacb63f9270eb5b2ee0de315b9e8",
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
    ): "350c6c9438aeda300c41363939815e9bfa46932d382b804392408c7d95bcd3f4",
    (
        "Flat stakes",
        "longshot-heavy",
        0,
        400,
    ): "6a0fc3eda4c6d69145c322349023cc3113b6edfafe3b153d07caae81bebd2500",
    (
        "Flat stakes",
        "longshot-heavy",
        1,
        7,
    ): "fa7a647ede9eb43751282b1bdeb8f112d48d86e34327a978df6306f2017b5766",
    (
        "Flat stakes",
        "longshot-heavy",
        1,
        53,
    ): "6dd71734334d5933d42251984ad1ae10c17dced75794ba9fa5ab9d515a806ec1",
    (
        "Flat stakes",
        "longshot-heavy",
        1,
        400,
    ): "849a9ee5a7851352c3305608e2891378c350058b434618039d8953febcfedf63",
    (
        "Flat stakes",
        "longshot-heavy",
        42,
        7,
    ): "3476a3f2cacd990e2af5eceb43a371d294b3f1e25a14c6d8e0eb32d5ab8c8127",
    (
        "Flat stakes",
        "longshot-heavy",
        42,
        53,
    ): "f21e47494e3c53e31fe9c95a648c1925ba03f12175732e7a2644a93f9a7f8ea4",
    (
        "Flat stakes",
        "longshot-heavy",
        42,
        400,
    ): "5fcb480de4cd3e13dfecc19efa715efb2206ef64eafaa07249bb56e9de1a479f",
    (
        "Flat stakes",
        "longshot-heavy",
        20260803,
        7,
    ): "874466d7c34d8cd6db9614b1980e3b1a514e1f0bc83944391b8a4e0487d4fff0",
    (
        "Flat stakes",
        "longshot-heavy",
        20260803,
        53,
    ): "22785ce402da441691e527624f3ece4c103c88978c5bfe03d946df75b0a62277",
    (
        "Flat stakes",
        "longshot-heavy",
        20260803,
        400,
    ): "6f27123bab6d5c85c760aaeff3e6ac9fdfd173e7e0f53a1338b8c5e82a8dfc77",
    (
        "Flat stakes",
        "standard",
        0,
        7,
    ): "a75a47cc68fb10577ecb01c87992085e99732b268a4565fe87d134146ef47103",
    (
        "Flat stakes",
        "standard",
        0,
        53,
    ): "e647d16b314b5a08cfbefef0a24357067e63b7104328eac99375be53c95e9113",
    (
        "Flat stakes",
        "standard",
        0,
        400,
    ): "0760ff709ba5cc1d3d4c2b1931321ffb776e929f67c841489684fb01c0edafcd",
    (
        "Flat stakes",
        "standard",
        1,
        7,
    ): "d927316f221a2c3fc627b2e7562d14149a1bd3fdd73ac278497283cc73274690",
    (
        "Flat stakes",
        "standard",
        1,
        53,
    ): "cf6daeea0a302570d776ce154ffaff44664dafd3a50af4c0573ff8382d0eb623",
    (
        "Flat stakes",
        "standard",
        1,
        400,
    ): "a514a89d24b05d86353e693474df6829725603964001a65f746cdc5fdf26a975",
    (
        "Flat stakes",
        "standard",
        42,
        7,
    ): "dbf7dc59a0aee0b518f816d2014a6112c0b53405e15a155b82436470a99e0b91",
    (
        "Flat stakes",
        "standard",
        42,
        53,
    ): "f013d56de60426e5de004f7d01da351e024fa2c51e1e79f0ef1ed3a10c296eca",
    (
        "Flat stakes",
        "standard",
        42,
        400,
    ): "05f957c24011ead82d85ca7262b11c2565822a53636e1bca67904c65a84785ae",
    (
        "Flat stakes",
        "standard",
        20260803,
        7,
    ): "15d55aaa34eba40de10e3b9d8fff0ab3b54dd99a0849d3688f1b9588a24efe7a",
    (
        "Flat stakes",
        "standard",
        20260803,
        53,
    ): "c1668057397cdce05fb3f16ad046989965302009bf9e08e40feebbe0561e1877",
    (
        "Flat stakes",
        "standard",
        20260803,
        400,
    ): "e9a6287a110309dcdcf08cd071ec53026feb647d917c0613186fe639319727fc",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        0,
        7,
    ): "36c751bb179998a50c8e0202735eaffdded414ec3a74807c33aaa890349398d5",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        0,
        53,
    ): "36c751bb179998a50c8e0202735eaffdded414ec3a74807c33aaa890349398d5",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        0,
        400,
    ): "36c751bb179998a50c8e0202735eaffdded414ec3a74807c33aaa890349398d5",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        1,
        7,
    ): "ed9874c98118d6a1b91bf69adbfb0e0637d6482eb666aeb5b6fc3550141dd54b",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        1,
        53,
    ): "ed9874c98118d6a1b91bf69adbfb0e0637d6482eb666aeb5b6fc3550141dd54b",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        1,
        400,
    ): "ed9874c98118d6a1b91bf69adbfb0e0637d6482eb666aeb5b6fc3550141dd54b",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        42,
        7,
    ): "88b92edfaf40d1f56f4cd2fc0cf1133531563fff0c3ccb76d0e3bd5f7eba5e81",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        42,
        53,
    ): "3e852e6ff8fedd67eb0cd600adf4a1947347295087a45da6fb7ea3a2b5f8c24a",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        42,
        400,
    ): "3e852e6ff8fedd67eb0cd600adf4a1947347295087a45da6fb7ea3a2b5f8c24a",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        20260803,
        7,
    ): "70512a88e8d181a16bb3c6221bc2a93f510b1c0be3e3ecbef0bddca086f0539a",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        20260803,
        53,
    ): "70512a88e8d181a16bb3c6221bc2a93f510b1c0be3e3ecbef0bddca086f0539a",
    (
        "Losing-batch tracking",
        "drawdown-tight",
        20260803,
        400,
    ): "70512a88e8d181a16bb3c6221bc2a93f510b1c0be3e3ecbef0bddca086f0539a",
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
    ): "c5e23a18efbdedada648d04d69a3fe511f48f5cfe6ea8caea7265d7fb767f1be",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        0,
        400,
    ): "c5e23a18efbdedada648d04d69a3fe511f48f5cfe6ea8caea7265d7fb767f1be",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        1,
        7,
    ): "74f989e7c888fd75ae9c86606f3c25048ac7e0227637331d04c287345de9ceaf",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        1,
        53,
    ): "1d5ebf4efbddbeb1c8e9d0a7cf4e8e891e66cfd19ac2815bf49337217ea2f747",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        1,
        400,
    ): "1d5ebf4efbddbeb1c8e9d0a7cf4e8e891e66cfd19ac2815bf49337217ea2f747",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        42,
        7,
    ): "fbfe1e9f5ef248a599f55af47039604f8d0ba92425cb253f350caed7bbd627d5",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        42,
        53,
    ): "083e77aefdbb102e3d72b19db2d65eba0b4fb2dba2011a55b62b95ff5172bb38",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        42,
        400,
    ): "083e77aefdbb102e3d72b19db2d65eba0b4fb2dba2011a55b62b95ff5172bb38",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        20260803,
        7,
    ): "efabaf523991c2ab82342f52070746403e739b6f7e8fe532823d4e5dfc01812f",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        20260803,
        53,
    ): "a34c8edd63c37272be4940bac58bfcf5e4dda9bf7aea46fff7e772073cb9fd2b",
    (
        "Losing-batch tracking",
        "longshot-heavy",
        20260803,
        400,
    ): "a34c8edd63c37272be4940bac58bfcf5e4dda9bf7aea46fff7e772073cb9fd2b",
    (
        "Losing-batch tracking",
        "standard",
        0,
        7,
    ): "4c0c52da80087783a9344396cdb19a53b0ceee321aaf1f8387e2b2e38aef04dd",
    (
        "Losing-batch tracking",
        "standard",
        0,
        53,
    ): "ad9484304afec37ac2117f624f37defcac6c50a7c38f656d55a20efe210bbc66",
    (
        "Losing-batch tracking",
        "standard",
        0,
        400,
    ): "bfca6804b1a60a3947d0b6ac08593a0f7b92827f4944e74d7485134f224b741f",
    (
        "Losing-batch tracking",
        "standard",
        1,
        7,
    ): "30e11d76e7d771e198fab0a4183ca7ea982a3e2d62ce5837f2277fd9e9f6bf34",
    (
        "Losing-batch tracking",
        "standard",
        1,
        53,
    ): "9bcc4d1d61cf783018bfb4483e3616071e7c3e8bbbbb325906710c21a60402b2",
    (
        "Losing-batch tracking",
        "standard",
        1,
        400,
    ): "d567eca752dbe89d3c5b8178dceea53a26a015260635afd4b227e1ff0f913f93",
    (
        "Losing-batch tracking",
        "standard",
        42,
        7,
    ): "5af0d98037f5c5e081fcb72434f64f582b2946493914f260502f589dcf5be105",
    (
        "Losing-batch tracking",
        "standard",
        42,
        53,
    ): "78fb203685beda9c064086df5267a3e3385bec29bc9b532a71c5e042b2759ef8",
    (
        "Losing-batch tracking",
        "standard",
        42,
        400,
    ): "fb5120df9a7590283be128a241d4e45770e87c84be94e62539199ddccf123b6a",
    (
        "Losing-batch tracking",
        "standard",
        20260803,
        7,
    ): "0ff875892ecad794c1fd3dc5796f2311ea5d5a502a221bac741a8f6aaee36aab",
    (
        "Losing-batch tracking",
        "standard",
        20260803,
        53,
    ): "f92c83b7e39fc96b61321663ec6321703d2748bab64a244aa55ce900b63aa29a",
    (
        "Losing-batch tracking",
        "standard",
        20260803,
        400,
    ): "8df5e9ec19305d43d2055541d2b3590f11b07c1d0b003fa73ac414fc1a90a4d0",
    (
        "Probability weighted",
        "drawdown-tight",
        0,
        7,
    ): "6a4119081eb2ddb6ac2abc886fad4860404ee7480dd7ad38e2d92aaea7f3def6",
    (
        "Probability weighted",
        "drawdown-tight",
        0,
        53,
    ): "6a4119081eb2ddb6ac2abc886fad4860404ee7480dd7ad38e2d92aaea7f3def6",
    (
        "Probability weighted",
        "drawdown-tight",
        0,
        400,
    ): "6a4119081eb2ddb6ac2abc886fad4860404ee7480dd7ad38e2d92aaea7f3def6",
    (
        "Probability weighted",
        "drawdown-tight",
        1,
        7,
    ): "7d3bde5c0d1cfaebb21f6e9dff8fbd852f1bc1c8c0b4c44e3ff9235252f79cc0",
    (
        "Probability weighted",
        "drawdown-tight",
        1,
        53,
    ): "7d3bde5c0d1cfaebb21f6e9dff8fbd852f1bc1c8c0b4c44e3ff9235252f79cc0",
    (
        "Probability weighted",
        "drawdown-tight",
        1,
        400,
    ): "7d3bde5c0d1cfaebb21f6e9dff8fbd852f1bc1c8c0b4c44e3ff9235252f79cc0",
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
    ): "873ce5e24a667d563d95caed3095663bc7dcbc77c9ed6f6deb9189c3a12b6265",
    (
        "Probability weighted",
        "longshot-heavy",
        1,
        53,
    ): "be14e7e2b0b2d4818dd4603ed5b946b8f9c30a6ec5e66431a16760bf4b180209",
    (
        "Probability weighted",
        "longshot-heavy",
        1,
        400,
    ): "be14e7e2b0b2d4818dd4603ed5b946b8f9c30a6ec5e66431a16760bf4b180209",
    (
        "Probability weighted",
        "longshot-heavy",
        42,
        7,
    ): "cc3365dad60ae5244b878b010c805c7d712e30b78657c798c316b2e2be181152",
    (
        "Probability weighted",
        "longshot-heavy",
        42,
        53,
    ): "cc3365dad60ae5244b878b010c805c7d712e30b78657c798c316b2e2be181152",
    (
        "Probability weighted",
        "longshot-heavy",
        42,
        400,
    ): "cc3365dad60ae5244b878b010c805c7d712e30b78657c798c316b2e2be181152",
    (
        "Probability weighted",
        "longshot-heavy",
        20260803,
        7,
    ): "a6aa08093c6fdf4ea9fe1e87054a429dde256e136e1426228ae77cafa1feb0bf",
    (
        "Probability weighted",
        "longshot-heavy",
        20260803,
        53,
    ): "a6aa08093c6fdf4ea9fe1e87054a429dde256e136e1426228ae77cafa1feb0bf",
    (
        "Probability weighted",
        "longshot-heavy",
        20260803,
        400,
    ): "a6aa08093c6fdf4ea9fe1e87054a429dde256e136e1426228ae77cafa1feb0bf",
    (
        "Probability weighted",
        "standard",
        0,
        7,
    ): "4a56525f74f1294389ff42287a2560e578254a3075dfc8492bc8ccf357710710",
    (
        "Probability weighted",
        "standard",
        0,
        53,
    ): "ff8ff69bb46e299a9dc95267009c751693ffcf88b5cda727dfc759ab9d04fcdf",
    (
        "Probability weighted",
        "standard",
        0,
        400,
    ): "c9d1ba0f9b5450972f9a19a0d8680016671d6b7198526586b74381dc22ceb9c6",
    (
        "Probability weighted",
        "standard",
        1,
        7,
    ): "c43f9d9d6cc1d49257104080049efb7836aa5bd5a66f7a1dce07b1d6219e8079",
    (
        "Probability weighted",
        "standard",
        1,
        53,
    ): "a6cfe4bb93764d7eaedd92b7b97c0bdfb54d381dfc27686f9b87f9698ebe94d9",
    (
        "Probability weighted",
        "standard",
        1,
        400,
    ): "d6b2076bd8e4d95e0bbdca8d9fc3890afb131b77af9f9eace8e15965cd8064fb",
    (
        "Probability weighted",
        "standard",
        42,
        7,
    ): "3e42271491e132fac54cff603134e3c6a983c8baf5cb4c633cf1521716702782",
    (
        "Probability weighted",
        "standard",
        42,
        53,
    ): "c1b9c25622510eb5c069adecd9d59dd0cda4e3b84263b33098d1b66b5dc5d65c",
    (
        "Probability weighted",
        "standard",
        42,
        400,
    ): "c01e0458030950f3f2b7fa9260b8dfd8347d2175761b1103e44bc6aabb78fe73",
    (
        "Probability weighted",
        "standard",
        20260803,
        7,
    ): "24f073c6c4745fcb533427fcd2ea8546674ffb000eea1d8a46a39a6956d59758",
    (
        "Probability weighted",
        "standard",
        20260803,
        53,
    ): "6d55a4f482bc166bff26b0372addafc1622d4dee22dc31db1a676403a8dd3359",
    (
        "Probability weighted",
        "standard",
        20260803,
        400,
    ): "4feecd3ed9fa7e4611c5df974629ea6e6f1982baa8007059b223b8ab81be2f89",
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
