"""
Allocation simulator: replays allocators through keeks' bankroll machinery.

:class:`AllocationSimulator` is what makes the allocation layer observable in
keeks' house style - without it the allocators would be a pile of closed-form
functions bolted onto a bankroll library. It takes any
:class:`~keeks.allocation.models.JointReturnModel` as the source of joint
simple-return realizations, asks an allocator for one long-only weight per
option each trial, and settles the portfolio through :class:`~keeks.bankroll.BankRoll`
exactly the way the multi-outcome simulators settle theirs: batch-net, one
deposit or withdrawal per settled period, so bankruptcy and drawdown
safeguards evaluate the period once.

The simulator is the static/online divide made runnable: a static allocator
is replayed as-is, while an online allocator additionally receives the
realized joint simple-return vector through ``record_settlement`` - the
allocation layer's only stateful channel - after each settled period, so
:class:`~keeks.allocation.online.ExponentialGradient` and
:class:`~keeks.allocation.online.OnlineNewtonStep` adapt their weights from
settled outcomes the way the binary and multi-outcome strategies adapt
through the same hook.
"""

import operator
import warnings

import numpy as np

from keeks.allocation.base import _validate_strategy_scenarios, _validate_weights
from keeks.allocation.models import _validate_draws
from keeks.utils import (
    RuinError,
    _require_finite,
    _validate_simulator_seed,
    validate_probabilities,
)

__author__ = "willmcginnis"


def _validate_model(model):
    """
    Check that a simulator's model exposes the sampling contract.

    >>> from keeks.allocation import scenario_model
    >>> _validate_model(scenario_model([[0.01, 0.02]])).scenarios.shape
    (1, 2)

    >>> _validate_model([[0.01, 0.02]])
    Traceback (most recent call last):
        ...
    ValueError: Model must expose sample(n_samples, rng), the JointReturnModel sampling contract - wrap a raw matrix with keeks.allocation.scenario_model
    """
    if not callable(getattr(model, "sample", None)):
        raise ValueError(
            "Model must expose sample(n_samples, rng), the JointReturnModel "
            "sampling contract - wrap a raw matrix with "
            "keeks.allocation.scenario_model"
        )
    return model


def _model_option_count(model):
    """
    Best-effort option count from a model's public surface, without sampling.

    Every built-in model states its option count without drawing: the
    sequence descriptors (``bets``, ``marginals``) hold one entry per
    option, ``scenarios`` holds one row per observation with one column per
    option, and ``moments()`` is contract exact when it returns a pair. The
    count drives the simulator's pre-draw length gate - an invalid weight
    vector is rejected before any draw is consumed - so models that reveal
    their option count only by sampling (many user callables) get the gate
    at the first settlement instead.

    >>> from keeks.allocation import binary_bets_model
    >>> _model_option_count(binary_bets_model([(0.5, 2.0, 1.0), (0.3, 1.5, 0.5)]))
    2

    >>> from keeks.allocation import scenario_model
    >>> _model_option_count(scenario_model([[0.01, 0.02], [-0.02, 0.01]]))
    2

    >>> _model_option_count(lambda n, rng: None) is None
    True
    """
    for attribute in ("bets", "marginals"):
        descriptor = getattr(model, attribute, None)
        if descriptor is not None:
            return len(descriptor)
    scenarios = getattr(model, "scenarios", None)
    if scenarios is not None:
        scenarios = np.asarray(scenarios)
        if scenarios.ndim != 2:
            raise ValueError("Model scenarios must be a two-dimensional matrix")
        return int(scenarios.shape[1])
    moments = getattr(model, "moments", None)
    if callable(moments):
        exact = moments()
        if exact is not None:
            mean, _covariance = exact
            mean = np.asarray(mean)
            if mean.ndim != 1:
                raise ValueError(
                    "Model moments() must return a (mean, covariance) pair "
                    "with a one-dimensional mean vector"
                )
            return int(mean.size)
    return None


class AllocationSimulator:
    """
    Replay an allocator over scenario realizations through a bankroll.

    Each trial draws one joint simple-return realization from the model,
    asks the allocation for its weights, and settles the portfolio through
    :class:`~keeks.bankroll.BankRoll` with keeks' batch-net convention: the
    option stakes are the weights times the bankroll's bettable funds as
    they stood when the trial began, the option returns turn the stakes
    into signed amounts, and the amounts net into exactly one deposit or
    withdrawal - so bankruptcy and drawdown safeguards evaluate the period
    once, and the simulation is the single rounding site per period.

    Residual probability mass is an all-cash period: with ``probabilities``
    given, each trial draws one uniform and reads it against the cumulative
    bands, and a draw beyond the total mass is a period where nothing is
    realized - no transaction, no fee, and an all-``None`` outcome vector
    with a zero return vector reported through the settlement hook.

    Parameters
    ----------
    model : JointReturnModel
        Source of joint simple-return realizations - anything exposing
        ``sample(n_samples, rng) -> (n_samples, N)``. Empirical scenario
        rows, keeks-native binary bets, parametric marginals, and user
        callables all enter through the same contract; a raw matrix wraps
        with :func:`~keeks.allocation.scenario_model`.
    probabilities : sequence of float, optional
        Probability of each model state, validated like every keeks
        probability vector (finite, nonnegative, summing to no more than
        one within ``PROBABILITY_SUM_TOLERANCE``). ``None`` (the default)
        makes the model's draws the trial sequence: trial ``t`` settles the
        ``t``-th realization row.
    fee_per_bet : float
        Flat cost charged once per staked period, the house convention.
        Turnover-based charging (costs that scale with weight changes
        between periods) is deliberately deferred.
    trials : int
        Maximum number of periods to replay.
    seed : int, optional
        Seed for the simulator's private random stream. See the
        reproducibility contract below.

    Raises
    ------
    ValueError
        If the model does not expose the sampling contract, the
        probabilities fail validation, the transaction costs are negative
        or non-finite, the trial count is not a nonnegative integer, or the
        seed is not a nonnegative integer or ``None``.

    Examples
    --------
    >>> from keeks import BankRoll
    >>> from keeks.allocation import AllocationSimulator, FixedWeights
    >>> from keeks.allocation import scenario_model
    >>> model = scenario_model(
    ...     [[0.03, 0.01], [-0.01, 0.02], [0.01, -0.005], [0.0, 0.0]]
    ... )
    >>> simulator = AllocationSimulator(model, trials=200, seed=42)
    >>> bankroll = BankRoll(initial_funds=1000.0, max_transaction_loss=None)
    >>> simulator.evaluate_strategy(FixedWeights([0.4, 0.2]), bankroll)
    >>> round(float(bankroll.total_funds), 2)
    2583.92
    >>> len(bankroll.history)
    201

    Notes
    -----
    Reproducibility contract (public behavior, matching the multi-outcome
    simulators): with a ``seed``, every draw the simulator makes comes from
    one private :class:`numpy.random.Generator` on a spawned
    :class:`numpy.random.SeedSequence` child -
    ``default_rng(SeedSequence(seed).spawn(1)[0])`` - so a seeded run
    replays byte-identically: same realizations, same hook calls, same
    bankroll history. Seeded runs consume nothing from numpy's global
    generator. Without a seed, a fresh generator drives each
    ``evaluate_strategy`` call and no replay is promised.

    Realization semantics: the realizations are drawn from the model in one
    sampling call, lazily, the first time a trial needs one - a run whose
    every trial is skipped or rejected consumes no draws at all. With no
    ``probabilities``, the drawn matrix is the trial sequence: trial ``t``
    settles row ``t``. This one-call, in-order shape is deliberate - it is
    what lets a binary-bets book
    (:class:`~keeks.allocation.models.BinaryBetsModel`) replay
    :class:`~keeks.multi_outcome.PortfolioSimulator`'s outcome streams
    draw-for-draw under matched seeding, because the model keys each bet's
    per-bet stream exactly as the portfolio simulator does and the simulator
    hands each bet its trials in order. With ``probabilities``, the matrix
    holds one row per probability band and each staked trial reads one
    uniform against the cumulative bands.

    Each staked trial then runs in a fixed order:

    1. Stop when the bankroll is depleted (``total_funds <= 0``).
    2. Fire the allocation's ``update_bankroll`` hook when it has one.
    3. Validate the weight vector returned by ``evaluate``: finite, one
       weight per option, each in ``[0, 1]``, sum at most
       ``1 + PROBABILITY_SUM_TOLERANCE``. An invalid vector aborts the run
       before any draw is consumed whenever the model's option count is
       known without sampling (every built-in model); models that reveal it
       only by sampling get the length gate at the first settlement, but no
       settlement ever sees an invalid vector. A trial staking nothing
       (every weight zero) is skipped entirely: no draw is consumed, no fee
       is charged, and no settlement is reported.
    4. Draw one realization (see the realization semantics above). An
       all-cash period - residual probability mass - reports a zero vector
       through the settlement hook and moves to the next trial with the
       bankroll untouched.
    5. Settle the batch net: exactly one deposit (net gain) or withdrawal
       (net loss) of ``bankroll * w'R(omega)`` minus the flat transaction
       cost, written to the history once. When a bankroll safeguard refuses
       the settlement (:class:`~keeks.utils.RuinError` - bankruptcy or the
       configured ``max_transaction_loss`` cap), the period leaves the
       bankroll unchanged and the simulation stops after it completes.
    6. Fire ``record_settlement(won, realized_returns)`` once with the
       period's per-option outcome vector (``True`` when the option's
       realized return is positive, ``False`` otherwise) and the realized
       joint simple-return vector - the state channel for online
       allocators. A refused settlement still reports the market's
       realization: the refusal blocks the bankroll transfer, not the
       market's move.

    A scenario-bound allocator - one carrying a ``scenarios`` descriptor,
    like :class:`~keeks.allocation.scenarios.MeanCVaR` - must be sized with
    the scenarios the simulator's model settles with: the simulator runs the
    same descriptor-equality gate the multi-outcome simulators run on odds
    (:func:`~keeks.allocation.base._validate_strategy_scenarios`), and a
    mismatch raises before any draw is consumed. Allocators bound to other
    descriptors pass untouched, as do models without scenario rows - their
    compatibility stays the caller's responsibility.
    """

    def __init__(
        self,
        model,
        probabilities=None,
        fee_per_bet=0.0,
        trials=1000,
        seed=None,
    ):
        self.model = _validate_model(model)
        if probabilities is None:
            self.probabilities: np.ndarray | None = None
            self._cumulative: np.ndarray | None = None
        else:
            self.probabilities = validate_probabilities(probabilities)
            self._cumulative = np.cumsum(self.probabilities)
        fee_per_bet = _require_finite(fee_per_bet, "Fee per bet")
        if fee_per_bet < 0:
            raise ValueError("Fee per bet must be non-negative")
        self.fee_per_bet: float = fee_per_bet
        try:
            trials = operator.index(trials)
        except TypeError as exc:
            raise ValueError("Trials must be a nonnegative integer") from exc
        if trials < 0:
            raise ValueError("Trials must be a nonnegative integer")
        self.trials: int = trials
        self.seed: int | None = _validate_simulator_seed(seed)
        self._rng: np.random.Generator | None = (
            np.random.default_rng(np.random.SeedSequence(self.seed).spawn(1)[0])
            if self.seed is not None
            else None
        )
        self._option_count: int | None = _model_option_count(model)

    def _draw_realizations(self, rng):
        """
        Draw the run's realizations from the model in one sampling call.

        With no probabilities the matrix holds one row per trial; with
        probabilities it holds one row per band. The model's option count,
        when unknown, is adopted from the draws - and a later disagreement
        with it is a model contract violation.
        """
        count = self.trials if self.probabilities is None else len(self.probabilities)
        draws = _validate_draws(self.model.sample(count, rng), count)
        if self._option_count is None:
            self._option_count = int(draws.shape[1])
        elif draws.shape[1] != self._option_count:
            raise ValueError(
                f"Model draws carry {draws.shape[1]} options but its option "
                f"count is {self._option_count}; a model's option count is fixed"
            )
        return draws

    def evaluate_strategy(self, allocation, bankroll) -> None:
        """
        Replay the allocation over the simulator's realizations, in place.

        Parameters
        ----------
        allocation : BaseAllocationStrategy
            The allocator to replay. Online allocators are resolved for
            their ``update_bankroll`` and ``record_settlement`` hooks.
        bankroll : BankRoll
            The bankroll the portfolio settles through.

        Raises
        ------
        ValueError
            If a scenario-bound allocation does not match the model's
            scenarios, or a trial returns an invalid weight vector.
        """
        # Descriptor-equality gate: a scenario-bound allocator must size
        # with the scenarios it settles with.
        model_scenarios = getattr(self.model, "scenarios", None)
        if model_scenarios is not None:
            _validate_strategy_scenarios(allocation, model_scenarios)

        # State-dependent hooks resolve getattr-style, the house pattern.
        update_bankroll = getattr(allocation, "update_bankroll", None)
        if not callable(update_bankroll):
            update_bankroll = None
        record_settlement = getattr(allocation, "record_settlement", None)
        if not callable(record_settlement):
            record_settlement = None

        # One stream drives the run: the seeded construction's spawned
        # generator, or a fresh generator when unseeded (no replay promised).
        rng = self._rng if self._rng is not None else np.random.default_rng()
        realizations: np.ndarray | None = None
        next_row = 0

        for _ in range(self.trials):
            total_funds = bankroll.total_funds
            if total_funds <= 0:
                break

            if update_bankroll is not None:
                update_bankroll(total_funds)

            weights = _validate_weights(
                allocation.evaluate(total_funds), option_count=self._option_count
            )
            if not any(weights):
                # Nothing staked: no draw, no fee, no settlement.
                continue

            if realizations is None:
                realizations = self._draw_realizations(rng)
            if len(weights) != realizations.shape[1]:
                # An opaque model's option count only became known at the
                # draw; the length gate lands here, before anything settles.
                raise ValueError(
                    f"Strategy must return exactly {realizations.shape[1]} "
                    f"weights, got {len(weights)}"
                )

            if self.probabilities is None:
                # The model's draws are the trial sequence: row t settles
                # trial t, which is what makes a binary-bets book replay
                # PortfolioSimulator's streams under matched seeding.
                realized = realizations[next_row]
                next_row += 1
            else:
                band = int(np.searchsorted(self._cumulative, rng.random(), "right"))
                if band >= len(self.probabilities):
                    # Residual probability mass: an all-cash period.
                    if record_settlement is not None:
                        record_settlement(
                            tuple([None] * realizations.shape[1]),
                            tuple(np.zeros(realizations.shape[1])),
                        )
                    continue
                realized = realizations[band]

            # Batch-net settlement: one deposit or withdrawal per period.
            bettable_funds = bankroll.bettable_funds
            amounts = [
                bettable_funds * weight * option_return
                for weight, option_return in zip(weights, realized, strict=True)
            ]
            batch_ruined = False
            try:
                net = sum(amounts) - self.fee_per_bet
                if net >= 0:
                    bankroll.deposit(net)
                else:
                    bankroll.withdraw(-net)
            except RuinError as exc:
                # Refuse-then-stop: the settlement is refused, the bankroll
                # is unchanged, and the simulation stops after this period.
                # Warn loudly: the message names the attempted amount, the
                # configured limit, and current funds.
                warnings.warn(
                    f"Settlement refused; the simulation stops after "
                    f"this period: {exc}",
                    stacklevel=2,
                )
                batch_ruined = True

            if record_settlement is not None:
                # The outcome flag reads off the realized return: a positive
                # simple return won its stake, zero or negative lost it.
                won = tuple(bool(r > 0) for r in realized)
                record_settlement(won, tuple(realized.tolist()))

            if batch_ruined:
                break
