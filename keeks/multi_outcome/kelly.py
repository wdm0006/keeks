"""
Log-growth optimal stakes on the legs of one mutually exclusive market.

The Kelly criterion generalizes from one binary bet to N legs. Staking the
fractions ``f_1..f_N`` of the bankroll on legs quoted at decimal odds
``payoff_i`` (a winning leg pays ``payoff_i`` times its stake, stake
included), the net win per unit staked on leg ``i`` is
``a_i = payoff_i - 1 - transaction_cost`` and every losing leg costs
``l = loss + transaction_cost`` per unit. Exactly one leg realizes, so the
bankroll multiplier when leg ``j`` wins is
``1 + a_j f_j - l * sum_{k != j} f_k`` and the log-growth optimal allocation
maximizes::

    sum_i p_i * log(1 + a_i f_i - l * sum_{k != i} f_k)

At two legs the optimum of this objective is a single-leg stake whenever the
book is fully priced (the probabilities sum to one) and only one leg prices a
positive binary Kelly fraction - the realistic case, since both legs pricing
positive is an inconsistent (arbitrage) book - so
:class:`MultiOutcomeKellyCriterion` falls back to exact
:class:`keeks.binary_strategies.KellyCriterion` sizing there and is
golden-tested to reproduce it exactly. When the probabilities leave void mass
(binary Kelly would count the refund branch as a loss), when both legs price
positive, or at three legs and beyond, the joint problem is solved
numerically by deterministic coordinate ascent.
"""

import math
from collections.abc import Sequence

import numpy as np

from keeks.binary_strategies.kelly import KellyCriterion
from keeks.multi_outcome.base import BaseMultiOutcomeStrategy, _validate_stake_fractions
from keeks.utils import (
    PROBABILITY_SUM_TOLERANCE,
    _require_finite,
    normalize_probabilities,
)

__author__ = "willmcginnis"

_MAX_BISECTIONS = 200
_MAX_SWEEPS = 100
# Coordinate changes below this are float noise on a [0, 1] fraction.
_COORDINATE_TOLERANCE = 1e-15
# Hard floor on every positive-probability leg's winning multiplier. The log
# domain is open (a zeroed multiplier is unconditional ruin), so stakes must
# back off from its boundary; the floor only ever binds for subnormal-
# probability legs, whose log terms are weighted by that tiny probability.
_MULTIPLIER_FLOOR = 1e-12
# Slack for the delegation KKT check: marginal values within this of zero
# are float noise on the binary root, not real improvement directions.
_KKT_SLACK = 1e-12


def _slice_slope(t, prob, weight, alpha, other_probs, other_multipliers, cost):
    """
    dU/df_j of the log-growth objective at ``f_j = t``, other legs fixed.

    ``alpha`` is the winning leg's own multiplier at ``t = 0`` and
    ``other_multipliers`` are the other legs' multipliers at ``t = 0``; both
    are positive inside the log's domain.
    """
    if other_probs.size:
        rooms = other_multipliers - cost * t
        if rooms.min() <= 0.0:
            # Past another leg's log domain: its multiplier would vanish, and
            # the objective falls to -infinity, so the slope is -infinity.
            return -math.inf
        drag = cost * float(np.sum(other_probs / rooms))
    else:
        drag = 0.0
    own_room = alpha + weight * t
    if own_room <= 0.0:
        # Only reachable at t = 0 with a zero own multiplier; the leg's own
        # term dominates in the limit.
        return math.inf
    return prob * weight / own_room - drag


def _multipliers_within_floor(stakes, net_gains, cost, probabilities):
    """
    The legs' winning multipliers for a stake vector, or ``None``.

    ``None`` means at least one positive-probability leg's multiplier falls
    below ``_MULTIPLIER_FLOOR`` - a stake vector this close to guaranteed
    ruin must never leave the strategy, so both the two-leg delegation and
    the joint solver validate candidates through this one check.
    """
    stakes = np.asarray(stakes, dtype=float)
    multipliers = (
        1.0
        - cost * float(np.sum(stakes))
        + (np.asarray(net_gains, dtype=float) + cost) * stakes
    )
    relevant = np.asarray(probabilities, dtype=float) > 0.0
    if bool(np.any(multipliers[relevant] < _MULTIPLIER_FLOOR)):
        return None
    return multipliers


def _delegation_is_joint_optimum(stakes, net_gains, cost, probabilities, max_total):
    """
    Whether a fully priced delegation candidate is already the joint optimum.

    ``stakes`` holds at most one positive per-leg binary Kelly stake. The
    candidate is the joint optimum only if no feasible direction improves
    the log-growth objective at that point: no other positive-probability
    leg may grow into the remaining aggregate cap on its own, and none may
    beat the priced leg in a fixed-total trade. For an uncapped priced leg
    the priced marginal is exactly zero and the trade condition reduces to
    ``p_k * w_k * w_j <= (1 - p_j) * l**2`` per other leg - the closed form
    that separates genuinely binary-equivalent books from two-sided books
    that grow faster by staking both legs (which must be solved jointly).
    """
    cost = float(cost)
    stakes = [float(stake) for stake in stakes]
    total = float(sum(stakes))
    multipliers = [
        1.0 + float(net_gains[leg]) * stakes[leg] - cost * (total - stakes[leg])
        for leg in range(len(stakes))
    ]
    if any(
        multipliers[leg] <= 0.0
        for leg in range(len(stakes))
        if float(probabilities[leg]) > 0.0
    ):
        # A realized outcome would zero the bankroll: the candidate is
        # infeasible, hence certainly not the joint optimum. (Callers that
        # check the multiplier floor first never reach this, but the gate
        # must stay safe standing alone.)
        return False
    gradient = []
    for leg in range(len(stakes)):
        if float(probabilities[leg]) <= 0.0:
            # A zero-probability leg contributes no log term; staking it
            # only burns fees, so it never improves the objective.
            gradient.append(-math.inf)
            continue
        own = float(probabilities[leg]) * float(net_gains[leg]) / multipliers[leg]
        drag = cost * sum(
            float(probabilities[i]) / multipliers[i]
            for i in range(len(stakes))
            if i != leg and float(probabilities[i]) > 0.0
        )
        gradient.append(own - drag)

    priced = [leg for leg, stake in enumerate(stakes) if stake > 0.0]
    if not priced:
        return all(g <= _KKT_SLACK for g in gradient)

    priced_leg = priced[0]
    room = float(max_total) - total
    for leg in range(len(stakes)):
        if leg == priced_leg:
            continue
        if gradient[leg] <= _KKT_SLACK:
            continue
        if room > 0.0:
            # The leg can grow on its own without touching the priced leg.
            return False
        if gradient[leg] - gradient[priced_leg] > _KKT_SLACK:
            # Trading the priced leg's stake into this leg improves.
            return False
    return True


def _optimal_single_leg_stake(
    prob, weight, alpha, other_probs, other_multipliers, cost, upper
):
    """
    Maximize the objective's 1-D slice over ``f_j`` on ``[0, upper]``.

    The slice is strictly concave in ``f_j`` (a sum of logs of affine
    functions with positive coefficients on the log arguments), so its
    derivative crosses zero at most once and bisection on the slope sign
    finds the maximum to float precision.
    """
    if upper <= 0.0:
        return 0.0
    args = (prob, weight, alpha, other_probs, other_multipliers, cost)
    if _slice_slope(0.0, *args) <= 0.0:
        return 0.0
    if _slice_slope(upper, *args) > 0.0:
        # The aggregate cap binds before the interior root is reached.
        stake = upper
    else:
        lo, hi = 0.0, upper
        for _ in range(_MAX_BISECTIONS):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if _slice_slope(mid, *args) > 0.0:
                lo = mid
            else:
                hi = mid
        stake = 0.5 * (lo + hi)
    # Never land on the log-domain boundary itself: a root or cap that rounds
    # onto it would zero another leg's multiplier in float arithmetic, so
    # back off by whole ulps until every room is strictly positive.
    while stake > 0.0 and bool(np.any(other_multipliers - cost * stake <= 0.0)):
        stake = math.nextafter(stake, 0.0)
    return stake


def _optimal_pairwise_transfer(
    prob_to,
    spread_to,
    multiplier_to,
    prob_from,
    spread_from,
    multiplier_from,
    available,
):
    """
    Maximize the objective along a stake transfer between two active legs.

    Moving ``delta`` stake from one leg to the other keeps the total stake -
    and therefore every other leg's multiplier - fixed, so the objective
    along the move is ``p_to * log(m_to + spread_to * delta) +
    p_from * log(m_from - spread_from * delta)``, strictly concave in
    ``delta``. Returns the maximizing transfer, in ``[0, available]``.
    """
    if available <= 0.0 or multiplier_to <= 0.0 or multiplier_from <= 0.0:
        # Defensive: a transfer is meaningless from a state that has already
        # lost a log domain, and the receiver's slope divides by its
        # multiplier below.
        return 0.0

    def slope(delta):
        room = multiplier_from - spread_from * delta
        if room <= 0.0:
            # The donor leg's multiplier would vanish: past its log domain.
            return -math.inf
        return prob_to * spread_to / (multiplier_to + spread_to * delta) - (
            prob_from * spread_from / room
        )

    if slope(0.0) <= _COORDINATE_TOLERANCE:
        return 0.0
    upper = min(available, multiplier_from / spread_from)
    if slope(upper) > 0.0:
        # The donor leg gives everything it has (or its domain allows).
        delta = upper
    else:
        lo, hi = 0.0, upper
        for _ in range(_MAX_BISECTIONS):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if slope(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        delta = 0.5 * (lo + hi)
    # Keep the donor off the log-domain boundary: rounding onto it would zero
    # (or flip negative) its multiplier once fractions are recomputed.
    while delta > 0.0 and multiplier_from - spread_from * delta <= 0.0:
        delta = math.nextafter(delta, 0.0)
    return delta


def _solve_log_growth_allocation(net_gains, cost, probabilities, max_total):
    """
    Solve the N-leg log-growth allocation by coordinate ascent.

    Maximizes ``sum_i p_i * log(1 + w_i f_i - cost * sum_{k != i} f_k)`` over
    stakes ``f >= 0`` with ``sum f <= max_total``. Legs with no probability or
    no net gain (``w_i <= 0``) are held at exactly zero: the objective strictly
    decreases in their stake, so zero is optimal for them and for the whole
    problem they are simply absent.

    Each sweep alternates two moves. A single-leg move grows one stake up to
    the remaining aggregate cap, which reaches every interior optimum; a
    pairwise transfer moves stake between two legs at a fixed total, the only
    way to improve once the cap binds (no single-leg move can trade). A point
    optimal for both move types is optimal along every direction of the
    feasible cone, hence - the objective being concave - globally optimal.

    Returns one stake per leg as a tuple, in leg order.
    """
    stakes = [0.0] * len(net_gains)
    gains = np.asarray(net_gains, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    spreads = gains + cost
    # Legs that carry probability mass enter the objective through their own
    # log term, so their multipliers must stay positive even when they are
    # never staked: a no-edge leg that realizes still multiplies the bankroll
    # by 1 - cost * F, and letting that hit zero is unconditional ruin.
    relevant = probs > 0.0
    active = [leg for leg in range(len(gains)) if relevant[leg] and gains[leg] > 0.0]
    if not active:
        return tuple(stakes)

    fractions = np.zeros(len(gains), dtype=float)
    multipliers = 1.0 - cost * float(fractions.sum()) + spreads * fractions

    def _apply_capped(candidate):
        """Accept as much of a proposed move as the multiplier floor allows.

        Multipliers are affine in the stakes, so floor-feasible move sizes
        form an interval: bisecting the move size recovers the frontier
        exactly where a proposed move rounds past it. The floor is enforced
        on recomputed multipliers, which makes it authoritative over the
        helpers' internal room models.
        """
        nonlocal fractions, multipliers
        trial = _multipliers_within_floor(candidate, gains, cost, probs)
        if trial is not None:
            fractions = np.asarray(candidate, dtype=float)
            multipliers = trial
            return True
        lo, hi = 0.0, 1.0
        best = None
        for _ in range(_MAX_BISECTIONS):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            trial_fractions = fractions + mid * (np.asarray(candidate) - fractions)
            trial = _multipliers_within_floor(trial_fractions, gains, cost, probs)
            if trial is None:
                hi = mid
            else:
                lo = mid
                best = (trial_fractions, trial)
        if best is None:
            return False
        fractions, multipliers = best
        return True

    for _ in range(_MAX_SWEEPS):
        worst_change = 0.0
        for leg in active:
            rest = float(fractions.sum() - fractions[leg])
            rooms = relevant.copy()
            rooms[leg] = False
            stake = _optimal_single_leg_stake(
                prob=float(probs[leg]),
                weight=float(gains[leg]),
                alpha=1.0 - cost * rest,
                other_probs=probs[rooms],
                other_multipliers=1.0 - cost * rest + spreads[rooms] * fractions[rooms],
                cost=cost,
                upper=max_total - rest,
            )
            previous = float(fractions[leg])
            if stake == previous:
                continue
            candidate = fractions.copy()
            candidate[leg] = stake
            if _apply_capped(candidate):
                worst_change = max(worst_change, abs(stake - previous))

        for to in active:
            for hero in active:
                if to == hero:
                    continue
                delta = _optimal_pairwise_transfer(
                    prob_to=float(probs[to]),
                    spread_to=float(spreads[to]),
                    multiplier_to=float(multipliers[to]),
                    prob_from=float(probs[hero]),
                    spread_from=float(spreads[hero]),
                    multiplier_from=float(multipliers[hero]),
                    available=float(fractions[hero]),
                )
                if delta > 0.0:
                    candidate = fractions.copy()
                    candidate[to] += delta
                    candidate[hero] -= delta
                    if _apply_capped(candidate):
                        worst_change = max(worst_change, delta)

        if worst_change <= _COORDINATE_TOLERANCE:
            break

    return tuple(fractions.tolist())


class MultiOutcomeKellyCriterion(BaseMultiOutcomeStrategy):
    """
    Kelly criterion for mutually exclusive markets with N legs.

    Sizes one stake fraction per leg to maximize the expected log growth of
    the bankroll: ``sum_i p_i * log(1 + a_i f_i - l * sum_{k != i} f_k)``
    where ``a_i = payoff_i - 1 - transaction_cost`` is the net win per unit
    staked on leg ``i`` (the payoffs are decimal odds: a winning leg pays its
    payoff times its stake, stake included) and ``l = loss +
    transaction_cost`` is the per-unit charge on every losing leg. Exactly one
    leg realizes per round, so staking two legs is a hedge inside one market,
    not two independent bets.

    Two legs are special: for a fully priced book (the probabilities sum to
    one) where at most one leg prices a positive binary Kelly stake - which
    holds for every consistent two-outcome book - the joint optimum is that
    leg's exact binary Kelly fraction and zero on the other, so
    :class:`MultiOutcomeKellyCriterion` computes it with
    :class:`keeks.binary_strategies.KellyCriterion` (including its
    ``min_probability`` gate) and reproduces the binary strategy exactly.
    When the probabilities leave void mass the delegation no longer applies -
    binary Kelly would count the refund branch as a loss - and the joint
    solver below runs instead; the same happens when both legs price a
    positive binary stake (an inconsistent, arbitrage two-sided book).

    At three legs and up the objective has no closed form in general and is
    maximized by deterministic coordinate ascent, converged to roughly
    ``1e-15`` of a stake unit. The ``min_probability`` gate is a
    :class:`~keeks.binary_strategies.KellyCriterion` feature of the two-leg
    fallback only; the general optimizer sizes every leg that carries a net
    win on its own merits, including legs below even odds.

    Parameters
    ----------
    payoffs : sequence of float
        The decimal-odds multiplier paid by each mutually exclusive leg, in
        leg order, on top of the stake's own return. Every payoff must be
        finite and greater than 0; a leg whose payoff cannot beat its costs
        (``payoff <= 1 + transaction_cost``) is never staked.
    loss : float
        The loss multiplier applied to every losing leg's stake.
    transaction_cost : float, optional
        The transaction cost as a fraction of each unit staked, by default 0.
        This is a per-unit *fractional* cost that enters the sizing formulas
        alongside ``payoffs`` and ``loss``, so ``0.01`` means one percent of
        the stake. Note this differs in unit from the simulators' flat
        per-settlement ``transaction_costs`` fee.
    min_probability : float, default=0.5
        The minimum leg probability for the two-leg binary fallback to place
        a stake, mirroring
        :class:`~keeks.binary_strategies.KellyCriterion`'s lossy gate. It has
        no effect at three legs and up.

    Raises
    ------
    ValueError
        If ``payoffs`` is not a non-empty one-dimensional sequence of finite
        numbers greater than 0, if ``loss`` or ``transaction_cost`` is not a
        finite nonnegative number with ``loss + transaction_cost > 0``, or if
        ``min_probability`` is outside ``[0, 1]``.

    Examples
    --------
    A two-leg market reproduces the binary Kelly stake exactly. Leg 1 at
    decimal odds 3.0 is a net win of 2.0 per unit staked - the same bet
    :class:`~keeks.binary_strategies.KellyCriterion` sizes from its net
    payoff:

    >>> strategy = MultiOutcomeKellyCriterion(payoffs=(3.0, 1.5), loss=1.0)
    >>> strategy.evaluate([0.5, 0.5], 1000.0)
    (0.25, 0.0)
    >>> KellyCriterion(payoff=2.0, loss=1.0, transaction_cost=0).evaluate(0.5, 1000.0)
    0.25

    A 1X2 market where only leg 0 has a standalone edge still splits the
    stake: leg 1's odds are rich enough that once leg 0 is staked its
    coupled marginal value turns positive, while no-edge leg 2 stays at
    zero. Only the joint solve sees the coupling - the per-leg binary
    answer for leg 0 alone is the first coordinate:

    >>> strategy = MultiOutcomeKellyCriterion(payoffs=(3.2, 3.4, 2.4), loss=1.0)
    >>> stakes = strategy.evaluate([0.42, 0.27, 0.28], 1000.0)
    >>> tuple(round(stake, 6) for stake in stakes)
    (0.203681, 0.06253, 0.0)
    >>> KellyCriterion(payoff=2.2, loss=1.0, transaction_cost=0, min_probability=0.4).evaluate(0.42, 1000.0)
    0.15636363636363632
    """

    def __init__(
        self,
        payoffs: Sequence[float],
        loss: float,
        transaction_cost: float = 0,
        min_probability: float = 0.5,
    ):
        """
        Initialize the strategy.

        Parameters
        ----------
        payoffs : sequence of float
            The decimal-odds multiplier paid by each mutually exclusive leg,
            in leg order. Every payoff must be finite and greater than 0.
        loss : float
            The loss multiplier applied to every losing leg's stake.
        transaction_cost : float, optional
            The transaction cost as a fraction of each unit staked, by default 0.
        min_probability : float, default=0.5
            The minimum leg probability for the two-leg binary fallback to
            place a stake.

        Raises
        ------
        ValueError
            If any constructor argument is outside its documented range.
        """
        super().__init__(payoffs, loss, transaction_cost)
        if not 0 <= min_probability <= 1:
            raise ValueError("Minimum probability must be between 0 and 1")
        self.min_probability = min_probability
        # Net win per unit staked on each leg: decimal odds minus one, less
        # the per-unit fractional cost. Shared per-unit charge on every
        # losing leg.
        self._net_gains = tuple(
            payoff - 1.0 - transaction_cost for payoff in self.payoffs
        )
        self._total_cost = loss + transaction_cost
        # At N=2 the allocation is exact binary Kelly per leg, scored by the
        # same class the golden tests pin. A leg whose decimal payoff cannot
        # return a positive profit has no scorer and is never staked.
        self._binary_scorers: tuple[KellyCriterion | None, ...] | None = None
        if len(self.payoffs) == 2:
            self._binary_scorers = tuple(
                KellyCriterion(payoff - 1.0, loss, transaction_cost, min_probability)
                if payoff > 1.0
                else None
                for payoff in self.payoffs
            )

    def evaluate(
        self, probabilities: Sequence[float], current_bankroll: float
    ) -> tuple[float, ...]:
        """
        Calculate the log-growth optimal stake fraction per leg.

        Parameters
        ----------
        probabilities : sequence of float
            The probability of each mutually exclusive leg, in leg order.
            Must be a valid probability vector per
            :func:`keeks.utils.normalize_probabilities` and match the number
            of payoffs.
        current_bankroll : float
            The current bankroll amount. The optimal fractions do not depend
            on its size; a nonpositive bankroll has no safe stake at all.

        Returns
        -------
        tuple of float
            One stake fraction per leg: ``len(result) ==
            len(probabilities)``, every element finite and within ``[0, 1]``,
            and ``sum(result) <= 1 + PROBABILITY_SUM_TOLERANCE``.

        Raises
        ------
        ValueError
            If the probability vector is malformed or its length does not
            match the number of payoffs, or if ``current_bankroll`` is not
            finite.
        """
        probabilities = normalize_probabilities(probabilities)
        current_bankroll = _require_finite(current_bankroll, "Current bankroll")
        if len(probabilities) != len(self.payoffs):
            raise ValueError(
                f"Probabilities ({len(probabilities)}) must match the number of "
                f"payoffs ({len(self.payoffs)})"
            )
        if current_bankroll <= 0:
            return (0.0,) * len(probabilities)

        # The per-leg fallback is exact only for a fully priced book: when the
        # probabilities leave void mass, binary Kelly counts the refund branch
        # as a loss and understakes. Only when the probabilities sum to one
        # does the binary "lose" branch coincide with "another leg realizes".
        if (
            self._binary_scorers is not None
            and abs(sum(probabilities) - 1.0) <= PROBABILITY_SUM_TOLERANCE
        ):
            stakes = [
                scorer.evaluate(float(probability), current_bankroll)
                if scorer is not None
                else 0.0
                for scorer, probability in zip(
                    self._binary_scorers, probabilities, strict=False
                )
            ]
            if (
                sum(stake > 0.0 for stake in stakes) <= 1
                and _multipliers_within_floor(
                    stakes, self._net_gains, self._total_cost, probabilities
                )
                is not None
                and _delegation_is_joint_optimum(
                    stakes,
                    self._net_gains,
                    self._total_cost,
                    probabilities,
                    self._max_safe_total_fraction,
                )
            ):
                # At most one leg prices a positive binary stake, the result
                # keeps every other leg off the ruin boundary, and no other
                # leg has an improving direction at that point: the delegated
                # stakes are the exact joint optimum.
                return _validate_stake_fractions(stakes)
            # Otherwise solve the joint problem: both legs may price a
            # positive binary stake (an inconsistent arbitrage book), a
            # binary stake may drive a positive-probability leg through the
            # ruin floor (a float-rounded sure thing can hide a subnormal
            # branch), or - most importantly - the other legs' coupled
            # marginal value may be positive at the delegated point. For a
            # fully priced two-leg book that marginal is proportional to
            # ``w_0 * w_1 - l**2``, so two-sided books with rich payoffs on
            # both sides grow faster by staking BOTH legs than binary Kelly
            # on either one alone.

        stakes = _solve_log_growth_allocation(
            net_gains=self._net_gains,
            cost=self._total_cost,
            probabilities=probabilities,
            max_total=self._max_safe_total_fraction,
        )
        return _validate_stake_fractions(stakes)
