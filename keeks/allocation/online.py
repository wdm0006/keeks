"""
Online allocation strategies: weights that adapt from settled returns.

The online family sizes a portfolio the way sequential portfolio selection
does: the allocator holds a current weight vector, each staked period
settles with a realized joint simple-return vector, and that vector is the
only stateful channel - ``record_settlement(won, realized_returns)``, called
once per staked period. ``evaluate`` never mutates state; it only reads it,
so the weights a period stakes are the weights the previous settlements
produced.

Three members ship here:

- :class:`FixedWeights` - the constant rebalanced portfolio: the same bound
  weights every period, the regret benchmark every adaptive method is
  measured against.
- :class:`ExponentialGradient` - the follow-the-loser multiplicative
  (exponentiated gradient) update.
- :class:`OnlineNewtonStep` - the second-order follow-the-loser update with
  gradient outer-product state.

The family shares one sign convention - follow the loser. Each option's
realized simple return is read as a cost signal: options that realized low
returns gain weight, options that realized high returns lose weight. When
returns mean-revert - this period's laggard tends to be next period's
leader - that is the profitable direction, and it is the direction the
online portfolio selection "follow the loser" family (exponential gradient,
online Newton step) is built to exploit. On trending streams an adaptive
method can lag the best fixed portfolio, which is exactly the comparison
the :class:`FixedWeights` benchmark exists to expose.

All three honor the house weight contract: one long-only weight per option,
each in ``[0, 1]``, summing to no more than one, residual cash at zero
return, and all zeros for a nonpositive bankroll. The adaptive members
start from the uniform portfolio and stay fully invested (their weights
always sum to one); cash enters through :class:`FixedWeights`, whose bound
weights may sum to less than one. The option count binds at construction
because ``evaluate`` must return one weight per option from its very first
call - a simulator evaluates before the first settlement exists to learn
the count from.

Every update here is closed-form numpy; nothing in this module needs scipy.
"""

from collections.abc import Sequence

import numpy as np

from keeks.allocation.base import BaseAllocationStrategy, _validate_weights

__author__ = "willmcginnis"

# Cap on the multiplicative update's log-tilt. Beyond 50 log-points the tilt
# is absolute concentration anyway; the cap keeps an extreme realized return
# from overflowing exp() into inf or nan weights.
_MAX_LOG_TILT = 50.0


def _validate_option_count(option_count):
    """
    Validate an option count and return it as an ``int``.

    Parameters
    ----------
    option_count : int
        The number of options the allocator sizes across. Must be a true
        integer (not a bool) of at least one.

    Returns
    -------
    int
        The validated option count.

    Raises
    ------
    ValueError
        If the count is not an integer or is below one.

    Examples
    --------
    >>> _validate_option_count(3)
    3
    """
    if isinstance(option_count, bool) or not isinstance(option_count, int):
        raise ValueError("Option count must be an integer")
    if option_count < 1:
        raise ValueError("Option count must be at least one")
    return option_count


def _validate_positive_finite(value, label):
    """
    Validate a positive finite number and return it as a ``float``.

    The shared gate for the family's scalar hyperparameters (learning rates,
    the Newton step's ridge): zero, negative, NaN, and infinite values would
    each break the update they parameterize in its own way, so all four are
    rejected up front.

    Parameters
    ----------
    value : float
        The hyperparameter value.
    label : str
        Human-readable parameter name used in the error message.

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    ValueError
        If the value is not a positive finite number.

    Examples
    --------
    >>> _validate_positive_finite(0.05, "Learning rate")
    0.05
    """
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite number") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return value


def _validate_realized_returns(realized_returns, option_count=None):
    """
    Coerce a realized joint simple-return vector to a finite float array.

    One entry per option: the simple return each option realized over the
    settled period. This is the returns payload of the ``record_settlement``
    hook - the online family's only stateful channel - so it is validated at
    the door: a non-one-dimensional, empty, non-finite, or wrong-length vector
    is rejected rather than silently misaligning the update.

    Parameters
    ----------
    realized_returns : sequence of float
        The realized simple return of each option.
    option_count : int, optional
        When given, the vector must carry exactly that many entries - one
        per option the allocator sizes across.

    Returns
    -------
    numpy.ndarray
        The validated returns as a one-dimensional float array.

    Raises
    ------
    ValueError
        If the returns are not a non-empty one-dimensional sequence of
        finite numbers, or do not match ``option_count``.

    Examples
    --------
    >>> returns = _validate_realized_returns([0.01, -0.02])
    >>> returns.shape
    (2,)
    """
    try:
        returns = np.asarray(realized_returns, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Realized returns must be a finite sequence") from exc
    if returns.ndim != 1:
        raise ValueError("Realized returns must be one-dimensional")
    if returns.size == 0:
        raise ValueError("Realized returns must be non-empty")
    if option_count is not None and returns.size != option_count:
        raise ValueError(
            f"Realized returns must carry exactly {option_count} entries, "
            f"got {returns.size}"
        )
    if not np.all(np.isfinite(returns)):
        raise ValueError("Realized returns must contain only finite values")
    return returns


def _project_to_simplex(vector):
    """
    Project a vector onto the simplex ``{w : w >= 0, sum(w) == 1}``.

    Euclidean projection via the standard sort-based algorithm: find the
    largest rank whose shifted coordinate stays positive, subtract the
    common shift, and floor at zero. Deterministic and numpy-only. This is
    what keeps the second-order update's raw step - which can overshoot the
    weight bounds arbitrarily - inside the house weight contract.

    Parameters
    ----------
    vector : sequence of float
        The unconstrained update result, one entry per option. Must be
        non-empty; entries may be any finite values.

    Returns
    -------
    numpy.ndarray
        The projected weights: nonnegative, summing to one. The input is
        never modified.

    Examples
    --------
    >>> _project_to_simplex([0.7, 0.7]).tolist()
    [0.5, 0.5]
    >>> _project_to_simplex([-1.0, 2.0]).tolist()
    [0.0, 1.0]
    """
    vector = np.asarray(vector, dtype=float)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    ranks = np.arange(1, vector.size + 1)
    # Rank 1 is always feasible (v_max - (v_max - 1) == 1 > 0), so the
    # selection below is never empty.
    feasible = ordered - (cumulative - 1.0) / ranks > 0
    rho = ranks[feasible][-1]
    theta = (cumulative[rho - 1] - 1.0) / rho
    return np.maximum(vector - theta, 0.0)


class _OnlineAllocator(BaseAllocationStrategy):
    """
    Shared plumbing for the online family.

    Holds the current weight array and implements the house ``evaluate``
    contract once: scale-free weights that ignore the bankroll level, all
    zeros for a nonpositive bankroll, and the validated tuple on the way
    out. Subclasses replace the weight array through their settlement
    updates - they never mutate it in place - so ``record_settlement``
    remains the only stateful channel and repeated ``evaluate`` calls are
    pure reads.
    """

    def __init__(self, weights: np.typing.ArrayLike) -> None:
        self._weights = np.asarray(_validate_weights(weights), dtype=float)
        self._option_count = self._weights.size

    @property
    def weights(self) -> np.ndarray:
        """
        The allocator's current weight vector.

        The bound weights for :class:`FixedWeights`; for the adaptive
        members, the adaptation state - the vector the settlements so far
        produced. Also the readback for the constructor parameter of the
        same name, per the ``ParameterMixin`` convention; assigning through
        it validates like construction does.
        """
        return self._weights

    @weights.setter
    def weights(self, value: np.typing.ArrayLike) -> None:
        self._weights = np.asarray(
            _validate_weights(value, option_count=self._option_count), dtype=float
        )

    @property
    def option_count(self) -> int:
        """
        The number of options the allocator sizes across.

        Structural: the count binds at construction because ``evaluate``
        must return one weight per option from its very first call, so
        ``set_params`` can only re-assign the current count, never change
        it - reprice by fresh construction.
        """
        return self._option_count

    @option_count.setter
    def option_count(self, value: int) -> None:
        count = _validate_option_count(value)
        if count != self._option_count:
            raise ValueError(
                f"option_count is structural: it binds at construction, so it "
                f"cannot change from {self._option_count} to {count}; rebind "
                "by fresh construction"
            )

    def evaluate(self, current_bankroll: float) -> tuple[float, ...]:
        """
        Evaluate the strategy for the current bankroll.

        Returns
        -------
        tuple of float
            One long-only weight per option: every element finite and
            within ``[0, 1]``, the sum at most
            ``1 + PROBABILITY_SUM_TOLERANCE``. All zeros when
            ``current_bankroll <= 0``.
        """
        if current_bankroll <= 0:
            return _validate_weights(
                [0.0] * self._option_count, option_count=self._option_count
            )
        return _validate_weights(self._weights, option_count=self._option_count)


class FixedWeights(_OnlineAllocator):
    """
    Constant rebalanced portfolio: the same weight vector every period.

    The bound weights are staked every period forever - the best constant
    rebalanced portfolio that the adaptive online methods measure their
    regret against. It is also the family's cash door: a bound vector
    summing to less than one holds the residual at zero return, while the
    adaptive members are always fully invested.

    The weights bind at construction and are immutable thereafter, validated
    like every weight vector: finite, each in ``[0, 1]``, summing to no more
    than one. ``record_settlement`` validates the realized returns - the
    house hook contract - but updates nothing: the benchmark has no state.

    Examples
    --------
    >>> allocator = FixedWeights([0.25, 0.25, 0.5])
    >>> allocator.evaluate(1000.0)
    (0.25, 0.25, 0.5)
    >>> allocator.evaluate(0.0)
    (0.0, 0.0, 0.0)
    """

    def __init__(self, weights: np.typing.ArrayLike) -> None:
        super().__init__(weights)

    def record_settlement(
        self, won: Sequence[bool] | None, realized_returns: Sequence[float]
    ) -> None:
        """
        Validate the settlement; the benchmark holds no state.

        Parameters
        ----------
        won : sequence of bool or None
            The settled period's per-option outcome vector; the benchmark
            updates from the returns alone and ignores it.
        realized_returns : sequence of float
            The realized joint simple-return vector of the settled period.

        Raises
        ------
        ValueError
            If the vector is not a non-empty one-dimensional sequence of
            finite numbers matching the option count.
        """
        del won  # the benchmark reads the returns alone
        _validate_realized_returns(realized_returns, option_count=self._option_count)


class ExponentialGradient(_OnlineAllocator):
    """
    Follow-the-loser multiplicative update on realized returns.

    The exponentiated-gradient update: after each settled period with
    realized joint simple returns ``r``, every weight is scaled by an
    exponential of its option's realized return and the vector is
    renormalized to sum to one,

        ``w_i <- w_i * exp(-learning_rate * r_i)`` , renormalized.

    Under the follow-the-loser convention the realized return is a cost
    signal, so the minus sign sends weight toward the options that just
    underperformed - the profitable direction when returns mean-revert. The
    exponent is clipped at ``±_MAX_LOG_TILT`` log-points in magnitude, so an
    extreme settlement concentrates the portfolio instead of overflowing.

    The option count binds at construction and the portfolio starts uniform
    and fully invested. ``record_settlement`` is the only stateful channel:
    ``evaluate`` never mutates state. Once an option's weight reaches zero
    it stays there - multiplicative updates cannot revive it - though the
    uniform start and the positive renormalizer keep every weight strictly
    positive under any finite settlement history.

    Parameters
    ----------
    option_count : int
        The number of options to size across; binds at construction.
    learning_rate : float
        Positive adaptivity rate - larger values tilt harder toward the
        most recent losers. Defaults to ``0.05``.

    Examples
    --------
    >>> allocator = ExponentialGradient(option_count=2, learning_rate=0.5)
    >>> allocator.evaluate(1000.0)
    (0.5, 0.5)
    >>> allocator.record_settlement((True, False), [0.2, -0.1])
    >>> weights = allocator.evaluate(1000.0)
    >>> weights[1] > weights[0]
    True
    >>> sum(weights) <= 1 + 1e-12
    True
    """

    def __init__(self, option_count: int, learning_rate: float = 0.05) -> None:
        count = _validate_option_count(option_count)
        super().__init__(np.full(count, 1.0 / count))
        self.learning_rate = _validate_positive_finite(learning_rate, "Learning rate")

    def record_settlement(
        self, won: Sequence[bool] | None, realized_returns: Sequence[float]
    ) -> None:
        """
        Advance the weights with the settled joint simple returns.

        Parameters
        ----------
        won : sequence of bool or None
            The settled period's per-option outcome vector; the gradient
            update reads the returns alone and ignores it.
        realized_returns : sequence of float
            The realized joint simple-return vector of the settled period;
            one entry per option.

        Raises
        ------
        ValueError
            If the vector is not a non-empty one-dimensional sequence of
            finite numbers matching the option count.
        """
        del won  # the gradient update reads the returns alone
        returns = _validate_realized_returns(
            realized_returns, option_count=self._option_count
        )
        log_tilt = np.clip(-self.learning_rate * returns, -_MAX_LOG_TILT, _MAX_LOG_TILT)
        tilted = self._weights * np.exp(log_tilt)
        self._weights = tilted / tilted.sum()
        # Fail loudly at the settlement site if the update ever produced
        # something outside the weight contract.
        _validate_weights(self._weights, option_count=self._option_count)


class OnlineNewtonStep(_OnlineAllocator):
    """
    Second-order follow-the-loser update with gradient outer-product state.

    The online Newton step: the per-period follow-the-loser cost is the
    linear ``<w, r>`` whose gradient at any ``w`` is the realized return
    vector ``r`` itself. The allocator accumulates those gradients' outer
    products into a regularized Gram matrix,

        ``A = epsilon * I + sum_t r_t r_t'``

    and takes a Newton-style step on the cost before projecting back onto
    the simplex:

        ``y = w - learning_rate * A^{-1} r`` ,  ``w <- project(y)``.

    The inverse outer-product mass shrinks the step along directions the
    return stream has already explored and leaves it near full size along
    new ones, so adaptation concentrates where the stream still varies. The
    settlement that triggers the update contributes its outer product
    before the step, so early steps scale with the observed return
    magnitude instead of being amplified by ``1 / epsilon``.

    Larger ``learning_rate`` values adapt faster; ``epsilon`` is the small
    ridge that keeps ``A`` invertible when the return stream never spans
    some direction. The raw step can overshoot the weight bounds -
    projecting onto the simplex keeps every weight long-only and fully
    invested - so early adaptation is aggressive on purpose: raise
    ``epsilon`` or lower ``learning_rate`` to temper it.

    Parameters
    ----------
    option_count : int
        The number of options to size across; binds at construction.
    learning_rate : float
        Positive step size. Defaults to ``0.5``.
    epsilon : float
        Positive ridge added to the Gram matrix's diagonal. Defaults to
        ``1e-6``.

    Examples
    --------
    >>> allocator = OnlineNewtonStep(option_count=2)
    >>> allocator.evaluate(1000.0)
    (0.5, 0.5)
    >>> allocator.record_settlement((True, False), [0.05, -0.05])
    >>> weights = allocator.evaluate(1000.0)
    >>> weights[1] > weights[0]
    True
    >>> sum(weights) <= 1 + 1e-12
    True
    """

    def __init__(
        self, option_count: int, learning_rate: float = 0.5, epsilon: float = 1e-6
    ) -> None:
        count = _validate_option_count(option_count)
        super().__init__(np.full(count, 1.0 / count))
        self.learning_rate = _validate_positive_finite(learning_rate, "Learning rate")
        self.epsilon = _validate_positive_finite(epsilon, "Epsilon")
        self._outer_products = np.zeros((count, count))

    def record_settlement(
        self, won: Sequence[bool] | None, realized_returns: Sequence[float]
    ) -> None:
        """
        Advance the weights with the settled joint simple returns.

        Parameters
        ----------
        won : sequence of bool or None
            The settled period's per-option outcome vector; the Newton step
            reads the returns alone and ignores it.
        realized_returns : sequence of float
            The realized joint simple-return vector of the settled period;
            one entry per option.

        Raises
        ------
        ValueError
            If the vector is not a non-empty one-dimensional sequence of
            finite numbers matching the option count.
        """
        del won  # the Newton step reads the returns alone
        returns = _validate_realized_returns(
            realized_returns, option_count=self._option_count
        )
        self._outer_products += np.outer(returns, returns)
        gram = self.epsilon * np.eye(self._option_count) + self._outer_products
        newton_direction = np.linalg.pinv(gram) @ returns
        candidate = self._weights - self.learning_rate * newton_direction
        self._weights = _project_to_simplex(candidate)
        # Fail loudly at the settlement site if the update ever produced
        # something outside the weight contract.
        _validate_weights(self._weights, option_count=self._option_count)
