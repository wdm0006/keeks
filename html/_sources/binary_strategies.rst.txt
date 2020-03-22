Binary Strategies
=================

Binary strategies are strategies that reason about singular binary bets. These bets have some payoff if they win, some
loss if they lose, and a transaction cost regardless. Bets are not continuous, they either win or lose.

.. autoclass:: keeks.binary_strategies.simple.NaiveStrategy
    :members:

.. autoclass:: keeks.binary_strategies.kelly.KellyCriterion
    :members:

.. autoclass:: keeks.binary_strategies.kelly.FractionalKellyCriterion
    :members:
