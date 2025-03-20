Binary Strategies
=================

Binary strategies are strategies that reason about singular binary bets. These bets have some payoff if they win, some
loss if they lose, and a transaction cost regardless. Bets are not continuous, they either win or lose.

Base Strategy
-------------

.. autoclass:: keeks.binary_strategies.base.BaseStrategy
    :members:
    :undoc-members:
    :show-inheritance:

Naive Strategy
--------------

.. autoclass:: keeks.binary_strategies.simple.NaiveStrategy
    :members:
    :undoc-members:
    :show-inheritance:

Fixed Fraction Strategy
-----------------------

.. autoclass:: keeks.binary_strategies.simple.FixedFractionStrategy
    :members:
    :undoc-members:
    :show-inheritance:

CPPI Strategy
-------------

.. autoclass:: keeks.binary_strategies.simple.CPPIStrategy
    :members:
    :undoc-members:
    :show-inheritance:

Dynamic Bankroll Management
---------------------------

.. autoclass:: keeks.binary_strategies.simple.DynamicBankrollManagement
    :members:
    :undoc-members:
    :show-inheritance:

Kelly Criterion
---------------

.. autoclass:: keeks.binary_strategies.kelly.KellyCriterion
    :members:
    :undoc-members:
    :show-inheritance:

Fractional Kelly Criterion
--------------------------

.. autoclass:: keeks.binary_strategies.kelly.FractionalKellyCriterion
    :members:
    :undoc-members:
    :show-inheritance:

Drawdown-Adjusted Kelly
-----------------------

.. autoclass:: keeks.binary_strategies.kelly.DrawdownAdjustedKelly
    :members:
    :undoc-members:
    :show-inheritance:

OptimalF
--------

.. autoclass:: keeks.binary_strategies.simple.OptimalF
    :members:
    :undoc-members:
    :show-inheritance:
