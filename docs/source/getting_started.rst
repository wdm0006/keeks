Getting Started
===============

Keeks sizes bets and bankrolls. Given an estimated win probability, payoff, loss, and transaction cost, its strategies return the fraction of your bankroll to stake, and its simulators replay that rule over repeated trials to show the bankroll path it produces. This page walks through installation and a first simulation; for portfolios of distribution-valued options, Keeks also provides an allocation layer that sizes many options together — see `Next: Sizing a Portfolio`_ at the end of this page.

Installation
------------

You can install keeks using pip:

.. code-block:: bash

    pip install keeks

Basic Usage
-----------

Here's a simple example of how to use keeks to simulate a betting strategy:

.. code-block:: python

    from keeks.bankroll import BankRoll
    from keeks.binary_strategies.kelly import KellyCriterion
    from keeks.simulators.repeated_binary import RepeatedBinarySimulator

    # Create a bankroll with initial funds
    bankroll = BankRoll(initial_funds=1000.0, max_transaction_loss=0.3)

    # Create a Kelly Criterion strategy
    # Parameters: payoff, loss, transaction_cost_rate
    strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost_rate=0.01)

    # Create a simulator with a fixed probability
    # Parameters: payoff, loss, fee_per_bet, probability, trials
    simulator = RepeatedBinarySimulator(
        payoff=1.0, 
        loss=1.0, 
        fee_per_bet=0.01, 
        probability=0.55,  # 55% chance of winning
        trials=1000
    )

    # Run the simulation
    simulator.evaluate_strategy(strategy, bankroll)

    # Plot the results
    bankroll.plot_history()

Using Different Strategies
--------------------------

Keeks provides several betting strategies:

1. **Kelly Criterion**: Optimal bet sizing based on probability and odds
   
   .. code-block:: python
   
      from keeks.binary_strategies import KellyCriterion
      strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost_rate=0.01)
   
2. **Fractional Kelly**: A more conservative version of Kelly
   
   .. code-block:: python
   
      from keeks.binary_strategies import FractionalKellyCriterion
      strategy = FractionalKellyCriterion(payoff=1.0, loss=1.0, transaction_cost_rate=0.01, fraction=0.5)
   
3. **Drawdown-Adjusted Kelly**: Kelly variant that accounts for drawdown tolerance
   
   .. code-block:: python
   
      from keeks.binary_strategies import DrawdownAdjustedKelly
      strategy = DrawdownAdjustedKelly(payoff=1.0, loss=1.0, transaction_cost_rate=0.01, max_transaction_loss=0.2)
   
4. **OptimalF**: Ralph Vince's method for maximizing geometric growth rate
   
   .. code-block:: python
   
      from keeks.binary_strategies.simple import OptimalF
      strategy = OptimalF(payoff=1.0, loss=1.0, transaction_cost_rate=0.01, win_rate=0.55)
   
5. **Fixed Fraction**: Simple strategy that bets a constant percentage
   
   .. code-block:: python
   
      from keeks.binary_strategies import FixedFractionStrategy
      strategy = FixedFractionStrategy(
          fraction=0.05,
          payoff=1.0,
          loss=1.0,
          transaction_cost_rate=0.01,
          min_probability=0.5,
      )
   
6. **CPPI**: Constant Proportion Portfolio Insurance for capital preservation
   
   .. code-block:: python
   
      from keeks.binary_strategies import CPPIStrategy
      strategy = CPPIStrategy(
          floor_fraction=0.5,
          multiplier=2.0,
          initial_bankroll=1000.0,
          payoff=1.0,
          loss=1.0,
          transaction_cost_rate=0.01,
      )
      
      # Remember to update the CPPI strategy with the current bankroll value
      # before each evaluation
      strategy.update_bankroll(current_bankroll)
   
7. **Dynamic Bankroll Management**: Adaptive allocation based on recent performance
   
   .. code-block:: python
   
      from keeks.binary_strategies import DynamicBankrollManagement
      strategy = DynamicBankrollManagement(
          base_fraction=0.1, payoff=1.0, loss=1.0, transaction_cost_rate=0.01
      )

      # After each settled bet, update the strategy with the outcome
      # (one-entry won and realized-return vectors for a single bet)
      strategy.record_settlement((True,), (0.05,))
   
8. **Naive Strategy**: Simple strategy that bets full amount when expected value is positive
   
   .. code-block:: python
   
      from keeks.binary_strategies import NaiveStrategy
      strategy = NaiveStrategy(payoff=1.0, loss=1.0, transaction_cost_rate=0.01)

Using Different Simulators
--------------------------

Keeks provides several simulators:

1. **RepeatedBinarySimulator**: Simulates repeated bets with a fixed probability
2. **RandomBinarySimulator**: Simulates bets with random probabilities
3. **RandomUncertainBinarySimulator**: Adds uncertainty to the actual outcome probabilities

Next: Sizing a Portfolio
------------------------

The strategies and simulators above size one bet at a time. For portfolios of options described by return distributions rather than single bets, the `allocation layer <https://keeks.mcginniscommawill.com/allocation.html>`_ builds one joint-return model from what you have and sizes the portfolio at once, returning one long-only weight per option. For a complete worked example, see `examples/allocation_etfs.py <https://github.com/wdm0006/keeks/blob/master/examples/allocation_etfs.py>`_ in the repository.