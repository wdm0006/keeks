Getting Started
===============

Installation
-----------

You can install keeks using pip:

.. code-block:: bash

    pip install keeks

Basic Usage
----------

Here's a simple example of how to use keeks to simulate a betting strategy:

.. code-block:: python

    from keeks.bankroll import BankRoll
    from keeks.binary_strategies.kelly import KellyCriterion
    from keeks.simulators.repeated_binary import RepeatedBinarySimulator

    # Create a bankroll with initial funds
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=0.3)

    # Create a Kelly Criterion strategy
    # Parameters: payoff, loss, transaction_cost
    strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.01)

    # Create a simulator with a fixed probability
    # Parameters: payoff, loss, transaction_costs, probability, trials
    simulator = RepeatedBinarySimulator(
        payoff=1.0, 
        loss=1.0, 
        transaction_costs=0.01, 
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
2. **Fractional Kelly**: A more conservative version of Kelly
3. **Naive Strategy**: Simple strategy that bets full amount when expected value is positive

Using Different Simulators
--------------------------

Keeks provides several simulators:

1. **RepeatedBinarySimulator**: Simulates repeated bets with a fixed probability
2. **RandomBinarySimulator**: Simulates bets with random probabilities
3. **RandomUncertainBinarySimulator**: Adds uncertainty to the actual outcome probabilities