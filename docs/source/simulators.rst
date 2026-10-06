Simulators
==========

Simulators are classes that take in a bankroll and a strategy and apply that strategy to a situation. They can be used
to evaluate the efficacy of strategies in certain domains.

Repeated Binary Simulator
-------------------------

.. autoclass:: keeks.simulators.repeated_binary.RepeatedBinarySimulator
    :members:
    :undoc-members:
    :show-inheritance:

Random Binary Simulator
-----------------------

.. autoclass:: keeks.simulators.random_binary.RandomBinarySimulator
    :members:
    :undoc-members:
    :show-inheritance:

Random Uncertain Binary Simulator
---------------------------------

.. autoclass:: keeks.simulators.random_uncertain_binary.RandomUncertainBinarySimulator
    :members:
    :undoc-members:
    :show-inheritance:

Historical Binary Simulator
---------------------------

Replays a log of real bets: each trial uses a recorded probability and settles on the
recorded outcome, so you can ask what a strategy would have done with your own bets.
It draws no random numbers and takes no ``seed``; the number of trials is the length
of the log. Settlement matches the repeated binary simulator.

.. code-block:: python

    from keeks import BankRoll, HistoricalBinarySimulator, KellyCriterion

    simulator = HistoricalBinarySimulator(
        payoff=1.0, loss=1.0, fee_per_bet=0.0,
        probabilities=[0.55, 0.60, 0.52],
        outcomes=[True, False, True],
    )
    bankroll = BankRoll(initial_funds=1000)
    simulator.evaluate_strategy(KellyCriterion(payoff=1.0, loss=1.0, transaction_cost_rate=0.0), bankroll)

.. autoclass:: keeks.simulators.historical_binary.HistoricalBinarySimulator
    :members:
    :undoc-members:
    :show-inheritance:
