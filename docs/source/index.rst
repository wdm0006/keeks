.. keeks documentation master file, created by
   sphinx-quickstart on Sat Mar 21 15:10:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Keeks
===============

Keeks is a specialized Python library for optimal bankroll allocation and betting strategies, with a focus on the Kelly Criterion and its variants.

What is Keeks?
--------------

Keeks provides tools for implementing and testing various betting and investment strategies. It includes:

- **Bankroll management**: Track and manage your funds with built-in protection against excessive losses
- **Betting strategies**: Implement mathematically optimal strategies like the Kelly Criterion
- **Simulation**: Test your strategies under different conditions before risking real money

Whether you're a sports bettor, a financial trader, or a researcher in decision theory, Keeks provides the tools to make more informed decisions about capital allocation.

Why Use Keeks?
--------------

- **Mathematically sound**: Based on proven mathematical principles like the Kelly Criterion
- **Risk management**: Built-in protection against ruin with configurable drawdown limits
- **Simulation-driven**: Test strategies in various scenarios before applying them with real money
- **Flexible**: Supports different types of betting scenarios and probability distributions
- **Educational**: Learn about optimal betting strategies through practical implementation

**Disclaimer**: This library is for educational purposes only. It is not intended to provide investment, legal, or tax advice. Always be responsible and consult with a professional before applying these strategies to real-world betting or investment scenarios. The authors and contributors of this library are not liable for any financial losses or damages that may result from the use of this software.

Installation
------------

.. code-block:: bash

   pip install keeks

Quick Example
-------------

.. code-block:: python

   from keeks.bankroll import BankRoll
   from keeks.binary_strategies.kelly import KellyCriterion
   from keeks.simulators.repeated_binary import RepeatedBinarySimulator

   # Create a bankroll with initial funds
   bankroll = BankRoll(initial_funds=1000.0, max_draw_down=0.3)

   # Create a Kelly Criterion strategy
   strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.01)

   # Create a simulator with a fixed probability
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

Available Strategies
--------------------

Keeks implements various bankroll allocation strategies:

- **Kelly Criterion**: The mathematically optimal strategy for maximizing the logarithm of wealth
- **Fractional Kelly**: A more conservative version of Kelly that reduces volatility
- **Drawdown-Adjusted Kelly**: A Kelly variant that adjusts bet sizing based on risk tolerance
- **OptimalF (Ralph Vince)**: Strategy that maximizes geometric growth rate
- **Fixed Fraction**: Simple strategy that bets a constant percentage of the bankroll
- **CPPI (Constant Proportion Portfolio Insurance)**: Strategy that protects a floor value while allowing upside exposure
- **Dynamic Bankroll Management**: Adaptive strategy based on recent performance
- **Naive Strategy**: A simple strategy that bets the full amount when expected value is positive

Each strategy offers different tradeoffs between risk and reward, allowing you to select the approach that best matches your investment goals and risk tolerance.

Applications
------------

Keeks can be applied to various domains:

- **Sports Betting**: Optimize your bet sizing based on your edge
- **Financial Trading**: Apply Kelly principles to portfolio management
- **Gambling**: Understand the mathematics behind optimal betting
- **Research**: Study the behavior of different betting strategies
- **Education**: Learn about probability, statistics, and risk management

Development
-----------

The source code for Keeks is hosted on GitHub:

.. code-block:: bash

   git clone https://github.com/wdm0006/keeks.git
   cd keeks
   pip install -e ".[dev]"

References
----------

- `A New Interpretation of Information Rate <http://www.herrold.com/brokerage/kelly.pdf>`_ - The original Kelly Criterion paper
- `Fortune's Formula <https://www.amazon.com/Fortunes-Formula-Scientific-Betting-Casinos/dp/0809045990>`_ - The untold story of the scientific betting system that beat the casinos and Wall Street

---


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   binary_strategies
   simulators
   bankroll
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
