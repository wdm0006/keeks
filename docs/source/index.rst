.. keeks documentation master file, created by
   sphinx-quickstart on Sat Mar 21 15:10:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Keeks
================

Keeks is a Python library for bet sizing and bankroll simulation, with a portfolio-allocation layer for sizing many options together.

**Size a single bet.** Given your estimated win probability, payoff, loss, and transaction cost, Keeks' strategies — the Kelly Criterion and eight variants — return the fraction of your bankroll to stake, and simulators replay that same rule over repeated trials to show the bankroll path it produces.

**Size a portfolio.** For portfolios of options described by return distributions rather than single bets, the :doc:`allocation <allocation>` layer builds one joint-return model from what you have and sizes the portfolio at once, returning one long-only weight per option.

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
   bankroll = BankRoll(initial_funds=1000.0, max_transaction_loss=0.3)

   # Create a Kelly Criterion strategy
   strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost_rate=0.01)

   # Create a simulator with a fixed probability
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

Available Strategies
--------------------

Keeks implements nine betting strategies:

- **Kelly Criterion**: The mathematically optimal strategy for maximizing the logarithm of wealth
- **Fractional Kelly**: A more conservative version of Kelly that reduces volatility
- **Drawdown-Adjusted Kelly**: A Kelly variant that adjusts bet sizing based on risk tolerance
- **OptimalF (Ralph Vince)**: Strategy that maximizes geometric growth rate
- **Fixed Fraction**: Simple strategy that bets a constant percentage of the bankroll
- **CPPI (Constant Proportion Portfolio Insurance)**: Strategy that protects a floor value while allowing upside exposure
- **Dynamic Bankroll Management**: Adaptive strategy based on recent performance
- **MertonShare**: A CRRA risk-aversion rule adapted to binary outcomes
- **Naive Strategy**: A simple strategy that bets the full amount when expected value is positive

For portfolios of distribution-valued options, the :doc:`allocation <allocation>` layer provides mean-variance, risk budgeting, HRP, mean-CVaR, and online methods over one configurable joint-return input model.

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

.. toctree::
   :caption: Getting Started
   :maxdepth: 2

   getting_started

.. toctree::
   :caption: Learn
   :maxdepth: 1

   Kelly Criterion in Python <kelly-criterion-python>
   Full Kelly vs Fractional Kelly <fractional-kelly-vs-kelly>
   Keeks vs betting-math-kit <keeks-vs-betting-math-kit>
   Strategy Benchmark <strategy_benchmark>

.. toctree::
   :caption: Core Reference
   :maxdepth: 2

   Binary Strategies <binary_strategies>
   Simulators <simulators>
   Multi-Outcome <multi_outcome>
   Bankroll <bankroll>
   Utilities <utils>

.. toctree::
   :caption: Portfolio Allocation
   :maxdepth: 2

   Allocation <allocation>
   Visualization <allocation_plots>

.. toctree::
   :caption: Examples
   :maxdepth: 1

   Allocation over ETF returns <examples/allocation_etfs>
   1X2 betting with multi-outcome Kelly <examples/multi_outcome_1x2>
   The St. Petersburg paradox <examples/st_petersburg_paradox>
   Strategies on a marginal edge <examples/strategy_comparison>
   Strategy lab notebook <examples/strategy_lab>


.. toctree::
   :caption: Contributing
   :maxdepth: 1

   Contract Checks <checks>
