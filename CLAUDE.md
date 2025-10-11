# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Keeks is a Python library for optimal bankroll allocation and betting strategies, focusing on the Kelly Criterion and its variants. The library provides tools for:
- Bankroll management with risk protection
- Mathematical betting strategies (Kelly, Fractional Kelly, etc.)
- Simulation of betting scenarios with different probability distributions

## Development Commands

This project uses `uv` as the package manager. All commands are available via the Makefile.

### Setup
```bash
make setup         # Install uv and create virtual environment
make install-dev   # Install package with development dependencies
```

### Testing
```bash
make test          # Run pytest
make test-cov      # Run tests with coverage report
make test-all      # Run tests on all Python versions using tox

# Run a single test file
uv run pytest tests/test_kelly_criterion.py

# Run a specific test
uv run pytest tests/test_kelly_criterion.py::test_basic_functionality

# Custom pytest arguments
make test PYTEST_ARGS="tests/test_bankroll.py -v"
```

### Code Quality
```bash
make lint          # Run ruff checks
make lint-fix      # Auto-fix linting issues (includes unsafe fixes)
make format        # Format code with ruff
```

### Documentation
```bash
make docs          # Build Sphinx documentation and open in Chrome
```

### Other
```bash
make build         # Build package distributions
make clean         # Clean build artifacts
make examples      # Run example scripts (st_petersburg_comparison.py)
```

## Architecture

### Four-Component Design

The library follows a four-component architecture:

1. **BankRoll** (`keeks/bankroll.py`): Central state manager
   - Tracks total funds and bettable funds (via `percent_bettable`)
   - Maintains transaction history
   - Enforces risk limits via `max_draw_down` parameter
   - Raises `RuinError` when drawdown limit exceeded
   - Key methods: `deposit()`, `withdraw()`, `bet()`, `add_funds()`, `remove_funds()`

2. **Strategies** (`keeks/binary_strategies/`): Decision engines
   - All inherit from `BaseStrategy` abstract class
   - Must implement `evaluate(probability, current_bankroll)` returning bet fraction
   - Strategy constructors take: `payoff`, `loss`, `transaction_cost`
   - Available strategies: `KellyCriterion`, `FractionalKellyCriterion`, `DrawdownAdjustedKelly`, `OptimalF`, `FixedFractionStrategy`, `CPPIStrategy`, `DynamicBankrollManagement`, `MertonShare`, `NaiveStrategy`

3. **Simulators** (`keeks/simulators/`): Test harnesses
   - `RepeatedBinarySimulator`: Fixed probability across all trials
   - `RandomBinarySimulator`: Normally distributed probabilities
   - `RandomUncertainBinarySimulator`: Adds uncertainty to outcomes
   - All simulators have `evaluate_strategy(strategy, bankroll)` method that runs trials and updates bankroll in-place

4. **Utility Functions** (`keeks/utils.py`): Decision theory tools
   - `crra_utility(wealth, risk_aversion)`: Calculate CRRA utility values
   - `expected_utility(outcomes, probabilities, current_wealth, entry_price, risk_aversion)`: Calculate expected utility of a gamble
   - `find_indifference_price(outcomes, probabilities, current_wealth, risk_aversion)`: Find maximum price willing to pay for a gamble
   - Used for one-time decision problems (e.g., St. Petersburg paradox) vs. repeated betting strategies

### Key Design Patterns

- **Strategy Pattern**: Betting strategies are interchangeable implementations of `BaseStrategy`
- **State Management**: `BankRoll` maintains all financial state; strategies are stateless (except for state-dependent variants)
- **In-Place Updates**: Simulators modify the `BankRoll` object directly rather than returning new values
- **Safety-First**: Strategies use `get_max_safe_bet()` to prevent negative bankroll scenarios

### Important Constraints

- **No negative bets**: All strategies validated in `BaseStrategy.__init__()` to ensure `payoff > 0`, `loss >= 0`, `transaction_cost >= 0`
- **Drawdown protection**: `BankRoll.max_draw_down` enforces maximum loss per transaction; raises `RuinError` when violated
- **Test isolation**: NEVER add test-specific handling or special case logic in library code (see python_standards.mdc)

## Code Standards

### Python Style
- Line length: 88 characters (ruff/Black default)
- Use snake_case for functions/variables, CamelCase for classes
- Type hints preferred for public APIs
- Docstrings: NumPy/Google style with parameters, returns, and examples

### Testing Standards
- Tests in `tests/` directory, named `test_*.py`
- Use `pytest.approx()` for floating-point comparisons
- Random seed (`random.seed(42)`) for reproducible simulations
- Target 80%+ code coverage
- Run specific tests: `uv run pytest tests/test_file.py::test_name`

### Import Organization
1. Standard library (random, abc)
2. Third-party (numpy, matplotlib, pandas, pytest)
3. Local imports (keeks modules)

## Common Patterns

### Testing a Strategy
```python
# Pattern used throughout tests
bankroll = BankRoll(initial_funds=1000, percent_bettable=0.5, max_draw_down=0.3)
strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.01)
simulator = RepeatedBinarySimulator(
    payoff=1.0, loss=1.0, transaction_costs=0.01,
    probability=0.55, trials=1000
)
simulator.evaluate_strategy(strategy, bankroll)
```

### Strategy Implementation
```python
class CustomStrategy(BaseStrategy):
    def evaluate(self, probability: float, current_bankroll: float) -> float:
        # Calculate bet fraction (0.0 to 1.0)
        fraction = ...  # your logic
        # Always respect safety limits
        return min(fraction, self.get_max_safe_bet(current_bankroll))
```

### Using MertonShare Strategy
```python
# MertonShare uses CRRA utility (Constant Relative Risk Aversion)
# Formula: f* = μ / (γ × σ²) where:
#   μ = expected return
#   γ = risk aversion coefficient (1=aggressive, 2=moderate, 5=conservative)
#   σ² = variance of returns

from keeks.binary_strategies.simple import MertonShare

# Moderate risk aversion (γ=2.0 is empirically common)
strategy = MertonShare(
    payoff=1.0,
    loss=1.0,
    transaction_cost=0.01,
    risk_aversion=2.0,  # Higher values = more conservative
    min_probability=0.5,
    max_fraction=1.0
)

# MertonShare is typically more conservative than Kelly Criterion
# Useful for risk-averse investors or when parameter uncertainty is high
```

### Using Utility Functions for One-Time Decisions
```python
# For one-time gamble decisions (not repeated betting)
from keeks.utils import find_indifference_price

# St. Petersburg paradox example: what would you pay to play?
# Generate outcomes and probabilities for St. Petersburg game
outcomes = [2**n for n in range(1, 31)]  # Payouts: $2, $4, $8, ..., $2^30
probabilities = [(0.5)**n for n in range(1, 31)]  # Probs: 0.5, 0.25, 0.125, ...

max_price = find_indifference_price(
    outcomes=outcomes,
    probabilities=probabilities,
    current_wealth=10000,
    risk_aversion=2.0,  # Moderate risk aversion
    tolerance=0.01
)

print(f"Maximum willing to pay: ${max_price:.2f}")
# Output: Maximum willing to pay: $12.80

# For comparison, Kelly bettor (γ=1.0) would pay ~$14.24
# Conservative bettor (γ=5.0) would pay ~$11.26
```

## Project-Specific Notes

- **Version**: Managed in `pyproject.toml` (currently 0.3.0)
- **Python Support**: 3.8, 3.9, 3.10, 3.11
- **Documentation site**: [keeks.mcginniscommawill.com](https://keeks.mcginniscommawill.com)
- **Educational focus**: Library includes disclaimer about educational use only
- **Examples**:
  - `strategy_comparison.py`: Compares all strategies with favorable odds
  - `st_petersburg_paradox.py`: True St. Petersburg paradox - shows rational payment amounts
