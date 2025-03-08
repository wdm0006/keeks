# Keeks Examples

This directory contains example scripts that demonstrate how to use the keeks library for various financial simulation and strategy evaluation scenarios.

## Available Examples

### St. Petersburg Paradox Simulation (`st_petersburg_comparison.py`)

This example uses a simplified version of the St. Petersburg paradox to compare different bankroll management strategies. The St. Petersburg paradox is a theoretical gambling game with infinite expected value, but in practice has finite outcomes.

In our simulation:
- We use a binary betting system with a 55% win probability
- The payoff is 1.1:1 (win 1.1x your bet)
- We compare multiple binary betting strategies available in keeks
- Results are presented as both data tables and visualizations

#### Sample Output

![St. Petersburg Paradox Strategy Comparison](output/st_petersburg_strategies_comparison.png)

*This chart shows the distribution of final bankrolls across different betting strategies after multiple simulations. Notice how Optimal-F and Kelly Criterion achieved the highest returns but with greater volatility, while more conservative strategies like Quarter Kelly had more consistent (but lower) returns.*

#### Running the Example

To run the St. Petersburg paradox simulation:

```bash
python examples/st_petersburg_comparison.py
```

The script will:
1. Run simulations for all available binary strategies
2. Output comparative performance statistics to the console
3. Generate a visualization showing the distribution of final bankrolls
4. Save the results to CSV and PNG files in the `examples/output` directory

#### Required Dependencies

- matplotlib
- pandas
- numpy

## Adding New Examples

When adding new examples to this directory, please follow these guidelines:

1. Include detailed documentation and comments in your code
2. Provide a clear explanation of what the example demonstrates
3. Add your example to this README.md file with a brief description
4. Include any special requirements or dependencies needed to run the example 