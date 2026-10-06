"""
Simulators for evaluating betting strategies in the keeks package.

This module provides various simulators for testing betting strategies:
- HistoricalBinarySimulator: Replays recorded probabilities and outcomes
- RandomBinarySimulator: Simulates bets with random probabilities
- RandomUncertainBinarySimulator: Adds uncertainty to the actual outcome probabilities
- RepeatedBinarySimulator: Simulates repeated bets with a fixed probability

These simulators can be used to evaluate the performance of different betting strategies
under various conditions.
"""

from keeks.simulators.historical_binary import HistoricalBinarySimulator
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

__all__ = [
    "HistoricalBinarySimulator",
    "RandomBinarySimulator",
    "RandomUncertainBinarySimulator",
    "RepeatedBinarySimulator",
]
