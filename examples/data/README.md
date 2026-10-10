# Synthetic bet log

`synthetic_bet_log.csv` contains 60 **synthetic**, chronological even-money
bets. It contains no real bettors or market data. Estimates are probabilities
available before settlement; `outcome` is `1` for a win and `0` for a loss.

The committed rows were generated once with Python's `random.Random(140)`.
For each row, draw `p = rng.choice([0.52, 0.55, 0.58, 0.60])`, then draw
`int(rng.random() < p)` as the outcome. Probabilities are written to two decimal
places. Replay reads the committed file and needs no generator or random seed.

All bets use net win payoff 1.0, loss 1.0 and zero fees. This single synthetic
path illustrates a workflow; it does not establish a strategy's expected return
or validate probability calibration. Educational only, not financial advice.
