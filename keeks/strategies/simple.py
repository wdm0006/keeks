from keeks.strategies.base import BaseStrategy

__author__ = 'willmcginnis'


class AllOnBest(BaseStrategy):
    def __init__(self, bankroll, verbose=0):
        self.bankroll = bankroll
        self.verbose = verbose

    def evaluate_opportunities(self, opps):
        # first get the basic allocations, ignoring negative ones.
        if len(opps) > 0:
            opps = sorted(opps, key=lambda x: x.probability, reverse=True)
            return {opps[0].label: round(self.bankroll.bettable_funds, 2)}
        else:
            return dict()


class AllOnBestExpectedValue(BaseStrategy):
    def __init__(self, bankroll, verbose=0):
        self.bankroll = bankroll
        self.verbose = verbose

    def evaluate_opportunities(self, opps):
        # first get the basic allocations, ignoring negative ones.
        if len(opps) > 0:
            opps = sorted(opps, key=lambda x: x.probability * x.odds, reverse=True)
            return {opps[0].label: round(self.bankroll.bettable_funds, 2)}
        else:
            return dict()