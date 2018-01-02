from keeks.strategies.base import BaseStrategy


class Blended(BaseStrategy):
    def __init__(self, bankroll, strategies, verbose=0):
        self.bankroll = bankroll
        self.strategies = strategies
        self.verbose = verbose

    def evaluate_opportunities(self, opps):
        strats = [strat.evaluate_opportunities(opps) for strat in self.strategies]
        strategy = dict()
        for strat in strats:
            for opp, amt in strat.items():
                if opp in strategy:
                    strategy[opp] += amt
                else:
                    strategy[opp] = amt

        scale = sum(strategy.values())
        return self.filter_dual_sided({k: round((v / scale) * self.bankroll.bettable_funds, 2) for k, v in strategy.items()})

    def evaluate_and_issue(self, opps, dry_run=False):
        for strat in self.strategies:
            strat.evaluate_and_issue(opps, dry_run=True)

        super().evaluate_and_issue(opps, dry_run=dry_run)

    def simulate_and_issue(self, opps, dry_run=False):
        for strat in self.strategies:
            strat.simulate_and_issue(opps, dry_run=True)

        super().simulate_and_issue(opps, dry_run=dry_run)