import abc

__author__ = 'willmcginnis'


class BaseStrategy:
    @abc.abstractmethod
    def evaluate_opportunities(self, opps):
        pass

    def evaluate_and_issue(self, opps, dry_run=False):
        # first grab our bets
        strat = self.evaluate_opportunities(opps)
        for opp, amt in strat.items():
            if amt is not None:
                if self.verbose > 0 and not dry_run:
                    print('evaluating %s on %s' % (amt, opp,))
                opp.evaluate(amt, self.bankroll, dry_run=dry_run)

    def simulate_and_issue(self, opps, dry_run=False):
        # first grab our bets
        strat = self.evaluate_opportunities(opps)
        for opp, amt in strat.items():
            if amt is not None:
                if self.verbose > 0 and not dry_run:
                    print('simulating %s on %s' % (amt, opp,))
                opp.simulate(amt, self.bankroll, dry_run=dry_run)

    @property
    def total_funds(self):
        return self.bankroll.total_funds

    def filter_dual_sided(self, strategy):
        filtered_strategy = dict()
        for opp, amt in strategy.items():
            for opp2, amt2 in strategy.items():
                if opp.expected_winner == opp2.expected_loser and opp.expected_loser == opp2.expected_winner:
                    if amt2 > amt:
                        filtered_strategy[opp2] = amt2
                    else:
                        filtered_strategy[opp] = amt
                    break
            else:
                filtered_strategy[opp] = amt

        return filtered_strategy