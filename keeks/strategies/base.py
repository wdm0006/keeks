import abc

__author__ = 'willmcginnis'


class BaseStrategy:
    @abc.abstractmethod
    def evaluate_opportunities(self, opps):
        pass

    def evaluate_and_issue(self, opps):
        # first grab our bets
        strat = self.evaluate_opportunities(opps)
        for opp in opps:
            amt = strat.get(opp.label)
            if amt is not None:
                if self.verbose > 0:
                    print('evaluating %s on %s' % (amt, opp,))
                opp.evaluate(amt, self.bankroll)

    def simulate_and_issue(self, opps):
        # first grab our bets
        strat = self.evaluate_opportunities(opps)
        for opp in opps:
            amt = strat.get(opp.label)
            if amt is not None:
                if self.verbose > 0:
                    print('simulating %s on %s' % (amt, opp,))
                opp.simulate(amt, self.bankroll)

    @property
    def total_funds(self):
        return self.bankroll.total_funds