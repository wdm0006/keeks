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
            return {opps[0]: round(self.bankroll.bettable_funds, 2)}
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
            return {opps[0]: round(self.bankroll.bettable_funds, 2)}
        else:
            return dict()


class Momentum(BaseStrategy):
    def __init__(self, bankroll, verbose=0):
        self.bankroll = bankroll
        self.verbose = verbose
        self.history = []

    def depth_of_wins(self, corr_id):
        cnt = 0
        for iter in reversed(self.history):
            if corr_id in {x.correlation_id for x in iter}:
                cnt += 1
            else:
                break
        return cnt

    def evaluate_opportunities(self, opps):
        # if there is no history, bet on everything

        if len(self.history) == 0:
            return dict()
        else:
            wins = {x.correlation_id for x in self.history[-1] if x.correlation_id is not None}
            if len(wins) == 0:
                wins = {x.correlation_id: 1 for x in opps}
            else:
                wins = {x: self.depth_of_wins(x) for x in wins}

            out = {opp: self.bankroll.bettable_funds * wins.get(opp.correlation_id) for opp in opps if opp.correlation_id in wins}
            scale = sum(out.values())
            return self.filter_dual_sided({k: round((v / scale) * self.bankroll.bettable_funds, 2) for k, v in out.items()})

    def evaluate_and_issue(self, opps, dry_run=False):
        # first grab our bets
        strat = self.evaluate_opportunities(opps)
        res = []
        for opp, amt in strat.items():
            if amt is not None:
                if self.verbose > 0:
                    print('evaluating %s on %s' % (amt, opp,))
                opp.evaluate(amt, self.bankroll, dry_run=dry_run)

        for opp in opps:
            if opp.evaluate(1, self.bankroll, dry_run=True):
                res.append(opp)

        self.history.append(res)

    def simulate_and_issue(self, opps, dry_run=False):
        # first grab our bets
        strat = self.evaluate_opportunities(opps)
        res = []
        for opp, amt in strat.items():
            if amt is not None:
                if self.verbose > 0:
                    print('simulating %s on %s' % (amt, opp,))
                opp.simulate(amt, self.bankroll, dry_run=dry_run)

        for opp in opps:
            if opp.simulate(1, self.bankroll, dry_run=True):
                res.append(opp)

        self.history.append(res)


class AllOnMostMomentum(Momentum):
    def evaluate_opportunities(self, opps):
        # if there is no history, bet on everything

        if len(self.history) == 0:
            return dict()
        else:
            wins = {x.correlation_id for x in self.history[-1] if x.correlation_id is not None}
            if len(wins) == 0:
                wins = {x.correlation_id: 1 for x in opps}
            else:
                wins = {x: self.depth_of_wins(x) for x in wins}

            if len(wins) != 0:
                max_depth = max(wins.values())
                wins = {k: v for k, v in wins.items() if v == max_depth}

            out = {opp: self.bankroll.bettable_funds * wins.get(opp.correlation_id) for opp in opps if opp.correlation_id in wins}

            out = self.filter_dual_sided(out)

            out = dict(sorted([(a, b) for a, b in out.items()], reverse=True, key=lambda x: x[0].probability)[:1])

            scale = sum(out.values())

            return {k: round((v / scale) * self.bankroll.bettable_funds, 2) for k, v in out.items()}