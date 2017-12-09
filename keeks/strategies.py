import abc


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

class KellyCriterion(BaseStrategy):
    def __init__(self, bankroll, scale_bets=True, verbose=0):
        self.bankroll = bankroll
        self.scale_bets = scale_bets
        self.verbose = verbose

    def evaluate_opportunities(self, opps):
        # first get the basic allocations, ignoring negative ones.
        bets = {}
        for opp in opps:
            if opp.b == 0:
                bets[opp.label] = 0
            else:
                allo = ((opp.probability * opp.b) - (1 - opp.probability)) / opp.b
                if allo > 0:
                    bets[opp.label] = allo

        # now if the sum of all allocations is less than 1, then we are good to go else we scale them all down to 1
        scale = sum([v for k, v in bets.items()])
        if scale > 1:
            if self.scale_bets:
                bets = {k: v / scale for k, v in bets.items()}
            else:
                out = {}
                bets = sorted([(k, v) for k, v in bets.items()], reverse=True, key=lambda x: x[1])
                tot = 0
                for bet in bets:
                    if tot + bet[1] < 1:
                        out[bet[0]] = bet[1]
                        tot += bet[1]
                    else:
                        break
                bets = out

        return {k: round(v * self.bankroll.bettable_funds, 2) for k, v in bets.items()}
