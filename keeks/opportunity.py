import random


class Opportunity:
    def __init__(self, expected_winner, expected_loser, probability=0.0, odds=3.0, truth=None, verbose=0):
        self.expected_winner = expected_winner
        self.expected_loser = expected_loser
        self.label = '%s over %s' % (expected_winner, expected_loser, )
        self.probability = probability
        self.odds = odds
        self.truth = truth
        self.verbose = verbose
        self.correlation_id = expected_winner

    def __str__(self):
        return 'Opportunity: %s' % (self.label, )

    def simulate(self, amt, bank, dry_run=False):
        if random.random() <= self.probability:
            if not dry_run:
                bank.deposit(amt * self.odds - amt)
            return True
        else:
            if not dry_run:
                bank.withdraw(amt)
            return False

    @property
    def b(self):
        return self.odds * 1 - 1

    def evaluate(self, amt, bank, dry_run=False):
        if self.truth:
            if self.verbose > 0 and not dry_run:
                print('depositing %s in winnings' % (round(amt * self.odds - amt, 2), ))

            if not dry_run:
                bank.deposit(amt * self.odds - amt)

            return True
        else:
            if self.verbose > 0 and not dry_run:
                print('withdrawing %s in loses' % (amt, ))

            if not dry_run:
                bank.withdraw(amt)

            return False


