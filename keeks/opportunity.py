import random


class Opportunity:
    def __init__(self, label, probability=0.0, odds=3.0, truth=None, verbose=0):
        self.label = label
        self.probability = probability
        self.odds = odds
        self.truth = truth
        self.verbose = verbose

    def __str__(self):
        return 'Opportunity: %s' % (self.label, )

    def simulate(self, amt, bank):
        if random.random() <= self.probability:
            bank.deposit(amt * self.odds - amt)
        else:
            bank.withdraw(amt)

    @property
    def b(self):
        return self.odds * 1 - 1

    def evaluate(self, amt, bank):
        if self.truth:
            if self.verbose > 0:
                print('depositing %s in winnings' % (round(amt * self.odds - amt, 2), ))
            bank.deposit(amt * self.odds - amt)
        else:
            if self.verbose > 0:
                print('withdrawing %s in loses' % (amt, ))
            bank.withdraw(amt)


