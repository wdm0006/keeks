from keeks.opportunity import Opportunity
from keeks.strategies import KellyCriterion
from keeks.bankroll import BankRoll
import random

bank = BankRoll(1000, percent_bettable=0.3)
strategy = KellyCriterion(bankroll=bank)

for _ in range(10):
    opps = [
        Opportunity('a', probability=random.random(), odds=2),
        Opportunity('b', probability=random.random(), odds=2),
        Opportunity('c', probability=random.random(), odds=1.1),
        Opportunity('d', probability=random.random(), odds=1.2),
        Opportunity('e', probability=random.random(), odds=5)
    ]

    print('strategy for the round:')
    print(strategy.evaluate_opportunities(opps))
    strategy.evaluate_and_issue(opps)
    print('bankroll: %s' % (strategy.total_funds, ))