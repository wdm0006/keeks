import matplotlib.pyplot as plt

from keeks.utils import RuinError


class BankRoll:
    def __init__(
        self, initial_funds=0.0, percent_bettable=1.0, max_draw_down=0.3, verbose=0
    ):
        self._bank = initial_funds
        self.percent_bettable = percent_bettable
        self.max_draw_down = max_draw_down
        self.verbose = verbose
        self.history = [initial_funds]

    def update_history(self):
        self.history.append(self.total_funds)

    @property
    def bettable_funds(self):
        return round(self._bank * self.percent_bettable, 2)

    @property
    def total_funds(self):
        return round(self._bank, 2)

    def deposit(self, amt):
        self._bank += amt

        self.update_history()

    def withdraw(self, amt):
        self._bank -= amt
        if self.max_draw_down and amt > self.max_draw_down * self.total_funds:
            raise RuinError("You lost too much money buddy, slow down.")

        self.update_history()

    def plot_history(self, fname=None):
        plt.figure()
        plt.plot(list(range(len(self.history))), self.history, fmt="bo-")
        if fname:
            plt.savefig(fname)
        else:
            plt.show()
