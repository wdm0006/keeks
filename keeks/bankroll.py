import matplotlib.pyplot as plt

from keeks.utils import RuinError


class BankRoll:
    """
    A class representing a bankroll for betting or investment strategies.

    The BankRoll class manages funds, tracks history, and enforces drawdown limits
    to prevent catastrophic losses. It provides methods for depositing and withdrawing
    funds, as well as visualizing the bankroll history.

    Parameters
    ----------
    initial_funds : float, default=0.0
        The starting amount of money in the bankroll.
    percent_bettable : float, default=1.0
        The percentage of total funds that can be used for betting (0.0 to 1.0).
    max_draw_down : float, default=0.3
        The maximum percentage of funds that can be lost in a single withdrawal (0.0 to 1.0).
        If None, no drawdown limit is enforced.
    verbose : int, default=0
        Controls the verbosity level of the bankroll operations.

    Attributes
    ----------
    history : list
        A list tracking the total funds after each transaction.
    """

    def __init__(
        self, initial_funds=0.0, percent_bettable=1.0, max_draw_down=0.3, verbose=0
    ):
        self._bank = initial_funds
        self.percent_bettable = percent_bettable
        self.max_draw_down = max_draw_down
        self.verbose = verbose
        self.history = [initial_funds]

    def update_history(self):
        """
        Update the history list with the current total funds.

        This method is called automatically after deposits and withdrawals.
        """
        self.history.append(self.total_funds)

    @property
    def bettable_funds(self):
        """
        Calculate the amount of funds available for betting.

        Returns
        -------
        float
            The amount of funds available for betting, calculated as
            total funds multiplied by the percent_bettable parameter.
        """
        return round(self._bank * self.percent_bettable, 2)

    @property
    def total_funds(self):
        """
        Get the total amount of funds in the bankroll.

        Returns
        -------
        float
            The total amount of funds, rounded to 2 decimal places.
        """
        return round(self._bank, 2)

    def deposit(self, amt):
        """
        Add funds to the bankroll.

        Parameters
        ----------
        amt : float
            The amount to deposit into the bankroll.
        """
        self._bank += amt
        self.update_history()

    def withdraw(self, amt):
        """
        Remove funds from the bankroll.

        This method enforces the max_draw_down limit if set, raising a RuinError
        if the withdrawal would exceed the allowed drawdown.

        Parameters
        ----------
        amt : float
            The amount to withdraw from the bankroll.

        Raises
        ------
        RuinError
            If the withdrawal would exceed the maximum allowed drawdown.
        """
        self._bank -= amt
        if self.max_draw_down and amt > self.max_draw_down * self.total_funds:
            raise RuinError("You lost too much money buddy, slow down.")

        self.update_history()

    def bet(self, amount):
        """
        Place a bet with the specified amount.

        Parameters
        ----------
        amount : float
            The amount to bet.

        Raises
        ------
        ValueError
            If the bet amount exceeds the bettable funds.
        """
        if amount > self.bettable_funds:
            raise ValueError("Bet amount exceeds bettable funds")
        self._bank -= amount
        self.update_history()

    def add_funds(self, amount):
        """
        Add funds to the bankroll after a winning bet.

        Parameters
        ----------
        amount : float
            The amount to add to the bankroll.
        """
        self._bank += amount
        self.update_history()

    def remove_funds(self, amount):
        """
        Remove funds from the bankroll after a losing bet.

        Parameters
        ----------
        amount : float
            The amount to remove from the bankroll.

        Raises
        ------
        RuinError
            If the removal would exceed the maximum allowed drawdown.
        """
        if self.max_draw_down and amount > self.max_draw_down * self.total_funds:
            raise RuinError("You lost too much money buddy, slow down.")
        self._bank -= amount
        self.update_history()

    def plot_history(self, fname=None):
        """
        Plot the history of the bankroll over time.

        Creates a line plot showing the bankroll value after each transaction.

        Parameters
        ----------
        fname : str, optional
            If provided, saves the plot to the specified filename instead of displaying it.
        """
        plt.figure()
        plt.plot(list(range(len(self.history))), self.history, fmt="bo-")
        if fname:
            plt.savefig(fname)
        else:
            plt.show()
