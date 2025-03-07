import abc

__author__ = "willmcginnis"


class BaseStrategy(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, probability):
        pass
