from abc import ABC, abstractmethod
import numpy as np

class Quadrature(ABC):

    def kc_list(self):
        pass

    @abstractmethod
    def weights(self):
        pass

    @abstractmethod
    def rates(self):
        pass

class TrapezoidalQuadrature(Quadrature):

    def __init__(self, kc_list, rates_interface):
        self._kc_list = kc_list

        if (len(kc_list)) == 1:
            self._weights = np.array([1.0])
        else:
            self._weights = [0.5 * (kc_list[1] - kc_list[0])]

            for i in range(1, len(kc_list) - 1):
                self._weights.append(kc_list[i + 1] - kc_list[i])

            self._weights.append(0.5 * (kc_list[-1] - kc_list[-2]))

        self._rates = []

        for kc in self._kc_list:
            self._rates.append(rates_interface.get_rates(kc))

    def kc_list(self):
        return self._kc_list

    def weights(self):
        return self._weights

    def rates(self):
        return self._rates

