from abc import ABC, abstractmethod
import numpy as np
from nuMSM_solver.rates import Rates_Jurai
from nuMSM_solver.common import f_nu
import quadpy as qp

class Quadrature(ABC):

    @abstractmethod
    def kc_list(self):
        pass

    @abstractmethod
    def weights(self):
        pass

class TrapezoidalQuadrature(Quadrature):

    # Can use with either rates class
    def __init__(self, kc_list):
        self._kc_list = kc_list

        if (len(kc_list)) == 1:
            self._weights = np.array([1.0])
        else:
            self._weights = [0.5 * (kc_list[1] - kc_list[0])]

            for i in range(1, len(kc_list) - 1):
                self._weights.append(kc_list[i + 1] - kc_list[i])

            self._weights.append(0.5 * (kc_list[-1] - kc_list[-2]))

    def kc_list(self):
        return self._kc_list

    def weights(self):
        return self._weights

class GaussianQuadrature(Quadrature):

    def __init__(self, n_points, kc_min, kc_max, qscheme="legendre"):

        if qscheme == "legendre":
            gq = qp.c1.gauss_legendre(n_points)
        elif qscheme == "radau":
            gq = qp.c1.gauss_radau(n_points)
        elif qscheme == "lobatto":
            gq = qp.c1.gauss_lobatto(n_points)
        gq_points = gq.points
        gq_weights = gq.weights

        self._weights = 0.5*(kc_max - kc_min)*gq_weights

        self._kc_list = np.array(list(map(
            lambda x: 0.5*(kc_max - kc_min)*x + 0.5*(kc_max + kc_min),
            gq_points
        )))

    def kc_list(self):
        return self._kc_list

    def weights(self):
        return self._weights
