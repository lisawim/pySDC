"""
Base module for approximating a MacLaurin series.
"""
import numpy as np
import math


class MacLaurinApproximation:
    def __init__(self, u0, is_diff):
        self.u0 = u0
        self.is_diff = is_diff

        self.lambda_d = -2.0
        self.lambda_a = 1.0
    
    def _coefficients(self, degree):
        if self.is_diff:
            coeffs = np.array(
                [((2.0 * self.lambda_d) ** j) / math.factorial(j) for j in range(1, degree + 1)]
            )
        else:
            fac = self.lambda_d / self.lambda_a
            coeffs = np.array(
                [(fac * (2.0 * self.lambda_d) ** j) / math.factorial(j) for j in range(1, degree + 1)]
            )
        return coeffs
    
    def eval(self, t, degree):
        t = np.asarray(t)
        coeffs = self._coefficients(degree)
        exponents = np.arange(1, degree + 1)

        powers = t[..., np.newaxis] ** exponents
        return self.u0 + powers @ coeffs
