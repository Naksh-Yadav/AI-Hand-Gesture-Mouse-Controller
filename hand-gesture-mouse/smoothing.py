import numpy as np

class OneEuroFilter:
    """
    One Euro Filter for jitter reduction.
    https://github.com/stefanmilowanov/OneEuroFilter
    """
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * self.freq * cutoff)
        return 1.0 / (1.0 + tau)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        # Calculate derivative
        dx = (x - self.x_prev) * self.freq

        # Filter derivative
        alpha_d = self._alpha(self.dcutoff)
        dx_hat = self.dx_prev + alpha_d * (dx - self.dx_prev)

        # Cutoff based on velocity
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff)

        # Filter signal
        x_hat = self.x_prev + alpha * (x - self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat

class ExponentialSmoothing:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.last_val = None

    def filter(self, val):
        if self.last_val is None:
            self.last_val = val
            return val

        res = self.alpha * val + (1 - self.alpha) * self.last_val
        self.last_val = res
        return res
