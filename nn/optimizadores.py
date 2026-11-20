import numpy as np


class SGD:
    """p <- p - lr * g"""

    def __init__(self, lr=0.1):
        self.lr = lr

    def paso(self, params):
        for p, g in params:
            p -= self.lr * g


class Momentum:
    """Acumula una 'velocidad': los pasos en la misma direccion se suman."""

    def __init__(self, lr=0.05, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def paso(self, params):
        if self.v is None:
            self.v = [np.zeros_like(p) for p, _ in params]
        for v, (p, g) in zip(self.v, params):
            v *= self.beta
            v += g
            p -= self.lr * v


class Adam:
    """Momentum + un paso distinto para cada parametro segun el tamano de sus gradientes."""

    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None
        self.t = 0

    def paso(self, params):
        if self.m is None:
            self.m = [np.zeros_like(p) for p, _ in params]
            self.v = [np.zeros_like(p) for p, _ in params]
        self.t += 1
        for m, v, (p, g) in zip(self.m, self.v, params):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1**self.t)
            v_hat = v / (1 - self.b2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


