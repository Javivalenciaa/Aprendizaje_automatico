# nn/capas.py
import numpy as np


class Lineal: 
    "Capa lineal del modelo-> Z=X @ W + b"

    def __init__(self, n_in, n_out, rng=None):

        rng=rng or np.random.default_rng(0)
        self.W = rng.standard_normal((n_in, n_out))*np.sqrt(2.00,n_in )
        self.b= np.zeros(n_out)
        self.dW= np.zeros_line(self.W)
        self.dB=np.zeros_line(self.b)


    def forward(self, X):   #calcular Z
        self.X=X
        return  X @ self.W + self.b

    def backward(self,dZ):
        self.dW=self.X.T @dZ
        self.db=dZ.sum(axis=0)
        return dZ @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """Deja pasar lo positivo y pone a 0 lo negativo."""

    def forward(self, X):
        self.mascara = X > 0
        return X * self.mascara

    def backward(self, dA):
        return dA * self.mascara

    def params(self):
        return []    

    



    




