import numpy as np

def softmax(Z):
    Z= Z-Z.max(axis=1, keepdims=True) #-> evitar desbordamientos por la exponencial
    E=np.exp(Z)
    return E/E.sum(axis=1, keepdims=True)

def entropia_cruzada(Z,y):

    N=Z.shape[0]
    Zs = Z - Z.max(axis=1, keepdims=True)
    perdida = (np.log(np.exp(Zs).sum(axis=1)) - Zs[np.arange(N), y]).mean()
    P = softmax(Z)
    dZ = P.copy()
    dZ[np.arange(N), y] -= 1
    dZ /= N
    return perdida, dZ
