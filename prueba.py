import numpy as np
from nn.red import crear_mlp

red = crear_mlp(40, [64, 32], 10)
print([type(c).__name__ for c in red.capas])
print(sum(p.size for p, _ in red.params()))
print(red.forward(np.zeros((5, 40))).shape)
print(crear_mlp(40, [], 10).forward(np.zeros((5, 40))).shape)

import numpy as np
from nn.red import crear_mlp, guardar, cargar_pesos

red = crear_mlp(40, [64, 32], 10)
guardar(red, 'prueba_modelo.npz', generos=np.array(['a', 'b']))
otra = crear_mlp(40, [64, 32], 10, semilla=5)      # otra red con otros números al azar
X = np.random.default_rng(0).standard_normal((3, 40))
print(np.allclose(red.forward(X), otra.forward(X)))   # False: son redes distintas
cargar_pesos(otra, 'prueba_modelo.npz')
print(np.allclose(red.forward(X), otra.forward(X)))   # True: ahora son la misma
