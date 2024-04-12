import numpy as np
import time

A = np.array([[0, 0], [0, 1], [0, 3], [1, 2], [2, 2]])
B = np.array([[ 0.1,  0.4,  0.5],
    [ 0.7,  0.0,  0.4],
    [ 0.8,  0.4,  0.7],
    [ 0.9,  0.3,  0.8]])
C = np.array([[ 0.9,  0.8,  0.9],
    [ 0.3,  0.9,  0.5],
    [ 0.3,  0.4,  0.8],
    [ 0.5,  0.4,  0.3]])

# your approach A
start = time.perf_counter()
print(list(map( lambda x: np.sqrt( (B[x[0]] - C[x[1]]).dot(B[x[0]] - C[x[1]]) ), A)))  # outer list because of py3
print('used: ', time.perf_counter() - start)

# your approach B
start = time.perf_counter()
print(list(map( lambda x: np.linalg.norm((B[x[0]] - C[x[1]])), A)))  # outer list because of py3
print('used: ', time.perf_counter() - start)

# new approach
start = time.perf_counter()
print(np.linalg.norm(B[A[:,0]] - C[A[:,1]], axis=1))
print('used: ', time.perf_counter() - start)

print(lambda x: np.linalg.norm((B[x[0]] - C[x[1]])))