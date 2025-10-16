import numpy as np


M = np.random.randint(0, 10, size=(4 + 1, 5 + 1))
M2 = M**2


def get_sum(M: np.ndarray) -> np.ndarray:
    SM = np.zeros_like(M)
    for i in range(1, M.shape[0]):
        for j in range(1, M.shape[1]):
            SM[i, j] = M[i, j] + SM[i - 1, j] + SM[i, j - 1] - SM[i - 1, j - 1]
    return SM


def get_mean(SM: np.ndarray, I0: tuple, I1: tuple) -> np.ndarray:
    return (
        SM[I0[1], I1[1]] - SM[I0[1], I1[0]] - SM[I0[0], I1[1]] + SM[I0[0], I1[0]]
    ) / ((I0[1] - I0[0]) * (I1[1] - I1[0]))


def get_var(SM2: np.ndarray, SM: np.ndarray, I0: tuple, I1: tuple) -> np.ndarray:
    m2 = get_mean(SM2, I0, I1)
    m = get_mean(SM, I0, I1)
    return m2 - m**2


SM = get_sum(M)
SM2 = get_sum(M2)

m = get_mean(SM, (0, 1), (2, 4))
v = get_var(SM2, SM, (0, 1), (2, 4))


pass
