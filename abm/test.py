import numpy as np, hellocuda
import time

A = np.random.rand(1_000_000_000).astype(np.float32)
B = np.random.rand(1_000_000_000).astype(np.float32)
t0 = time.perf_counter()
C = hellocuda.vecadd(A, B, chunked=True)
t1 = time.perf_counter()
print(C[:5], np.allclose(C, A + B), t1 - t0)
t0 = time.perf_counter()
C = hellocuda.vecadd(A, B, chunked=False)
t1 = time.perf_counter()
print(C[:5], np.allclose(C, A + B), t1 - t0)

import numpy as np
from hellocuda import Dense

layer = Dense(4, 3)
x = np.random.randn(2, 4).astype(np.float32)
y = layer.forward(x)
print("out:", y)
