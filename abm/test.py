import numpy as np, hellocuda

A = np.random.rand(1_000_000_000).astype(np.float32)
B = np.random.rand(1_000_000_000).astype(np.float32)
C = hellocuda.vecadd(A, B, chunked=True)
print(C[:5], np.allclose(C, A + B))
