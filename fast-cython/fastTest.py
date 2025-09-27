import time
import numpy as np
import fast

# NOTE: don't allocate a billion doubles unless you really have 8+ GB free.
x = np.random.rand(1_000_000_000).astype(np.float64)  # 1e6 ~ 8 MB
print("START")
t0 = time.perf_counter()
res = fast.sumsq(x)
t1 = time.perf_counter()
print("result:", res)
print("elapsed:", t1 - t0, "seconds")
