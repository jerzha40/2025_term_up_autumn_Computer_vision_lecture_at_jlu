import time, numpy as np, fastcuda_nvrtc as f

f.init()

x = np.random.rand(1_000_000).astype(np.float64)
print("START")
t0 = time.perf_counter()
y = f.sumsq(x)
t1 = time.perf_counter()
print("GPU:", y, "sec:", t1 - t0)

print("START")
t0 = time.perf_counter()
y = float((x * x).sum())
t1 = time.perf_counter()
print("CPU:", y, "sec:", t1 - t0)
