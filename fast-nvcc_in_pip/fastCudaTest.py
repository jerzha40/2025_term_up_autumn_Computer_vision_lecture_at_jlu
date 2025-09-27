# tes.py
import os, glob, site, ctypes
import numpy as np


# 1) 加载你的 DLL（和 tes.py 同目录）
dll = ctypes.CDLL(os.path.abspath("fastgpu.dll"))

# 2) 绑定导出的函数签名（按你 .cu 里的函数名来）
# 如果你按我之前示例写的是：extern "C" __declspec(dllexport) double sumsq_cuda64(const double*, size_t);
dll.sumsq_cuda64.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)
dll.sumsq_cuda64.restype = ctypes.c_double

import time

# 3) 准备数据并调用
x = np.random.rand(1_000_000_000).astype(np.float64)  # double
ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

print("START")
t0 = time.perf_counter()
y = res = dll.sumsq_cuda64(ptr, x.size)
t1 = time.perf_counter()
print("GPU:", y, "sec:", t1 - t0)
print("START")
t0 = time.perf_counter()
y = res = dll.sumsq_cuda64(ptr, x.size)
t1 = time.perf_counter()
print("GPU:", y, "sec:", t1 - t0)
print("START")
t0 = time.perf_counter()
y = res = dll.sumsq_cuda64(ptr, x.size)
t1 = time.perf_counter()
print("GPU:", y, "sec:", t1 - t0)
print("START")
t0 = time.perf_counter()
y = res = dll.sumsq_cuda64(ptr, x.size)
t1 = time.perf_counter()
print("GPU:", y, "sec:", t1 - t0)
print("START")
t0 = time.perf_counter()
y = res = dll.sumsq_cuda64(ptr, x.size)
t1 = time.perf_counter()
print("GPU:", y, "sec:", t1 - t0)
print("START")
t0 = time.perf_counter()
y = res = dll.sumsq_cuda64(ptr, x.size)
t1 = time.perf_counter()
print("GPU:", y, "sec:", t1 - t0)

print("START")
t0 = time.perf_counter()
y = float((x * x).sum())
t1 = time.perf_counter()
print("CPU:", y, "sec:", t1 - t0)
print("START")
t0 = time.perf_counter()
y = float((x * x).sum())
t1 = time.perf_counter()
print("CPU:", y, "sec:", t1 - t0)
print("START")
t0 = time.perf_counter()
y = float((x * x).sum())
t1 = time.perf_counter()
print("CPU:", y, "sec:", t1 - t0)
print("START")
t0 = time.perf_counter()
y = float((x * x).sum())
t1 = time.perf_counter()
print("CPU:", y, "sec:", t1 - t0)
print("START")
t0 = time.perf_counter()
y = float((x * x).sum())
t1 = time.perf_counter()
print("CPU:", y, "sec:", t1 - t0)
print("START")
t0 = time.perf_counter()
y = float((x * x).sum())
t1 = time.perf_counter()
print("CPU:", y, "sec:", t1 - t0)
