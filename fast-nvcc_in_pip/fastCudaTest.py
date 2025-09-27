# tes.py
import os, glob, site, ctypes
import numpy as np


# 让系统能找到 cudart64_*.dll（来自 pip 的 nvidia-cuda-runtime*）
def add_cuda_runtime_dirs():
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for p in glob.glob(
            os.path.join(sp, "nvidia", "cuda_runtime*", "bin")
        ) + glob.glob(os.path.join(sp, "nvidia", "cuda_runtime*", "lib", "x64")):
            if os.path.isdir(p):
                try:
                    os.add_dll_directory(p)  # Py3.8+
                except Exception:
                    os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


add_cuda_runtime_dirs()

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
