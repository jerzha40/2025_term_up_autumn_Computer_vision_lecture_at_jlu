# tes.py
import os, glob, site, ctypes
import numpy as np


def add_cuda_runtime_dirs():
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for p in glob.glob(
            os.path.join(sp, "nvidia", "cuda_runtime*", "bin")
        ) + glob.glob(os.path.join(sp, "nvidia", "cuda_runtime*", "lib", "x64")):
            if os.path.isdir(p):
                try:
                    os.add_dll_directory(p)
                except Exception:
                    os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


add_cuda_runtime_dirs()

dll = ctypes.CDLL(os.path.abspath("fastgpu.dll"))

# double
dll.sumsq_cuda64_chunked.argtypes = (
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
)
dll.sumsq_cuda64_chunked.restype = ctypes.c_double

# float
dll.sumsq_cuda32_chunked.argtypes = (
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
)
dll.sumsq_cuda32_chunked.restype = ctypes.c_float

# --- 测试：10^9 太大就改成合适的 N ---
N = 1000_000_000  # 例：1e8，f32 约 0.4GB；f64 约 0.8GB（建议分块）
x32 = np.random.rand(N).astype(np.float32)
x64 = x32.astype(np.float64)

# 预热（创建上下文等）
ptr32 = x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
_ = dll.sumsq_cuda32_chunked(ptr32, x32.size, 64 * 1024 * 1024 // 4)

import time

print("START GPU f32")
t0 = time.perf_counter()
y32 = dll.sumsq_cuda32_chunked(ptr32, x32.size, 256 * 1024 * 1024 // 4)  # 每块 256MB
t1 = time.perf_counter()
print("GPU f32:", y32, "sec:", t1 - t0)
print("START GPU f32")
t0 = time.perf_counter()
y32 = dll.sumsq_cuda32_chunked(ptr32, x32.size, 256 * 1024 * 1024 // 4)  # 每块 256MB
t1 = time.perf_counter()
print("GPU f32:", y32, "sec:", t1 - t0)

print("START CPU f32")
t0 = time.perf_counter()
c32 = float((x32 * x32).sum())
t1 = time.perf_counter()
print("CPU f32:", c32, "sec:", t1 - t0)
print("START CPU f32")
t0 = time.perf_counter()
c32 = float((x32 * x32).sum())
t1 = time.perf_counter()
print("CPU f32:", c32, "sec:", t1 - t0)

# 如需 f64：
ptr64 = x64.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
print("START GPU f64")
t0=time.perf_counter()
y64 = dll.sumsq_cuda64_chunked(ptr64, x64.size, 128*1024*1024//8)  # 每块 128MB
t1=time.perf_counter()
print("GPU f64:", y64, "sec:", t1-t0)
print("START CPU f64")
t0=time.perf_counter(); c64 = float((x64*x64).sum()); t1=time.perf_counter()
print("CPU f64:", c64, "sec:", t1-t0)
