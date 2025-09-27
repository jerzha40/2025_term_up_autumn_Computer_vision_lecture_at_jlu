import ctypes, numpy as np, os, sys
import time

lib = ctypes.CDLL(os.path.abspath("fast.dll"))
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
N = 1_000_000_000  # 例：1e8，f32 约 0.4GB；f64 约 0.8GB（建议分块）
x32 = np.random.rand(N).astype(np.float32)
x64 = x32.astype(np.float64)
# 预热（创建上下文等）
ptr32 = x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
_ = dll.sumsq_cuda32_chunked(ptr32, x32.size, 64 * 1024 * 1024 // 4)

# declare signature: double sumsq(const double*, int)
lib.sumsq.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_int)
lib.sumsq.restype = ctypes.c_double

x = np.random.rand(1_000_000_000).astype(np.float64)
ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
print("START CPU c/c++")
t0 = time.perf_counter()
res = lib.sumsq(ptr, x.size)
t1 = time.perf_counter()
print(f"elapsed:{res}", t1 - t0, "seconds")
t0 = time.perf_counter()
res = lib.sumsq(ptr, x.size)
t1 = time.perf_counter()
print(f"elapsed:{res}", t1 - t0, "seconds")
t0 = time.perf_counter()
res = lib.sumsq(ptr, x.size)
t1 = time.perf_counter()
print(f"elapsed:{res}", t1 - t0, "seconds")
print("END CPU\n")

print("START GPU f32 cuda")
t0 = time.perf_counter()
y32 = dll.sumsq_cuda32_chunked(ptr32, x32.size, 256 * 1024 * 1024 // 4)  # 每块 256MB
t1 = time.perf_counter()
print("GPU f32:", y32, "sec:", t1 - t0)
t0 = time.perf_counter()
y32 = dll.sumsq_cuda32_chunked(ptr32, x32.size, 256 * 1024 * 1024 // 4)  # 每块 256MB
t1 = time.perf_counter()
print("GPU f32:", y32, "sec:", t1 - t0)
t0 = time.perf_counter()
y32 = dll.sumsq_cuda32_chunked(ptr32, x32.size, 256 * 1024 * 1024 // 4)  # 每块 256MB
t1 = time.perf_counter()
print("GPU f32:", y32, "sec:", t1 - t0)
print("END GPU f32\n")

# 如需 f64：
ptr64 = x64.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
print("START GPU f64 cuda")
t0 = time.perf_counter()
y64 = dll.sumsq_cuda64_chunked(ptr64, x64.size, 128 * 1024 * 1024 // 8)  # 每块 128MB
t1 = time.perf_counter()
print("GPU f64:", y64, "sec:", t1 - t0)
t0 = time.perf_counter()
y64 = dll.sumsq_cuda64_chunked(ptr64, x64.size, 128 * 1024 * 1024 // 8)  # 每块 128MB
t1 = time.perf_counter()
print("GPU f64:", y64, "sec:", t1 - t0)
t0 = time.perf_counter()
y64 = dll.sumsq_cuda64_chunked(ptr64, x64.size, 128 * 1024 * 1024 // 8)  # 每块 128MB
t1 = time.perf_counter()
print("GPU f64:", y64, "sec:", t1 - t0)
print("END GPU f64\n")

print("START CPU f32 numpy")
t0 = time.perf_counter()
c32 = float((x32 * x32).sum())
t1 = time.perf_counter()
print("CPU f32:", c32, "sec:", t1 - t0)
t0 = time.perf_counter()
c32 = float((x32 * x32).sum())
t1 = time.perf_counter()
print("CPU f32:", c32, "sec:", t1 - t0)
t0 = time.perf_counter()
c32 = float((x32 * x32).sum())
t1 = time.perf_counter()
print("CPU f32:", c32, "sec:", t1 - t0)
print("END CPU f32\n")

print("START CPU f64 numpy")
t0 = time.perf_counter()
c64 = float((x64 * x64).sum())
t1 = time.perf_counter()
print("CPU f64:", c64, "sec:", t1 - t0)
t0 = time.perf_counter()
c64 = float((x64 * x64).sum())
t1 = time.perf_counter()
print("CPU f64:", c64, "sec:", t1 - t0)
t0 = time.perf_counter()
c64 = float((x64 * x64).sum())
t1 = time.perf_counter()
print("CPU f64:", c64, "sec:", t1 - t0)
print("END CPU f64\n")
