import ctypes, numpy as np, os, sys
import time

libname = "libfast.so" if sys.platform != "win32" else "fast.dll"
lib = ctypes.CDLL(os.path.abspath(libname))

# declare signature: double sumsq(const double*, int)
lib.sumsq.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_int)
lib.sumsq.restype = ctypes.c_double

x = np.random.rand(1_000_000_000).astype(np.float64)
ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
print("START")
t0 = time.perf_counter()
res = lib.sumsq(ptr, x.size)
t1 = time.perf_counter()
print("result:", res)
print("elapsed:", t1 - t0, "seconds")
print(res)
