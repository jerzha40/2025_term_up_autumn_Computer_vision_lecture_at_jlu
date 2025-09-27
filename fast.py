import ctypes, numpy as np, os, sys

libname = "libfast.so" if sys.platform != "win32" else "fast.dll"
lib = ctypes.CDLL(os.path.abspath(libname))

# declare signature: double sumsq(const double*, int)
lib.sumsq.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_int)
lib.sumsq.restype = ctypes.c_double

x = np.random.rand(1_000_000_000).astype(np.float64)
ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
print("START")
res = lib.sumsq(ptr, x.size)
print(res)
