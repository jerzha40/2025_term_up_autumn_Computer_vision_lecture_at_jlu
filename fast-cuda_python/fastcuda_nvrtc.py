# fastcuda_nvrtc.py
import numpy as np, ctypes, atexit
from cuda.bindings import driver as cu, nvrtc

# ---- globals (cached once) ----
_CTX = None
_MOD = None
_FN64 = None  # float64 kernel
_FN32 = None  # float32 kernel


def _to_code(ret):
    if isinstance(ret, (tuple, list)):
        ret = ret[0]
    if hasattr(ret, "value"):
        ret = ret.value
    return int(ret)


def _check(ret, msg):
    # normalize return code
    c = ret[0] if isinstance(ret, (tuple, list)) else ret
    c = int(getattr(c, "value", c))
    if c != int(cu.CUresult.CUDA_SUCCESS):
        try:
            # wrap as CUresult enum for the binding
            enum_c = cu.CUresult(c)
            _, name = cu.cuGetErrorName(enum_c)
            _, desc = cu.cuGetErrorString(enum_c)
            if isinstance(name, bytes):
                name = name.decode()
            if isinstance(desc, bytes):
                desc = desc.decode()
            raise RuntimeError(f"{msg}: {name} - {desc} ({c})")
        except Exception:
            raise RuntimeError(f"{msg} (error {c})")


_SRC64 = rb"""
extern "C" __global__
void sumsq64(const double* __restrict__ x, double* __restrict__ partials, size_t n){
    extern __shared__ double sm[];
    unsigned tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + tid;
    double v = 0.0; if (i < n){ double t = x[i]; v = t*t; }
    sm[tid] = v; __syncthreads();
    for (unsigned s = blockDim.x>>1; s>0; s>>=1){ if (tid < s) sm[tid]+=sm[tid+s]; __syncthreads(); }
    if (tid==0) partials[blockIdx.x] = sm[0];
}
"""
_SRC32 = rb"""
extern "C" __global__
void sumsq32(const float* __restrict__ x, float* __restrict__ partials, size_t n){
    extern __shared__ float sm[];
    unsigned tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + tid;
    float v = 0.0f; if (i < n){ float t = x[i]; v = t*t; }
    sm[tid] = v; __syncthreads();
    for (unsigned s = blockDim.x>>1; s>0; s>>=1){ if (tid < s) sm[tid]+=sm[tid+s]; __syncthreads(); }
    if (tid==0) partials[blockIdx.x] = sm[0];
}
"""


def _arch_opt(dev):
    err, maj = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
    )
    _check(err, "get CC major")
    err, minr = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
    )
    _check(err, "get CC minor")
    return f"--gpu-architecture=compute_{maj}{minr}".encode()


def _compile_one(src, dev, name_b):
    err, prog = nvrtc.nvrtcCreateProgram(src, b"k.cu", 0, [], [])
    _check(err, "nvrtcCreateProgram")
    opts = (b"--std=c++14", _arch_opt(dev))
    err = nvrtc.nvrtcCompileProgram(prog, len(opts), list(opts))
    if _to_code(err) != int(cu.CUresult.CUDA_SUCCESS):
        _, log_sz = nvrtc.nvrtcGetProgramLogSize(prog)
        buf = bytearray(max(1, log_sz))
        nvrtc.nvrtcGetProgramLog(prog, buf)
        nvrtc.nvrtcDestroyProgram(prog)
        raise RuntimeError(
            "NVRTC compile failed:\n" + bytes(buf).decode(errors="replace")
        )
    _, ptx_sz = nvrtc.nvrtcGetPTXSize(prog)
    ptx = bytearray(max(1, ptx_sz))
    nvrtc.nvrtcGetPTX(prog, ptx)
    nvrtc.nvrtcDestroyProgram(prog)

    err, mod = cu.cuModuleLoadData(bytes(ptx))
    _check(err, "cuModuleLoadData")
    err, fn = cu.cuModuleGetFunction(mod, name_b)
    _check(err, "cuModuleGetFunction")
    return mod, fn


def init():
    """Create context + compile kernels once."""
    global _CTX, _MOD, _FN64, _FN32
    if _CTX is not None:
        return
    _check(cu.cuInit(0), "cuInit")
    err, dev = cu.cuDeviceGet(0)
    _check(err, "cuDeviceGet")
    err, ctx = cu.cuCtxCreate(0, dev)
    _check(err, "cuCtxCreate")
    _CTX = ctx

    # compile both precisions; reuse one module for both by linking both kernels
    # easiest is compile two small modules; they’re tiny and one-time.
    _MOD, _FN64 = _compile_one(_SRC64, dev, b"sumsq64")
    mod32, _FN32 = _compile_one(_SRC32, dev, b"sumsq32")
    # keep only one module handle alive for simplicity (both are fine to keep)
    # unload the extra module handle if you want; not necessary.
    atexit.register(shutdown)


def shutdown():
    global _CTX, _MOD
    if _MOD is not None:
        try:
            cu.cuModuleUnload(_MOD)
        except Exception:
            pass
        _MOD = None
    if _CTX is not None:
        try:
            cu.cuCtxDestroy(_CTX)
        except Exception:
            pass
        _CTX = None


def _maybe_get_device_ptr(x):
    """Return (device_ptr:int, nbytes:int) if x is a CuPy/torch CUDA tensor via __cuda_array_interface__; else (None,None)."""
    iface = getattr(x, "__cuda_array_interface__", None)
    if not iface:
        return None, None
    ptr, ro = iface["data"]
    return int(ptr), int(np.asarray(x).nbytes)


def _sum_impl(x: np.ndarray, use_float32: bool):
    init()  # ensure one-time init/compile

    # path A: zero-copy if x is already on GPU (CuPy, PyTorch CUDA tensor)
    d_x_ptr, nbytes = _maybe_get_device_ptr(x)
    on_device = d_x_ptr is not None

    if use_float32:
        arr = np.asarray(x, dtype=np.float32, order="C")
        elem_size = 4
        block = 256
        fn = _FN32
    else:
        arr = np.asarray(x, dtype=np.float64, order="C")
        elem_size = 8
        block = 256
        fn = _FN64

    n = arr.size
    if n == 0:
        return 0.0
    grid = (n + block - 1) // block
    shared = block * elem_size

    # allocate inputs/partials
    if on_device:
        d_x = d_x_ptr
    else:
        _check(
            cu.cuMemAlloc(arr.nbytes), "alloc?"
        )  # placeholder to query return signature
        err, d_x = cu.cuMemAlloc(arr.nbytes)
        _check(err, "cuMemAlloc x")
        # optional: register pinned host memory for faster H2D
        pinned = False
        try:
            _check(
                cu.cuMemHostRegister(arr.ctypes.data, arr.nbytes, 0),
                "cuMemHostRegister",
            )
            pinned = True
        except Exception:
            pinned = False
        _check(cu.cuMemcpyHtoD(d_x, arr.ctypes.data, arr.nbytes), "H2D")
        if pinned:
            try:
                cu.cuMemHostUnregister(arr.ctypes.data)
            except Exception:
                pass

    err, d_partials = cu.cuMemAlloc(grid * elem_size)
    _check(err, "cuMemAlloc partials")

    try:
        # kernel args (void**)
        if use_float32:
            arg0 = ctypes.c_uint64(int(d_x))
            arg1 = ctypes.c_uint64(int(d_partials))
            arg2 = ctypes.c_size_t(n)
        else:
            arg0 = ctypes.c_uint64(int(d_x))
            arg1 = ctypes.c_uint64(int(d_partials))
            arg2 = ctypes.c_size_t(n)

        kparams = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.byref(arg0), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(arg1), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(arg2), ctypes.c_void_p),
        )
        kparams_addr = ctypes.addressof(kparams)

        _check(
            cu.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, shared, 0, kparams_addr, 0),
            "launch",
        )
        _check(cu.cuCtxSynchronize(), "sync")

        # fetch partials and finalize on host (few KB)
        if use_float32:
            partials = np.empty(grid, dtype=np.float32)
        else:
            partials = np.empty(grid, dtype=np.float64)

        _check(
            cu.cuMemcpyDtoH(partials.ctypes.data, d_partials, grid * elem_size),
            "D2H partials",
        )
        return float(partials.sum())

    finally:
        cu.cuMemFree(d_partials)
        if not on_device:
            cu.cuMemFree(d_x)


def sumsq(x: np.ndarray) -> float:
    """float64 sum of squares (keeps CPU parity)."""
    return _sum_impl(x, use_float32=False)


def sumsq_f32(x: np.ndarray) -> float:
    """float32 sum of squares (faster on most GPUs)."""
    return _sum_impl(x, use_float32=True)
