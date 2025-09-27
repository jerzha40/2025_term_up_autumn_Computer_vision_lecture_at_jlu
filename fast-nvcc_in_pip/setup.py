from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path
import os, site, glob, subprocess
import numpy as np
import pybind11


def find_nvcc():
    """
    优先从 pip 的 nvidia-cuda-nvcc / nvidia-cuda-nvcc-cu12 里找 nvcc.exe，
    其次查 CUDA_PATH / CONDA_PREFIX，最后查 PATH。
    """
    # 1) pip wheels
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        print(sp)
        for pat in ("nvidia/cu13/bin/nvcc.exe", "nvidia/cuda_nvcc*/bin/nvcc"):
            for p in Path(sp).glob(pat):
                if p.exists():
                    return str(p)

    # 2) CUDA Toolkit
    for key in ("CUDA_PATH", "CUDA_HOME"):
        base = os.environ.get(key)
        if base:
            cand = Path(base) / "bin" / "nvcc.exe"
            if cand.exists():
                return str(cand)

    # 3) conda
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        cand = Path(cp) / "Library" / "bin" / "nvcc.exe"
        if cand.exists():
            return str(cand)

    # 4) PATH
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(d) / "nvcc.exe"
        if cand.exists():
            return str(cand)

    return None


def find_cuda_inc_lib():
    """
    在 pip 的 nvidia-cuda-runtime* 里查 include 与 lib\x64（含 cudart.lib）。
    同时兼容 CUDA_PATH / CONDA_PREFIX。
    """
    incs, libs = set(), set()

    # pip runtime wheels
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        # include: .../nvidia/cuda_runtime*/include/cuda_runtime.h
        for p in Path(sp).glob("nvidia/cu*/include"):
            if (Path(p) / "cuda_runtime.h").exists():
                incs.add(str(p))
        # libs: .../nvidia/cuda_runtime*/lib/x64/cudart.lib
        for p in Path(sp).glob("nvidia/cu*/lib/x64"):
            if (Path(p) / "cudart.lib").exists():
                libs.add(str(p))

    # CUDA Toolkit
    for key in ("CUDA_PATH", "CUDA_HOME"):
        base = os.environ.get(key)
        if base:
            inc = Path(base) / "include"
            lib = Path(base) / "lib" / "x64"
            if (inc / "cuda_runtime.h").exists():
                incs.add(str(inc))
            if (lib / "cudart.lib").exists():
                libs.add(str(lib))

    # conda
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        inc = Path(cp) / "Library" / "include"
        lib = Path(cp) / "Library" / "lib" / "x64"
        if (inc / "cuda_runtime.h").exists():
            incs.add(str(inc))
        if (lib / "cudart.lib").exists():
            libs.add(str(lib))

    if not incs or not libs:
        raise RuntimeError(
            "未找到 CUDA 头文件或 cudart.lib。\n"
            "请在当前 venv 中安装匹配版本的 runtime：\n"
            "  CUDA 13.x:  python -m pip install -U nvidia-cuda-runtime\n"
            "  CUDA 12.x:  python -m pip install -U nvidia-cuda-runtime-cu12\n"
            "或确保已正确设置 CUDA_PATH/CONDA_PREFIX。"
        )
    return sorted(incs), sorted(libs)


NVCC = find_nvcc()
if not NVCC:
    raise RuntimeError(
        "未找到 nvcc。\n"
        "你已安装 nvidia-cuda-nvcc，但可能未在当前 venv 或路径未被检出。\n"
        "请确认包安装在此 venv，或安装 CUDA Toolkit/conda 的 nvcc。"
    )

CUDA_INCS, CUDA_LIBS = find_cuda_inc_lib()


class BuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            objs, new_srcs = [], []
            for src in ext.sources:
                if src.endswith(".cu"):
                    objs += self.compile_with_nvcc(src, ext)
                else:
                    new_srcs.append(src)
            ext.sources = new_srcs
            ext.extra_objects = getattr(ext, "extra_objects", []) + objs
        super().build_extensions()

    def compile_with_nvcc(self, src, ext):
        btemp = Path(self.build_temp)
        btemp.mkdir(parents=True, exist_ok=True)
        obj = btemp / (Path(src).stem + ".obj")

        cmd = [
            NVCC,
            "-c",
            src,
            "-o",
            str(obj),
            "-O3",
            "-std=c++14",
            "--compiler-options",
            "/MD,/EHsc",
        ]
        # includes
        for inc in ext.include_dirs or []:
            cmd += ["-I", inc]
        for inc in CUDA_INCS:
            cmd += ["-I", inc]

        # 选常见架构；根据你显卡可改，比如 30 系列用 sm_86，40 系列 sm_89
        cmd += [
            "-gencode",
            "arch=compute_86,code=sm_86",
            "-gencode",
            "arch=compute_89,code=sm_89",
        ]

        print("NVCC:", " ".join(cmd))
        subprocess.check_call(cmd)
        return [str(obj)]


ext = Extension(
    "fastcuda",
    sources=["fastcuda.cu"],  # 你的 .cu 文件
    include_dirs=[pybind11.get_include(), np.get_include()],
    library_dirs=CUDA_LIBS,
    libraries=["cudart"],  # 链接 CUDA runtime
    extra_compile_args=["/O2"],
)

setup(
    name="fastcuda",
    version="0.0.1",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
