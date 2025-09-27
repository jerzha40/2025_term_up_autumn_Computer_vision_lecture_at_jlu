# setup.py
from setuptools import setup, Extension
import numpy as np

setup(
    name="fast",
    version="0.1.0",
    ext_modules=[
        Extension(
            "fast",
            sources=["fastmodule.c", "fast.c"],
            include_dirs=[np.get_include()],
            # extra_compile_args=["/O2"],  # MSVC
            # extra_link_args=[],
        )
    ],
)
