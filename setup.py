# setup.py
"""Build script for the log3 C++17 BitNet b1.58 & Autograd Python package.

Compiles `binding/binding.cpp` together with all C++17 implementation files in
`src/**/*.cpp` into the native extension module `log3._C` with zero external
C++ library dependencies.

Usage:
    pip install -e .
    # or:
    python3 setup.py build_ext --inplace
"""

from __future__ import annotations

import glob
import os
import platform
import sys
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class Log3BuildExt(build_ext):
    """Custom build_ext that applies strict C++17, -fno-exceptions, -fno-rtti, and SIMD flags."""

    def build_extensions(self) -> None:
        compiler_type = self.compiler.compiler_type
        system = platform.system().lower()
        machine = platform.machine().lower()
        enable_native = os.environ.get("L3_ENABLE_NATIVE", "1") != "0"

        for ext in self.extensions:
            if compiler_type == "msvc":
                ext.extra_compile_args = [
                    "/std:c++17",
                    "/O2",
                    "/W4",
                    "/MP",
                    "/Zc:__cplusplus",
                    "/EHs-c-",
                    "/GR-",
                ]
                ext.define_macros.extend([
                    ("_HAS_EXCEPTIONS", "0"),
                    ("_CRT_SECURE_NO_WARNINGS", "1"),
                    ("NOMINMAX", "1"),
                    ("WIN32_LEAN_AND_MEAN", "1"),
                    ("L3_STATIC", "1"),
                    ("L3_MEMORY_TRACKING", "1"),
                ])
                if enable_native and machine in ("amd64", "x86_64", "x86"):
                    ext.extra_compile_args.append("/arch:AVX2")
            else:
                ext.extra_compile_args = [
                    "-std=c++17",
                    "-O3",
                    "-Wall",
                    "-Wextra",
                    "-Wno-unused-parameter",
                    "-Wno-missing-field-initializers",
                    "-fno-exceptions",
                    "-fno-rtti",
                    "-fPIC",
                    "-pthread",
                ]
                ext.extra_link_args = ["-pthread"]
                ext.define_macros.extend([
                    ("L3_STATIC", "1"),
                    ("L3_MEMORY_TRACKING", "1"),
                ])
                if enable_native and "android" not in system:
                    ext.extra_compile_args.append("-march=native")
                if system == "linux":
                    ext.libraries.append("m")

        super().build_extensions()


def get_sources() -> list[str]:
    root = os.path.dirname(os.path.abspath(__file__))
    cpp_sources = sorted(
        glob.glob(os.path.join(root, "src", "**", "*.cpp"), recursive=True)
    )
    rel_sources = [os.path.relpath(p, root) for p in cpp_sources]
    rel_sources.append(os.path.join("binding", "binding.cpp"))
    return rel_sources


ext_modules = [
    Extension(
        name="log3._C",
        sources=get_sources(),
        include_dirs=["include"],
        language="c++",
    )
]

setup(
    name="log3",
    version="0.4.0",
    description=(
        "High-performance C++17 BitNet b1.58 (2-bit ternary), N-D Tensor, "
        "SIMD Kernels, GQA Transformer, Zero-Copy .l3 I/O, and Reverse-Mode Autograd library"
    ),
    author="Ansh Raj",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(include=["log3", "log3.*"]),
    package_data={"log3": ["py.typed", "*.pyi", "_C.pyi"]},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": Log3BuildExt},
    zip_safe=False,
)
