#!/usr/bin/env python
"""Setuptools entry point that drives the CMake build for librevras."""

from __future__ import annotations

import multiprocessing
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """A setuptools Extension stub that knows where the CMake project lives."""

    def __init__(self, name: str, sourcedir: str = ".") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()


class CMakeBuild(build_ext):
    """Invoke CMake to compile the pybind11 module."""

    def run(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError("CMake >= 3.27 is required to build librevras") from exc
        super().run()

    def build_extension(self, ext: CMakeExtension) -> None:  # type: ignore[override]
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DBUILD_EXECUTABLES=OFF",
            "-DBUILD_PYTHON_LIB=ON",
            "-DBUILD_OFFSCREEN_VERSION=ON",
        ]

        extra_args = os.environ.get("CMAKE_ARGS")
        if extra_args:
            cmake_args += shlex.split(extra_args)

        build_args: list[str] = []
        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            build_args += ["--config", cfg]
        else:
            cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in env:
            build_args += ["-j", str(multiprocessing.cpu_count())]

        subprocess.check_call(
            ["cmake", "-S", str(ext.sourcedir), "-B", str(build_temp), *cmake_args],
            env=env,
        )
        subprocess.check_call(
            ["cmake", "--build", str(build_temp), *build_args],
            env=env,
        )


here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="reversible-rasterizer",
    version="0.1.0",
    description="Pybind11 bindings for the reversible rasterizer renderer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scikit-learn>=1.0",
    ],
    ext_modules=[CMakeExtension("librevras", sourcedir=str(here))],
    cmdclass={"build_ext": CMakeBuild},
    py_modules=["reversible_rasterizer_api"],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
        "Development Status :: 3 - Alpha",
    ],
)

