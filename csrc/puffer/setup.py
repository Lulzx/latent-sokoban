"""Build the Sokoban Ocean extension against the installed PufferLib.

    python csrc/puffer/setup.py build_ext --inplace

Builds out-of-tree: env_binding.h is pulled from wherever pufferlib is
installed, so nothing inside the pufferlib package has to be modified.
"""

import os
from pathlib import Path

import numpy as np
import pufferlib
from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent
PUFFER = Path(os.path.dirname(pufferlib.__file__))

setup(
    name="sokoban-ocean",
    ext_modules=[
        Extension(
            "binding",
            sources=[str(HERE / "binding.c")],
            include_dirs=[
                str(PUFFER / "ocean"),      # env_binding.h
                str(HERE),
                str(HERE.parent),           # sokoban.h, solver.h
                np.get_include(),
            ],
            extra_compile_args=["-O2", "-std=c99"],
        )
    ],
)
