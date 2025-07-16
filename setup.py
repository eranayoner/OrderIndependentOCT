from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Specify the Cython extension module
extensions = [
    Extension(
        "Pricing_Branching",
        sources=["Pricing.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    )
]

# Compile the Cython code
setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)