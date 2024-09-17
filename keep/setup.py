from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "numpy_position_buffer",
        ["numpy_position_buffer.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules=cythonize(extensions)
)


# python setup.py build_ext --inplace