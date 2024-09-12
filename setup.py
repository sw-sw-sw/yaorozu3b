from setuptools import setup, Extension
import pybind11
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        'predator_prey_module',
        ['predator_prey.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11'],  # C++11を使用
    ),
]

setup(
    name='predator_prey_module',
    ext_modules=ext_modules,
    zip_safe=False,
)
"""
ビルド方法

python setup.py build_ext --inplace


"""
 