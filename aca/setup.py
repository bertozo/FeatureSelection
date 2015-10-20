from distutils.core import setup
from Cython.Build import cythonize

setup(name='_k_modes', ext_modules = cythonize('_k_modes.pyx'))