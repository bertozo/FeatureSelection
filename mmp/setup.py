from distutils.core import setup
from Cython.Build import cythonize

setup(name='_mmp', ext_modules = cythonize('_mmp.pyx'))