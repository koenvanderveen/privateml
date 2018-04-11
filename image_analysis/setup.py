from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
      Extension('im2col/im2col_cython_float', ['im2col/im2col_cython_float.pyx'],
                    include_dirs = [numpy.get_include()]),
    Extension('im2col/im2col_cython_object', ['im2col/im2col_cython_object.pyx'],
              include_dirs = [numpy.get_include()]),
]

setup(
        ext_modules = cythonize(extensions),
)
