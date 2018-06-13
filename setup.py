from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension('image_analysis.im2col.im2col_cython_float', ['image_analysis/im2col/im2col_cython_float.pyx'],
              include_dirs=[numpy.get_include()]),
    Extension('image_analysis.im2col.im2col_cython_object', ['image_analysis/im2col/im2col_cython_object.pyx'],
              include_dirs=[numpy.get_include()]),
]

setup(
    ext_modules=cythonize(extensions), requires=['matplotlib', 'numpy', 'seaborn', 'Cython', 'keras', 'IPython',
                                                 'nbformat']
)
