from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np
import os

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

NAME = "UMPA"
VERSION = "0.2"
DESCR = "Unified Modulated Pattern Analysis"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Pierre Thibault, Fabio De Marco, Sara Savatovic, Ronan Smith"
EMAIL = "pthibault@units.it"

LICENSE = "GPL 3.0"

SRC_DIR = "UMPA"
PACKAGES = [SRC_DIR]

ext_1 = Extension(SRC_DIR + ".model",
                  [SRC_DIR + "/model.pyx"],
                  language="c++",
                  libraries=["m"],
                  extra_compile_args=["-std=c++17", "-O3", "-ffast-math", "-march=native", "-fopenmp" ],
                  extra_link_args=['-fopenmp'],
                  include_dirs=[np.get_include()])

EXTENSIONS = [ext_1]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          #url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=EXTENSIONS,
          include_package_data=True,
          package_data={'': ['test/logo.npy']}
          )

