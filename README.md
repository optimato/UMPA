# UMPA
Code for the "Unified Modulated Pattern Analysis" (UMPA) method of processing speckle-based X-ray imaging data.

The original (Python-based) implementation of the method is published in
[M.-C. Zdora, et al., “X-ray Phase-Contrast Imaging and Metrology through Unified Modulated Pattern Analysis,” Phys. Rev. Lett., **118** 203903 (2017)](http://dx.doi.org/10.1103/PhysRevLett.118.203903), and is available at https://github.com/pierrethibault/UMPA. It is also included in this repository, in the file [speckle_matching.py](https://github.com/optimato/UMPA/blob/main/UMPA/speckle_matching.py).

This repository contains an improved, faster implementation of the UMPA model, which is achieved through the use of C++ and Cython. A publication describing and explaining this implementation is in currently in review. This readme will be updated upon acceptance.

This package is also used by an extension to directional dark-field (link to repository here!). That work is described in the publication "X-ray directional dark-field imaging using Unified Modulated Pattern Analysis" by Ronan Smith et al., which is to be published in PLOS ONE shortly. (update references when ready!)

## Installation instructions
In the current state, this repository is only compatible with Linux, and has only been tested with the `gcc` / `g++` compiler.

### Linux

Installation via
```
python setup.py install
```
or
```
pip install UMPA/
```
should be possible.

### Windows
Compilation with Microsoft Visual Studio under Windows is possible, but requires the modification of [setup.py](https://github.com/optimato/UMPA/blob/main/setup.py).

* Download "Visual Studio 2022, community version", select "Desktop development with C++" under "Workloads" when starting Visual Studio.
* In [setup.py](https://github.com/optimato/UMPA/blob/main/setup.py), modify
```
ext_1 = Extension(SRC_DIR + ".model",
                  [SRC_DIR + "/model.pyx"],
                  language="c++",
                  libraries=["m"],
                  extra_compile_args=["-std=c++17", "-O3", "-ffast-math", "-march=native", "-fopenmp" ],
                  extra_link_args=['-fopenmp'],
                  include_dirs=[np.get_include()])
```
to
```
ext_1 = Extension(SRC_DIR + ".model",
                  [SRC_DIR + "/model.pyx"],
                  language="c++",
                  extra_compile_args=["/std:c++17", "/O2", "/fp:fast", "/favor:INTEL64", "/openmp" ],
                  include_dirs=[np.get_include()])
```
This should work, it was tested on a Windows 11 laptop, multithreading works too. I'll modify setup.py to do this change automatically depending on architecture...

### Mac
Installation on Mac currently fails due to unknown reasons.

