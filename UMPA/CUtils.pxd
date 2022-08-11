cdef extern from "lib/Utils.cpp":
    pass

cdef extern from "lib/Utils.h":
    T gaussian_kernel[T](int i, int j, T a, T b, T c)
    T gaussian_blur[T](T* image, int i, int j, int dim[2], int Nk, T a, T b, T c)
    T gaussian_blur_isotropic[T](T* image, int i, int j, int dim[2], int Nk, T sigma)
    T convolve[T](T* image, int i, int j, int dim[2], T* kernel, int Nk)
