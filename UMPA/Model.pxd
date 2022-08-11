from libcpp.vector cimport vector

cdef extern from "lib/Model.cpp":
    pass

cdef extern from "lib/Optim.h" nogil:
    cdef cppclass minimizer_debug[T] nogil:
        minimizer_debug() nogil
        T d[25]
        T a[16]
        int Ncalls
    
    cdef cppclass error_status nogil:
        unsigned int ok
        unsigned int bound_error
        unsigned int dimension
        unsigned int positive

    
cdef extern from "lib/Model.h" nogil:
    cdef cppclass CostArgsDFKernel[T] nogil:
        T t
        T a
        T b
        T c
        T* kernel
        CostArgsDFKernel() nogil
        CostArgsDFKernel(int i, int j, T a, T b, T c) nogil

cdef extern from "lib/Model.h" namespace "models":
    cdef cppclass ModelBase[T]:
        int Na, Nw, max_shift, padding, reference_shift, subpx_func
        vector[int*] dim
        vector[T*] ref
        vector[T*] sam
        vector[T*] mask
        vector[int*] pos
        T* win
        T Im
        ModelBase()
        ModelBase(int Na, vector[int*] dim, vector[T*] sams, vector[T*] refs, vector[T*] masks, vector[int*] pos, int Nw, T* win, int max_shift, int padding) except +
        double test()
        void set_window(T* new_win, int new_Nw)
        error_status coverage(T* out, int i, int j) nogil
        error_status min(int i, int j, T* values, minimizer_debug[T]* db) nogil
        error_status min(int i, int j, T* values, T* uv, minimizer_debug[T]* db) nogil
        error_status cost_interface(int i, int j, int shift_i, int shift_j, T* values)


    cdef cppclass ModelNoDF[T](ModelBase[T]):
        ModelNoDF()
        ModelNoDF(int Na, vector[int*] dim, vector[T*] sams, vector[T*] refs, vector[T*] masks, vector[int*] pos, int Nw, T* win, int max_shift, int padding) except +
        error_status min(int i, int j, T* values, minimizer_debug[T]* db) nogil
        error_status min(int i, int j, T* values, T* uv, minimizer_debug[T]* db) nogil
        error_status cost_interface(int i, int j, int shift_i, int shift_j, T* values)

    cdef cppclass ModelDF[T](ModelBase[T]):
        ModelDF()
        ModelDF(int Na, vector[int*] dim, vector[T*] sams, vector[T*] refs, vector[T*] masks, vector[int*] pos, int Nw, T* win, int max_shift, int padding) except +
        error_status min(int i, int j, T* values, minimizer_debug[T]* db) nogil
        error_status min(int i, int j, T* values, T* uv, minimizer_debug[T]* db) nogil
        error_status cost_interface(int i, int j, int shift_i, int shift_j, T* values)

    cdef cppclass ModelDFKernel[T](ModelBase[T]):
        ModelDFKernel()
        ModelDFKernel(int Na, vector[int*] dim, vector[T*] sams, vector[T*] refs, vector[T*] masks, vector[int*] pos, int Nw, T* win, int max_shift, int padding) except +
        error_status min(int i, int j, T* values, minimizer_debug[T]* db) nogil
        error_status min(int i, int j, T* values, T* uv, minimizer_debug[T]* db) nogil
        error_status cost_interface(int i, int j, int shift_i, int shift_j, T* values)
