cdef extern from "lib/Optim.cpp":
    pass

cdef extern from "lib/Optim.h":
    T spmin[T](T* a, T* pos)
    T spmin_quad[T](T* a, T* pos)

    cdef cppclass minimizer_debug[T]:
        T d[25]
        T a[16]
        int Ncalls
    
    cdef cppclass error_status:
        unsigned int ok
        unsigned int bound_error
        unsigned int dimension
        unsigned int positive

    # error_status discrete_2d_minimizer[T, OBJ, FCT](T* out, T* uv, OBJ obj, FCT f, int N, T* args, minimizer_debug[T]* db);
