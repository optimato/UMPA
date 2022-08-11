#ifndef OPTIM_H
#define OPTIM_H

#include <functional>

// Error status structure returned by distance functions
struct error_status {
    unsigned int ok : 1;
    unsigned int bound_error : 1;
    unsigned int dimension : 1;
    unsigned int positive : 1;
};


template <class T>
struct minimizer_debug {
    minimizer_debug() {}; 
    T d[25];  // Function neighborhood
    T a[16];  // Function neighborhood for the subpixel minimizer
    int Ncalls;    // Number of function calls
};

// Sub-pixel minimum position
template <class T>
T spmin(T* a, T* pos);

template <class T>
T spmin_quad(T* a, T* pos);

// 2d minimizer
template <typename T, typename OBJ, typename COST_ARGS, error_status(OBJ::*f)(T*, int*, COST_ARGS* args)>
error_status discrete_2d_minimizer(T* out, T* uv, OBJ *obj, COST_ARGS* args, minimizer_debug<T>* db);

#endif
