#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "Optim.h"

const int KERNEL_WINDOW_SIZE = 8;

// TODO: I think the `Im` attribute can be deleted everywhere, we're not using
//       this anymore.

/*
 Cost function argument base class: contains only the (input) coordinates `ij`
 where the cost function is to be computed.
*/
template <class T>
class CostArgs {
    public:
        int ij[2];
        CostArgs();
        CostArgs(int i, int j);
        ~CostArgs();
};

/*
 Derived class for without-dark-field cost function: contains field `t` for
 evaluated transmittance.
*/
template <class T>
class CostArgsNoDF : public CostArgs<T> {
    public:
        T t;
        //using CostArgs<T>::CostArgs;
        CostArgsNoDF();
        CostArgsNoDF(int i, int j);
        ~CostArgsNoDF();
};

/*
 Derived class for with-dark-field cost function: contains fields "t" for
 evaluated transmittance and "v" for dark-field.
*/
template <class T>
class CostArgsDF : public CostArgs<T> {
    public:
        T t;
        T v;
        //using CostArgs<T>::CostArgs;
        CostArgsDF();
        CostArgsDF(int i, int j);
        ~CostArgsDF();
};

/*
 Derived class for kernel-dark-field cost function: contains fields `t` for
 evaluated transmittance and `a`, `b`, `c` for dark-field kernel parameters.
*/
template <class T>
class CostArgsDFKernel : public CostArgs<T> {
    public:
        T t;
        T a;
        T b;
        T c;
        T kernel[(2*KERNEL_WINDOW_SIZE+1)*(2*KERNEL_WINDOW_SIZE+1)];
        //using CostArgs<T>::CostArgs;
        CostArgsDFKernel();
        CostArgsDFKernel(int i, int j, T a, T b, T c);
        CostArgsDFKernel<T>& operator=(const CostArgsDFKernel<T> & rhs);
        ~CostArgsDFKernel();
};


namespace models {
    /*
    Model base class
     */
    template <class T>
    class ModelBase {
        public:
            int Na;                     // The number of frames
            int Nw;                     // Window size
            std::vector<T*> ref;   // list of reference frames
            std::vector<T*> sam;   // list of reference frames
            std::vector<T*> mask;  // list of reference frames
            std::vector<int*> dim; // list of Frame dimensions
            std::vector<int*> pos; // list of Frame positions
            T* win;                // Window
            int max_shift;              // maximum shift
            int padding;            // Number of pixels removed from the sides
            int subpx_func;         // sara: Subpixel optimization function to use
            int reference_shift;    // Model the sample translation with fixed reference instead of the opposite.
            

            ModelBase();
            
            ModelBase(
                int Na, std::vector<int*> &dim, std::vector<T*> &sams,
                std::vector<T*> &refs, std::vector<T*> &masks,
                std::vector<int*> &pos, int Nw, T* win, int max_shift,
                int padding);
            
            ~ModelBase();
            
            virtual void set_window(T* new_win, int new_Nw);

            virtual T test();

            error_status coverage(T* out, int i, int j);

            virtual error_status min(
                int i, int j, T* values, minimizer_debug<T>* db);

            virtual error_status min(
                int i, int j, T* values, T uv[2], minimizer_debug<T>* db) = 0;

            virtual error_status cost_interface(
                int i, int j, int shift_i, int shift_j, T* values) = 0;

            // virtual error_status cost(T* out, int* ij, T* args) = 0;
    };

    /*
    Model definition for without-dark-field case.
    */
    template <class T>
    class ModelNoDF: public ModelBase<T> {
        public:
            using ModelBase<T>::ModelBase;
            
            error_status cost(T* out, int* shift_ij, CostArgsNoDF<T>* args);

            error_status min(
                int i, int j, T* values, T uv[2], minimizer_debug<T>* db);

            error_status cost_interface(
                int i, int j, int shift_i, int shift_j, T* values);
            // error_status min(
            //  int i, int j, T* D, T* t, T* sx, T* sy, minimizer_debug<T>* db);
    };

    /*
    Model definition for with-dark-field case.
    */
    template <class T>
    class ModelDF: public ModelBase<T> {
        public:
            T Im; // Mean intensity
            //using ModelBase<T>::ModelBase;
            ModelDF(
                int Na, std::vector<int*> &dim, std::vector<T*> &sams,
                std::vector<T*> &refs, std::vector<T*> &masks,
                std::vector<int*> &pos, int Nw, T* win, int max_shift,
                int padding);
            
            error_status cost(T* out, int* shift_ij, CostArgsDF<T>* args);
            
            error_status min(
                int i, int j, T* values, T uv[2], minimizer_debug<T>* db);

            error_status cost_interface(
                int i, int j, int shift_i, int shift_j, T* values);

    };

    /*
    Model definition for 3-parameter kernel dark-field method
    */

    template <class T>
    class ModelDFKernel: public ModelBase<T> {
        public:
            int Nk; // dimensions of kernel
            //using ModelBase<T>::ModelBase;
            
            ModelDFKernel(
                int Na, std::vector<int*> &dim, std::vector<T*> &sams,
                std::vector<T*> &refs, std::vector<T*> &masks,
                std::vector<int*> &pos, int Nw, T* win, int max_shift,
                int padding);
            
            error_status cost(T* out, int* shift_ij, CostArgsDFKernel<T>* args);
            
            error_status min(
                int i, int j, T* values, T uv[2], minimizer_debug<T>* db);
            
            error_status cost_interface(
                int i, int j, int shift_i, int shift_j, T* values);
    };

}

#endif
