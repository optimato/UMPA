#include <cmath>
#include "Optim.h"
#include <iostream>

template <class T>
const T d0[25] = {-1., -1., -1., -1., -1.,
                  -1., -1., -1., -1., -1.,
                  -1., -1., -1., -1., -1.,
                  -1., -1., -1., -1., -1.,
                  -1., -1., -1., -1., -1.};

// Limit number of function calls to prevent getting stuck in a loop
// (which should not happen though...)
const int MAX_CALLS = 500;

/*
========== spmin() ==========
Sub-pixel minimum position based on convolution with the cross-correlation of
two bilinear interpolating kernels (see paper for more detail)!

Input arguments:
----------------
    a: T[16]
        (Flattened) 4x4 array of cost function values, where the minimum is
        known to be contained in the central square.
    pos: T[2]
        Initial search position 
        (must be the integer position of the minimum in `a`)

Output arguments:
-----------------
    pos : T[2]
        Sub-pixel position of the interpolated function.
        Yes, `pos` is both an input parameter and output parameter!

Return argument:
----------------
    c : T
        Interpolated minimum value
*/
template <class T>
T spmin(T *a, T *pos)
{
    // Coefficients of the polynomial representation of the 4x4 array // fdm: this is my modification of the matrix, see issue #2 on GitHub...
/*    
    T c[] = {                                               36*a[5],
                       -12*a[1]                            -18*a[5]                            +36*a[9]                               - 6*a[13],
                        18*a[1]                            -36*a[5]                            +18*a[9],
                        -6*a[1]                            +18*a[5]                            -18*a[9]                               + 6*a[13],
                                                  -12*a[4] -18*a[5] +36*a[6] - 6*a[7],
                4*a[0] + 6*a[1] -12*a[2] + 2*a[3] + 6*a[4] + 9*a[5] -18*a[6] + 3*a[7] -12*a[8] -18*a[9] +36*a[10] - 6*a[11] + 2*a[12] + 3*a[13] - 6*a[14] + 1*a[15],
               -6*a[0] - 9*a[1] +18*a[2] - 3*a[3] +12*a[4] +18*a[5] -36*a[6] + 6*a[7] - 6*a[8] - 9*a[9] +18*a[10] - 3*a[11],
                2*a[0] + 3*a[1] - 6*a[2] + 1*a[3] - 6*a[4] - 9*a[5] +18*a[6] - 3*a[7] + 6*a[8] + 9*a[9] -18*a[10] + 3*a[11] - 2*a[12] - 3*a[13] + 6*a[14] - 1*a[15],
                                                   18*a[4] -36*a[5] +18*a[6],
               -6*a[0] +12*a[1] - 6*a[2]          - 9*a[4] +18*a[5] - 9*a[6]          +18*a[8] -36*a[9] +18*a[10]           - 3*a[12] + 6*a[13] - 3*a[14],
                9*a[0] -18*a[1] + 9*a[2]          -18*a[4] +36*a[5] -18*a[6]          + 9*a[8] -18*a[9] + 9*a[10],
               -3*a[0] + 6*a[1] - 3*a[2]          + 9*a[4] -18*a[5] + 9*a[6]          - 9*a[8] +18*a[9] - 9*a[10]           + 3*a[12] - 6*a[13] + 3*a[14],
                                                   -6*a[4] +18*a[5] -18*a[6] + 6*a[7],
                2*a[0] - 6*a[1] + 6*a[2] - 2*a[3] + 3*a[4] - 9*a[5] + 9*a[6] - 3*a[7] - 6*a[8] +18*a[9] -18*a[10] + 6*a[11] + 1*a[12] - 3*a[13] + 3*a[14] - 1*a[15],
               -3*a[0] + 9*a[1] - 9*a[2] + 3*a[3] + 6*a[4] -18*a[5] +18*a[6] - 6*a[7] - 3*a[8] + 9*a[9] - 9*a[10] + 3*a[11],
                1*a[0] - 3*a[1] + 3*a[2] - 1*a[3] - 3*a[4] + 9*a[5] - 9*a[6] + 3*a[7] + 3*a[8] - 9*a[9] + 9*a[10] - 3*a[11] - 1*a[12] + 3*a[13] - 3*a[14] + 1*a[15]
             };
*/
    T c[] = {  a[0] + 4*a[1] + a[2] + 4*a[4] + 16*a[5] + 4*a[6] + a[8] + 4*a[9] + a[10],
                    -3*a[0] - 12*a[1] - 3*a[2] + 3*a[8] + 12*a[9] + 3*a[10],
                    3*a[0] + 12*a[1] + 3*a[2] - 6*a[4] - 24*a[5] - 6*a[6] + 3*a[8] + 12*a[9] + 3*a[10],
                    -a[0] - 4*a[1] - a[2] + 3*a[4] + 12*a[5] + 3*a[6] - 3*a[8] - 12*a[9] - 3*a[10] + a[12] + 4*a[13] + a[14],
                    -3*a[0] + 3*a[2] - 12*a[4] + 12*a[6] - 3*a[8] + 3*a[10],
                    9*a[0] - 9*a[2] - 9*a[8] + 9*a[10],
                    -9*a[0] + 9*a[2] + 18*a[4] - 18*a[6] - 9*a[8] + 9*a[10],
                    3*a[0] - 3*a[2] - 9*a[4] + 9*a[6] + 9*a[8] - 9*a[10] - 3*a[12] + 3*a[14],
                    3*a[0] - 6*a[1] + 3*a[2] + 12*a[4] - 24*a[5] + 12*a[6] + 3*a[8] - 6*a[9] + 3*a[10],
                    -9*a[0] + 18*a[1] - 9*a[2] + 9*a[8] - 18*a[9] + 9*a[10],
                    9*a[0] - 18*a[1] + 9*a[2] - 18*a[4] + 36*a[5] - 18*a[6] + 9*a[8] - 18*a[9] + 9*a[10],
                    -3*a[0] + 6*a[1] - 3*a[2] + 9*a[4] - 18*a[5] + 9*a[6] - 9*a[8] + 18*a[9] - 9*a[10] + 3*a[12] - 6*a[13] + 3*a[14],
                    -a[0] + 3*a[1] - 3*a[2] + a[3] - 4*a[4] + 12*a[5] - 12*a[6] + 4*a[7] - a[8] + 3*a[9] - 3*a[10] + a[11],
                    3*a[0] - 9*a[1] + 9*a[2] - 3*a[3] - 3*a[8] + 9*a[9] - 9*a[10] + 3*a[11],
                    -3*a[0] + 9*a[1] - 9*a[2] + 3*a[3] + 6*a[4] - 18*a[5] + 18*a[6] - 6*a[7] - 3*a[8] + 9*a[9] - 9*a[10] + 3*a[11],
                    a[0] - 3*a[1] + 3*a[2] - a[3] - 3*a[4] + 9*a[5] - 9*a[6] + 3*a[7] + 3*a[8] - 9*a[9] + 9*a[10] - 3*a[11] - a[12] + 3*a[13] - 3*a[14] + a[15]
                    };
 
    T x = pos[0];
    T y = pos[1];
    // Tolerance TODO: could be an input argument instead.
    T tol = 1e-8;
    T det;

    T dx, dy, fx, fy, fxx, fxy, fyy;

    // Newton-Raphson
    for(int i=0; i<=20; i++)
    {
        // Compute first and second derivatives   // fdm: ok, I checked them
        fx = c[1] + 2*c[2]*x + 3*c[3]*x*x + c[5]*y + 2*c[6]*x*y + 3*c[7]*x*x*y +
             c[9]*y*y + 2*c[10]*x*y*y + 3*c[11]*x*x*y*y + c[13]*y*y*y + 2*c[14]*x*y*y*y + 3*c[15]*x*x*y*y*y;
        fy = c[4] + c[5]*x + c[6]*x*x + c[7]*x*x*x + 2*c[8]*y + 2*c[9]*x*y +
             2*c[10]*x*x*y + 2*c[11]*x*x*x*y + 3*c[12]*y*y + 3*c[13]*x*y*y + 3*c[14]*x*x*y*y + 3*c[15]*x*x*x*y*y;
        fxx = 2*c[2] + 6*c[3]*x + 2*c[6]*y + 6*c[7]*x*y + 2*c[10]*y*y + 6*c[11]*x*y*y + 2*c[14]*y*y*y + 6*c[15]*x*y*y*y;
        fxy = c[5] + 2*c[6]*x + 3*c[7]*x*x + 2*c[9]*y + 4*c[10]*x*y + 6*c[11]*x*x*y + 3*c[13]*y*y + 6*c[14]*x*y*y + 9*c[15]*x*x*y*y;
        fyy = 2*c[8] + 2*c[9]*x + 2*c[10]*x*x + 2*c[11]*x*x*x + 6*c[12]*y + 6*c[13]*x*y + 6*c[14]*x*x*y + 6*c[15]*x*x*x*y;

        // Move to new minimum
        det = fxx*fyy - fxy*fxy;    // x_k+1 - x_k = -inv(hess(f)) .* grad(f). "det" is the determinant of hess(f)
        dx = (fxy*fy - fyy*fx)/det; 
        dy = (fxy*fx - fxx*fy)/det;
        x += dx;
        y += dy;

        /*if ((x<0.) | (x>1.) | (y<0.) | (y>1.))
        {
            // Fallback to quadratic
            //return spmin_quad(a, pos);

            
            // Give up if we get out of the box
            x = pos[0];
            y = pos[1];
            break;
            
        }*/

        // Exit loop if tolerance is reached
        if(dx*dx + dy*dy < tol) break;
    }
    pos[0] = x;
    pos[1] = y;
    return (c[0] + c[1]*x + c[2]*x*x + c[3]*x*x*x + c[4]*y + c[5]*x*y + c[6]*x*x*y +
            c[7]*x*x*x*y + c[8]*y*y + c[9]*x*y*y + c[10]*x*x*y*y + c[11]*x*x*x*y*y +
            c[12]*y*y*y + c[13]*x*y*y*y + c[14]*x*x*y*y*y + c[15]*x*x*x*y*y*y)/36.;
}


/*
========== spmin_quad() ==========
Calculates sub-pixel minimum position based on a quadratic fit.

Input parameters:
-----------------
    a: T[16]
        (flattened) 4x4 pixel region of the cost function landscape surrounding
        the minimum. Minimum must be one of the central 2x2 pixels.

Output parameters:
------------------
    pos: double[2]
        Sub-pixel position of minimum of the interpolation of `a`.
        Equivalent to `uv` in `discrete_2d_minimizer()`

Return parameter:
-----------------
    Interpolated minimum value

TODO: catch all failed cases (hessian non positive-definite)
 */
template <class T>
T spmin_quad(T* a, T* pos)
{
    // Pseudo-inverse applied to the array
    /*
    T p[] = { -75*a[0] + 50*a[1] + 50*a[2] - 75*a[3] + 50*a[4] + 175*a[5] + 175*a[6] + 50*a[7] + 50*a[8] + 175*a[9] + 175*a[10] + 50*a[11] - 75*a[12] + 50*a[13] + 50*a[14] - 75*a[15],
              -60*a[0] - 20*a[1] + 20*a[2] + 60*a[3] - 60*a[4] - 20*a[5] + 20*a[6] + 60*a[7] - 60*a[8] - 20*a[9] + 20*a[10] + 60*a[11] - 60*a[12] - 20*a[13] + 20*a[14] + 60*a[15],
              -60*a[0] - 60*a[1] - 60*a[2] - 60*a[3] - 20*a[4] - 20*a[5] - 20*a[6] - 20*a[7] + 20*a[8] + 20*a[9] + 20*a[10] + 20*a[11] + 60*a[12] + 60*a[13] + 60*a[14] + 60*a[15],
              50*a[0] - 50*a[1] - 50*a[2] + 50*a[3] + 50*a[4] - 50*a[5] - 50*a[6] + 50*a[7] + 50*a[8] - 50*a[9] - 50*a[10] + 50*a[11] + 50*a[12] - 50*a[13] - 50*a[14] + 50*a[15],
              72*a[0] + 24*a[1] - 24*a[2] - 72*a[3] + 24*a[4] + 8*a[5] - 8*a[6] - 24*a[7] - 24*a[8] - 8*a[9] + 8*a[10] + 24*a[11] - 72*a[12] - 24*a[13] + 24*a[14] + 72*a[15],
              50*a[0] + 50*a[1] + 50*a[2] + 50*a[3] - 50*a[4] - 50*a[5] - 50*a[6] - 50*a[7] - 50*a[8] - 50*a[9] - 50*a[10] - 50*a[11] + 50*a[12] + 50*a[13] + 50*a[14] + 50*a[15]};
    */
    // fdm: my version of the matrix (common denominator 400)
    // TODO: why is this version of the matrix used?
    T p[] = { 14*a[0]+48*a[1]+32*a[2]-34*a[3]+48*a[4]+86*a[5]+74*a[6]+12*a[7]+32*a[8]+74*a[9]+66*a[10]+ 8*a[11]-34*a[12]+12*a[13]+ 8*a[14]-46*a[15],
             -73*a[0]-61*a[1]-49*a[2]-37*a[3]+ 9*a[4]+13*a[5]+17*a[6]+21*a[7]+41*a[8]+37*a[9]+33*a[10]+29*a[11]+23*a[12]+11*a[13]- 1*a[14]-13*a[15],
             -73*a[0]+ 9*a[1]+41*a[2]+23*a[3]-61*a[4]+13*a[5]+37*a[6]+11*a[7]-49*a[8]+17*a[9]+33*a[10]- 1*a[11]-37*a[12]+21*a[13]+29*a[14]-13*a[15],
              25*a[0]+25*a[1]+25*a[2]+25*a[3]-25*a[4]-25*a[5]-25*a[6]-25*a[7]-25*a[8]-25*a[9]-25*a[10]-25*a[11]+25*a[12]+25*a[13]+25*a[14]+25*a[15],
              36*a[0]+12*a[1]-12*a[2]-36*a[3]+12*a[4]+ 4*a[5]- 4*a[6]-12*a[7]-12*a[8]- 4*a[9]+ 4*a[10]+12*a[11]-36*a[12]-12*a[13]+12*a[14]+36*a[15],
              25*a[0]-25*a[1]-25*a[2]+25*a[3]+25*a[4]-25*a[5]-25*a[6]+25*a[7]+25*a[8]-25*a[9]-25*a[10]+25*a[11]+25*a[12]-25*a[13]-25*a[14]+25*a[15]};
    
    T det = (4*p[3]*p[5] - p[4]*p[4]);
    //pos[0] = .5 - (2*p[3]*p[2] - p[4]*p[1])/det;
    //pos[1] = .5 - (2*p[5]*p[1] - p[4]*p[2])/det;
    // fdm: my version of pos should be without a 0.5 offset (?)
    pos[0] = - (2*p[3]*p[2] - p[4]*p[1]) / det;
    pos[1] = - (2*p[5]*p[1] - p[4]*p[2]) / det;
    //return (p[0] + .5*(p[2]*(pos[0]-.5) + p[1]*(pos[1]-.5)))/800;
    // fdm: altered version of the function value
    return (p[0] + .5 * (p[2] * pos[0] + p[1] * pos[1])) / 400;
}


/*
=========== discrete_2d_minimizer() ===========
Returns the (sub-pixel) minimum of the interpolated cost function, as well as
the location of this minimum. This minimization occurs in two steps:

1) Discrete minimization on a grid of integer dpc shifts.
2) When a discrete minimum is found, the neighborhood of the cost function is
   interpolated and the sub-pixel minimum is determined with a steepest-descent
   approach. This step is performed by `spmin()` (or `spmin_quad()`).

This function is written to minimise the number of function calls.
A minimum of 16 calls will be made because `spmin()` requires
a 4x4 neighbourhood of the local minimum.

Input arguments:
----------------
    f : function with signature `error_status s = f(T* out, int ij[2], COST_ARGS* args)`.
        Function to minimize. `COST_ARGS` is one of: `CostArgsNoDF`,
        `CostArgsDF`, `CostArgsDFKernel`.
    obj : Instance of one of: `ModelNoDF`, `ModelDF`, `ModelDFKernel`
        Model instance (see Model.cpp for definition)
    args : Instance of one of: `CostArgsNoDF`, `CostArgsDF`, `CostArgsDFKernel`
        CostArgs instance (see Model.cpp for definition)
    db : instance of `minimizer_debug` (defined in Optim.h).
        Contains a 4x4 array `d`, a 5x5 array `a` and an integer `Ncalls`.

Output arguments:
-----------------
    out : T
        Interpolated value of the function at the sub-pixel minimum
    uv : T[2]
        Sub-pixel minimum of the sample-to-ref shift
    
Notes:
------
 * `args` can be used for additional input parameters or output parameters. For
   this reason, the value of `args` is set to that returned by the function at
   the nearest discrete minimum.
 * To avoid `malloc`/`free`, `args` is currently limited to a hard-coded length
   of 50.
 * `f` is a cost function and must return only positive values.
*/

//template <class T>
//error_status discrete_2d_minimizer(T* out, T* uv, std::function<error_status(T*, int*, T*)> f, int N, T* args, minimizer_debug<T>* db)
template <typename T, typename OBJ, typename COST_ARGS, error_status(OBJ::*f)(T*, int*, COST_ARGS* args)>
error_status discrete_2d_minimizer(T* out, T* uv, OBJ *obj, COST_ARGS* args, minimizer_debug<T>* db)
{
    T* d;                    // Pointer to array to hold evaluated function and neighborhood
    T* a;                    // Pointer to array for sub-pixel refinement (subset of d once minimum is found)
    int c_m, c_p;            // Index in d for (m)inus or (p)lus 1 (vertically or horizontally)
    int ij[2];               // integer version of uv
    int ijc[2] = {0};        // Working array to call the function
    int ij_m[2] = {0};       // Working array to call the function
    int ij_p[2] = {0};       // Working array to call the function
    T tol = 1e-8;            // A small number to get out of cases where neighboring pixels have the same value
    int min_dir[2] = {0, 0}; // 0 if unknown, -1 or +1 give direction of found local minimum
    int dim = 0;             // 0 or 1, current search axis
    int min_m, min_p;        // bool to describe if new evaluated point is smaller than current value.
    error_status s;          // return status from function f.
    int i,j,ip,jp;           // working variables
    COST_ARGS args_copy;

    // Initialise d to all -1 (from global d0)
    memcpy(db->d, d0<T>, 25*sizeof(T));      // fdm: copy d0 (a 5x5 array of values, see l. 5) to db->d.
    d = db->d;
    a = db->a;
    db->Ncalls = 0;

    // Cast T uv T coordinate to ij int.
    ij[0] = (int) std::round(uv[0]);
    ij[1] = (int) std::round(uv[1]);

    // First function call in the middle     // fdm: "in the middle" = the center of the 5x5 neighborhood
    s = (obj->*f)(&(d[2*5 + 2]), ij, args);  // fdm: basically s, d[...] = f(...). I think "s" is just a program flow variable.
    db->Ncalls += 1;
    if(!s.ok) return s;                      // fdm: not sure what kind of check this is..? Maybe s.ok becomes 0 for NaN or something.
    args_copy = *args;

    while(db->Ncalls < MAX_CALLS)
    {
    start:                                   // fdm: this is used for a goto statement in l. 295!
        // c minus 1
        if(dim)
        {
            // Looking south: (-1, 0)
            c_m = (2-1)*5 + 2;               // fdm: c_m is an absolute index for the d environment. 
            ij_m[0] = ij[0]-1;               //      here: 1 element "south" of the center.
            ij_m[1] = ij[1];
        }
        else
        {
            // Looking west: (0, -1)
            c_m = 2*5 + (2-1);
            ij_m[0] = ij[0];
            ij_m[1] = ij[1]-1;
        }

        // If value has not been computed yet, call the function
        if(d[c_m] < -.5)                     // fdm: because it's initialized to -1 and f() can't be negative
        {
            s = (obj->*f)(&(d[c_m]), ij_m, args);       // fdm: "s, d[c_m] = f(ij_m, args)"
            db->Ncalls += 1;
            if(!s.ok) return s;
            // Here "tol" is used to impose a small bias to avoid infinite loops in cases
            // where neighboring pixels have the same value.
            min_m = d[c_m] > d[2 + 2*5] + tol;          // fdm: check: "is d[c_m] greater than d[center]?"
            // If the value lower, copy output values
            if(!min_m) args_copy = *args;    // fdm: so "args" are output values? I thought they were inputs...
        }
        else
        {
            min_m = d[c_m] > d[2 + 2*5] + tol;      // fdm: like line 213
        }

        // c plus 1
        if(dim)                                     // fdm: now we look in the opposite direction
        {
            // Looking north: (1, 0)
            c_p = (2+1)*5 + 2;
            ij_p[0] = ij[0]+1;
            ij_p[1] = ij[1];
        }
        else
        {
            // Looking east: (0, 1)
            c_p =  2*5 + (2+1);
            ij_p[0] = ij[0];
            ij_p[1] = ij[1]+1;
        }

        // If value has not been computed yet, call the function   // fdm: like block starting in l. 206
        if(d[c_p] < -.5)
        {
            s = (obj->*f)(&(d[c_p]), ij_p, args);
            db->Ncalls += 1;
            if(!s.ok) return s;
            min_p = d[c_p] > d[2 + 2*5] - tol;
            // If the value lower, copy output values
            if(!min_p) args_copy = *args;
        }
        else
        {
            min_p = d[c_p] > d[2 + 2*5] - tol;
        }

        if(min_m & min_p)          // fdm: "&" is bitwise AND
        {
            // Found minimum in at least one direction
            min_dir[dim] = d[c_m] < d[c_p] ? -1 : 1;

            if(min_dir[1-dim] != 0)
            {
                // Found in both direction. Time to grab all info and run sub-pixel minimizer

                // Find which quadrant to refine
                ip = d[2+3*5] < d[2 + 1*5] ? 1 : 0;
                jp = d[3+2*5] < d[1 + 2*5] ? 1 : 0;


                // Copy known data (and call f for unknown values) into a (4x4 array)
                for(i=0; i<4; i++)
                {
                    for(j=0; j<4; j++)
                    {
                        if(d[5*(ip+i) + jp+j] < -.9)
                        {
                            ijc[0] = ij[0] + ip + i - 2;
                            ijc[1] = ij[1] + jp + j - 2;
                            s = (obj->*f)(&(a[4*i + j]), ijc, args);     // fdm: s, a[4*i+j] = f(ijc, args)
                            db->Ncalls += 1;                             //      a is (becomes) the 4x4 array used in spmin, see l. 26
                            if(!s.ok) return s;

                            // Store also in d
                            d[5*(ip+i) + jp+j] = a[4*i + j];  // fdm: confusing, the 4x4 is selected from the 5x5
                                                              //      depending on the location of the minimum
                            if(a[4*i + j] < d[2 + 2*5])
                            {
                                // We have missed lower value (probably along a diagonal)
                                // TODO: figure the best way to shift known values of d
                                // Here, doing a hard restart instead.
                                ij[0] = ijc[0];
                                ij[1] = ijc[1];
                                memcpy(d, d0<T>, 25*sizeof(T));
                                d[2*5 + 2] = a[4*i + j];
                                *args = args_copy;
                                min_dir[0] = 0;
                                min_dir[1] = 0;
                                goto start;
                            }
                        }
                        else
                        {
                            a[4*i + j] = d[5*(ip+i) + jp+j];
                        }
                    }
                }

                *args = args_copy;

                // Call sub_pixel optimiser
                // fdm: I thought it should be (2-ip / 2-ip) (since those are the indices
                // of the center of the 5x5 in the shifted 4x4 box). However, since I defined
                // the coordinate grid to be [-1, 0, 1, 2], (1-ip / 1-jp) was coincidentally already correct!
                // (ip / jp) might also work, that would be one pixel diagonally in the direction of the minimum
                // No, actually that doesn't work at all...
                
                uv[0] = 1. - ip;
                uv[1] = 1. - jp;

                switch(obj->subpx_func){ // sara: add possibility of choosing sub_pixel optimization method
                    case 0: *out = *uv; // sara: no sub_pixel optimizer
                            break;
                    case 1: *out = spmin_quad(a, uv); // sara: quadratic fit, supposedly same as in pythonUMPA
                            break;
                    default:
                            *out = spmin(a, uv); // fdm: the call modifies uv in-place.
                    }

                uv[0] += ij[0] + ip - 1.;  // fdm: I assume this shifts the retrieved value
                uv[1] += ij[1] + jp - 1.;  //      to the original grid. uv isn't explicitly
                                           //      returned, but actually the important output.
                return s;
            }
            else
            {
                // Switch dimension and continue
                dim = 1-dim;
                continue;
            }
        }

        // Best values up to now (will be used if MAX_SHIFT is reached).
        uv[0] = ij[0];
        uv[1] = ij[1];
        *out = d[2*5 + 2];

        if((!min_p) & (!min_m))
        {
            // Local maximum !?
            min_m = d[c_p] < d[c_m];
        }

        if(min_m)
        {
            // Move in positive direction
            ij[1-dim] += 1;

            // Shift values in d
            if(dim)
            {
                // pos, dim=1
                for(i=5; i<25; i++) d[i-5] = d[i];
                for(i=0; i<5; i++) d[i + 5*4] = -1.;
            }
            else
            {
                // pos, dim=0
                for(i=1; i<25; i++) d[i-1] = d[i];
                for(i=0; i<5; i++) d[4 + 5*i] = -1.;
            }

            // Found minimum in other direction not valid
            min_dir[1-dim] = 0;
        }
        else
        {
            // Move in negative direction
            ij[1-dim] -= 1;

            // Shift values in d
            if(dim)
            {
                // neg, dim=1
                for(i=0; i<20; i++) d[24-i] = d[19-i];
                for(i=0; i<5; i++) d[i] = -1.;
            }
            else
            {
                // neg, dim=0
                for(i=1; i<25; i++) d[25-i] = d[24-i];
                for(i=0; i<5; i++) d[0 + 5*i] = -1.;
            }

            // Found minimum in other direction not valid
            min_dir[1-dim] = 0;
        }
    }
    // If we are here, too many calls were made
    s.ok = 0;
    return s;
}
