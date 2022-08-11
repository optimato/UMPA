#include <iostream>
#include <vector>
#include <cmath>
#include <exception>
#include "Model.h"
#include "Optim.h"
#include "Utils.h"

/*
fdm: 
I think the sign of the dpc signal should be flipped.
We use the definition that i_sam, j_sam = i_ref - ui, j_sam - uj, so if we get
a positive shift signal (ui, uj), that means that the matching sample window
is actually in the direction of negative ui, uj, relative to the ref window.
I.e., the sample refracted the speckle pattern towards NEGATIVE ui, uj, but
UMPA returns POSITIVE ui, uj, which is unintuitive.
*/

using namespace std;

/* CostArgs:
 * Class to hold the coordinates to a pixel (tuple of 2 integers)
 * Contains:
 *  - this->ij: array of 2 integers.
 * 
 * CostArgsNoDF:
 * Class to hold all arguments of the ModelNoDF cost function.
 * Derived class of CostArgs.
 * Contains:
 *  - this->ij: array of 2 integers (inherited from CostArgs)
 *  - this->t:  single number, type T (transmission value)
 * 
 * CostArgsDF:
 * Class to hold all arguments of the ModelDF cost function.
 * Derived class of CostArgs.
 * Contains:
 *  - this->ij: array of 2 integers (inherited from CostArgs)
 *  - this->t:  single number of type T (transmission value)
 *  - this->v:  single number of type T (dark-field value)
 * 
 * CostArgsDFKernel:
 * Class to hold all arguments of the ModelDFKernel cost function.
 * In the constructor (see below), `kernel` is generated according to the
 * coefficients `a`, `b`, `c`.
 * Derived class of CostArgs.
 * Contains:
 *  - this->ij:     array of 2 integers (inherited from CostArgs)
 *  - this->t:      single number of type T (transmission value)
 *  - this->a:      single number of type T
 *  - this->b:      single number of type T
 *  - this->c:      single number of type T
 *  - this->kernel: array of type T, (2*KERNEL_WINDOW_SIZE+1)^2 values
*/

// Constructors and destructors for the CostArgs* classes:
template <class T> CostArgs<T>::CostArgs() {};

template <class T> CostArgs<T>::CostArgs(int i, int j)
{
    this->ij[0] = i;
    this->ij[1] = j;
};

template <class T> CostArgs<T>::~CostArgs() {};

template <class T> CostArgsNoDF<T>::CostArgsNoDF() {};

template <class T> CostArgsNoDF<T>::CostArgsNoDF(int i, int j) : CostArgs<T>(i, j), t(0.) {};

template <class T> CostArgsNoDF<T>::~CostArgsNoDF() {};

template <class T> CostArgsDF<T>::CostArgsDF() {};

template <class T> CostArgsDF<T>::CostArgsDF(int i, int j) : CostArgs<T>(i, j), t(0.), v(0.) {};

template <class T> CostArgsDF<T>::~CostArgsDF() {};

template <class T> CostArgsDFKernel<T>::CostArgsDFKernel() {};

/*
========== CostArgsDFKernel: constructor =========
Builds a kernel according to the parameters `a`, `b`, `c`:
As given in `gaussian_kernel()`, the kernel is defined as:
exp(-a*i*i - b*i*j - c*j*j).
The extent of the constructed kernel is given by `KERNEL_WINDOW_SIZE`, i.e.
-KERNEL_WINDOW_SIZE <= i, j <= KERNEL_WINDOW_SIZE.
*/
template <class T> CostArgsDFKernel<T>::CostArgsDFKernel(
    int i, int j, T a, T b, T c) :
    
    CostArgs<T>(i, j),
    t(0.),
    a(a),
    b(b),
    c(c)
    {
        T norm = 0;
        int kws = KERNEL_WINDOW_SIZE;
        /* Generate kernel*/
        for(int k=0; k < 2*kws+1; k++)     // kernel rows
        {
            for(int l=0; l < 2*kws+1; l++) // kernel columns
            {
                kernel[k*(2*kws+1)+l] = gaussian_kernel<T>(k-kws, l-kws, a, b, c);
                norm += kernel[k*(2*kws+1)+l];
            }
        }

        /* Normalise */
        for(int k=0; k < 2*kws+1; k++)     // kernel rows
        {
            for(int l=0; l < 2*kws+1; l++) // kernel columns
            {
                kernel[k*(2*kws+1)+l] /= norm;
            }
        }
    };

template <class T> CostArgsDFKernel<T>& CostArgsDFKernel<T>::operator=(const CostArgsDFKernel<T> & rhs)
{
    if (this != &rhs)
    {
        t = rhs.t;
        a = rhs.a;
        b = rhs.b;
        c = rhs.c;
        this->ij[0] = rhs.ij[0];
        this->ij[1] = rhs.ij[1];
        for (int i = 0; i < (2*KERNEL_WINDOW_SIZE+1)*(2*KERNEL_WINDOW_SIZE+1); i++)
        {
            kernel[i] = rhs.kernel[i];
        }
    }
    return *this;
};


template <class T> CostArgsDFKernel<T>::~CostArgsDFKernel() {};

namespace models {

/*
========================================
============== ModelBase ===============
========================================
    PARENT CLASS FOR ALL UMPA MODELS

This class is the parent for ModelNoDF, ModelDF, and ModelDFKernel.
The below arguments are shared by the constructors for all these classes.
(ModelDFKernel has an additional argument).
*/

// Nullary constructor
template <class T> ModelBase<T>::ModelBase() {}


/*
======== Constructor for ModelBase =========
This constructor is also used for ModelNoDF.

Arguments:
----------
    Na : int
        Number of image frames (diffuser steps or sample steps)
    dim : vector, type int, shape (Na, 2) ?
        Dimensions of each image frame (in pixels)
    sams : vector, type T, (List of 2d arrays?)
        Sample image data.
    refs : List of 2d arrays?
        Reference image data (with sandpaper, without sample).
        Must be same shape as sams (?)
    masks : List of 2d arrays, type T
        Maps of fit weights. If supplied, cost is calculated as weighted sum of
        squares. Must be same shape as sams and refs (?)
    pos : List of [Na][2] ints
        Per-frame sample translation in the image plane. All zeroes if diffuser
        stepping is used.
    Nw : int
        Size parameter of the analysis window. Window size is (2*Nw+1, 2*Nw+1).
    win : array of T, size (2*Nw+1)^2
        Window weights to use. Unlike `masks`, these values are relative to the
        current position of the analysis window. Usually, a 2d Hamming window is
        used (which nearly falls off to zero at the edge of the analysis window)
    max_shift : int
        Maximum allowable shift (i.e., DPC signal in pixels) before cost
        function optimization is aborted. Abort condition is fulfilled if
        abs(dpcx) or abs(dpcy) exceeds max_shift.
    padding : int
        Number of pixels at the edge of the raw data frames to be excluded from
        UMPA analysis. Necessary to avoid analysis windows extending into the
        area beyond the limits of the raw data arrays.
    */
template <class T>
ModelBase<T>::ModelBase(
    int Na, vector<int*> &dim, vector<T*> &sams, vector<T*> &refs,
    vector<T*> &masks, vector<int*> &pos, int Nw, T* win, int max_shift,
    int padding) :
    /*
    note fdm: this is a "member initializer list", see e.g.
    https://en.cppreference.com/w/cpp/language/constructor
    So "sam { sams }" means: ModelBase::sam = sams (or something like that)
    I don't understand the necessity of "Na (Na)" or "win (win)"...
    */ 
    Na (Na),  // doesn't this just mean ModelBase::Na = ModelBase::Na?
    dim { dim },
    Nw (Nw),
    max_shift (max_shift),
    sam { sams },
    ref { refs },
    mask { masks },
    win ( win ),
    pos { pos },
    padding ( padding ),
    subpx_func (-1),  
    reference_shift (0)
    {}

/* sara: add subpixel method minimization function selection
            0:            no subpixel minimization
            1:            old quadratic fit (spmin_quad())
            default (-1): convolution with B*B (spmin()) */

// Destructor
template <class T> ModelBase<T>::~ModelBase () {}

template <class T>
T ModelBase<T>::test()
{
    cout << "Sub_pixel optimizator: " << this->subpx_func << endl;
    cout << this->sam[0] << " address of first element." << endl;
    cout << this->sam[0][0] << " first element." << endl;
    cout << this->dim[0][0] << ", " << this->dim[0][1] << " dimensions." << endl;
    return (T) sam.size();
}

/*
Set or reset the window pointer and dimensions
*/
template <class T>
void ModelBase<T>::set_window(T* new_win, int new_Nw)
{
    if (new_Nw < 0) throw runtime_error("Nw must be non-negative.");
    this->win = new_win;
    this->Nw = new_Nw;
    return;
}

/*
========== ModelBase, coverage() ==========
Calculates the coverage map (number of contributing frames)
at a single pixel.

Input parameters:
-----------------
    i, j: int
        Pixel coordinates (excluding padding, ROIs, etc.)

Output parameters:
------------------
    out: T*
        Pointer for output value of the coverage

Notes:
------
* If this->mask is empty, each contributing frame adds 1 to the output value.
  If this->mask is set, it contributes an amount equal to the value of
  the "mask" array in that pixel.
* When used without weighting, this function is only relevant for sample stepping.
  The coverage in an (unweighted) diffuser-stepping dataset is equal to the
  total number of frames (Na).

*/
template <class T>
error_status ModelBase<T>::coverage(T* out, int i, int j)
{
    T w, wt=0.;
    error_status s = {0};

    if(this->mask.empty())
    {
        for(int kk=0; kk<this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if((i - pos_i - this->padding) < 0) continue;
            if((i - pos_i + this->padding) > this->dim[kk][0]) continue;
            if((j - pos_j - this->padding) < 0) continue;
            if((j - pos_j + this->padding) > this->dim[kk][1]) continue;
            wt += 1.;
        }
    }
    else
    {
        for(int kk=0; kk<this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if((i - pos_i - this->padding) < 0) continue;
            if((i - pos_i + this->padding) > this->dim[kk][0]) continue;
            if((j - pos_j - this->padding) < 0) continue;
            if((j - pos_j + this->padding) > this->dim[kk][1]) continue;
            // fdm: I'm not convinced that these coordinates are correct.
            //      Furthermore, I'm not sure that it makes sense to include
            //      the weights in the coverage like this. There should be
            //      an option to calculate the "normal", or the "weighted"
            //      coverage.
            w = this->mask[kk][(i-pos_i)*this->dim[kk][1]+(j-pos_j)];
            wt += w;
        }
    }
    *out = wt;
    s.ok = 1;
    return s;
}

template <class T>
error_status ModelBase<T>::min(int i, int j, T* values, minimizer_debug<T>* db)
{
    T uv[2] = {0.};
    return min(i, j, values, uv, db);
}

/*
========================================
============== ModelNoDF ===============
========================================
       MODEL WITHOUT DARK FIELD

This model assumes that the effect of a sample is described as a translation
(refraction), and an overall attenuation. It assumes that speckle amplitude is
not reduced, i.e., that there is no dark-field.

There is no constructor here because ModelNoDF uses the one of ModelBase.
*/


/*
========== ModelNoDF, cost() ==========
Calculates one value of the cost function without dark-field, minimized with
respect to transmittance, for a given image pixel and sample-to-reference shift.
Note that elements of args, a CostArgsDF instance, are used as input and
output parameters.
 
Input arguments:
----------------
    shift_ij : array of 2 ints
        Displacement between reference and sample arrays ("u" in the paper)
    args->ij : array of 2 ints
        Pixel position in the frame ("r" in the paper)

Output arguments:
-----------------
    args->t : T
        Transmittance value that minimizes the cost function
    out : T
        Cost function value for the given shift (`shift_ij`), after being
        minimized w.r.t. transmittance (args->t) and dark-field (args->v).
*/
template <class T>
error_status ModelNoDF<T>::cost(T* out, int* shift_ij, CostArgsNoDF<T>* args)
{
    int i, j, ia, ib, ja, jb;
    int px_ref_i, px_ref_j, px_ref, px_sam_i, px_sam_j, px_sam;
    int ii, jj, kk;
    T t1 = 0.;
    T t3 = 0.;
    T t5 = 0.;
    T w, wt, wij, sij, rij; 
    error_status s = {0};
    int Nw = this->Nw;

    if(shift_ij[0] <= -this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 0;
        s.positive = 0;
        return s;
    }
    if(shift_ij[0] >= this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 0;
        s.positive = 0;
        return s;
    }
    if(shift_ij[1] <= -this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 1;
        s.positive = 0;
        return s;
    }
    if(shift_ij[1] >= this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 1;
        s.positive = 1;
        return s;
    }

    // Pixel coordinate
    // fdm: I think rounding is unnecessary here because
    //      args->ij is already of type int.
    i = (int) round(args->ij[0]);  
    j = (int) round(args->ij[1]);

    // Offset coordinate
    if(this->reference_shift)
    {
        ia = i;
        ja = j;
        ib = i - shift_ij[0];
        jb = j - shift_ij[1];
    }
    else
    {
        ia = i + shift_ij[0];
        ja = j + shift_ij[1];
        ib = i;
        jb = j;
    }

    if(this->mask.empty())
    {
        wt = this->Na; 
        for(kk=0; kk<this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if((i - pos_i - this->padding) < 0) continue;
            if((i - pos_i + this->padding) > this->dim[kk][0]) continue;  // fdm: not dim[kk][0]-1?
            if((j - pos_j - this->padding) < 0) continue;
            if((j - pos_j + this->padding) > this->dim[kk][1]) continue;
            for(ii=0; ii<2*Nw+1; ii++)
            {
                for(jj=0; jj<2*Nw+1; jj++)
                {
                    // pixel coordinates:
                    px_ref_i = ii - Nw + ia - pos_i;
                    px_ref_j = jj - Nw + ja - pos_j;
                    px_ref = px_ref_i * this->dim[kk][1] + px_ref_j;
                    px_sam_i = ii - Nw + ib - pos_i;
                    px_sam_j = jj - Nw + jb - pos_j;
                    px_sam = px_sam_i * this->dim[kk][1] + px_sam_j;
                    
                    // array values:
                    wij = this->win[ii*(2*Nw+1)+jj];
                    // "sij = sam[kk, ii-Nw+ib-pos_i, jj-Nw+jb-pos_j]"
                    sij = this->sam[kk][px_sam];
                    // "rij = ref[kk, ii-Nw+ia-pos_i, jj-Nw+ja-pos_j]"
                    rij = this->ref[kk][px_ref];

                    // cost function terms:
                    t1 += wij * sij * sij;
                    t3 += wij * rij * rij;
                    t5 += wij * rij * sij;
                }
            }
        }
    }
    else
    {
        wt = 0;
        for(kk=0; kk<this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if((i - pos_i - this->padding) < 0) continue;
            if((i - pos_i + this->padding) > this->dim[kk][0]) continue;  // fdm: not dim[kk][0]-1?
            if((j - pos_j - this->padding) < 0) continue;
            if((j - pos_j + this->padding) > this->dim[kk][1]) continue;
            for(ii=0; ii<2*Nw+1; ii++)
            {
                for(jj=0; jj<2*Nw+1; jj++)
                {
                    // pixel coordinates:
                    px_ref_i = ii - Nw + ia - pos_i;
                    px_ref_j = jj - Nw + ja - pos_j;
                    px_ref = px_ref_i * this->dim[kk][1] + px_ref_j;
                    px_sam_i = ii - Nw + ib - pos_i;
                    px_sam_j = jj - Nw + jb - pos_j;
                    px_sam = px_sam_i * this->dim[kk][1] + px_sam_j;

                    // array values:
                    wij = this->win[ii*(2*Nw+1)+jj];
                    sij = this->sam[kk][px_sam];
                    rij = this->ref[kk][px_ref];
                    w   =  combine_weights(this->mask[kk][px_ref],
                                            this->mask[kk][px_sam]);
                    
                    // cost function terms:
                    t1 += w * wij * sij * sij;
                    t3 += w * wij * rij * rij;
                    t5 += w * wij * rij * sij;
                    wt += w * wij;
                }
            }
        }
    }

    // Set transmission function
    args->t = t5 / t3;

    // Set cost function value
    *out = (t1 - t5*args->t) / wt;

    s.ok = 1;
    return s;
}

/* 
========== ModelNoDF, cost_interface() ==========
Calculates a single cost function value.
Accessible via the cost() functions in model.pyx.

Input arguments:
----------------
    i, j : int
        Pixel coordinates in sample frame
    shift_i, shift_j : int
        Displacement between reference and sample frames

Output arguments:
-----------------
    values : array of T
        o values[0]: cost function value
        o values[1]: transmission

Return argument:
----------------
    s : error_status instance.
*/
template <class T>
error_status ModelNoDF<T>::cost_interface(
    int i, int j, int shift_i, int shift_j, T* values)
{
    int shift_ij[2] = {shift_i, shift_j};
    CostArgsNoDF<T> args = CostArgsNoDF<T>(i, j);
    error_status s = this->cost(&values[0], shift_ij, &args);
    values[1] = args.t;
    return s;
}

/* 
========== ModelNoDF, min() ========== 
Performs UMPA optimisation on a single pixel.

Input arguments:
----------------
    i, j : int
        Pixel position

Output arguments:
-----------------
    values : T array, 4 values
        Optimized values (cost function, sample transmission, dpcx, dpcy)

Return argument:
----------------
    s : error_status instance
*/
template <class T>
error_status ModelNoDF<T>::min(
    int i, int j, T* values, T uv[2], minimizer_debug<T>* db)
{
    T D;
    error_status s;
    CostArgsNoDF<T> args = CostArgsNoDF<T>(i, j);

    s = discrete_2d_minimizer<T, ModelNoDF, CostArgsNoDF<T>, &ModelNoDF::cost>(
        &D, &uv[0], this, &args, db);

    values[0] = D;
    values[1] = args.t;
    values[2] = uv[1];
    values[3] = uv[0];
    return s;
}

/*
==========================================
================ ModelDF =================
==========================================
          MODEL WITH DARK FIELD

This approach assumes that the effect of a sample is described as a downscaling
of speckle amplitude (dark-field), a translation (refraction), and an
overall attenuation.

The `ModelDF::cost()` function applies these three steps, and minimizes
the cost function with respect to transmittance and dark-field.

Arguments:
----------
    see constructor of `ModelBase`.
*/
template <class T>
ModelDF<T>::ModelDF(
    int Na, vector<int*> &dim, vector<T*> &sams, vector<T*> &refs,
    vector<T*> &masks, vector<int*> &pos, int Nw, T* win, int max_shift,
    int padding) :
ModelBase<T>(Na, dim, sams, refs, masks, pos, Nw, win, max_shift, padding)
{
}

/*
========== ModelDF, cost() ==========
Calculates one value of the cost function with dark-field, minimized with
respect to transmittance and dark-field, for a given image pixel and
sample-to-reference shift.
Note that elements of args, a CostArgsDF instance, are used as input and
output parameters.
 
Input arguments:
----------------
    shift_ij : array of 2 ints
        Displacement between reference and sample arrays ("u" in the paper)
    args->ij : array of 2 ints
        Pixel position in the frame ("r" in the paper)

Output arguments:
-----------------
    args->t : T
        Transmittance value that minimizes the cost function
    args->v : T
        Dark-field value that minimizes the cost function
    out : pointer to T
        Cost function value for the given shift (`shift_ij`), after being
        minimized w.r.t. transmittance (args->t) and dark-field (args->v).
*/
template <class T>
error_status ModelDF<T>::cost(T* out, int* shift_ij, CostArgsDF<T>* args)
{
    int i, j, ia, ib, ja, jb;
    int px_ref_i, px_ref_j, px_ref, px_sam_i, px_sam_j, px_sam;
    int ii, jj, kk;
    T t1 = 0.;
    T t2 = 0.;
    T t3 = 0.;
    T t4 = 0.;
    T t5 = 0.;
    T t6 = 0.;
    T t2_term = 0.;   // variable for intermediate summation step
    T t4_term = 0.;   // variable for intermediate summation step
    T t6_term = 0.;   // variable for intermediate summation step
    T ref_mean = 0.;  // mean of ref intensity over window
    //T ref_denom = 0.;  // number of pixels in the mean calculation
    T beta, K, wij, sij, rij;
    T w, wt; 
    T c, wt_c, denom;
    error_status s = {0};
    int Nw = this->Nw;

    if(shift_ij[0] <= -this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 0;
        s.positive = 0;
        return s;
    }
    if(shift_ij[0] >= this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 0;
        s.positive = 0;
        return s;
    }
    if(shift_ij[1] <= -this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 1;
        s.positive = 0;
        return s;
    }
    if(shift_ij[1] >= this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 1;
        s.positive = 1;
        return s;
    }

    // Pixel coordinate
    i = (int) round(args->ij[0]);
    j = (int) round(args->ij[1]);

    // Offset coordinate
    if(this->reference_shift)
    {
        ia = i;
        ja = j;
        ib = i - shift_ij[0];
        jb = j - shift_ij[1];
    }
    else
    {
        ia = i + shift_ij[0];
        ja = j + shift_ij[1];
        ib = i;
        jb = j;
    }

    // fdm: I wonder if we can extract the calculation of the mean from
    //      the if-bracket. This would mean that the this->mask.empty()
    //      test ends up in the kk loop, i.e. it's run much more often,
    //      but is that a performance problem?
    //      If I extract that part we have to repeat less code...

    if(this->mask.empty())  // mask stuff moved here by Sara
    {
        wt = this->Na;
        for(kk=0; kk<this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if((i - pos_i - this->padding) < 0) continue;
            if((i - pos_i + this->padding) > this->dim[kk][0]) continue;
            if((j - pos_j - this->padding) < 0) continue;
            if((j - pos_j + this->padding) > this->dim[kk][1]) continue;

            // Change due to per-frame means: calculate the per-frame mean
            // before the cost function terms!
            ref_mean = 0.;
            denom = 0;
            for(ii=0; ii<2*Nw+1; ii++)
            {
                for(jj=0; jj<2*Nw+1; jj++)
                {
                    wij = this->win[ii*(2*Nw+1)+jj];
                    px_ref_i = ii - Nw + ia - pos_i;
                    px_ref_j = jj - Nw + ja - pos_j;
                    px_ref = px_ref_i * this->dim[kk][1] + px_ref_j;
                    //sara: I think we should take the window into account for the mean
                    ref_mean += wij * this->ref[kk][px_ref];
                    denom += wij;
                }
            }
            //ref_mean /= (2*Nw+1)*(2*Nw+1);
            ref_mean /= denom;

            /* additional change: The summation of the t4 and t6 terms
                must be split: sum the terms over the window in the current
                frame, then multiply with the mean in the frame's window.
                i.e. t4 changed from: <I0> * sum_k,i,j(I_kij)
                                    to: sum_k <I0>_k sum_i,j I_k,i,j     */
            t4_term = 0.;
            t6_term = 0.;
            for(ii=0; ii<2*Nw+1; ii++)
            {
                for(jj=0; jj<2*Nw+1; jj++)
                {
                    px_ref_i = ii - Nw + ia - pos_i;
                    px_ref_j = jj - Nw + ja - pos_j;
                    px_ref = px_ref_i * this->dim[kk][1] + px_ref_j;
                    px_sam_i = ii - Nw + ib - pos_i;
                    px_sam_j = jj - Nw + jb - pos_j;
                    px_sam = px_sam_i * this->dim[kk][1] + px_sam_j;

                    wij = this->win[ii*(2*Nw+1)+jj];
                    sij = this->sam[kk][px_sam];
                    rij = this->ref[kk][px_ref];

                    t1      += wij * sij * sij;
                    t3      += wij * rij * rij;
                    t4_term += wij * sij;
                    t5      += wij * rij * sij;
                    t6_term += wij * rij;
                }
            }
            t2 += ref_mean * ref_mean;
            t4 += ref_mean * t4_term;
            t6 += ref_mean * t6_term;
        }
    }
    else
    {
        wt = 0;
        for(kk=0; kk<this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if((i - pos_i - this->padding) < 0) continue;
            if((i - pos_i + this->padding) > this->dim[kk][0]) continue;
            if((j - pos_j - this->padding) < 0) continue;
            if((j - pos_j + this->padding) > this->dim[kk][1]) continue;

            // Change due to per-frame means: calculate the per-frame mean
            // before the cost function terms!
            ref_mean = 0.;
            denom = 0;
            for(ii=0; ii<2*Nw+1; ii++)
            {
                for(jj=0; jj<2*Nw+1; jj++)
                {
                    // pixel indices for reference
                    px_ref_i = ii - Nw + ia - pos_i;
                    px_ref_j = jj - Nw + ja - pos_j;
                    px_ref = px_ref_i * this->dim[kk][1] + px_ref_j;

                    wij = this->win[ii*(2*Nw+1)+jj];
                    rij = this->ref[kk][px_ref];
                    // Sara: I think we should take the window into account
                    //       for the mean
                    ref_mean += wij * this->ref[kk][px_ref];
                    denom += wij;
                }
            }
            ref_mean /= denom;

            t2_term = 0.;
            t4_term = 0.;
            t6_term = 0.;
            for(ii=0; ii<2*Nw+1; ii++)
            {
                for(jj=0; jj<2*Nw+1; jj++)
                {
                    // 1d and 2d indices for the pixel under examination
                    // for the reference and sample window:
                    px_ref_i = ii - Nw + ia - pos_i;
                    px_ref_j = jj - Nw + ja - pos_j;
                    px_ref = px_ref_i * this->dim[kk][1] + px_ref_j;
                    px_sam_i = ii - Nw + ib - pos_i;
                    px_sam_j = jj - Nw + jb - pos_j;
                    px_sam = px_sam_i * this->dim[kk][1] + px_sam_j;

                    // pixel values of window, sam, ref, weights:
                    wij = this->win[ii*(2*Nw+1)+jj];
                    sij = this->sam[kk][px_sam];
                    rij = this->ref[kk][px_ref];
                    w   = combine_weights(this->mask[kk][px_ref],
                                            this->mask[kk][px_sam]);

                    // cost function terms:
                    t1      += w * wij * sij * sij;
                    t2_term += w * wij;
                    t3      += w * wij * rij * rij;
                    t4_term += w * wij * sij;
                    t5      += w * wij * rij * sij;
                    t6_term += w * wij * rij;
                    wt += w * wij;
                }
            }
            t2 += ref_mean * ref_mean * t2_term;
            t4 += ref_mean * t4_term;
            t6 += ref_mean * t6_term;
        }
    }
    
    K = (t2*t5 - t4*t6) / (t2*t3 - t6*t6);
    beta = (t3*t4 - t5*t6) / (t2*t3 - t6*t6);

    // Transmission function and dark field
    args->t = beta + K;
    args->v = K / args->t;


    // Set cost function value
    *out = (t1 + beta*beta*t2 + K*K*t3 - 2*beta*t4 - 2*K*t5 + 2*beta*K*t6) / wt;

    s.ok = 1;
    return s;
}

/* 
========== ModelDF, cost_interface() ==========
Calculates a single cost function value.
Accessible via the cost() functions in model.pyx.

Input arguments:
----------------
    i, j : int
        Pixel coordinates in sample frame
    shift_i, shift_j : int
        Displacement between reference and sample frames

Output arguments:
-----------------
    values : array of T
        values[0]: Cost function value
        values[1]: Transmittance
        values[2]: Dark-field

Return argument:
----------------
    s : error_status instance.
*/
template <class T>
error_status ModelDF<T>::cost_interface(
    int i, int j, int shift_i, int shift_j, T* values)
{
    int shift_ij[2] = {shift_i, shift_j};
    CostArgsDF<T> args = CostArgsDF<T>(i, j);
    error_status s = this->cost(&values[0], shift_ij, &args);
    values[1] = args.t;
    values[2] = args.v;
    return s;
}


/* 
========== ModelNoDF, min() ========== 
Performs UMPA optimisation on a single pixel.

Input arguments:
----------------
    i, j : int
        Pixel position
    db : ?
        ?

Output arguments:
-----------------
    values : T array, 5 values
        Optimized values (cost function, sample transmission, dpcx, dpcy,
        dark-field)
    uv : T[2]
        Equivalent to values[2], values[3], just used as intermediate array

Return argument:
----------------
    s : error_status instance
*/
template <class T>
error_status ModelDF<T>::min(
    int i, int j, T* values, T uv[2], minimizer_debug<T>* db)
{
    T D;
    error_status s;

    CostArgsDF<T> args = CostArgsDF<T>(i, j);

    s = discrete_2d_minimizer<T, ModelDF, CostArgsDF<T>, &ModelDF::cost>(&D, &uv[0], this, &args, db);

    values[0] = D;
    values[1] = args.t;
    values[2] = uv[1];
    values[3] = uv[0];
    values[4] = args.v;
    return s;
}

/*
============== ModelDFKernel ===============
  MODEL WITH 3-PARAMETER KERNEL DARK FIELD

This approach assumes that the effect of a sample is described as a combination
of a blur, a translation, and an attenuation of the reference speckle pattern.

The `ModelDFKernel::cost()` function applies these three steps, but minimizes
the cost function only with respect to transmittance. A full solution of the
problem requires an optimization of the used blur kernel (through minimization
of the cost function). However, this functionality is not provided in this
package.

Arguments:
----------
    Nk : int
        Size parameter of blur kernel. Actual size: (2*Nk+1)^2. Initialized to
        the constant KERNEL_WINDOW_SIZE.
    all other parameters:
        see constructor of `ModelBase`.
*/
template <class T>
ModelDFKernel<T>::ModelDFKernel(
    int Na, vector<int*> &dim, vector<T*> &sams, vector<T*> &refs,
    vector<T*> &masks, vector<int*> &pos, int Nw, T* win, int max_shift,
    int padding) :
    ModelBase<T>(Na, dim, sams, refs, masks, pos, Nw, win, max_shift, padding), Nk(KERNEL_WINDOW_SIZE) { }


/*
========== ModelDFKernel, cost() ==========
Calculates one value of the cost function with kernel-based dark-field,
minimized with respect to transmittance, for a given image pixel,
sample-to-reference shift, and blur kernel.
Note that elements of args, a CostArgsDFKernel instance, are used as input and
output parameters.
 
Input arguments:
----------------
    shift_ij : array of 2 ints
        Displacement between reference and sample arrays ("u" in the paper)\
    args->ij : array of 2 ints
        Pixel position in the frame ("r" in the paper)
    args->kernel : array of type T, (2*KERNEL_WINDOW_SIZE+1)^2 values
        Blur kernel


Output arguments:
-----------------
    args->t : T
        Transmittance value that minimizes the cost function
    out : pointer to T
        Cost function value for the given shift (`shift_ij`), after being
        minimized for transmittance(?).
*/
template <class T>
error_status ModelDFKernel<T>::cost(
    T* out, int* shift_ij, CostArgsDFKernel<T>* args)
{
    int i, j, ia, ib, ja, jb;
    int px_ref, px_ref_i, px_ref_j, px_sam, px_sam_i, px_sam_j;
    int ii, jj, kk;
    T t1 = 0.;
    T t3 = 0.;
    T t5 = 0.;
    T w, wt, wij, sij, rij, blurred_ref;
    error_status s = {0};
    int Nw = this->Nw;

    if(shift_ij[0] <= -this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 0;
        s.positive = 0;
        return s;
    }
    if(shift_ij[0] >= this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 0;
        s.positive = 0;
        return s;
    }
    if(shift_ij[1] <= -this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 1;
        s.positive = 0;
        return s;
    }
    if(shift_ij[1] >= this->max_shift)
    {
        s.bound_error = 1;
        s.dimension = 1;
        s.positive = 1;
        return s;
    }

    // Pixel coordinate
    i = (int) round(args->ij[0]);
    j = (int) round(args->ij[1]);

    // Offset coordinate
    if(this->reference_shift)
    {
        ia = i;
        ja = j;
        ib = i - shift_ij[0];
        jb = j - shift_ij[1];
    }
    else
    {
        ia = i + shift_ij[0];
        ja = j + shift_ij[1];
        ib = i;
        jb = j;
    }

    if(this->mask.empty())
    {
        wt = this->Na;
        for(kk=0; kk<this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if((i - pos_i - this->padding) < 0) continue;
            if((i - pos_i + this->padding) > this->dim[kk][0]) continue;
            if((j - pos_j - this->padding) < 0) continue;
            if((j - pos_j + this->padding) > this->dim[kk][1]) continue;

            for(ii=0; ii<2*Nw+1; ii++)
            {
                for(jj=0; jj<2*Nw+1; jj++)
                {
                    // 1d and 2d indices for the pixel under examination
                    // for the reference and sample window:
                    px_ref_i = ii - Nw + ia - pos_i;
                    px_ref_j = jj - Nw + ja - pos_j;
                    //px_ref = px_ref_i * this->dim[kk][1] + px_ref_j;
                    px_sam_i = ii - Nw + ib - pos_i;
                    px_sam_j = jj - Nw + jb - pos_j;
                    px_sam = px_sam_i * this->dim[kk][1] + px_sam_j;
                    
                    // pixel values of window, sam, ref:
                    wij = this->win[ii*(2*Nw+1)+jj];
                    sij = this->sam[kk][px_sam];
                    blurred_ref = convolve<T>(
                        this->ref[kk], px_ref_i, px_ref_j, this->dim[kk],
                        args->kernel, this->Nk);

                    // cost function terms:
                    t1 += wij * sij         * sij;
                    t3 += wij * blurred_ref * blurred_ref;
                    t5 += wij * blurred_ref * sij;
                }
            }
        }
    }
    else
    {
        wt = 0;
        for(kk=0; kk<this->Na; kk++)
        {
            int pos_i = this->pos[kk][0];
            int pos_j = this->pos[kk][1];
            if((i - pos_i - this->padding) < 0) continue;
            if((i - pos_i + this->padding) > this->dim[kk][0]) continue;
            if((j - pos_j - this->padding) < 0) continue;
            if((j - pos_j + this->padding) > this->dim[kk][1]) continue;

            for(ii=0; ii<2*this->Nw+1; ii++)
            {
                for(jj=0; jj<2*this->Nw+1; jj++)
                {
                    // 1d and 2d indices for the pixel under examination
                    // for the reference and sample window:
                    px_ref_i = ii - Nw + ia - pos_i;
                    px_ref_j = jj - Nw + ja - pos_j;
                    px_ref = px_ref_i * this->dim[kk][1] + px_ref_j;
                    px_sam_i = ii - Nw + ib - pos_i;
                    px_sam_j = jj - Nw + jb - pos_j;
                    px_sam = px_sam_i * this->dim[kk][1] + px_sam_j;
                    
                    // pixel values of window, sam, ref, weighting:
                    wij = this->win[ii*(2*Nw+1)+jj];
                    w   = combine_weights(this->mask[kk][px_ref],
                                            this->mask[kk][px_sam]);
                    sij = this->sam[kk][px_sam];
                    blurred_ref = weighted_convolve<T>(
                        this->ref[kk], this->mask[kk], px_ref_i, px_ref_j,
                        this->dim[kk], args->kernel, this->Nk);

                    // cost function terms:
                    t1 += w * wij * sij         * sij;
                    t3 += w * wij * blurred_ref * blurred_ref;
                    t5 += w * wij * blurred_ref * sij;
                    wt += w * wij;
                }
            }
        }
    }
    // Set transmission function
    args->t = t5 / t3;

    // Set cost function value
    *out = (t1 - t5 * args->t) / wt;

    s.ok = 1;
    return s;
}

/* 
========== ModelDFKernel, cost_interface() ==========
Calculates a single cost function value.
Accessible via the cost() functions in model.pyx.
Note that the kernel coefficients have to be provided as input!

Input arguments:
----------------
    i, j : int
        Pixel coordinates in sample frame
    shift_i, shift_j : int
        Displacement between reference and sample frames
    values[2], values[3], values[4] : T, T, T
        Kernel parameters `a`, `b`, `c`. Kernel is defined as
        exp(-a*i*i - b*i*j - c*j*j).

Output arguments:
-----------------
    values[0] : T
        Cost function value
    values[1] : T
        Transmittance


Return argument:
----------------
    s : error_status instance.
*/
template <class T>
error_status ModelDFKernel<T>::cost_interface(
    int i, int j, int shift_i, int shift_j, T* values)
{
    int shift_ij[2] = {shift_i, shift_j};
    // constructs the kernel from values[2], values[3], values[4]:
    CostArgsDFKernel<T> args = CostArgsDFKernel<T>(
        i, j, values[2], values[3], values[4]);
    error_status s = this->cost(&values[0], shift_ij, &args);
    values[1] = args.t;
    return s;
}


/* 
========== ModelNoDF, min() ========== 
Performs UMPA optimisation on a single pixel.

Input arguments:
----------------
    i, j : int
        Pixel position
    values[4], values[5], values[6] : T
        Kernel parameters `a`, `b`, `c`
    db : ?

Output arguments:
-----------------
    values[0] : T
        Cost function
    values[1] : T
        Transmittance
    values[2], values[3] : T, T
        dpcx, dpcy
    uv : T[2]
        Equivalent to values[2], values[3], just used as intermediate array
    
Return argument:
----------------
    s : error_status instance
*/
template <class T>
error_status ModelDFKernel<T>::min(
    int i, int j, T* values, T uv[2], minimizer_debug<T>* db)
{
    T D;
    error_status s;
    CostArgsDFKernel<T> args = CostArgsDFKernel<T>(
        i, j, values[4], values[5], values[6]);

    s = discrete_2d_minimizer<T, ModelDFKernel, CostArgsDFKernel<T>, &ModelDFKernel::cost>(
        &D, &uv[0], this, &args, db);
    values[0] = D;
    values[1] = args.t;
    values[2] = uv[1];
    values[3] = uv[0];
    return s;
}
}
