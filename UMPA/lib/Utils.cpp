#include "Utils.h"
#include <math.h>


template <class T> wAB<T>::wAB() {}
template <class T> wAB<T>::~wAB() {}

template <class T>
wAB<T>::wAB(int Nw, T* win) :
    Nw (Nw),
    win (win)
    {}

template <class T>
T wAB<T>::sumA(T* A, int Astride, int Ai, int Aj)
{
    T out = 0;
    for(int ii=0; ii<2*this->Nw+1; ii++)
    {
        for(int jj=0; jj<2*this->Nw+1; jj++)
        {
            out += this->win[ii*(2*this->Nw+1)+jj] * A[(ii-this->Nw+Ai)*Astride+(jj-this->Nw+Aj)];
        }
    }
    return out;
}

template <class T>
T wAB<T>::sumAB(T* A, int Astride, int Ai, int Aj, T* B, int Bstride, int Bi, int Bj)
{
    T out = 0;
    for(int ii=0; ii<2*this->Nw+1; ii++)
    {
        for(int jj=0; jj<2*this->Nw+1; jj++)
        {
            out += this->win[ii*(2*this->Nw+1)+jj] * A[(ii-this->Nw+Ai)*Astride+(jj-this->Nw+Aj)] * B[(ii-this->Nw+Bi)*Bstride+(jj-this->Nw+Bj)];
        }
    }
    return out;
}

/*
Value of 2d gaussian kernel at point (i, j), with parameters (a, b, c)
-> could be replaced with table look-up, etc.
*/
template <class T>
T gaussian_kernel(int i, int j, T a, T b, T c)
{
    return exp(-a*i*i - b*i*j - c*j*j);
}

// 2d convolution at position (i, j) in image (dimensions dim[2])
// with gaussian kernel of dimensions (2*Nk+1, 2*Nk+1)
// and parameters (a, b, c).
// Absolutely no bound checking
template <class T>
T gaussian_blur(T* image, int i, int j, int dim[2], int Nk, T a, T b, T c)
{
    T out = 0.;
    T norm = 0.;
    T kernel_value;
    for(int k=-Nk; k < Nk+1; k++)     // kernel rows
    {
        for(int l=-Nk; l < Nk+1; l++) // kernel columns
        {
          kernel_value = gaussian_kernel<T>(k, l, a, b, c);
          norm += kernel_value;
          out += kernel_value * image[(i+k)*dim[1] + (j+l)];
        }
    }
    return out/norm;
}

// 2d convolution at position (i, j) in image (dimensions dim[2])
// with isotropic kernel of dimensions (2*Nk+1, 2*Nk+1) and width sigma
// Absolutely no bound checking
template <class T>
T gaussian_blur_isotropic(T* image, int i, int j, int dim[2], int Nk, T sigma)
{
    return gaussian_blur<T>(image, i, j, dim, Nk, .5/sigma/sigma, 0., .5/sigma/sigma);
}

// 2d convolution at position (i, j) in image (dimensions dim[2])
// with provided kernel of dimensions (2*Nk+1, 2*Nk+1)
template <class T>
T convolve(T* image, int i, int j, int dim[2], T* kernel, int Nk)
{
    T out = 0;
    for(int k=-Nk; k < Nk+1; k++)     // kernel rows
    {
        for(int l=-Nk; l < Nk+1; l++) // kernel columns
        {
          out += kernel[(2*Nk+1)*(k+Nk) + l+Nk] * image[(i+k)*dim[1] + (j+l)];
        }
    }
    return out;
}


// weighted 2d convolution at position (i, j) in image (dimensions dim[2])
// with provided kernel of dimensions (2*Nk+1, 2*Nk+1)
// image and weight are of the same type and dimensions.
template <class T>
T weighted_convolve(T* image, T* weight, int i, int j, int dim[2], T* kernel, int Nk)
{
    T out = 0;
    T w = 0;
    for(int k=-Nk; k < Nk+1; k++)     // kernel rows
    {
        for(int l=-Nk; l < Nk+1; l++) // kernel columns
        {
          out += kernel[(2*Nk+1)*(k+Nk) + l+Nk] * image[(i+k)*dim[1] + (j+l)] * weight[(i+k)*dim[1] + (j+l)];
          w += kernel[(2*Nk+1)*(k+Nk) + l+Nk] * weight[(i+k)*dim[1] + (j+l)];
        }
    }
    return out/w;
}

/*
Combination rule for masks. Statistical weights combine as the reciprocal of the
covariance, which is just the reciprocal of the individual variances. In the special
but common case where the mask is 0 or 1 (corresponding to variances of infinity or constant),
this reduces to a multiplication.
*/
template <class T>
static inline T combine_weights(T a, T b)
{
    return a * b / (a + b + 1e-8);
    //return a * b;
}


/*
************
COULD BE USED TO ACCELERATE EXP CALCULATIONS.
From Schraudolph, Nicol N. "A fast, compact approximation of the exponential function." Neural Computation 11.4 (1999): 853-862.
https://doi.org/10.1162/089976699300016467
************

#include <math.h>
static union
{
double d;
struct
{
#ifdef LITTLE_ENDIAN
int j, i;
#else
int i, j;
#endif
} n;
} _eco;
#define EXP_A (1048576/M_LN2)
 // use 1512775 for integer version
#define EXP_C 60801
 // see text for choice of c values
#define EXP(y) (_eco.n.i = EXP_A*(y) + (1072693248 - EXP_C), _eco.d)
*/