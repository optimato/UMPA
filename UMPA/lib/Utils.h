#ifndef UTILS_H
#define UTILS_H

/*
For lack of a better name, wAB represent the operation of summing over the window and two arrays (w * A * B),
with a given offset in A and B.
*/
template <class T>
class wAB {
    public:
        int Nw;                // Window size
        T* win;                // Window

        wAB();
        wAB(int Nw, T* win);
        ~wAB();
        T sumA(T* A, int Astride, int Ai, int Aj);
        T sumAB(T* A, int Astride, int Ai, int Aj, T* B, int Bstride, int Bi, int Bj);
};


template <class T> T gaussian_kernel(int i, int j, T a, T b, T c);
template <class T> T gaussian_blur(T* image, int i, int j, int dim[2], int Nk, T a, T b, T c);
template <class T> T gaussian_blur_isotropic(T* image, int i, int j, int dim[2], int Nk, T sigma);
template <class T> T convolve(T* image, int i, int j, int dim[2], T* kernel, int Nk);
template <class T> T weighted_convolve(T* image, T* weight, int i, int j, int dim[2], T* kernel, int Nk);
template <class T> static inline T combine_weights(T a, T b);


#endif