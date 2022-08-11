#distutils: language = c++
#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: unraisable_tracebacks=True

"""
TODO: * inconsistent data type checks in _match(): np.float64, NPDOUBLE.
        Also, everything is generalized with templates in the C++ code, why not here?
"""

cimport numpy as cnp
cimport openmp
from multiprocessing import cpu_count
import numpy as np
from cython.parallel cimport prange, parallel
from libcpp.vector cimport vector
from libc.string cimport memcpy
from libc.math cimport round

from Model cimport ModelNoDF, ModelBase, ModelDF, ModelDFKernel, CostArgsDFKernel
from Optim cimport error_status, minimizer_debug, spmin, spmin_quad
from CUtils cimport gaussian_kernel, convolve

DEF DEBUG = True

NPSINGLE = np.dtype('float32')
NPDOUBLE = np.dtype('float64')

def spm(a):
    """
    Wrapper for C++ spmin_quad: sub-pixel minimum position of 4x4 array
    based on quadratic fit.
    `a` must be a 4x4 double or single precision array.
    """
    cdef cnp.ndarray[double, ndim = 1] pos = np.zeros((2,), dtype=NPDOUBLE)
    cdef cnp.ndarray[float, ndim = 1] pos_f = np.zeros((2,), dtype=NPSINGLE)
    cdef double c
    cdef float c_f
    cdef cnp.ndarray[double, ndim = 2] a_c
    cdef cnp.ndarray[float, ndim = 2] a_c_f
    if a.shape != (4,4):
        raise RuntimeError('input array must be (4,4)')
    if a.dtype is NPDOUBLE:
        a_c = np.asarray(a, dtype=NPDOUBLE)
        c = spmin_quad[double](&a_c[0, 0], &pos[0])
        return pos, c
    elif a.dtype is NPSINGLE:
        a_c_f = np.asarray(a, dtype=NPSINGLE)
        c_f = spmin_quad[float](&a_c_f[0, 0], &pos_f[0])
        return pos_f, c_f
    else:
        raise RuntimeError('Unsupported data type')


def spmq(a):
    """
    Wrapper for C++ spmin_quad: sub-pixel minimum position of 4x4 array
    based on convolved linear interpolation model.
    `a` must be a 4x4 double or single precision array.
    """
    cdef cnp.ndarray[double, ndim = 1] pos = np.zeros((2,), dtype=NPDOUBLE)
    cdef cnp.ndarray[float, ndim = 1] pos_f = np.zeros((2,), dtype=NPSINGLE)
    cdef double c
    cdef float c_f
    cdef cnp.ndarray[double, ndim = 2] a_c
    cdef cnp.ndarray[float, ndim = 2] a_c_f
    if a.shape != (4,4):
        raise RuntimeError('input array must be (4,4)')
    if a.dtype is NPDOUBLE:
        a_c = np.asarray(a, dtype=NPDOUBLE)
        c = spmin[double](&a_c[0, 0], &pos[0])
        return pos, c
    elif a.dtype is NPSINGLE:
        a_c_f = np.asarray(a, dtype=NPSINGLE)
        c_f = spmin[float](&a_c_f[0, 0], &pos_f[0])
        return pos_f, c_f
    else:
        raise RuntimeError('Unsupported data type')

def gaussian_kernel_test(int Nk, double a, double b, double c):
    """
    Wrapper for C++ gaussian_kernel function.
    Generates a numpy array with the Gaussian kernel.
    """
    cdef cnp.ndarray[double, ndim = 2] kernel = np.empty((2*Nk+1, 2*Nk+1), dtype=NPDOUBLE)
    cdef int i, j
    for i in range(-Nk, Nk+1):
        for j in range(-Nk, Nk+1):
            kernel[i+Nk, j+Nk] = gaussian_kernel[double](i, j, a, b, c)
    return kernel

def test_convolve(image, int i, int j, kernel):
    """
    Wrapper for C++ convolve function.
    """
    cdef cnp.ndarray[double, ndim = 2] c_kernel = kernel
    cdef cnp.ndarray[double, ndim = 2] c_image = image
    cdef cnp.ndarray[int, ndim = 1] c_dim = np.array(image.shape).astype(np.int32)
    cdef int Nk = (kernel.shape[0]-1)//2
    return convolve[double](&c_image[0,0], i, j, &c_dim[0], &c_kernel[0,0], Nk)

def test_CostArgsDFKernel(int i, int j, double a, double b, double c):
    """
    Wrapper for C++ convolve function.
    """
    cdef CostArgsDFKernel[double] ca = CostArgsDFKernel[double](i, j, a, b, c)
    cdef int Nk = 8
    cdef cnp.ndarray[double, ndim = 2] kernel = np.empty((2*Nk+1, 2*Nk+1), dtype=NPDOUBLE)
    for i in range(2*Nk+1):
        for j in range(2*Nk+1):
            kernel[i, j] = ca.kernel[(2*Nk+1)*i + j]
    return kernel

cdef class UMPAModelBase:
    """
    Base class interface with C++ model classes.
    """
    # The c++ classes
    cdef ModelBase[double]* c_model

    # Keeping python objects as private attributes to avoid garbage collection
    cdef object sam_list
    cdef object ref_list
    cdef object mask_list
    cdef object pos_list
    cdef object shape_list
    cdef object window

    # Additional parameters
    cdef (int, int) dim    # Frame shape
    cdef object ROI        # Region to reconstruct
    cdef int Nparam        # Number of parameters in the model
    cdef int safe_crop     # Additional cropping
                           # (needed for kernel convolution to avoid boundaries)

    def __init__(self, sam_list, ref_list, mask_list=None, pos_list=None,
                 int window_size=2, int max_shift=4, ROI=None):
        """
        Initialize the UMPA model. This is the base class that the different
        actual models are derived from, and thus not called explicitly.

        todo: which element in pos_list is for x / y shift?
        todo: what actually happens when estimated shift reaches max_shift?
        todo: the normalization of pos_list (greater zero, minimum zero)
              should maybe be done internally?
              Also maybe give some "flip_pos" parameter(s) whether or not
              to flip the sign of the position values.

        Parameters
        ----------
        sam_list : List of K 2D numpy arrays, or a 3D numpy array (K, X, Y)
            List of speckle projections with the sample in the beam.
            Each image corresponds either to a different diffuser
            position or a different sample position. If a list of 2D
            arrays is used, the images may differ in shape.
        ref_list : List of K 2D numpy arrays, or a 3D numpy array (K, X, Y)
            List of speckle projections without a sample in the beam.
            As for `sam_list`, each image corresponds to a different
            diffuser or sample position.
            The number of projections in `ref_list` should be the same
            as in `sam_list`, and the n-th image in each list should
            correspond to the same diffuser / sample position.
            The n-th image from `sam_list` and `ref_list` must be of the
            same shape, but the n-th and (n+1)-th images may differ in shape.
        mask_list : List of K 2D numpy arrays, or a 3D numpy array (K, X, Y),
                    boolean (?), optional
            List of per-projection masks, identical in size and number
            of entries to `sam_list` and `ref_list`. It should be of
            boolean data type (?): Only pixels in `sam_list` and `ref_list`
            where `mask_list` is True will be included in the calculation
            of the cost function. This can thus be used to exclude "bad"
            image regions from calculation. If omitted, all pixel values
            of `sam_list` and `ref_list` are used.
        pos_list : List of K tuples of int or float, or (K, 2) 2d numpy array,
                   optional
            Relative lateral sample positions (in pixels) of each of the K
            images in `sam_list` and `ref_list`. This is required for
            sample-stepping measurements. In diffuser stepping measurements, the
            sample position doesn't vary and this parameter can then be omitted
            (which is equivalent to values (0, 0) for each frame).
            The values must be >= 0, and start at 0, i.e. the minimum of the
            `pos_list` across all steps must be 0 (separately for both
            dimensions).
        window_size : int, optional
            Size parameter for UMPA analysis window. A Hamming-weighted
            window of (2 * window_size + 1) x (2 * window_size + 1) pixels
            will be used. The parameter must thus be greater or equal to zero.
        max_shift : int, optional
            Maximum allowable shift (in pixels) of matching windows between
            `sam_list` and `ref_list`. This limit is understood as a magnitude,
            and separately for x and y shift, i.e. `max_shift`=4 will try to
            minimize until the magnitude of either x or y shift exceeds 4 (at
            which point the current shift estimate will be written to the DPC
            output?)
        ROI : 2d slice (i.e. tuple/list of 2 slice objects), or list/tuple of 2
              (start,stop,step) tuples.
            Slice defining a subsection of the whole field of view to
            reconstruct. For diffuser stepping, the "whole image" is equal to
            the size of the individual frames in `sam_list` and `ref_list`, and
            using the ROI here is equivalent to setting the same binary mask in
            each image of `mask_list`.
            For sample stepping however, the "whole image" is a function of the
            size of the frames, as well as the relative shifts (parameter
            `pos_list`). `ROI` is understood in this reference frame.
            For practical purposes, it is helpful to process the image once
            without defining `ROI`, and defining it by looking at the output
            images.

        
        """
        cdef int Nw = window_size
        cdef int Na = len(sam_list)
        cdef vector[int*] dim
        cdef vector[double*] refs
        cdef vector[double*] sams
        cdef vector[double*] masks
        cdef vector[int*] pos
        cdef double* win
        cdef cnp.ndarray[double, ndim = 2] a_c
        cdef cnp.ndarray[int, ndim = 1] aint_c
        cdef int padding

        # allocate shapes vector
        shape_list = [np.zeros((2,), dtype=np.int32) for _ in range(Na)]
        for i in range(Na):
            aint_c = np.asarray(shape_list[i], dtype=np.int32)
            dim.push_back(&aint_c[0])  # fdm: .push_back() is like .append() in Python
        self.shape_list = shape_list  # fdm: at this point shape_list and dim are filled with zeroes.

        # Transfer pointers from sample frame list (and dimensions)

        self._check_contiguous(sam_list)
        for k, s in enumerate(sam_list):
            a_c = np.asarray(s, dtype=NPDOUBLE)  # fdm: what does a_c mean?
            sams.push_back(&a_c[0,0])  # fdm: .push_back() is like .append() in Python
            dim[k][0] = s.shape[0]
            dim[k][1] = s.shape[1]
        # fdm: why isn't sams added here? can we not just delete `sam_list`?
        # Is it redundant?
        self.sam_list = sam_list

        # Transfer pointers for reference frames
        self._check_contiguous(ref_list)
        for k, r in enumerate(ref_list):
            a_c = np.asarray(r, dtype=NPDOUBLE)
            refs.push_back(&a_c[0,0])
            samsh = (dim[k][0], dim[k][1])
            if samsh != r.shape:
                raise RuntimeError('Incompatible shape between sample {0} and '
                                   'reference frames {1} (entry [{2}] in the '
                                   'datasets).'.format(samsh, r.shape, k))
        self.ref_list = ref_list

        # Transfer pointers of masks if they exist
        if mask_list is not None:
            self._check_contiguous(mask_list)
            for m in mask_list:
                a_c = np.asarray(m, dtype=NPDOUBLE)
                masks.push_back(&a_c[0,0])
        self.mask_list = mask_list

        # Transfer pointers of position arrays (allocate and set to [0,0] if None)
        if pos_list is None:
            pos_list = [np.zeros((2,), dtype=np.int32) for _ in range(Na)]
        else:
            pos_list = [p.astype(np.int32) for p in pos_list]
            # fdm: there appears to be no check for len(sam_list) == len(ref_list)?
            if len(pos_list) != Na:
                raise RuntimeError(
                    'Unexpected length for position list (len(pos_list)={0}, '
                    'len(sam_list)={1})'.format(len(pos_list), Na))
        if np.any(np.array(pos_list) < 0):
            raise RuntimeError(
                'Negative frame positions (entries in pos_list) are not allowed.')
        pmin = np.min(pos_list, axis=0)
        if not np.all(pmin == 0):
            raise RuntimeError('Positions should start at 0.')
        for p in pos_list:
            aint_c = np.asarray(p, dtype=np.int32)
            pos.push_back(&aint_c[0])
        self.pos_list = pos_list

        # Compute total padding
        padding = max_shift + Nw + self.safe_crop

        # Create window
        win = self._make_window(Nw)

        # Create c++ class - done in derived classes
        self.c_model = self._create_c_model(
            Na, dim, sams, refs, masks, pos, Nw, win, max_shift, padding)

        # Set ROI
        self._set_ROI(ROI)

    cdef ModelBase[double]* _create_c_model(
        self, int Na, vector[int*] dim, vector[double*] sams,
        vector[double*] refs, vector[double*] masks, vector[int*] pos, int Nw,
        double* win, int max_shift, int padding):

        raise NotImplementedError(
            'UMPAModelBase is not supposed to be called directly, use one of '
            'the subclasses "UMPAModelNoDF", "UMPAModelDF", or '
            '"UMPAModelDFKernel".')

    def __dealloc__(self):
        del self.c_model

    def _check_contiguous(self, a):
        """
        Check alignment and type of array list.
        """
        if any(not x.flags.c_contiguous for x in a):
            raise RuntimeError('The provided image frames are not C-contiguous.')
        return

    def min(self, int x, int y):
        """
        Minimise model at pixel (x, y) and return model-specific parameters at the optimum.
        """
        raise NotImplementedError

    cdef void _min(self, int i, int j, double* values):
        """
        Minimise model at pixel (i, j) and return values
        """
        cdef minimizer_debug[double] db;  # fdm: I don't think you need semicolons here
        cdef error_status s;
        s = self.c_model.min(i, j, values, &db)
        return

    def _match(self, step=None, input_values=None, dxdy=None, ROI=None,
               num_threads=None, quiet=False):
        """
        Generic match routine wrapped by the match method of sub-classes.
        step is ignored if ROI is not None.

        Parameters
        ----------
        step : int, optional
            "Stride" to use for UMPA, process every step-th pixel
            (in both dimensions). Ignored if ROI is not None.
        input_values : array of shape (N0, N1, Nparams)
            Input values for the UMPA fit parameters. Only useful if the
            `DFKernel` model is used, allows providing kernel parameters.
            N0, N1 = size of the image after padding and applying ROI, i.e. size
                     of UMPA output.
            For DFKernel, Nparams = 7.
        dxdy : array of shape (N0, N1, 2)
            Starting values for the UMPA minimization.
            N0, N1 = size of the image after padding and applying ROI, i.e. size
                     of UMPA output.
        ROI : 2d slice (tuple of 2 slices) or list/tuple of two
              (start, stop, step) tuples.
        num_threads : int > 0
            Number of CPU threads to use.
        quiet : bool, optional
            If True, disable print output about the number of used kernels.
        """
        cdef int start0, start1, end0, end1, step0, step1, N0, N1, offset, nthreads
        cdef Py_ssize_t xi, xj
        cdef cnp.ndarray[double, ndim = 3] values
        cdef cnp.ndarray[double, ndim = 3] uv
        cdef cnp.ndarray[double, ndim = 2] covermap
        cdef cnp.ndarray[int, ndim = 2] err
        cdef minimizer_debug[double]* db
        cdef error_status error
        cdef double cover_threshold

        if (ROI is not None) and (step is not None):
            print("Warning: 'ROI' and 'step' parameters are set simultaneously. "
                  "'step' parameter is ignored.")
            step = None

        # set number of threads to use:
        nthreads_default = cpu_count() // 2
        if num_threads is None:
            num_threads = nthreads_default
        try:
            nthreads = num_threads
        except:
            nthreads = nthreads_default
            if not quiet:
                print(f"Can't interpret '{num_threads}' as an integer, "
                      f"using default number of threads ({nthreads_default})")
        else:
            if num_threads < 1:
                if not quiet:
                    print("num_threads < 1 invalid, "
                        f"using default number of threads ({nthreads_default})")
                nthreads = nthreads_default
            else:
                if not quiet:
                    print(f"Using {nthreads} threads")

        # Generates (start, stop, step) tuples `s0` and `s1`, either from the
        # `ROI` or `step` parameters. If both are set, `step` is ignored.
        s0, s1 = self._convert_ROI_slice(ROI, step)
        
        # creates `self.ROI` from s0 and s1, has the shape
        # [(start, stop, step), (start, stop, step)].
        # Note that `_set_ROI()` is already run once in the constructor, with
        # `ROI` as the argument. I.e., it ignores the `step` parameter then.
        self._set_ROI((s0, s1))

        # Extract iteration ranges
        start0, end0, step0 = s0
        start1, end1, step1 = s1

        # The final images will have this size
        # (no. of elements in the range start:stop:step):
        N0 = 1 + (end0 - start0 - 1) // step0
        N1 = 1 + (end1 - start1 - 1) // step1
        sh = (N0, N1)

        # shape of `values`. When done in this order (`Nparam` at the end), all
        # parameters belonging to the same pixel are contiguous in memory.
        # This is used in the call to self.c_model.min() below
        # (`&values[xi, xj, 0]` is the address of the 0th parameter, and is
        # interpreted in the C++ code as the start of a 1d array of
        # length `Nparam`.) The same logic applies to `&uv[xi, xj, 0]`.
        shp = (N0, N1, self.Nparam)

        # Compute coverage
        covermap = self.coverage(ROI=((start0, end0, step0),
                                      (start1, end1, step1)))

        # fdm: what is the logic for this threshold?
        cover_threshold = .1 * covermap.max() / len(self.sam_list)  

        # Manage possible input parameters.
        # For the NoDF and DF models, `input_values` is irrelevant, because
        # `values` is never used as an input parameter by `c_model.min()`
        # in these cases, thus it's always fine to initialize `values` with zeroes.
        # In the DFKernel model however, 3 of the 7 parameters are used for
        # parameterizing the kernel, and the kernel is actually constructed from
        # these when `c_model.min()` is called.
        # I.e., the `input_values` variable is the interface to pass a guess of
        # kernel parameters to UMPA.
        if input_values is not None:
            if input_values.shape != shp:
                raise RuntimeError(
                    "Input values have the wrong shape: "
                    "%s, should be %s" % (input_values.shape, shp))
            
            if input_values.dtype != np.float64:
                raise RuntimeError(
                    "Input values have the wrong type: "
                    "%s, should be %s" % (input_values.dtype, np.float64))
            values = input_values
        
        else:
            values = np.zeros(shp, dtype=NPDOUBLE)

        offset = self.c_model.padding

        # Initial shifts in x and y: initialize with zeroes.
        # see also the comment at the "shp = ..." line
        shuv = (N0, N1, 2)
        uv = np.zeros(shuv, dtype=NPDOUBLE)
        if dxdy is not None:
            uv[:,:,0] = dxdy[0]
            uv[:,:,1] = dxdy[1]

        # Allocate and initialise output arrays
        err = np.zeros(sh, dtype=np.int32)
        cvr = np.zeros(sh, dtype=NPDOUBLE)

        IF DEBUG:
            cdef cnp.ndarray[double, ndim = 3] debug_d = np.zeros(sh + (25,), dtype=NPDOUBLE)
            cdef cnp.ndarray[double, ndim = 3] debug_a = np.zeros(sh + (16,), dtype=NPDOUBLE)
            cdef cnp.ndarray[int, ndim = 2] debug_Ncalls = np.zeros(sh, dtype=np.int32)

        with nogil, parallel(num_threads=nthreads):
            db = new minimizer_debug[double]()
            for xi in prange(N0, schedule='dynamic'):
                for xj in range(N1):
                    if covermap[xi,xj] < cover_threshold:
                        continue
                    error = self.c_model.min(offset + start0 + step0*xi,
                                             offset + start1 + step1*xj,
                                             &values[xi, xj, 0],
                                             &uv[xi, xj, 0],
                                             db)
                    err[xi, xj] = error.ok
                    IF DEBUG:
                        memcpy(&debug_d[xi, xj, 0], db.d, 25*sizeof(double))
                        memcpy(&debug_a[xi, xj, 0], db.a, 16*sizeof(double))
                        debug_Ncalls[xi, xj] = db.Ncalls
            del db
        IF DEBUG:
            return {'values': values, 'err': err, 'debug_d': debug_d,
                    'debug_a': debug_a, 'debug_Ncalls': debug_Ncalls}
        ELSE:
            return {'values': values, 'err': err}

    def coverage(self, step=None, ROI=None):
        """
        Return a coverage map, i.e. the amount of frame overlap.
        step is ignored if ROI is not None, in which case it is expected to be a
        2-tuple of python slices.
        """
        cdef int start0, end0, step0, N0, start1, end1, step1, N1, offset
        cdef Py_ssize_t xi, xj
        cdef error_status error
        cdef cnp.ndarray[double, ndim = 2] map

        s0, s1 = self._convert_ROI_slice(ROI, step)

        # Extract iteration ranges
        start0, end0, step0 = s0
        start1, end1, step1 = s1

        # The final images will have this size
        N0 = 1 + (end0 - start0 - 1) // step0
        N1 = 1 + (end1 - start1 - 1) // step1
        sh = (N0, N1)

        offset = self.c_model.padding

        map = np.zeros(sh, dtype=NPDOUBLE)
        for xi in range(N0):
            for xj in range(N1):
                error = self.c_model.coverage(&map[xi, xj],
                                              offset + start0 + step0 * xi,
                                              offset + start1 + step1 * xj)
        return map

    def _calculate_extent(self):
        """
        Compute the full extent of the reconstructible region.
        fdm: circumscribes rectangle around all frames, subtracts padding.
        fdm: might be interesting to have this available with parameters to use
             manually.
        """
        cdef int start, N0, N1

        padding = self.c_model.padding
        pmax = np.max(np.array(self.pos_list) + np.array(self.shape_list),
                      axis=0)

        # fdm: why is 1 added and subtracted?
        #      Seems easier / more obvious to me without that...
        N0 = 1 + (pmax[0] - 2 * padding - 1)
        N1 = 1 + (pmax[1] - 2 * padding - 1)

        return N0, N1

    def _convert_ROI_slice(self, ROI=None, step=None):
        """
        Convert ROI (a tuple of slice objects) to actual range.
        (fdm: a (start, stop, step) tuple for each dimension)
        * step and ROI cannot be defined simultaneously
          (fdm: should we get rid of the "step" parameter in `match()`?)
        * This method DOES NOT set self.ROI.
          (fdm: That is done by `_set_ROI()`, which is called in `match()`.)
        """

        # Total extent allowed by padding, positions and frame shapes
        N0, N1 = self._calculate_extent()

        if ROI is not None:
            # Use provided ROI
            if step is not None:
                raise RuntimeError(
                    'Step and ROI should not be specified simultaneously.')
            s0, s1 = ROI
            if type(s0) is slice:
                s0 = s0.indices(N0)
            if type(s1) is slice:
                s1 = s1.indices(N1)
        else:
            # Use existing ROI
            s0, s1 = self.ROI
            # Use step with existing ROI extent
            if step is not None:
                s0 = slice(s0[0], s0[1], step).indices(N0)
                s1 = slice(s1[0], s1[1], step).indices(N1)

        return s0, s1


    def test(self):
        return self.c_model.test()

    def coords(self, ROI=None):
        """
        Generate coordinates for the given ROI (or self.ROI) based on the
        natural extent of the data
        """
        offset = self.padding
        if ROI is not None:
            s0, s1 = self._convert_ROI_slice(self, ROI=ROI)
        else:
            s0, s1 = self.ROI

        return offset + np.arange(*s0), offset + np.arange(*s1)

    def _set_ROI(self, ROI=None):
        """
        Set reconstruction Region-of-interest. Full field of view if ROI is None
        """
        # Compute maximum allowed extent
        N0, N1 = self._calculate_extent()

        if ROI is None:
            self.ROI = ((0, N0, 1), (0, N1, 1))
        else:
            s0, s1 = ROI
            if type(s0) is slice:
                s0 = s0.indices(N0)
            if type(s1) is slice:
                s1 = s1.indices(N1)
            self.ROI = (s0, s1)

    def set_step(self, step):
        """
        Adjust ROI slices so that they have given step.
        """
        self._set_ROI(ROI=self._convert_ROI_slice(step=step))
        return self.ROI

    @property
    def extent(self):
        return self._calculate_extent()

    @property
    def ROI(self):
        """
        Reconstruction ROI.
        """
        return self.ROI

    @ROI.setter
    def ROI(self, new_ROI):
        self._set_ROI(new_ROI)

    @property
    def sh(self):
        """
        Reconstruction array shape
        """
        s0, s1 = self.ROI
        return ((s0[1]-s0[0]-1)//s0[2] + 1, (s1[1]-s1[0]-1)//s1[2] + 1)

    @property
    def Na(self):
        return self.c_model.Na

    @property
    def sam_list(self):
        """
        List of sample frames.
        """
        return self.sam_list

    @property
    def ref_list(self):
        """
        List of reference frames.
        """
        return self.ref_list

    @property
    def mask_list(self):
        """
        List of mask frames.
        """
        return self.mask_list

    @property
    def shape_list(self):
        """
        List of frame shapes.
        """
        return self.shape_list

    @property
    def pos_list(self):
        """
        List of positions.
        """
        return self.pos_list

    @property
    def window(self):
        return self.window

    cdef double* _make_window(self, int n):
        window = np.multiply.outer(np.hamming(2*n+1), np.hamming(2*n+1))
        window /= window.sum()
        self.window = window
        cdef cnp.ndarray[double, ndim = 2] a_c = np.asarray(self.window, dtype=NPDOUBLE)
        return &a_c[0,0]

    @property
    def Nw(self):
        """ Window size """
        return self.c_model.Nw
    @Nw.setter
    def Nw(self, int new_Nw):
        self.c_model.set_window(self._make_window(new_Nw), new_Nw)

    @property
    def max_shift(self):
        """Maximum shift"""
        return self.c_model.max_shift

    @property
    def padding(self):
        """ Frame padding """
        return self.c_model.padding

    @property
    def assign_coordinates(self):
        """
        Method for pixel assignment in the model.
        If 'ref' (default), UMPA output is assigned to the center of the
        analysis window in the reference image(s).
        If 'sam', it is assigned to the center of the window in the sample
        images.
        'sam' makes more sense physically, but produces greater artifacts in
        regions of strong edges or propagation fringes.
        """
        opts = {0: 'sam',
                1: 'ref'}
        
        return opts[self.c_model.reference_shift]

    @assign_coordinates.setter
    def assign_coordinates(self, new_mode):
        opts = {'sam': 0,
                'ref': 1}
        try:
            set_value = opts[new_mode]
        except KeyError:
            print('Option %s is not available, parameter was not changed.'
                  % repr(new_mode))
        else:
            self.c_model.reference_shift = set_value

    @property 
    def sub_pixel_mode(self):
        """
        Sub_pixel optimization mode in the model. By default -1, uses spmin().
        If 0, turns sub-pixel minimization off.
        If 1, uses spmin_quad() a quadratic fit to find the subpixel minimum.
        """
        return self.c_model.subpx_func

    @sub_pixel_mode.setter
    def sub_pixel_mode(self, new_mode):
        self.c_model.subpx_func = int(new_mode)


cdef class UMPAModelNoDF(UMPAModelBase):

    def __cinit__(self, *args, **kwargs):
        self.Nparam = 4  # f, dx, dy, T (?)
        self.safe_crop = 0

    cdef ModelBase[double]* _create_c_model(
        self, int Na, vector[int*] dim, vector[double*] sams,
        vector[double*] refs, vector[double*] masks, vector[int*] pos, int Nw,
        double* win, int max_shift, int padding):

        return new ModelNoDF[double](
            Na, dim, sams, refs, masks, pos, Nw, win, max_shift, padding)

    def min(self, int i, int j):
        """
        Minimisation at point (i, j)
        """
        cdef cnp.ndarray[double, ndim = 1] values = np.empty((self.Nparam,), dtype=NPDOUBLE)
        self._min(i, j, &values[0])
        return values

    def cost(self, int i, int j, double sx, double sy):
        """
        Returns cost function value and transmittance at point (`i`, `j`),
        assuming shift `sx` and `sy` (rounded to integer values)
        """
        cdef error_status error
        cdef double values[2]
        error = self.c_model.cost_interface(
            i, j, <int>round(sx), <int>round(sy), &values[0])
        return (values[0], values[1])


    def match(self, step=None, dxdy=None, ROI=None, num_threads=None,
              quiet=False):
        """
        Perform minimisation on the whole image or the given region of interest.
        step is ignored if ROI is not None.
        
        Parameters
        ----------
        
        step : int, optional
            "Stride" to use for UMPA, process every step-th pixel
            (in both dimensions). Ignored if ROI is not `None`.
        dxdy : ???
            Starting values for the UMPA minimization.
        ROI : 2d slice (tuple of 2 slices) or list/tuple of two
              (start, stop, step) tuples.
        num_threads : int
            Number of threads to use (OpenMP)
        """
        # Run base-class match
        result = self._match(step=step, dxdy=dxdy, ROI=ROI,
                             num_threads=num_threads, quiet=quiet)

        # Unpack the values
        values = result.pop('values')
        result['f'] = values[:,:,0].copy()
        result['T'] = values[:,:,1].copy()
        result['dx'] = values[:,:,2].copy()
        result['dy'] = values[:,:,3].copy()

        return result

cdef class UMPAModelDF(UMPAModelBase):

    def __cinit__(self, *args, **kwargs):
        self.Nparam = 5  # f, dx, dy, T, df (?)
        self.safe_crop = 0

    cdef ModelBase[double]* _create_c_model(
        self, int Na, vector[int*] dim, vector[double*] sams,
        vector[double*] refs, vector[double*] masks, vector[int*] pos, int Nw,
        double* win, int max_shift, int padding):

        return new ModelDF[double](
            Na, dim, sams, refs, masks, pos, Nw, win, max_shift, padding)

    def min(self, int i, int j):
        """
        Minimisation at point (i, j)
        """
        cdef cnp.ndarray[double, ndim = 1] values = np.empty((self.Nparam,), dtype=NPDOUBLE)
        self._min(i, j, &values[0])
        return values

    def cost(self, int i, int j, double sx, double sy):
        """
        Cost function at point (i, j) assuming shift sx and sy.
        """
        cdef error_status error
        cdef double values[3]
        
        error = self.c_model.cost_interface(
            i, j, <int>round(sx), <int>round(sy), &values[0])

        return (values[0], values[1], values[2])

    def match(self, step=None, dxdy=None, ROI=None, num_threads=None,
              quiet=False):
        """
        Perform minimisation on the whole image or the given region of interest.
        step is ignored if ROI is not None.
        
        Parameters
        ----------
        
        step : int, optional
            "Stride" to use for UMPA, process every step-th pixel
            (in both dimensions). Ignored if ROI is not `None`.
        dxdy : ???
            Starting values for the UMPA minimization.
        ROI : 2d slice (tuple of 2 slices) or list/tuple of two
              (start, stop, step) tuples.
        num_threads : int
            Number of threads to use (OpenMP)
        """
        # Run base-class match
        result = self._match(step=step, dxdy=dxdy, ROI=ROI,
                             num_threads=num_threads, quiet=quiet)

        # Unpack the values
        values = result.pop('values')
        result['f'] = values[:,:,0].copy()
        result['T'] = values[:,:,1].copy()
        result['dx'] = values[:,:,2].copy()
        result['dy'] = values[:,:,3].copy()
        result['df'] = values[:,:,4].copy()

        return result
    
    @property
    def Im(self):
        """
        Mean ref frame amplitude
        """
        return (<ModelDF[double]*> (self.c_model)).Im


cdef class UMPAModelDFKernel(UMPAModelBase):

    def __cinit__(self, *args, **kwargs):
        self.Nparam = 7  # f, dx, dy, T, a, b, c
        # Would be nicer to have this dynamically from the C++ class but ok for now
        self.safe_crop = 8

    cdef ModelBase[double]* _create_c_model(
        self, int Na, vector[int*] dim, vector[double*] sams,
        vector[double*] refs, vector[double*] masks, vector[int*] pos, int Nw,
        double* win, int max_shift, int padding):

        return new ModelDFKernel[double](
            Na, dim, sams, refs, masks, pos, Nw, win, max_shift, padding)

    def min(self, int i, int j, double a, double b, double c):
        """
        Minimisation at point (i, j) with gaussian parameters a,b,c
        """
        cdef cnp.ndarray[double, ndim = 1] values = np.empty((self.Nparam,), dtype=NPDOUBLE)
        values[4] = a
        values[5] = b
        values[6] = c
        self._min(i, j, &values[0])
        return values

    def cost(self, int i, int j, double sx, double sy,
             double a, double b, double c):
        """
        Cost function at point (i, j) assuming shift sx and sy,
        and kernel parameters a, b, c
        """
        cdef error_status error
        cdef double values[5]
        values[2] = a
        values[3] = b
        values[4] = c

        error = self.c_model.cost_interface(
            i, j, <int>round(sx), <int>round(sy), &values[0])

        return (values[0], values[1])

    def match(self, step=None, abc=None, dxdy=None, ROI=None, num_threads=None,
              quiet=False):
        """
        Perform minimisation on the whole image or the given region of interest.
        step is ignored if ROI is not None.
        
        Parameters
        ----------
        
        step : int, optional
            "Stride" to use for UMPA, process every step-th pixel
            (in both dimensions). Ignored if ROI is not `None`.
        dxdy : ???
            Starting values for the UMPA minimization.
        ROI : 2d slice (tuple of 2 slices) or list/tuple of two
              (start, stop, step) tuples.
        num_threads : int
            Number of threads to use (OpenMP)
        """
        s0, s1 = self._convert_ROI_slice(ROI, step)
        self._set_ROI((s0, s1))

        # Extract iteration ranges
        start0, end0, step0 = s0
        start1, end1, step1 = s1

        # The final images will have this size
        N0 = 1 + (end0-start0-1)//step0
        N1 = 1 + (end1-start1-1)//step1
        sh = (N0, N1)

        if abc is None:
            raise RuntimeError('abc array has to be provided')
        elif abc.shape != sh + (3,):
            raise RuntimeError('Wrong array shape for abc: %s, should be %s'
                               % (abc.shape, sh + (3,)))

        # Allocate output array
        shp = sh + (self.Nparam,)
        values = np.zeros(shp, dtype=np.float64)

        # Assign sigma to last slice as input argument
        values[:,:,-3:] = abc

        # Run base-class match
        result = self._match(step=step, input_values=values, dxdy=dxdy, ROI=ROI,
                             num_threads=num_threads, quiet=quiet)

        # Unpack the values
        values = result.pop('values')
        result['f'] = values[:,:,0].copy()
        result['T'] = values[:,:,1].copy()
        result['dx'] = values[:,:,2].copy()
        result['dy'] = values[:,:,3].copy()

        return result
