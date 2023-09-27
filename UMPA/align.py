import numpy as np
import scipy.fft as fft
import scipy.ndimage as ndi
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import multiprocessing as mp
import joblib as job
import UMPA

cpu = mp.cpu_count()

def UMPA_normal(sams, refs, window=1, shift=3, pos_list=None, mask_list=None, assign_coordinates='sam', num_threads=None, ROI=(slice(None,None,None), slice(None,None,None))):
    """
    Help function for UMPA without bias correction,
    with bad pixel correction.
    
    Parameters
    ----------
    sams, refs: 2D-array, list
        Input sample and reference images. 
    window: int, optional
        Analysis window size: 2*window+1 is used in UMPA.
    shift: int, optional
        Maximum shift for dpc signal.
    pos_list: list, optional
        List of motor positions for sample stepping.
    mask_list: 2D-array, list, optional
        List of masks for each sample and reference images.
    assign_coordinates: string, optional
        If 'sam' matches the sample to the reference, else 'ref' matches
        the reference on the sample model (would make more sense, but gives
        worse results, fringes in the way).
    num_threads: int, optional
        Number of threads to use.
    ROI: slice, optional
        Vertical and horizontal slicing for the pixel selection passed to UMPA.
        
    Returns
    ----------
    res: dictionary
        Output of UMPA ('T', 'dx', 'dy', 'df', 'pc', ...).
        Bad pixel correction (thereshold=shift) applied on DPC before 
        integration. 
    
    """

    if type(pos_list)!=type(None):
        if type(mask_list)!=type(None):
            model = UMPA.model.UMPAModelDF(sams, refs, window_size=window, max_shift=shift, pos_list=pos_list, mask_list=mask_list)
        else:
            model = UMPA.model.UMPAModelDF(sams, refs, window_size=window, max_shift=shift, pos_list=pos_list)
    else:
        if type(mask_list)!=type(None):
            model = UMPA.model.UMPAModelDF(sams, refs, window_size=window, max_shift=shift, mask_list=mask_list)
        else:
            model = UMPA.model.UMPAModelDF(sams, refs, window_size=window, max_shift=shift)
    model.assign_coordinates = assign_coordinates
    res = model.match(num_threads=num_threads, ROI=ROI, quiet=True)
    res['dx'] = correct_bad_pixels(res['dx'], shift)
    res['dy'] = correct_bad_pixels(res['dy'], shift)
    return res

def UMPA_nobias(sams, refs, window=1, shift=3, pos_list=None, mask_list=None, assign_coordinates='sam', num_threads=None, ROI=(slice(None,None,None), slice(None,None,None))):
    """
    Help function for UMPA with bias correction and
    with bad pixel correction.
    
    Parameters
    ----------
    sams, refs: 2D-array, list
        Input sample and reference images. 
    window: int, optional
        Analysis window size: 2*window+1 is used in UMPA.
    shift: int, optional
        Maximum shift for dpc signal.
    pos_list: list, optional
        List of motor positions for sample stepping.
    mask_list: 2D-array, list, optional
        List of masks for each sample and reference images.
    assign_coordinates: string, optional
        If 'sam' matches the sample to the reference, else 'ref' matches
        the reference on the sample model (would make more sense, but gives
        worse results, fringes in the way).
    num_threads: int, optional
        Number of threads to use.
    ROI: slice, optional
        Vertical and horizontal slicing for the pixel selection passed to UMPA.
        
    Returns
    ----------
    res: dictionary
        Output of UMPA ('T', 'dx', 'dy', 'df', 'pc', ...).
        Bad pixel correction (thereshold=shift) applied on DPC before 
        integration. 
    
    """

    if type(pos_list)!=type(None):
        if type(mask_list)!=type(None):
            model = UMPA.model.UMPAModelDF(sams, refs, window_size=window, max_shift=shift, pos_list=pos_list, mask_list=mask_list)
            bias = UMPA.model.UMPAModelDF(refs, refs, window_size=window, max_shift=shift, pos_list=pos_list, mask_list=mask_list)
        else:
            model = UMPA.model.UMPAModelDF(sams, refs, window_size=window, max_shift=shift, pos_list=pos_list)
            bias = UMPA.model.UMPAModelDF(refs, refs, window_size=window, max_shift=shift, pos_list=pos_list)
    else:
        if type(mask_list)!=type(None):
            model = UMPA.model.UMPAModelDF(sams, refs, window_size=window, max_shift=shift, mask_list=mask_list)
            bias = UMPA.model.UMPAModelDF(refs, refs, window_size=window, max_shift=shift, mask_list=mask_list)
        else:
            model = UMPA.model.UMPAModelDF(sams, refs, window_size=window, max_shift=shift)
            bias = UMPA.model.UMPAModelDF(refs, refs, window_size=window, max_shift=shift)
    model.assign_coordinates = assign_coordinates
    res = model.match(num_threads=num_threads, ROI=ROI, quiet=True)
    res_b = bias.match(num_threads=num_threads, ROI=ROI, quiet=True)
    res['dx'] = correct_bad_pixels(res['dx']-res_b['dx'], shift)
    res['dy'] = correct_bad_pixels(res['dy']-res_b['dy'], shift)
    return res

def shift_best(a, b, w=None, max_shift=None, return_params=True, numiter=1, also_return_null_shift = False, scale_coeff=True, warn=False):
    """
    Shifts and rescales ``b`` so that it best overlaps with ``a``.
    
    .. note::
        See :py:func:`shift_dist` for more documentation.

        If no improved minimum is found, a Null vector is return instead of ``b``.
    
    Parameters
    ----------
    a : 2D-numpy-array
        Array to compare with ``b``.
    
    b : 2D-numpy-array
        Array to compare with ``a``. 
        
    w : ndarray, Defualt=None, optional
        ``w`` can also be a tuple of two arrays (wa, wb) which are used to mask
        data from both ``a`` and ``b``.
    
    max_shift : int, Default=None, optional
        The maximum allowed shift distance (in pixel units). 
    
    return_params : bool, Default=True, optional
        If True, returns only the shifted version of b.
    
    numiter : int, Default=1, optional
        Number of used iterations.
    
    also_return_null_shift : 
        Will enforce that a vector is returned, even if no minimum was found.
        ``b`` is then unchanged.
    
    scale_coeff : bool, Default=Ture, optional
        Allows only for a phase factor.
        
    warn : bool, Default=False, optional
        If Ture, return warnings.
        
    Returns
    --------
    b : 2D-numpy-array
        Similar to a regular cross correlation.
        
    r0 : numpy-array, optional
        Translation vector.
        
    alpha0 : 2D-numpy-array
        Complex coefficient (:math:`\\alpha`) for minimizing.
"""
    
    QUAD_MAX_HW = 1
    QUAD_MAX_W = 2*QUAD_MAX_HW + 1

    sh = a.shape
    assert b.shape == sh

    ismatrix = isinstance(b,np.matrix)
    if ismatrix:
        a = np.asarray(a)
        b = np.asarray(b)

    ndim = a.ndim
    fake_1D = False
    if ndim==2 and 1 in sh:
        fake_1D = True
        a = a.ravel()
        b = b.ravel()
        sh_1D = sh
        sh = a.shape
        ndim = 1

    r0 = np.zeros((ndim,))

    qmaxslice = tuple([slice(0,QUAD_MAX_W) for dummy in range(ndim)])

    alpha0 = 1.
    for ii in range(numiter):
        # Compute distance
        cc = shift_dist(a,b,w,scale_coeff=scale_coeff)[0]

        if max_shift is not None:
            # Find minimum
            cc1 = cc.copy()
            if np.isscalar(max_shift):
                too_far_mask = (fvec2(sh) > max_shift**2)
            else:
                fg = fgrid(sh)
                too_far_mask = np.zeros(sh,dtype=bool)
                for idim in ndim:
                    too_far_mask += (abs(fg[idim]) > max_shift[idim]/2.)
            cc1[too_far_mask] = np.inf
            cmax = np.array(np.unravel_index(cc1.argmin(),sh))
            # sub-pixel center
            #cc_part = pshift(-np.real(cc1), cmax - QUAD_MAX_HW)[0:QUAD_MAX_W,0:QUAD_MAX_W]
            cc_part = pshift(-np.real(cc1), cmax - QUAD_MAX_HW)[qmaxslice]
            if np.any(np.isinf(cc_part)):
                warnings.warn("Failed to find a local minimum in shift_best.", RuntimeWarning)
                if also_return_null_shift is False:
                    return None
        else:
            # Find minimum
            cmax = np.array(np.unravel_index(cc.argmin(),sh))

        # sub-pixel center
        ccloc = pshift(-np.real(cc), cmax - QUAD_MAX_HW)[qmaxslice]
        mindist, r = quad_max(ccloc, warn=warn)
        if mindist is None:
            # mindist is None if the quadratic fit did give a local minimum.
            # Poor man's solution: re-run in 1D.
            mindist_d0, r_d0 = quad_max(ccloc[:,1], warn=warn)
            mindist_d1, r_d1 = quad_max(ccloc[1,:], warn=warn)
            if (mindist_d0 is None) or (mindist_d1 is None):
                # This should never happen if we are minimizing on a 3x3 array.
                raise RuntimeError('Could not find a local minimum!')
            else:
                mindist = min(mindist_d0, mindist_d1)
                r = np.array([r_d0,r_d1])
        r -= (QUAD_MAX_HW - cmax)

        # shift the array
        bshift = pshift(b,-r)

        # evaluate the coefficient for sub-pixel shift
        alpha = (a*np.conj(bshift)).sum()/norm2(bshift)
        if not scale_coeff:
            alpha = np.exp(1j * np.angle(alpha))

        # New b
        b = alpha*bshift
        alpha0 *= alpha
        r0 += r

    if fake_1D:
        b.resize(sh_1D)
    if ismatrix:
        b = np.asmatrix(b)

    # Store the optimum as an attribute in case it is useful
    shift_best.mindist = -mindist if mindist is not None else np.real(cc).min()

    if return_params:
        return b, -r0, alpha0
    else:
        return b
    
def quad_max(a, mask=None, return_hessian=False, warn=True):
    """
    Fits a parabola (or paraboloid) to ``a``.
    
    .. note::
        Fit model: y = c + (x-x0)' * H * (x-x0)\n
        where x is in pixel units.
        
        All entries are None upon failure. Failure occurs if:
            * A has a positive curvature (it then has a minimum, not a maximum).
            * A has a saddle point
            * the hessian of the fit is singular, that is A is (nearly) flat.
    
    Parameters
    ----------
    a : 1D or 2D-numpy-array
        Array with values for fitting.
    
    mask : bool, Defualt=None, optional
        Uses in the fit only the elements of ``a`` that have a True mask value (None means use all array). Same shape as ``a``. 
        
    return_hessian : bool, Defualt=False, optional
        If Ture, return also the hessian matrix.
    
    warn : bool, Default=Ture, optional
        Print out warnings.
        
    Returns
    --------
    c : float
        The value at the fitted optimum. f(x0)
        
    x0 : float
        The position of the optimum.
        
    h : nd-array, optional
        The hessian matrix (curvature in 1D)
    """        
    (c,x0,h) = quad_fit(a,mask)

    failed = False
    if a.ndim == 1:
        if h > 0:
            if warn: print('Warning: positive curvature!')
            failed = True
    else:
        if h[0,0] > 0:
            if warn: print('Warning: positive curvature along first axis!')
            failed = True
        elif h[1,1] > 0:
            if warn: print('Warning: positive curvature along second axis!')
            failed = True
        elif np.linalg.det(h) < 0:
            if warn: print('Warning: the provided data fits to a saddle!')
            failed = True

    if failed:
        c = None
    
    if return_hessian:
        return c,x0,h
    else:
        return c,x0
    
def quad_fit(a, mask=None,return_error=False):
    """
    Fits a parabola (or paraboloid) to ``a``.
    
    .. note::
        Fit model: y = c + (x-x0)' * H * (x-x0)\n
        where x is in pixel units. 
    
    Parameters
    ----------
    a : 1D or 2D-numpy-array
        Array with values for fitting.
    
    mask : bool, Defualt=None, optional
        Uses in the fit only the elements of ``a`` that have a True mask value (None means use all array). Same shape as ``a``.
        
    return_error : bool, Defualt=False, optional
        If Ture, return also the errors.
    
    Returns
    --------
    c : float
        The value at the fitted optimum. f(x0)
        
    x0 : float
        The position of the optimum.
        
    h : nd-array
        The hessian matrix (curvature in 1D)
        
    dc : float, optional
        Value error.
        
    dx0 : float, optional
        Position error.
        
    dh : nd-array, optional
        Hessian matrix error.
    """

    sh = a.shape

    if a.ndim == 1:
        # 1D fit 
        x = np.arange(len(a))
        if mask is not None:
            x = x[mask]
            a = a[mask]
    
        # Model: y = p(1) + p(2) x + p(3) x^2
        #          = c + h (x - x0)^2
        A = np.vstack([np.ones_like(x), x, x**2]).T
        r = np.linalg.lstsq(A, a, rcond=None)
        p = r[0]
        c = p[0] - .25*p[1]**2/p[2]
        x0 = -.5*p[1]/p[2]
        h = p[2]
        if not return_error:
            return (c,x0,h)

        mA = np.matrix(A)
        dp = np.sqrt(np.diag(np.linalg.pinv(mA.T*mA) * r[1][0]/2))
        dp2 = dp**2
        dc = np.sqrt( dp2[0] + dp2[1] * .25 * (p[1]/p[2])**2 + dp2[2] * .0625 * (p[1]/p[2])**4 )
        dx0 = np.sqrt( dp2[1] * .25 * (1/p[2])**2 + dp2[2] * .25 * p[1]/p[2]**2 )
        dh = dp[2]
    
    elif a.ndim == 2:
    
        # 2D fit
        i0, i1 = np.indices(sh)
        i0f = i0.flatten()
        i1f = i1.flatten()
        af = a.flatten()
    
        if mask is not None:
            mf = mask.flatten()
            i0f = i0f[mf]
            i1f = i1f[mf]
            af = af[mf]
        
        # Model = p(1) + p(2) x + p(3) y + p(4) x^2 + p(5) y^2 + p(6) xy
        #       = c + (x-x0)' h (x-x0)
        A = np.vstack([np.ones_like(i0f), i0f, i1f, i0f**2, i1f**2, i0f*i1f]).T
        r = np.linalg.lstsq(A, af, rcond=None)
        p = r[0]
        x0 = - (np.matrix([[2*p[3], p[5]],[p[5], 2*p[4]]]).I * np.matrix([p[1],p[2]]).T ).A1
        c = p[0] + .5*(p[1]*x0[0] + p[2]*x0[1])
        h = np.matrix([[p[3], .5*p[5]],[.5*p[5], p[4]]])
        if not return_error:
            return (c,x0,h)

        mA = np.matrix(A)
        mse = .5*r[1][0] 
        dp = np.sqrt(np.diag(np.linalg.pinv(mA.T*mA) * mse))
        
        h1 = p[3]
        h2 = .5*p[5]
        h3 = p[4]
        y1 = p[1]
        y2 = p[2]
        
        Dh1 = dp[3]**2
        Dh2 = .25*dp[5]**2
        Dh3 = dp[4]**2
        Dy1 = dp[1]**2
        Dy2 = dp[2]**2
        deth = h1*h3 - h2**2
        
        dx1dh1 = .5 * ((h3*y1 - h2*y2)*h3/deth) / deth
        dx1dh2 = .5 * (-2*(h3*y1 - h2*y2)*h2/deth + y2) / deth
        dx1dh3 = .5 * ((h3*y1 - h2*y2)*h1/deth - y1) / deth
        dx1dy1 = -.5*h3/deth
        dx1dy2 = .5*h2/deth
    
        dx2dh1 = .5 * ((h1*y2 - h2*y1)*h3/deth - y2) / deth
        dx2dh2 = .5 * (-2*(h1*y2 - h2*y1)*h2/deth + y1) / deth
        dx2dh3 = .5 * ((h1*y2 - h2*y1)*h1/deth) / deth
        dx2dy1 = .5 * h2/deth
        dx2dy2 = -.5 * h1/deth
    
        dcdh1 = .5 * (y1*dx1dh1 + y2*dx2dh1)
        dcdh2 = .5 * (y1*dx1dh2 + y2*dx2dh2)
        dcdh3 = .5 * (y1*dx1dh3 + y2*dx2dh3)
        dcdy1 = .5 * (x0[0] + y1*dx1dy1 + y2*dx2dy1)
        dcdy2 = .5 * (x0[1] + y1*dx1dy2 + y2*dx2dy2)
        
        dx0 = np.array([0, 0])
        dx0[0] = np.sqrt( Dy1*dx1dy1**2 + Dy2*dx1dy2**2 + Dh1*dx1dh1**2 + Dh2*dx1dh2**2 + Dh3*dx1dh3**2 )
        dx0[1] = np.sqrt( Dy1*dx2dy1**2 + Dy2*dx2dy2**2 + Dh1*dx2dh1**2 + Dh2*dx2dh2**2 + Dh3*dx2dh3**2 )
        dc = np.sqrt( dp[0]**2 + Dy1*dcdy1**2 + Dy2*dcdy2**2 + Dh1*dcdh1**2 + Dh2*dcdh2**2 + Dh3*dcdh3**2 )
        dh = np.matrix([[dp[3], .5*dp[5]],[.5*dp[5], dp[4]]])
    
    else:
        raise RuntimeError('quad_fit not implemented for higher than 2 dimensions!')

    return (c, x0, h, dc, dx0, dh)

def shift_dist(a, b, w=None, return_coeff=True, scale_coeff=True):
    """
    Computes a windowed distance between ``a`` and ``b`` for all relative
    shifts of ``a`` relative to ``b``. 
    
    More precisely, this relation returns
    
    .. math::
        D(r) = \\sum_{r'} w(r') (a(r') - \\alpha(r) b(r'-r))^2
        
    where :math:`\\alpha` is a complex coefficient that minimizes :math:`D(r)`.
    
    Parameters
    ----------
    a : 2D-numpy-array
        Array to compare with ``b``.
    
    b : 2D-numpy-array
        Array to compare with ``a``. 
        
    w : ndarray, Defualt=None, optional
        ``w`` can also be a tuple of two arrays (wa, wb) which are used to mask
        data from both ``a`` and ``b``.
        
    return_coeff : bool, Default=Ture, optional
        If True returns ``coeff`` :math:`\\alpha`.
    
    scale_coeff : bool, Default=Ture, optional
        Allows only for a phase factor.
        
    Returns
    --------
    cc : 2D-numpy-array
        Similar to a regular cross correlation.
        
    coeff : 2D-numpy-array
        Complex coefficient (:math:`\\alpha`) for minimizing.
    """
    if w is None:
        a2 = norm2(a)
        b2 = norm2(b)
        cab = fft.ifftn(fft.fftn(a) * np.conj(fft.fftn(b)))
        if not scale_coeff:
            coeff = np.exp(1j * np.angle(cab))
            cc = a2 + b2 - 2*np.abs(cab)
        else:
            coeff = cab / b2
            cc = a2 - b2 * abs2(coeff)
        if return_coeff:
            return cc,coeff
        else:
            return cc
    else:
        if len(w) == 2:
            # We have a tuple : two masks
            w,wb = w
            first_term = np.real(fft.ifftn(fft.fftn(w*abs2(a)) * np.conj(fft.fftn(wb))))
            b *= wb
        else:
            first_term = np.sum(w*abs2(a))
        #w = w.astype(float)
        fw = fft.fftn(w)
        fwa = fft.fftn(w * a)
        fb2 = fft.fftn(abs(b)**2)
        fb = fft.fftn(b)
        epsilon = 1e-10
        if not scale_coeff:
            coeff = np.exp(1j * np.angle( fft.ifftn(fwa * np.conj(fb)) ))
            cc = first_term + np.real(fft.ifftn(fw * np.conj(fb2))) - 2*np.abs(fft.ifftn(fwa * np.conj(fb)))
        else:
            coeff = fft.ifftn(fwa * np.conj(fb)) / (fft.ifftn(fw * np.conj(fb2)) + epsilon)
            cc = first_term - abs2(fft.ifftn(fwa * np.conj(fb)))/(fft.ifftn(fw * np.conj(fb2)) + epsilon)
        if return_coeff:
            return cc,coeff
        else:
            return cc

def norm2(a):
    """
    Squared array norm.
    """
    return float(np.real(np.vdot(a.ravel(),a.ravel())))

def abs2(a):
    """
    Absolute squared.
    """
    return np.abs(a)**2

def fgrid(sh, psize=None):
    """
    Returns Fourier-space coordinates for a N-dimensional array of shape ``sh`` (pixel units).
    
    Parameters
    ----------
    sh : nd-array
        Shape of array.
    
    psize : int, Defualt=None, optional
        Pixel size in each dimensions.
    
    Returns
    --------
    nd-array
        Returns Fourier-space coordinates.
    """
    if psize is None:
        return fft.ifftshift(np.indices(sh).astype(float) - np.reshape(np.array(sh)//2,(len(sh),) + len(sh)*(1,)), list(range(1,len(sh)+1)))
    else:
        psize = np.asarray(psize)
        if psize.size == 1:
            psize = psize * np.ones((len(sh),))
        psize = np.asarray(psize).reshape( (len(sh),) + len(sh)*(1,))
        return fft.ifftshift(np.indices(sh).astype(float) - np.reshape(np.array(sh)//2,(len(sh),) + len(sh)*(1,)), list(range(1,len(sh)+1))) * psize
    
def pshift(a, ctr, method='linear', fill=None):
    """
    Shift a multidimensional array so that ``ctr`` becomes the origin.
    
    Parameters
    ----------
    a : nd-array
        Input array.
        
    ctr : array
        Position of new origin.
    
    method : str={'linear', 'nearest', 'fourier'}, Default='linear', optional
        Shift method.
        
    fill : bool, Default=None, optional
        If fill is None, the shifted image is rolled periodically.
    
    .. note::
        Fourier used to be the default but it produced artifacts at strong edges.
    
    Returns
    -------
    out : nd-array
        Shiftet array. 
    """
    sh  = np.array(a.shape)
    out = np.empty_like(a)

    if method.lower() == 'nearest':
        ctri  = np.round(ctr).astype(int)
        ctri2 = -ctri % sh  # force swap shift direction and force shift indices to be positive

        if fill is not None:
            # the filling case
            if (np.abs(ctri) > sh).any():
                return out             # shift is out of volume
            out.fill(fill)
            ctri2 -= sh * (ctri == 0)  # restore the sign of the shift index
        
        # walk through all (but the first) combinations of 0 and 1 on a length of a.ndim,
        # which are all possible copies of the original image in the output:
        #   0 is the first possible copy of the image in one dimension
        #   1 the second one
        comb_num = 2**a.ndim
        for comb_i in range(comb_num):
            comb = np.asarray(tuple(("{0:0" + str(a.ndim) + "b}").format(comb_i)), dtype=int)

            ctri3 = ctri2 - sh * comb
            out[ tuple([slice(None,  s) if s >  0 else slice(s,  None) for s in ctri3]) ] = \
              a[ tuple([slice(-s, None) if s >= 0 else slice(None, -s) for s in ctri3]) ]
            
            if fill is not None:
                break  # only the first copy of the image wanted

    elif method.lower() == 'linear':
        ctri = np.floor(ctr).astype(int)
        ctrx = np.empty((2, a.ndim))
        ctrx[1,:] = ctr - ctri     # second weight factor
        ctrx[0,:] = 1 - ctrx[1,:]  # first  weight factor
        out.fill(0.)
        
        # walk through all combinations of 0 and 1 on a length of a.ndim:
        #   0 is the shift with shift index floor(ctr[d]) for a dimension d
        #   1 the one for floor(ctr[d]) + 1
        comb_num = 2**a.ndim
        for comb_i in range(comb_num):
            comb = np.asarray(tuple(("{0:0" + str(a.ndim) + "b}").format(comb_i)), dtype=int)
            
            # add the weighted contribution for the shift corresponding to this combination
            out += pshift(a, ctri + comb, method='nearest', fill=fill) * ctrx[comb,list(range(a.ndim))].prod()

    elif method.lower() == 'fourier':
        fout = np.fft.fftn(a.astype(complex))
        out = np.fft.ifftn(fout * np.exp(2j * np.pi * np.sum(fgrid(sh,ctr/sh),axis=0)))

    return out

def correct_bad_pixels(img_in, th=None, iterations=1, dims=(-2, -1), p=0.5):
    """
    Remove hot pixels in an n-dimensional array.
    Identfies hot pixels as values exceeding threshold `th`, and
    replaces them by the median of their neighbors. The directions /
    dimensions in which to consider the neighbors can be set with `dims`.
    
    Parameters
    ----------
    img_in : numpy.ndarray
        Input array to correct
    th : float / list, optional
        Threshold to identify hot pixels. Values of `img_in` exceeding
        `th` are considered "bad". If `distance` is set to True, then the
        values of `img_in - median(img_in)` exceeding `th` are considered
        bad. If two values are given it considers (lower, upper) threshold
        limits.
    iterations : int, optional
        Number of iterations
    dims : tuple / list, optional
        Dimensions of `img_in` along which neighbors are considered for
        the calculation of the median.
    p: float, optional
        Percentile for automated thresholding if no `th` is given.
    
    Returns
    -------
    img : numpy.ndarray
        Corrected version of `img_in`.
        
    Notes
    -----
    For a 3d stack of projections, for example, it doesn't make sense
    to consider the same pixel in the next projection as a neighbor.
    The default value for `dims` (-2, -1) is thus suitable for a stack
    of projections if they are stacked along axis 0.
    """
    
    img = img_in.copy()
    sh  = img.shape

    if th is None:
        th = [np.percentile(img, p), np.percentile(img, 100-p)]
    else:
        th = [-th, th]

    mask =  (img < min(th)) | (img > max(th))    
    indices = list(np.where(mask))
    num_bad = len(indices[0])
    if num_bad == 0:
        return img
    for i in range(iterations):
        # make array of neighboring values, calculate their mean at the very end:
        # there are 2n neighbors in a n-dimensional array.
        bad_values = np.zeros((2 * len(dims), num_bad))
        # Find the indices of the neighbors, ensure we don't exceed
        # the limits of the image with `np.clip`:   
        for j, dim in enumerate(dims):
            orig = indices[dim]
            # "up" neighbors: np.abs ensures reflection at edges
            indices[dim] = np.abs(orig - 1)
            bad_values[2 * j] = img[tuple(indices)]
            # "down" neighbors:
            indices[dim] = orig + 1
            # reflect edge values:
            indices[dim][indices[dim] == sh[dim]] = sh[dim] - 2
            bad_values[2 * j + 1] = img[tuple(indices)]
            # reset indices for the other dimensions:
            indices[dim] = orig
        # do the correction:
        img[tuple(indices)] = np.median(bad_values, 0)
    return img

def get_diff_pos(refs, plot=False):
    '''
    Function that returns a list of diffuser positions based on image registration of the reference images.
    
    Parameters
    ----------
    refs: ndarray
        Reference images at different diffuser positions.
    plot: bool
        If True returns a plot of the diffuser positions with labels.
    sample_fact: float
        Precision 1/sample_fact, sample factor (default 100).

    Returns
    -------
    out : ndarray
        Array of [y,x] shifts.   
    '''
    import warnings
    warnings.filterwarnings("ignore")
    
    diffuser_positions = []
    sh = refs[-1].shape
    for i in range(len(refs)):
        res = shift_best(refs[0],refs[i])[-2]
        res[0] = ((res[0] + sh[0]/2) % sh[0]) - sh[0]/2
        res[1] = ((res[1] + sh[1]/2) % sh[1]) - sh[1]/2
        diffuser_positions.append(np.round(res, 2))
    diffuser_positions = np.array(diffuser_positions)    
    if plot:    
        x = diffuser_positions[:,1]
        y = diffuser_positions[:,0]
        plt.figure(figsize=(11.7,8.3))
        plt.scatter(x, y, s=100, alpha=1, color='black', marker='o', linestyle='solid',linewidth=1)
        for i, txt in enumerate(diffuser_positions):
            plt.annotate(txt, (x[i], y[i]), xycoords='data',
                    xytext=(-25, 10),  textcoords='offset pixels', size=10)
        plt.gca().set_aspect('equal')
    return diffuser_positions

def find_shift(sams, refs, sample_pos, w=20, s=3, step=20, num_threads=cpu//2, data=False):
    # ROI and step are applied after UMPA padding!
    sh = (sams[-1].shape[0] - 2 * (w + s), sams[-1].shape[1] - 2 * (w + s) )
    est_shift = -np.diff(sample_pos, axis=0).astype(int)
    
    # The shifts that we evaluate are relative to the first ref
    # - this compares for different sample positions 
    shift = [[0,0]]
    dx = []
    dy = []
    if data: 
        ref_new = np.empty_like(refs)
        ref_new[0] = refs[0]
    for p in range(len(sams)-1):
        sl1 = np.s_[max(0, -est_shift[p][0]) : min(sh[0], sh[0] - est_shift[p][0]) : step,
            max(0, -est_shift[p][1]) : min(sh[1], sh[1] - est_shift[p][1]) : step]
        sl2 = np.s_[max(0, est_shift[p][0]) : min(sh[0], sh[0] + est_shift[p][0]) : step,
                max(0, est_shift[p][1]): min(sh[1], sh[1] + est_shift[p][1]) : step]
        sl = [sl1, sl2]
        # print(sl)
        for q in [0,1]:
            res = UMPA_normal([sams[p+q]], [refs[p+q]], window=w, shift=s, ROI=sl[q], num_threads=num_threads)
            dx.append(res['dx'])
            dy.append(res['dy'])
        shift.append( [(dy[-2] - dy[-1]).mean(), (dx[-2] - dx[-1]).mean()] )
        if data: ref_new[p+1] = ndi.shift(refs[p+1], np.array(shift).sum(axis=0), mode='wrap')
        
    if data:
        return shift, ref_new, dx, dy 
    else: 
        return shift
        
def overlap(mpos, size):
    """
    Calculate distances and overlap of frame positions.
    
    Parameters
    ----------
    mpos: (N, 2) array
        (z, x) coordinates for each position
    size: (2,) tuple
        Shape of image frame (z, x)
    
    Returns
    -------
    d0 : (N, N) array
        Vertical (z) distance for each pair of frame positions
    d1 : (N, N) array
        Horizontal (x) distance for each pair of frame positions
    ov : (N, N) array
        Overlap of each pair of frame positions, relative to the
        frame dimensions (overlap=1: fully overlapping)
    """
    offset = np.min(mpos, 0)
    mpos0 = mpos - offset
    # horiz. and vert. distance vector components:
    d0 = mpos0[:, None, 0] - mpos0[None, :, 0]
    d1 = mpos0[:, None, 1] - mpos0[None, :, 1]
    # overlap:
    # round the shifts because UMPA does not do sub-pixel sample shifts
    ov = (np.clip(size[0] - np.abs(np.round(d0)), 0, None)
          * np.clip(size[1] - np.abs(np.round(d1)), 0, None))
    return d0, d1, ov / (size[0] * size[1])

def cost(motor_pos_est, matches_list, found_shifts):
    """
    motor_pos_est:
        the (N, 2) array, with ravel() applied to it, so: 
        (z0, x0, z1, x1, ...)
    found_shifts:
        `dist_list`, (N, 2) array, no ravel().
    """
    mpos_z, mpos_x = motor_pos_est[::2], motor_pos_est[1::2]

    # use indexing for defining the sum:
    # mpos_z_picked and mpos_x_picked are (K, 2) arrays,
    # where K is the number of image pairs.
    # mpos_z_picked[j, 0] has the z coord of the first point
    # of the j-th pair, and
    # mpos_z_picked[j, 1] has the z coord of the second point
    # of the j-th pair.
    mpos_z_picked = mpos_z[np.array(matches_list)]
    mpos_x_picked = mpos_x[np.array(matches_list)]

    # dist_z and dist_x are (K,) arrays.
    # dist_z[j] is the z distance of the j-th point pair.s
    dist_z = mpos_z_picked[:,1] - mpos_z_picked[:,0]
    dist_x = mpos_x_picked[:,1] - mpos_x_picked[:,0]
    
    # found_shifts has the same structure as `motor_pos_est`,
    # i.e. a 1d array: [z0, x0, z1, x1, ...]
    ssd = (np.sum((dist_z - found_shifts[:, 0])**2) +
           np.sum((dist_x - found_shifts[:, 1])**2))

    return ssd

def shift_data(refs, shift_list, mode='nearest'):
    ref_new = np.empty_like(refs)
    for p in range(len(refs)):
        ref_new[p] = ndi.shift(refs[p], shift_list[p], mode=mode)
    return ref_new

def get_new_diff_pos(sams, refs, sample_pos=None, diff_pos=None, 
                     ov_thr=0.5, w=20, s=3, step=20, num_threads=cpu//2):
    """
    Estimate diffuser drift and get the relative positions. Works when the
    sample fills entire field of view too.
    Can be used for sample-stepping acquisitions and diffuser-stepping
    acquisitions.
    
    Parameters
    ----------
    sams: nD-array, list
        List of images with sample and speckles. 
    refs: nD-array, list
        List reference speckle images.
    sample_pos: nD-array, list, optional
        For sample-stepping scans, (vertical, horizontal) sample motor 
        positions, positive values, starting from 0.
    diff_pos: nD-array, list, optional
        For diffuser-stepping scans, (vertical, horizontal) diffuser motor 
        positions or estimated cross-correlation values.
    ov_thr: float, optional
        Threshold for the frame-overlap condition in the sample-stepping 
        coverage. Value between 0 and 1. By default 0.5, half of the total 
        frames.
    w: int, optional
        Window size for UMPA. Should be at least of the order of the speckle 
        size, or bigger. Default 20.
    s: int, optional
        Maximum shift for UMPA.
    step: int, optional
        Speed up UMPA by considering every step-th element. Lower values
        have better statistic but most of the time is not needed. Default 20.
    num_threads: int, optional
        Number of threads to use in UMPA. By default half of the available.
    
    Returns
    ----------
    diff_pos_new : nD-array, list
        Diffuser drift displacement positions, relative to diff_pos.
    """
    from scipy.optimize import minimize
    if type(sample_pos)==type(None): sample_pos = np.zeros((sams.shape[0],2)) 
    ov = overlap(sample_pos, sams[-1].shape)[2]
    matches = ov > ov_thr
    matches_list = []
    for i, row in enumerate(matches):
        m = np.where(row[i+1:])[0] + (i+1)
        for mm in m:
            matches_list.append([i, mm])
    matches_shifts = []
    for m in matches_list:
        matches_shifts.append(find_shift(sams[m], refs[m], sample_pos[m], w=w,
                                         s=s, step=step, data=False, 
                                         num_threads=num_threads)[1])
    matches_shifts = np.array(matches_shifts)
    if type(diff_pos)==type(None): diff_pos = np.zeros((sams.shape[0],2)) 
    fit = minimize(cost, diff_pos.flatten(), (matches_list, matches_shifts))
    diff_pos_new = np.reshape(fit.x, diff_pos.shape)
    return diff_pos_new

def find_sam_shift(T=None, sams=None, refs=None, sample_pos=None, w=1, s=3, p=99.9, data=False, num_threads=cpu//2):
    # ROI and step are applied after UMPA padding!
    if type(sams)!=type(None):
        sh = (sams[-1].shape[0] - 2 * (w + s), sams[-1].shape[1] - 2 * (w + s) )
        r = len(sams) 
    elif type(T)!=type(None):
        sh = np.array(T[-1].shape)
        r = len(T)
    else:
        print('Give sams and refs or T from UMPA.')
    if type(sample_pos)==type(None): sample_pos = np.zeros((r,2)) 
    est_shift = -np.diff(sample_pos, axis=0).astype(int)
    
    # The shifts that we evaluate are relative to the first ref
    # - this compares for different sample positions 
    shift = [np.array([0,0])]
    for p in range(r-1):
        sl1 = np.s_[max(0, -est_shift[p][0]) : min(sh[0], sh[0] - est_shift[p][0]),
            max(0, -est_shift[p][1]) : min(sh[1], sh[1] - est_shift[p][1])]
        sl2 = np.s_[max(0, est_shift[p][0]) : min(sh[0], sh[0] + est_shift[p][0]),
                max(0, est_shift[p][1]): min(sh[1], sh[1] + est_shift[p][1])]
        sl = [sl1, sl2]
        if type(T)!=type(None):
            ims = [T[p][sl[0]], T[p+1][sl[1]]]
        else:
            ims = [UMPA_normal([sams[p+q]], [refs[p+q]], window=w, shift=s, ROI=sl[q], num_threads=num_threads)['T'] for q in [0,1]]
        ims = [correct_bad_pixels(im, np.percentile(im,p)) for im in ims]
        res = shift_best(ims[0], ims[1])[-2]
        if type(sample_pos)!=type(None): sh = ims[-1].shape
        res[0] = ((res[0] + sh[0]/2) % sh[0]) - sh[0]/2
        res[1] = ((res[1] + sh[1]/2) % sh[1]) - sh[1]/2
        shift.append( -res )
    if data:
        return shift, ims
    else:
        return shift
    
def get_new_sam_pos(sams=None, refs=None, T=None, sample_pos=None, ov_thr=0.5, w=2, s=3, 
                    num_threads=cpu//2):
    """
    Refines sample positions.
    Can be used for sample-stepping acquisitions and diffuser-stepping
    acquisitions.
    
    Parameters
    ----------
    sams: nD-array, list
        List of images with sample and speckles. 
    refs: nD-array, list
        List reference speckle images.
    sample_pos: nD-array, list, optional
        For sample-stepping scans, (vertical, horizontal) sample motor 
        positions, positive values, starting from 0.
    ov_thr: float, optional
        Threshold for the frame-overlap condition in the sample-stepping 
        coverage. Value between 0 and 1. By default 0.5, half of the total 
        frames.
    w: int, optional
        Window size for UMPA. Should be at least of the order of the speckle 
        size, or bigger. Default 2.
    s: int, optional
        Maximum shift for UMPA.
    num_threads: int, optional
        Number of threads to use in UMPA. By default half of the available.
    
    Returns
    ----------
    sam_pos_new : nD-array, list
        Refined sample positions.
    """

    if type(sams)!=type(None):
        sh = sams[-1].shape
        r = len(sams)
    elif type(T)!=type(None):
        sh = T[-1].shape
        r = len(T)
    else:
        print('Give sams and refs or T from UMPA.')
    if type(sample_pos)==type(None):
        sample_pos = np.zeros((r,2))
    ov = overlap(sample_pos, sh)[2]
    matches = ov > ov_thr
    matches_list = []
    for i, row in enumerate(matches):
        m = np.where(row[i+1:])[0] + (i+1)
        for mm in m:
            matches_list.append([i, mm])
    
    if type(sams)!=type(None):
        matches_shifts = []
        for m in matches_list:
            matches_shifts.append(find_sam_shift(sams[m], refs[m], sample_pos[m], w=w,
                                             s=s, data=False, 
                                             num_threads=num_threads)[1])
        matches_shifts = np.array(matches_shifts)
    else:
        def get_sam_shift(match):
            return find_sam_shift(T=T[match], sample_pos=sample_pos[match])[1]

        matches_shifts = np.array(job.Parallel(n_jobs=cpu, backend='loky')(
                                    job.delayed(get_sam_shift)(m) for m in matches_list))

    fit = minimize(cost, sample_pos.flatten(), (matches_list, matches_shifts))
    sam_pos_new = np.reshape(fit.x, sample_pos.shape)
    return sam_pos_new

# USING THE FUNCTIONS

def info():
    print('EXAMPLE of sample motor position refinement: \n'
          'Ts = np.array([UMPA_normal([sams[i]], [refs[i]], '
          'num_threads=int(0.75*cpu))[T] '
          'for i in range(len(sams))]) \nsam_pos = get_new_sam_pos(T=Ts, '
          'sample_pos=None) \nsam_pos_new = sam_pos - sam_pos[0]\n'
          'sams_align = shift_data(sams, sam_pos_new)\n'
          'refs_align = shift_data(refs, sam_pos_new)')
    print('EXAMPLE of diffuser drift correction: \n'
          'diff_pos0 = get_diff_pos(refs)\n'
          'diff_pos = get_new_diff_pos(sams=sams, refs=refs, ' 
          'sample_pos=None, diff_pos=diff_pos0) \ndiff_pos =  diff_pos - ' 
          'diff_pos[0] \nrefs_new = shift_data(refs, diff_pos)')