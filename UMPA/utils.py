import numpy as np
import scipy
from scipy import signal as sig
from scipy import ndimage as ndi


def cc(A, B, mode='same'):
    """
    A fast cross-correlation based on scipy.signal.fftconvolve.

    :param A: The reference image
    :param B: The template image to match
    :param mode: one of 'same' (default), 'full' or 'valid' (see help
     for fftconvolve for more info)
    :return: The cross-correlation of A and B.
    """
    return sig.fftconvolve(A, B[::-1, ::-1], mode=mode)


def quad_fit(a):
    """\
    (c, x0, H) = quad_fit(A)
    Fits a parabola (or paraboloid) to A and returns the
    parameters (c, x0, H) such that

    a ~ c + (x-x0)' * H * (x-x0)

    where x is in pixel units. c is the value at the fitted optimum, x0
    is the position of the optimum, and H is the hessian matrix
    (curvature in 1D).
    """

    sh = a.shape

    i0, i1 = np.indices(sh)
    i0f = i0.flatten()
    i1f = i1.flatten()
    af = a.flatten()

    # Model = p(1) + p(2) x + p(3) y + p(4) x^2 + p(5) y^2 + p(6) xy
    #       = c + (x-x0)' h (x-x0)
    A = np.vstack([np.ones_like(i0f), i0f, i1f, i0f**2, i1f**2, i0f*i1f]).T
    r = np.linalg.lstsq(A, af, rcond=None)
    p = r[0]
    x0 = - (np.matrix([[2*p[3], p[5]], [p[5], 2*p[4]]]).I *
            np.matrix([p[1], p[2]]).T).A1
    c = p[0] + .5*(p[1]*x0[0] + p[2]*x0[1])
    h = np.matrix([[p[3], .5*p[5]], [.5*p[5], p[4]]])
    return c, x0, h


def quad_max(a):
    """\
    (c, x0) = quad_max(a)

    Fits a parabola (or paraboloid) to A and returns the
    maximum value c of the fitted function, along with its
    position x0 (in pixel units).
    All entries are None upon failure. Failure occurs if :
    * A has a positive curvature (it then has a minimum, not a maximum).
    * A has a saddle point
    * the hessian of the fit is singular, that is A is (nearly) flat.
    """

    c, x0, h = quad_fit(a)

    failed = False
    if a.ndim == 1:
        if h > 0:
            print('Warning: positive curvature!')
            failed = True
    else:
        if h[0, 0] > 0:
            print('Warning: positive curvature along first axis!')
            failed = True
        elif h[1, 1] > 0:
            print('Warning: positive curvature along second axis!')
            failed = True
        elif np.linalg.det(h) < 0:
            print('Warning: the provided data fits to a saddle!')
            failed = True

    if failed:
        c = None
    return c, x0


def pshift(a, ctr):
    """\
    Shift an array so that ctr becomes the origin.
    fdm: This performs linear interpolation on n-dimensional arrays.
    """
    sh  = np.array(a.shape)
    out = np.zeros_like(a)

    ctri = np.floor(ctr).astype(int)
    ctrx = np.empty((2, a.ndim))
    ctrx[1, :] = ctr - ctri     # second weight factor
    ctrx[0, :] = 1 - ctrx[1, :]  # first  weight factor

    # walk through all combinations of 0 and 1 on a length of a.ndim:
    #   0 is the shift with shift index floor(ctr[d]) for a dimension d
    #   1 the one for floor(ctr[d]) + 1
    comb_num = 2**a.ndim
    for comb_i in range(comb_num):
        # this is: [(0,0,0), (0,0,1), (0,1,0), etc.] for 3D arrays,
        # equivalently for other dimensions.
        # this selects one of the 1-pixel-shifted versions of the
        # input array "a".
        comb = np.asarray(tuple(("{0:0" + str(a.ndim) + "b}").format(comb_i)), dtype=int)

        # fdm: get the weighted contribution for the shift corresponding to this combination
        cc = ctri + comb
        out += np.roll(a, shift=-cc, axis=range(a.ndim)) * ctrx[comb,range(a.ndim)].prod()
    return out


def sub_pix_min_quad(a, width=1):
    """
    Find the position of the minimum in 2D array a with subpixel
    precision (using a paraboloid fit).

    :param a:
    :param width: 2*width+1 is the size of the window to apply the fit.
    :return:
    """

    sh = a.shape

    # Find the global minimum
    cmin = np.array(np.unravel_index(a.argmin(), sh))

    # Move away from edges
    if cmin[0] < width:
        cmin[0] = width
    elif cmin[0]+width >= sh[0]:
        cmin[0] = sh[0] - width - 1
    if cmin[1] < width:
        cmin[1] = width
    elif cmin[1]+width >= sh[1]:
        cmin[1] = sh[1] - width - 1

    # Sub-pixel minimum position.
    mindist, r = quad_max(-np.real(a[(cmin[0]-width):(cmin[0]+width+1), (cmin[1]-width):(cmin[1]+width+1)]))
    r -= (width - cmin)

    return r


# Polynomial coefficients for the autocorrelation of the triangular function
coeffs = np.array([[1., -3., 3., -1],
                   [4.,  0, -6.,  3.],
                   [1.,  3., 3., -3.],
                   [0.,  0., 0.,  1.]])
# Conversion matrix
M = np.array([np.multiply.outer(coeffs[j], coeffs[i]).ravel() for i in range(4) for j in range(4)]).T

# Exponent arrays for 0th, 1st and 2nd derivatives
e0 = np.array([0, 1, 2, 3])
e1 = np.array([0, 0, 1, 2])
e2 = np.array([0, 0, 0, 1])


# Interpolant and derivatives at x0
def allf(x0, c):
    x, y = x0
    return np.dot(np.array([np.multiply.outer(y**e0, x**e0).ravel(),
                            np.multiply.outer(y**e0, e0 * x**e1).ravel(),
                            np.multiply.outer(e0 * y**e1, x**e0).ravel(),
                            np.multiply.outer(y**e0, e1*e0 * x**e2).ravel(),
                            np.multiply.outer(e0 * y**e1, e0 * x**e1).ravel(),
                            np.multiply.outer(e0*e1 * y**e2, x**e0).ravel()]), c)


# def sub_pix_min(a):
#     """
#     Find the position of the minimum in 2D array a with subpixel precision.
#     """
#     sh = a.shape
#
#     # Find the global minimum
#     cmin = np.array(np.unravel_index(a.argmin(), sh))
#
#     # Move away from edges
#     if cmin[0] < 2:
#         cmin[0] = 2
#     elif cmin[0]+2 >= sh[0]:
#         cmin[0] = sh[0] - 3
#     if cmin[1] < 2:
#         cmin[1] = 2
#     elif cmin[1]+2 >= sh[1]:
#         cmin[1] = sh[1] - 3
#
#     # Sub-pixel minimum position.
#     a0 = a[(cmin[0]-2):(cmin[0]+3), (cmin[1]-2):(cmin[1]+3)]
#
#     # Find which quadrant to refine
#     jp = 1 if a0[2, 3] < a0[2, 1] else 0
#     ip = 1 if a0[3, 2] < a0[1, 2] else 0
#     af = a0[ip:ip+4, jp:jp+4].copy()
#
#     x0 = 1. - np.array([ip, jp])
#     f = cutils.sub_pix_cc_linear(af, x0)
#     r = x0 - 1 + np.array([ip, jp]) + cmin
#     return r, f


def sub_pix_cc_linear(a, x0):
    """

    :param a: a 4x4 array with the minimum in the central square.
    :param x0: starting point for search.
    :return: interpolated value of the minimum
    *** subpixel position is set in-place in x0! ***
    """
    # Generate vector
    c = np.dot(M, a.ravel())

    # Newton-Raphson
    tol = 1e-12
    for i in range(30):
        f, fx, fy, fxx, fxy, fyy = allf(x0, c)
        dx = -np.array([fyy*fx - fxy*fy, -fxy*fx + fxx*fy])/(fxx*fyy - fxy*fxy)
        dx.clip(-.5, .5, out=dx)
        x0 += dx
        if (dx*dx).sum() < tol:
            break

    return f/36.


def free_nf(w, l, z, pixsize=1.):
    """\
    Free-space propagation (near field) of the wavefield of a distance z.
    l is the wavelength.
    """
    if w.ndim != 2:
        raise RuntimeError("A 2-dimensional wave front 'w' was expected")

    sh = w.shape

    # Convert to pixel units.
    z = z / pixsize
    l = l / pixsize

    # Evaluate if aliasing could be a problem
    if min(sh) / np.sqrt(2.) < z * l:
        print("Warning: z > N/(sqrt(2)*lamda) = %.6g: this calculation could fail." % (min(sh) / (l * np.sqrt(2.))))
        print("(consider padding your array, or try a far field method)")

    q2 = np.sum((np.fft.ifftshift(
        np.indices(sh).astype(float) - np.reshape(np.array(sh) // 2, (len(sh),) + len(sh) * (1,)),
        range(1, len(sh) + 1)) * np.array([1. / sh[0], 1. / sh[1]]).reshape((2, 1, 1))) ** 2, axis=0)

    return np.fft.ifftn(np.fft.fftn(w) * np.exp(2j * np.pi * (z / l) * (np.sqrt(1 - q2 * l ** 2) - 1)))


def cdiff(array, axis, remap=True):
    """ Central difference with fixed phase wrapping of
    2pi-periodic quantities (for remap=True)."""
    # fd, bd: forward / backward difference
    fd, bd = np.zeros((2,) + array.shape)
    d = np.diff(array, 1, axis)
    if remap:
        d = (d + np.pi) % (2*np.pi) - np.pi
    sl = [slice(None,None,None)]*array.ndim  # slice object used multiple times
    # fd and bd differ only in where d is placed.
    sl[axis] = slice(1, None, None)
    fd[tuple(sl)] = d
    sl[axis] = slice(None, -1, None)
    bd[tuple(sl)] = d
    # central difference is mean of fd and bd.
    cd = (fd + bd) / 2.
    # fix edge cases: use only forward / backward difference
    sl[axis] = slice(0, 1, None)
    cd[tuple(sl)] = bd[tuple(sl)]
    sl[axis] = slice(-1, None, None)
    cd[tuple(sl)] = fd[tuple(sl)]
    return cd


def binning(arr, factor, axes=(-2, -1)):
    """
    Do quick binning of input array `arr` by `factor` along `axes`.
    I.e.: "bin(arr, 2, (-2, -1))" does 2x2 binning along axes (-2, -1).
    Fabio De Marco, 15.12.21.
    """
    
    assert type(factor) is int
    assert factor > 0
    # crop to make shape a multiple of `factor`:
    sh = arr.shape
    slice_crop = [slice(None, None, None)] * arr.ndim
    sh_crop = list(sh)
    for ax in axes:
        sh_crop[ax] = sh[ax] // factor * factor
        slice_crop[ax] = slice(0, sh_crop[ax], None)
    slice_crop = tuple(slice_crop)
    
    # We use the trick of reshaping and taking the mean to do fast rebinning.
    # It just looks complicated because we want this to work for arbitrary axes.
    new_shape = []
    mean_axes = []
    for ax in range(arr.ndim):
        new_shape.extend([sh_crop[ax], 1])
        mean_axes.append(2 * ax + 1)
    
    for ax in axes:
        new_shape[2 * ax] = sh_crop[ax] // factor
        new_shape[2 * ax + 1] = factor
    
    print(new_shape, mean_axes)
    return arr[slice_crop].reshape(new_shape).mean(tuple(mean_axes)).squeeze()


def prep_simul(sample_shift=False, steps=25, step_size=4, profile='flat',
               step_random=False, obj='sphere', shape=(500,600), bin_fact=1,
               energy=24.2, psize=1e-6, ssize=2e-6, speckles=True,
               dn=7.79552408E-07 - 1j * 1.2177146E-09, z=5e-2, noise=None,
               pyr_width=300e-6, sphere_radius=150e-6, cyl_radius=150e-6,
               wedge_width=300e-6, logo_height=50e-6):
    """
    Generate a simulated data set and ground truth.
    
    Parameters
    ----------
    sample_shift : bool, optional
        If True, shift sample and keep diffuser stationary. If False,
        move diffuser between acquisitions.
    steps : int or (N, 2) ndarray, optional
        If integer, gives number of steps. Displacement vector for each
        diffuser/sample step is then controlled by `step_random` and `step_size`.
        If (N, 2) ndarray, gives displacement vectors for each of the N steps.
    step_size : float, optional
        Step size (in px) between adjacent phase steps.
    profile : ['flat' | 'gauss']
        If `flat`, uses illumination with constant intensity. If 'gauss', uses
        a Gaussian intensity profile, with sigma_x, sigma_y equal to half of
        width of the FoV.
    step_random : bool, optional
        If True, select uniformly distributed random positions
        in a box of `step_size` x `step_size` pixels.
        If False, select step positions on a regular, roughly quadratic
        grid with `step_size` pixels between adjacent points.
    obj : ('sphere', 'cyl_x', 'cyl_y', 'pyramid', 'wedge_x', 'wedge_y', 'logo'), optional
        Type of sample to simulate. `logo` (UniTS logo) is larger than the
        default FoV of 500x600 and thus currently works only with `sample_shift`.
        'sphere' is a sphere in the center of the FoV. `cyl` is a vertical
        cylinder. `pyramid` is a 4-sided pyramid. 'wedge_x' and 'wedge_y' are
        what they sound like: a ramp with constant slope. `logo` is the UniTS
        logo.
    shape : (int, int), optional
        Shape of simulation space in pixels
    bin_fact : int > 0
        Binning factor for forward-propagated speckle data.
        (Hopefully) useful for simulating dark-field.
    energy : float, optional
        Photon energy in keV.
    psize : float, optional
        Pixel size in m.
    ssize : float, optional
        Approximate speckle size in m.
    speckles : bool, optional
    dn : complex, optional
        Index of refraction (1 - delta - i*beta) to assume for the sample.
        Default value is for SiO2.
    z : float, optional
        Propagation distance in m
    noise : float or None, optional
        If not None (default), assume this value to be the mean number of
        photons per pixel in the reference measurement, and add an appropriate
        amount of Poisson noise to the intensities.
    pyr_width, sphere_radius, cyl_radius, wedge_width, logo_height : float
        Dimensions of the respective test objects [m] (see parameter `obj`).
        
    Returns
    -------
    out, dict
        out['T'], out['dx'], out['dy']: Ground truth arrays for transmittance,
        dpcx, dpcy (in pixels), retrieved from the simulated wavefront.
        out['pos_diff']: Assumed diffuser displacement vectors.
        out['pos_sample']: Assumed sample displacement vectors.
        out['ref']: (N, X, Y) ndarray, Reference intensities
        out['meas']: (N, X, Y) ndarray, Sample intensities
        out['wf']: Wavefront at the detector, with sample, without diffuser
        out['sample_height']: Height map of sample, this times (delta * k)
                              should be equal to the phase shift?
    :return:
    """
    import os
    from scipy import ndimage as ndi
    from UMPA.utils import free_nf, pshift
    from UMPA import __file__
    from numpy.lib import scimath
    import scipy
    from scipy.ndimage import gaussian_filter
        
    lam = 12.406e-10 / energy  # wavelength [m]

    # Simulate speckle pattern
    if speckles:
        np.random.seed(10)
        speckle = ndi.gaussian_filter(np.random.normal(size=shape), ssize / psize) + 0j
        np.random.seed(11)
        speckle += 1j * ndi.gaussian_filter(np.random.normal(size=shape), ssize / psize)
    else:
        speckle = np.ones(shape, dtype=np.complex)
    
    yy, xx = np.indices(shape)
    
    if obj=='sphere':
        feature = 2 * psize * np.real(scimath.sqrt((sphere_radius / psize)**2
                                      - (xx - shape[1] / 2.)**2
                                      - (yy - shape[0] / 2.)**2))
    elif obj=='cyl_y':
        feature = 2 * psize * np.real(
            scimath.sqrt((cyl_radius / psize)**2 - (yy - shape[0] / 2.)**2))
    elif obj=='cyl_x':
        feature = 2 * psize * np.real(
            scimath.sqrt((cyl_radius / psize)**2 - (xx - shape[1] / 2.)**2))
    elif obj=='pyramid':
        pyr_height = pyr_width / 2.
        # "1-norm" / Manhattan norm:
        dist1 = np.max(np.array([np.abs(xx-shape[1] / 2.),
                                 np.abs(yy-shape[0] / 2.)]), 0) * psize 
        feature = np.clip(pyr_height - dist1 * pyr_height / (pyr_width / 2), 0, None)
    elif obj=='wedge_x':
        wedge_height = wedge_width
        feature = wedge_height / 2. + wedge_height / wedge_width * (xx-np.mean(xx)) * psize
        dist1 = np.max(np.array([np.abs(xx-shape[1] / 2.),
                                 np.abs(yy-shape[0] / 2.)]), 0) * psize
        feature[dist1 > wedge_width/2.] = 0
    elif obj=='wedge_y':
        wedge_height = wedge_width
        feature = wedge_height / 2. + wedge_height / wedge_width * (yy-np.mean(yy)) * psize
        dist1 = np.max(np.array([np.abs(xx-shape[1]/2.),
                                 np.abs(yy-shape[0]/2.)]), 0) * psize
        feature[dist1 > wedge_width/2.] = 0
    elif obj=='logo':
        module_path = os.path.split(__file__)[0]
        logo = np.load(os.path.join(module_path, 'test', 'logo.npy'))
        logo = gaussian_filter(logo, 2)
        feature  = np.pad(logo, ((shape[0] + 50, shape[0]),
                                 (shape[1] + 50, shape[1]))) * logo_height
    else:
        print(f'Object type "{obj}" not recognized.')
        return
    if profile=='flat':
        wf_flat = np.ones(shape, dtype=np.complex)
    elif profile=='gauss':
        sigma0, sigma1 = 0.5 * shape[0], 0.5 * shape[1]
        wf_flat = np.exp(-0.5 * (((xx - np.mean(xx)) / sigma1)**2 +
                                 ((yy - np.mean(yy)) / sigma0)**2))
    
    if obj=='logo':
        sl_fov = np.s_[shape[0]:2*shape[0], shape[1]:2*shape[1]]
    else:
        sl_fov = np.s_[:]
    # This is where the sample modulates the reference wavefront:
    t_sample = np.exp(-2j * np.pi * feature * dn / lam)
    wf_sample = wf_flat * t_sample[sl_fov]

    # Propagated without pattern.
    # todo fabio: normalize for non-flat profile...
    wf_sample_prop_nospeckle = free_nf(wf_sample, lam, z, psize)
    T = np.abs(wf_sample_prop_nospeckle)**2
    gy = cdiff(np.angle(wf_sample_prop_nospeckle), 0, True)
    gx = cdiff(np.angle(wf_sample_prop_nospeckle), 1, True)
    dx = -gx * z * lam / (2 * np.pi * psize**2)
    dy = -gy * z * lam / (2 * np.pi * psize**2)

    # Measurement positions, i.e., the lateral positions / shifts
    # of diffuser or sample.
    # pos = np.array([(0., 0.)] + [(np.round(15.*cos(pi*j/3)),
    #                np.round(15.*sin(pi*j/3))) for j in range(6)])
    if step_random:
        np.random.seed()
        pos = step_size * np.random.rand(steps * 2).reshape((2,-1)).T
    else:  # arrange steps in box pattern
        if type(steps) is np.ndarray:
            pos = steps
        elif type(steps) is int:
            root = np.sqrt(steps)
            iroot = int(root)
            if np.isclose(iroot, root):
                pos = step_size * np.indices((iroot, iroot)).reshape((2, -1)).T
            else:
                pos = step_size * np.indices((iroot+1, iroot+1)).reshape((2, -1)).T[:steps]   

    # UMPA needs the positions flipped,
    # and we need to correct for the binning factor:
    pos_flipped = np.max(pos, 0) - pos
    if sample_shift:
        pos_sample = pos_flipped / bin_fact
        pos_diff   = np.zeros((steps, 2))
    else:
        pos_sample = np.zeros((steps, 2))
        pos_diff   = pos / bin_fact
    
    # Simulate the measurements: include the flat wavefront!
    
    if sample_shift:
        m = []
        for p in pos:
            # Sample:
            shifted_sample = wf_flat * pshift(t_sample, -p)[sl_fov]
            wf_sample_prop = free_nf(shifted_sample * speckle, lam, z, psize)
            m.append(abs(wf_sample_prop)**2)
        I_sample = np.array(m)
        # Ref:
        wf_ref_prop = abs(free_nf(wf_flat * speckle, lam, z, psize))**2
        ref = wf_ref_prop.copy() + 0.
        I_ref = ref[None, :, :] * np.ones(len(pos))[:, None, None]
        #I_ref = np.array([ref for p in pos])  # duplicate the same reference len(pos) times
    else:
        m, r = [], []
        for p in pos:
            # Sample:
            shifted_speckle = pshift(speckle, p)
            wf_sample_prop = free_nf(wf_sample * shifted_speckle, lam, z, psize)
            m.append(abs(wf_sample_prop)**2)
            # Ref:
            # note: shift only the speckles, not the flat pattern!
            wf_ref_prop = free_nf(wf_flat * shifted_speckle, lam, z, psize)
            r.append(abs(wf_ref_prop)**2)
        I_sample, I_ref = np.array(m), np.array(r)
    
    # put binning here:
    if bin_fact > 1:
        I_sample = binning(I_sample, bin_fact, (-2, -1))
        I_ref    = binning(I_ref,    bin_fact, (-2, -1))
    
    if noise: # if not None, noise is in photons per pixel (in the ref.)
        # question: if we have non-flat profile, is this quantity good enough?
        fact = noise / np.mean(I_ref)
        np.random.seed()
        I_sample = np.random.poisson(I_sample * fact).astype(float) / fact
        np.random.seed()
        I_ref = np.random.poisson(I_ref * fact).astype(float) / fact

    return {'T': T,
            'dx': dx,
            'dy': dy,
            'pos_sample': pos_sample,
            'pos_diff': pos_diff,
            'ref': I_ref,
            'meas': I_sample,
            'wf': wf_sample_prop_nospeckle,
            'sample_height': feature}


def get_cost(model, i, j, N=2):
    """
    Evaluate cost function values of an UMPA model at one pixel (i,j) for a
    range of integer shifts (i.e., dx and dy values).
    """
    from . import model as m
    
    c, t, d = np.ones((3, 2*N+1, 2*N+1))
    dxs = np.arange(-N,N+1)
    dys = np.arange(-N,N+1)
    for k, dx in enumerate(dxs):
        for l, dy in enumerate(dys):
            out = model.cost(i,j,dx,dy)
            if type(model) is m.UMPAModelDF:
                c[k,l], t[k,l], d[k,l] = out
            elif type(model) is m.UMPAModelNoDF:
                c[k,l], t[k,l] = out
    return c, t, d  # d not set for df=False, thus all ones
  

def plot_cost(model, fit, i, j, window_size, max_shift, vmin=None, vmax=None, figsize=(14,4), do_plot=True):
    """
    Get UMPA cost function landscape for one pixel (i,j) (see get_cost()),
    and find the minimum via the UMPA sub-pixel minimizer.
    
    sam, ref: 3d arrays of sample and reference UMPA data.
    i,j : pixel coordinate (in the cropped UMPA images)
    sl : 
    """
    from matplotlib import pyplot as plt
    
    def imsh_cost(img):
        sh0, sh1 = img.shape
        ex = [-(sh0-1)//2-.5, (sh0-1)//2+.5, (sh1-1)//2+.5, -(sh1-1)//2-.5]
        plt.imshow(img, extent=ex)
        plt.xlabel('dx'); plt.ylabel('dy')

    c, t, d = get_cost(model, i+window_size+max_shift, j+window_size+max_shift, N=max_shift-1)

    if do_plot:
        plt.figure(figsize=figsize)
        plt.subplot(131); plt.title('dx, dot:(i,j)=(%d,%d)' % (i,j))
        plt.imshow(fit['dx'], vmin=vmin, vmax=vmax); plt.plot(j,i, 'ro')
        plt.subplot(132); plt.title('dy, dot:(i,j)=(%d,%d)' % (i,j))
        plt.imshow(fit['dy'], vmin=vmin, vmax=vmax); plt.plot(j,i, 'ro')
        plt.subplot(133); plt.title('Cost function at (i,j) = (%d,%d)\ndx=%.2f, dy=%.2f, f=%.3g, N_calls=%d' % (i,j,fit['dx'][i,j],fit['dy'][i,j],fit['f'][i,j],fit['debug_Ncalls'][i,j]))
        imsh_cost(c); plt.plot(fit['dx'][i,j], fit['dy'][i,j], 'ro')
        plt.tight_layout()
    return c, t, d

