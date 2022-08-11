import UMPA
import optimatools as opt
%matplotlib widget
from matplotlib.pyplot import *


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


def prep_simul(sample_shift=False, steps=25, step_size=4, step_random=False, obj='sphere',
               dn=2.0E-06 -1j*1.2177146E-09, z=5e-2, noise=None):
    """
    Generate a simulated data set and ground truth.
    
    Parameters
    ----------
    sample_shift : bool
        If True, shift sample and keep diffuser stationary. If False, move diffuser between acquisitions.
    steps : int or (N, 2) ndarray
        If integer, gives number of steps. Displacement vector for each
        diffuser/sample step is then controlled by `step_random` and `step_size`.
        If (N, 2) ndarray, gives displacement vectors for each of the N steps.
    step_size : float
        Step size (in px) between adjacent phase steps.
    step_random : bool
        If True, select uniformly distributed random positions
        in a box of `step_size` x `step_size` pixels.
        If False, select step positions on a regular, roughly quadratic grid
        with `step_size` pixels between adjacent points.
    obj : ('sphere', 'pyramid', 'wedge_x', 'wedge_y', 'logo')
        Type of sample to simulate. `logo` (UniTS logo) is larger than the
        default FoV of 500x600 and thus currently works only with `sample_shift`.
    dn : complex, optional
        Index of refraction (1 - delta - i*beta) to assume for the sample.
        Default value is for SiO2.
    z : float
        Propagation distance
    noise : float or None, optional
        If not None (default), assume this value to be the mean number of
        photons per pixel in the reference measurement, and add an appropriate
        amount of Poisson noise to the intensities.
        
    Returns
    -------
    out, dict
        out['T'], out['dx'], out['dy']: Ground truth arrays for transmittance,
        dpcx, dpcy (in pixels), retrieved from the simulated wavefront.
        out['positions']: Assumed sample / diffuser displacement vectors.
        out['ref']: (N, X, Y) ndarray, Reference intensities
        out['meas']: (N, X, Y) ndarray, Sample intensities
        out['wf']: Wavefront at the detector, with sample, without diffuser
    :return:
    """
    from scipy import ndimage as ndi
    from UMPA.utils import free_nf, pshift
    from numpy.lib import scimath
    import scipy
    from scipy.ndimage import gaussian_filter
    # Simulation of a SiO2 sphere
    #dn = 7.79552408E-07 - 1j * 1.2177146E-09  # index of refraction decrement
    energy = 24.2  # Photon energy [keV]
    lam = 12.406e-10 / energy  # wavelength [m]
    psize = 1e-6  # pixel size [m]
    ssize = 2e-6  # rough speckle size
    sh = (500, 600)
    

    # Simulate speckle pattern
    np.random.seed(10)
    speckle = ndi.gaussian_filter(np.random.normal(size=sh), ssize / psize) + 0j
    np.random.seed(11)
    speckle += 1j * ndi.gaussian_filter(np.random.normal(size=sh), ssize / psize)
    yy, xx = np.indices(sh)
    
    if obj=='sphere':
        sphere_radius = 150e-6
        feature = 2 * psize * np.real(scimath.sqrt((sphere_radius / psize) ** 2 - (xx - sh[1]/2.) ** 2 - (yy - sh[0]/2.) ** 2))
    elif obj=='pyramid':
        pyr_width, pyr_height = 300e-6, 150e-6
        dist1 = np.max(np.array([np.abs(xx-sh[1]/2.), np.abs(yy-sh[0]/2.)]),0) * psize # "1-norm" / Manhattan norm
        feature = np.clip(pyr_height - dist1 * pyr_height / (pyr_width / 2), 0, None)
    elif obj=='wedge_x':
        wedge_width, wedge_height = 300e-6, 300e-6
        feature = wedge_height / 2. + wedge_height / wedge_width * (xx-np.mean(xx)) * psize
        dist1 = np.max(np.array([np.abs(xx-sh[1]/2.), np.abs(yy-sh[0]/2.)]),0) * psize
        feature[dist1 > wedge_width/2.] = 0
    elif obj=='wedge_y':
        wedge_width, wedge_height = 300e-6, 300e-6
        feature = wedge_height / 2. + wedge_height / wedge_width * (yy-np.mean(yy)) * psize
        dist1 = np.max(np.array([np.abs(xx-sh[1]/2.), np.abs(yy-sh[0]/2.)]),0) * psize
        feature[dist1 > wedge_width/2.] = 0
    elif obj=='logo':
        height = 50e-6
        logo = np.load('/home/fabio/UMPAdev/UMPA/test/logo.npy')
        logo = gaussian_filter(logo, 2)
        feature  = np.pad(logo, ((sh[0]+50, sh[0]),(sh[1]+50, sh[1]))) * height
    else:
        print('Object type "%s" not recognized.' % obj)
        return
    sample = np.exp(-2j * np.pi * feature * dn / lam)

    # Propagated without pattern
    wavefront = free_nf(sample, lam, z, psize)
    T = np.abs(wavefront) ** 2
    gy = cdiff(np.angle(wavefront),0, True)
    gx = cdiff(np.angle(wavefront),1, True)
    dx = -gx * z * lam / (2 * np.pi * psize ** 2)
    dy = -gy * z * lam / (2 * np.pi * psize ** 2)

    # Measurement positions
    # fabio: meaning, the lateral positions / shifts of diffuser or sample.
    # pos = np.array( [(0., 0.)] + [(np.round(15.*cos(pi*j/3)), np.round(15.*sin(pi*j/3))) for j in range(6)] )
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

    # Simulate the measurements
    reference = abs(free_nf(speckle, lam, z, psize)) ** 2
    if sample_shift:
        if obj=='logo':
            sl_fov = np.s_[sh[0]:2*sh[0],sh[1]:2*sh[1]]
        else:
            sl_fov = np.s_[:]
        measurements = np.array([abs(free_nf(pshift(sample, -p)[sl_fov] * speckle, lam, z, psize)) ** 2 for p in pos])
        ref = reference.copy() + 0.
        sref = [ref for p in pos]
    else:
        measurements = np.array([abs(free_nf(sample * pshift(speckle, p), lam, z, psize)) ** 2 for p in pos])
        sref = np.array([pshift(reference, p) for p in pos])
    
    if noise: # if not None, noise is in photons per pixel (in the ref.)
        fact = noise / np.mean(reference)
        np.random.seed()
        measurements = np.random.poisson(measurements * fact).astype(float)
        np.random.seed()
        sref = np.random.poisson(sref * fact).astype(float)

    return {'T': T,
            'dx': dx,
            'dy': dy,
            'positions': pos,
            'ref': sref,
            'meas': measurements,
            'wf': wavefront}


x = np.arange(0, 1801, 150) * (-1)
y = np.arange(0, 1801, 150) * (-1)
i,j = np.meshgrid(x, y)
steps = np.dstack((i,j)).reshape((-1,2))


out = prep_simul(sample_shift=True, steps=steps, step_random=False, obj='logo')


pos_flipped = np.max(out['positions'], 0) - out['positions']
window_size = 3
max_shift = 10
model = UMPA.model.UMPAModelDF(sam_list=out['meas'], ref_list=out['ref'], pos_list=pos_flipped, window_size=window_size, max_shift=max_shift)
model.shift_mode=True
fit = model.match()


phase = opt.phase.phaseint_iter(fit['dx'], fit['dy'], numiter=100)


figure(figsize=(10,10))
subplot(221); opt.imsh(fit['dx'], cmap='gray'); title('dx')
subplot(222); opt.imsh(fit['dy'], cmap='gray'); title('dy')
subplot(223); opt.imsh(fit['T'], cmap='gray'); title('df')
subplot(224); opt.imsh(phase, cmap='gray'); title('Int. phase')
tight_layout()

