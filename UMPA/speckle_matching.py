# -*- coding: utf-8 -*-
"""
Speckle matching using a local minimisation routine implemented in c.

Author: Pierre Thibault
Date: April 2019
"""

from . import model


def match(Isample, Iref, Nw, mask=None, step=1, max_shift=4, df=True):
    """
    Speckle matching using the UMPA algorithm (Zdora et al PRL 2017, http://dx.doi.org/10.1103/PhysRevLett.118.203903).

    Parameters
    ----------
    Isample: stack of image frames containing the sample.
    Iref: stack of image frames without sample
    Nw: The width of the analysing window is 2*Nw + 1.
    mask: optional mask (or stack of masks) used in the analysis.
    step: Step size in pixels for the analysis (default is 1)
    max_shift: maximum allowed speckle displacement (THIS PARAMETER IS CURRENTLY IGNORED)
    df: If true, use the dark-field model.

    Returns
    -------
    Result dictionary:
     - 'T': transmission image
     - 'dx': differential phase contrast in x
     - 'dy': differential phase contrast in y
     - 'df': dark-field (will be 1 if not computed)
     - 'f': value of the cost function at the minimum. Can be used as a measure of goodness-of-fit.
    """
    if any(not x.flags.c_contiguous for x in Isample):
        print('Warning: provided list of sample frames are not c contiguous - working with a copy.')
        Isample = [x.copy() for x in Isample]

    if any(not x.flags.c_contiguous for x in Iref):
        print('Warning: provided list of reference frames are not c contiguous - working with a copy.')
        Iref = [x.copy() for x in Iref]

    if df:
        PM = model.UMPAModelDF(sam_list=Isample, ref_list=Iref, mask_list=mask, window_size=Nw)
    else:
        PM = model.UMPAModelNoDF(sam_list=Isample, ref_list=Iref, mask_list=mask, window_size=Nw)

    return PM.match(step=step)


def match_unbiased(Isample, Iref, Nw, mask=None, step=1, max_shift=4, df=True, bias=True):
    """
    Speckle matching including bias correction.
    """
    if bias is True:
        # First run analysis on Iref
        if df:
            PMref = model.UMPAModelDF(sam_list=Iref, ref_list=Iref, mask_list=mask, window_size=Nw)
        else:
            PMref = model.UMPAModelNoDF(sam_list=Iref, ref_list=Iref, mask_list=mask, window_size=Nw)
        bias_result = PMref.match(step=step)
        dx = bias_result['dx']
        dy = bias_result['dy']
    elif bias is False:
        # Nothing to do
        dx = 0.
        dy = 0.
    else:
        dx, dy = bias

    result = match(Isample=Isample, Iref=Iref, Nw=Nw, mask=mask, step=step, max_shift=max_shift, df=df)
    result['dx'] -= dx
    result['dy'] -= dy

    return result


def test_gaussian_abc(Nw=2, step=10, max_shift=4):
    import time
    import numpy as np
    from . import utils as u
    s = u.prep_simul()
    PM = model.UMPAModelDFKernel(sam_list=s['meas'], ref_list=s['ref'], mask_list=None, window_size=Nw)
    sh = PM.sh
    abc = np.zeros(sh+(3,))
    abc[:, :, 0] = .1
    abc[:, :, 2] = .1
    result = PM.match(step, max_shift, abc)
    # For a single pixel:
    # PM.min(i, j, a, b, c)
    # or
    # PM.cost(i, j, sx, sy, a, b, c)
    return result


def test(Nw=1, step=1, max_shift=4):
    import time
    import numpy as np
    from . import utils as u
    s = u.prep_simul()

    t0 = time.time()
    r1 = match(s['meas'], s['ref'], Nw, step=step, max_shift=max_shift)
    print('%s: "match" completed in %f seconds' % (__name__, time.time() - t0))

    t0 = time.time()
    r2 = match_unbiased(s['meas'], s['ref'], Nw, step=step, max_shift=max_shift)
    print('%s: "match_unbiased" completed in %f seconds' % (__name__, time.time() - t0))

    # Single precision
    smeas_f, sref_f = s['meas'].astype(np.single), s['ref'].astype(np.single)

    t0 = time.time()
    r1_f = match(smeas_f, sref_f, Nw, step=step, max_shift=max_shift)
    print('%s: "match" (single precision) completed in %f seconds' % (__name__, time.time() - t0))

    t0 = time.time()
    r2_f = match_unbiased(smeas_f, sref_f, Nw, step=step, max_shift=max_shift)
    print('%s: "match_unbiased" completed in %f seconds' % (__name__, time.time() - t0))

    # Simulate mask
    sh = s['ref'][0].shape
    np.random.seed(15)
    mask = (np.random.uniform(size=sh) < .95).astype(float)
    #mask = np.ones_like(s['ref'][0])

    mask_list = [u.pshift(mask, p) for p in s['positions']]

    t0 = time.time()
    r3 = match(s['meas'], s['ref'], Nw, mask=mask_list, step=step, max_shift=max_shift)
    print('%s: "match" with mask completed in %f seconds' % (__name__, time.time() - t0))

    t0 = time.time()
    r4 = match_unbiased(s['meas'], s['ref'], Nw, mask=mask_list, step=step, max_shift=max_shift)
    print('%s: "match_unbiased" with mask completed in %f seconds' % (__name__, time.time() - t0))

    mask_f = np.array(mask_list).astype(np.single)

    t0 = time.time()
    r5 = match(smeas_f, sref_f, Nw, mask=mask_f, step=step, max_shift=max_shift)
    print('%s: "match" (float) with mask completed in %f seconds' % (__name__, time.time() - t0))

    t0 = time.time()
    r6 = match_unbiased(smeas_f, sref_f, Nw, mask=mask_f, step=step, max_shift=max_shift)
    print('%s: "match_unbiased" (float) with mask completed in %f seconds' % (__name__, time.time() - t0))

    # Test new model map
    t0 = time.time()
    r7 = cutils.model_map(step, r1['dx'], r1['dy'], max_shift=max_shift)
    print('%s: "model_map" with mask completed in %f seconds' % (__name__, time.time() - t0))
