"""
Check behaviour of two different UMPA versions. Tests whether the output
(dx, dy, df, T, f) for identical parameters is identical (tested with
`numpy.allclose()`).
"""

import UMPA as u
import UMPAold as uo
import numpy as np
import time
from matplotlib.pyplot import *
import optimatools as opt

data = u.utils.prep_simul()
sam, ref = data['meas'], data['ref']
masks = np.ones_like(data['ref'])
N_THREADS = 4

def compare_output(fit1, fit2):
    keys = ['dx', 'dy', 'T', 'df', 'f']
    success = []
    for key in keys:
        same = np.allclose(fit1['dx'], fit2['dx'])
        success.append(same)
        if not same:
            print('%s: %s' % (key, same))
    if np.all(success):
        print("OK")
        return True
    else:
        return False


def run_both(sam, ref, num_threads, df=None, shift_mode=None, options={}):
    if df==True:
        from UMPA.model import UMPAModelDF as mnew
        from UMPAold.model import UMPAModelDF as mold
    elif df==False:
        from UMPA.model import UMPAModelNoDF as mnew
        from UMPAold.model import UMPAModelNoDF as mold
    else:
        print('select df!')
    mnewi = mnew(sam, ref, **options)
    mnewi.shift_mode = shift_mode
    moldi = mold(sam, ref, **options)
    moldi.shift_mode = shift_mode
    start = time.time()
    fitnew = mnewi.match(num_threads=num_threads, quiet=True)
    time_new = time.time() - start
    #print("Time new: %.2fs" % time_new)
    start = time.time()
    fitold = moldi.match(num_threads=num_threads)
    time_old = time.time() - start
    #print("Time old: %.2fs" % time_new)

    ok = compare_output(fitnew, fitold)
    return ok, fitnew, fitold

for df in [True, False]:
    for shift_mode in [True, False]:
        for m in [masks, None]:
            print('df: %s, shift_mode: %s, mask: %s' % (df, shift_mode, m is not None))
            ok, df_nomask_new, df_nomask_old = run_both(
                sam, ref, N_THREADS, df=df, shift_mode=shift_mode,
                options=dict(mask_list=m))

