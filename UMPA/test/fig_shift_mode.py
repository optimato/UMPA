import UMPA as u
import numpy as np
from matplotlib.pyplot import *

uopts = dict(window_size=2, max_shift=5, ROI=[(100,200,2), (100,200,2)])

data = u.utils.prep_simul(step_size=15, step_random=True)
m_f = u.model.UMPAModelDF(data['meas'], data['ref'], **uopts)
m_f.shift_mode=False
m_fb = u.model.UMPAModelDF(data['ref'], data['ref'], **uopts)
m_fb.shift_mode=False
fit_f = m_f.match(num_threads=4)
fit_fb = m_fb.match(num_threads=4)

m_t = u.model.UMPAModelDF(data['meas'], data['ref'], **uopts)
m_t.shift_mode=True
m_tb = u.model.UMPAModelDF(data['ref'], data['ref'], **uopts)
m_tb.shift_mode=True
fit_t = m_t.match(num_threads=4)
fit_tb = m_tb.match(num_threads=4)

sl = np.s_[50:-50, 100:-100]
dx_false = fit_f['dx'][sl] - fit_fb['dx'][sl]
dx_true  = fit_t['dx'][sl] - fit_tb['dx'][sl]
figure(figsize=(8,4))
subplot(121); title('shift_mode=False')
imshow(dx_false, vmin=-0.4, vmax=0.4, cmap='gray'); xticks([]); yticks([])
subplot(122); title('shift_mode=True')
imshow(dx_true, vmin=-0.4, vmax=0.4, cmap='gray'); xticks([]); yticks([])
tight_layout()
#savefig('/home/fabio/Documents/UMPA_paper/fig/shift_mode_umpa.pdf', dpi=300)



#uopts = dict(window_size=2, max_shift=5, ROI=[(0,400,2), (0,400,2)])
uopts = dict(window_size=2, max_shift=5, ROI=np.s_[:-50:2, :-50:2])
data = u.utils.prep_simul(step_size=15, step_random=True)
m_f = u.model.UMPAModelDF(data['meas'], data['ref'], **uopts)
m_f.shift_mode=False
fit_f = m_f.match(num_threads=4)
figure(); imshow(fit_f['dx'])