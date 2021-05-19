import numpy as np
import pylab as pl
import lensitbiases as lb
from lensitbiases import n0_fft, n1_fft, utils_n1
from scipy.interpolate import UnivariateSpline as spl


qekey = 'xtt'

cls_unl, cls_len, cls_grad = lb.get_default_cls() # default CMB cls (put here the one you are using)
lminbox=50
lmin_ivf, lmax_ivf, nlevt, nlevp, beam = (100, 2048, 35., 55., 6.5)
# N1 depends on the relation between the filtered and CMB maps (fal matrix):
fal = utils_n1.get_fal(clscmb_filt=cls_len, clscmb_dat=cls_len,
                       lmin_ivf=lmin_ivf, lmax_ivf=lmax_ivf, nlevt=nlevt, nlevp=nlevp, beam=beam, jt_tp=False)[1]
fal['ee'] *= 0.
fal['bb'] *= 0.
cls_len['te'] *= 0
cls_len['ee'] *= 0.
if 'te' in fal.keys(): fal['te'] *= 0.


# Building (unnormalized estimate) N1 on a grid of 50 points
lib_n1 = n1_fft.n1_fft(fal, cls_len, cls_grad, cls_unl['pp'], lminbox=lminbox, lmaxbox=6000, k2l='lensit')
Ls_n1 = np.linspace(lminbox, 2048, 25)
n1_s4like = np.array([lib_n1.get_n1(qekey, L, do_n1mat=False) for L in Ls_n1])

pl.plot(Ls_n1, n1_s4like)
pl.xlabel(r'$L$', fontsize=14)
pl.ylabel(r'$n_L^{(1)}$', fontsize=14)
pl.title(r'Unormalized QE $N^{(1)}$')


# the normalization should be the same as the one used in ones' analysis. Here's one way to get it with this package
(R_gg, R_cc), Ls_r = n0_fft.nhl_fft(fal, cls_grad, lminbox=lminbox, lmaxbox=6000, k2l='lensit').get_nhl(qekey.replace('x', 'p'))  # gradient and curl responses, and multipoles
pl.show()
pl.figure()
R = R_gg if qekey[0] == 'p' else R_cc
pl.plot(Ls_n1, Ls_n1 ** 2 * (Ls_n1 + 1) ** 2 * n1_s4like / spl(Ls_r, R, k=2, s=0, ext='zeros')(Ls_n1) ** 2 * 1e7 / 2. / np.pi)
pl.xlabel(r'$L$', fontsize=14)
pl.ylabel(r'$10^7\cdot L^4 N_L^{(1)} /2\pi$', fontsize=14)
pl.title(r'normalized QE $N^{(1)}$')

pl.show()