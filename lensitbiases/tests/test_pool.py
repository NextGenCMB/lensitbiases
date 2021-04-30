import numpy as np
import time
import pylab as pl
import lensitbiases as lb
from lensitbiases import n0_fft, n1_fft, utils_n1
from scipy.interpolate import UnivariateSpline as spl
from multiprocessing import Pool
cls_unl, cls_len, cls_grad = lb.get_default_cls() # default CMB cls (put here the one you are using)

lminbox, lmaxbox = (14, 4000)
# N1 depends on the relation between the filtered and CMB maps (fal matrix):
fal = utils_n1.get_fal(clscmb_filt=cls_len, clscmb_dat=cls_len,
                       lmin_ivf=30, lmax_ivf=3000, nlevt=1., nlevp=1. * np.sqrt(2.), beam=1., jt_tp=False)[1]

fal['tt'] *= 0  # using polarization only
if 'te' in fal.keys(): fal['te'] *= 0.
lib_n1 = n1_fft.n1_fft(fal, cls_len, cls_grad, cls_unl['pp'], lminbox=lminbox, lmaxbox=lmaxbox)
Ls_n1 = np.linspace(lminbox, 3000, 25)


def f(L_):
    return lib_n1.get_n1('p_p', L_, do_n1mat=False)

if __name__ == '__main__':
    # S4 polarization N1:


    # Building (unnormalized estimate) N1 on a grid of 50 points
    t0 = time.time()
    n1_s4like = np.array([lib_n1.get_n1('p_p', L, do_n1mat=False) for L in Ls_n1])
    dt = time.time()- t0

    t0 = time.time()
    with Pool() as p:
       n1_s4like_pool = p.map(f, Ls_n1)

    dtpool = time.time()- t0
    print(dt, dtpool)
    pl.ioff()
    pl.plot(Ls_n1, n1_s4like)
    pl.plot(Ls_n1, n1_s4like_pool)

    pl.xlabel(r'$L$', fontsize=14)
    pl.ylabel(r'$n_L^{(1)}$', fontsize=14)
    pl.title(r'Unormalized QE $N^{(1)}$')
    pl.show()
    #with Pool() as p:
    #    print(p.map(f, [1, 2, 3]))