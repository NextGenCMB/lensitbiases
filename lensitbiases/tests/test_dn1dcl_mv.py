import numpy as np
import pylab as pl
from time import time
from lensitbiases import n1_fft
import os
import plancklens
from plancklens.utils import camb_clfile
from lensitbiases import wrappers_lensit as wli
path =  os.path.join(os.path.abspath(os.path.dirname(plancklens.__file__)),'data', 'cls')
cls_grad = camb_clfile(os.path.join(path, 'FFP10_wdipole_gradlensedCls.dat'))
cls_len = camb_clfile(os.path.join(path, 'FFP10_wdipole_lensedCls.dat'))

cls_unl = camb_clfile(os.path.join(path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_weights = camb_clfile(os.path.join(path, 'FFP10_wdipole_lensedCls.dat'))
import healpy as hp

k = 'p'
jt_TP = k in ['p', 'x']
cf = wli.cmbconf(k.replace('x', 'p'), jt_TP, 35., 55., hp.gauss_beam(6. / 180. / 60. * np.pi, lmax=2048), 100, 2048, cls_cmbresponse=cls_len, cls_qeweight=cls_len)

if k[1:] in ['_p', 'tt']:
    cls_weights['te'] *= 0.
n1_lib = n1_fft.n1_fft(cf.fals, cls_weights, cls_len, cls_unl['pp'], lminbox=50, lmaxbox=2600)



Ls = np.linspace(50, 2048, 40)
lmax = 2048
dnmat = np.zeros((len(Ls), 4, lmax + 1), dtype=float)

dn = n1_lib.get_dn1(k, Ls[0]) # fftw planning
t0 = time()
for iL, L in enumerate(Ls):
    dn = n1_lib.get_dn1(k, L)
    dnmat[iL, 0, :] = dn['tt'][:lmax + 1]
    dnmat[iL, 1, :] = dn['te'][:lmax + 1]
    dnmat[iL, 2, :] = dn['ee'][:lmax + 1]
    dnmat[iL, 3, :] = dn['bb'][:lmax + 1]
print(' dN1 time per point %.2fs'%( (time()-t0) / len(Ls)))
t0 = time()
n1 = np.array([n1_lib.get_n1(k, L, do_n1mat=False) for L in Ls])
print(' N1 time per point %.2fs'%( (time()-t0) / len(Ls)))

pl.plot(Ls, n1)
n1_emp_tt = np.array([np.dot(dnmat[i, 0], cls_len['tt'][:lmax + 1]) for i in range(len(Ls))])
n1_emp_te = np.array([np.dot(dnmat[i, 1], cls_len['te'][:lmax + 1]) for i in range(len(Ls))])
n1_emp_ee = np.array([np.dot(dnmat[i, 2], cls_len['ee'][:lmax + 1]) for i in range(len(Ls))])
n1_emp_bb = np.array([np.dot(dnmat[i, 2], cls_len['bb'][:lmax + 1]) for i in range(len(Ls))])

pl.plot(Ls, n1_emp_tt, ls='--', label='tt')
pl.plot(Ls, n1_emp_te, ls='--', label='te')
pl.plot(Ls, n1_emp_ee, ls='--', label='ee')
pl.plot(Ls, n1_emp_bb, ls='--', label='bb')
pl.plot(Ls, n1_emp_tt + n1_emp_te + n1_emp_ee + n1_emp_bb, c='k', ls='-.')
pl.legend()
pl.title(k)
pl.figure()
pl.plot(Ls, (n1_emp_tt + n1_emp_te + n1_emp_ee + n1_emp_bb) / n1 -1., c='k')
pl.plot(Ls, n1_emp_tt / n1 -1., label='tt')
pl.plot(Ls, n1_emp_te / n1 -1., label='te')
pl.plot(Ls, n1_emp_ee / n1 -1., label='ee')
pl.plot(Ls, n1_emp_bb / n1 -1., label='bb')
pl.legend()
pl.title(k)

pl.semilogx()
pl.figure()
pl.plot(Ls, (n1_emp_tt + n1_emp_te + n1_emp_ee + n1_emp_bb) / n1 -1.)
pl.show()
pl.title(k)
