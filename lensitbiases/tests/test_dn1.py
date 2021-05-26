import numpy as np
import pylab as pl

from lensitbiases import n1_fft
import os
import plancklens
from plancklens.utils import camb_clfile
from lensitbiases import wrappers_lensit as wli
path =  os.path.join(os.path.abspath(os.path.dirname(plancklens.__file__)),'data', 'cls')
cls_grad = camb_clfile(os.path.join(path, 'FFP10_wdipole_gradlensedCls.dat'))
cls_len = camb_clfile(os.path.join(path, 'FFP10_wdipole_lensedCls.dat'))

cls_unl = camb_clfile(os.path.join(path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_weights = camb_clfile(os.path.join(path, 'FFP10_wdipole_gradlensedCls.dat'))
import healpy as hp

jt_TP = False
cf = wli.cmbconf('ptt', jt_TP, 35., 55., hp.gauss_beam(6./180 / 60 *np.pi, lmax=2048), 100, 2048, cls_cmbresponse=cls_len, cls_qeweight=cls_len)
n1_lib = n1_fft.n1_fft(cf.fals, {'tt': cls_len['tt']}, {'tt': cls_len['tt']}, cls_unl['pp'], lminbox=25, lmaxbox=4000)


def dn1(L):
    from plancklens import utils
    n1_lib._destroy_key('ptt')
    n1_lib._build_key('ptt', L, w=0. + 1e-10, sgn=1)  # l1, l2 = l, L-l
    WTT = n1_lib._W_ST('T', 'T')
    WTT_z0 = n1_lib._W_ST('T', 'T', ders_2=0)
    WTT_z1 = n1_lib._W_ST('T', 'T', ders_2=1)

    n1_lib._destroy_key('ptt')
    n1_lib._build_key('ptt', L, w=1. - 1e-10, sgn=1)
    # This absorbs the e^{-i r L/2} into the WTT_az, with l1, l2 = L+l, -l
    WTTr_0z = np.fft.ifft2(n1_lib._W_ST('T', 'T', ders_1=0))
    WTTr_1z = np.fft.ifft2(n1_lib._W_ST('T', 'T', ders_1=1))
    WTTr = np.fft.ifft2(n1_lib._W_ST('T', 'T'))
    n1_lib._destroy_key('ptt')
    n1_lib._build_key('ptt', L, w=1. - 1e-10)  # This absorbs the e^{-i r L/2} into the WTT_az, with l1, l2 = L+l, -l

    n0, n1 = np.meshgrid(n1_lib.box.ny_1d, n1_lib.box.ny_1d, indexing='ij')

    ret1   = 1j * n1 * WTT * np.fft.ifft2(WTTr_0z * n1_lib.xipp[1])
    ret1  += 1j * n0 * WTT * np.fft.ifft2(WTTr_0z * n1_lib.xipp[0])
    ret1 += ret1.T    # Same as, for all S, T:
    #ret1 += 1j * n0 * WTT * np.fft.ifft2(WTTr_1z * n1_lib.xipp[1])
    #ret1 += 1j * n1 * WTT * np.fft.ifft2(WTTr_1z * n1_lib.xipp[0].T)

    ret1 *= 2  # Up to here should be OK

    ret2 =  1j * n0 * WTT_z0 * np.fft.ifft2(WTTr * n1_lib.xipp[0])
    ret2 += 1j * n1 * WTT_z1 * np.fft.ifft2(WTTr * n1_lib.xipp[0].T)
    ret2 +=  2 * 1j * n0 * WTT_z1 * np.fft.ifft2(WTTr * n1_lib.xipp[1])
    #ret2 += 1j * n1 * WTT_z0 * np.fft.ifft2(WTTr * n1_lib.xipp[1])

    n1_lib._destroy_key('ptt')
    n1_lib._build_key('ptt', L, w=1e-10, sgn=1)  # -q, L + q
    WTTr = np.fft.ifft2(n1_lib._W_ST('T', 'T'))

    # This should be just the conj of ret2
    n1_lib._destroy_key('ptt')
    n1_lib._build_key('ptt', L, w=1. - 1e-10, sgn=-1)  # L- q, q
    WTT_0z = n1_lib._W_ST('T', 'T', ders_1=0)  # This is same as above WTTr_0z
    WTT_1z = n1_lib._W_ST('T', 'T', ders_1=1)

    ret3 = ret2.conj() # Same as : (for ST term this becomes TS.conj()
    #ret3 =  1j * n0 * WTT_0z * np.fft.ifft2(WTTr.conj() * n1_lib.xipp[0]).conj()
    #ret3 += 1j * n1 * WTT_1z * np.fft.ifft2(WTTr.conj() * n1_lib.xipp[0].T).conj()
    #ret3 += 1j * n0 * WTT_1z * np.fft.ifft2(WTTr.conj() * n1_lib.xipp[1]).conj()
    #ret3 += 1j * n1 * WTT_0z * np.fft.ifft2(WTTr.conj() * n1_lib.xipp[1]).conj()

    ret =  (- ret1 + ret2 + ret3)
    return [2 * 0.25 * n1_lib.box.sum_in_l(r.real) * n1_lib.norm  for r in [ret, ret1, ret2, ret3]]


Ls = np.arange(50, 2048, 100)
dn = -np.array([dn1(L) for L in Ls])
pl.plot(Ls, np.dot(dn[:, 0, :], cls_len['tt'][:len(dn[0, 0])]))
pl.plot(Ls, np.dot(dn[:, 1, :], cls_len['tt'][:len(dn[0, 0])]))

pl.plot(Ls, np.dot(dn[:, 2, :], cls_len['tt'][:len(dn[0, 0])]))
pl.plot(Ls, np.dot(dn[:, 3, :], cls_len['tt'][:len(dn[0, 0])]))
n1_lib._destroy_key('ptt')

n1_1 = np.array([n1_lib.get_n1('ptt', L, do_n1mat=False) for L in Ls])
pl.figure()
pl.plot(Ls, n1_1)
pl.plot(Ls, np.dot(dn[:, 0, :], cls_len['tt'][:len(dn[0, 0])]))

def get_n1(L, n1_lib):
    Xs = ['T']
    k = 'ptt'
    n1_lib._destroy_key(k)
    n1_lib._build_key(k, L, rfft=False)
    n1 = np.array([0., 0., 0.])
    n1_mat = 0.
    for a in [0, 1]:
        for b in [0, 1]:
            term1 = 0j
            term2 = 0j
            for T in Xs:
                for S in Xs:
                    term1 += np.fft.ifft2(n1_lib._W_ST(T, S, ders_1=a, ders_2=b)) * np.fft.ifft2(n1_lib._W_ST(S, T))
                    term2 += np.fft.ifft2(n1_lib._W_ST(T, S, ders_1=a)) * np.fft.ifft2(n1_lib._W_ST(S, T, ders_1=b))
            xipp = n1_lib.xipp[a + b] if (a + b) != 2 else n1_lib.xipp[0].T
            n1[0] += np.sum(xipp * (term1 - term2).real)
            n1[1] += np.sum(xipp * term1.real)
            n1[2] += np.sum(xipp * term2.real)
    return -n1_lib.norm* n1
n1s = np.array([get_n1(L, n1_lib) for L in Ls])
term1_d =  2 * np.dot(dn[:, 3, :], cls_len['tt'][:len(dn[0, 0])])
pl.figure()
pl.plot(Ls, n1s[:, 1] / term1_d)
term2_d =  np.dot(dn[:, 1, :], cls_len['tt'][:len(dn[0, 0])])
pl.plot(Ls, n1s[:, 2] / term2_d)
pl.axhline(1., c='k', ls='--')
pl.show()