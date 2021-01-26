import numpy as np
import pyfftw
import os
import plancklens
from plancklens import utils
import lensit as li
from n1.scripts import test2plancklens as tp

dL2did = lambda L: int(ell_mat.lsides[0] * L / (2 * np.pi))
dnpix2L = lambda npix: 2. * np.pi / ell_mat.lsides[0] * npix
def Freq(i, N):
    """
     Outputs the absolute integers frequencies [0,1,...,N/2,N/2-1,...,1]
     in numpy fft convention as integer i runs from 0 to N-1.
     Inputs can be numpy arrays e.g. i (i1,i2,i3) with N (N1,N2,N3)
                                  or i (i1,i2,...) with N
     Both inputs must be integers.
     All entries of N must be even.
    """
    assert (np.all(N % 2 == 0)), "This routine only for even numbers of points"
    return i - 2 * (i >= (N // 2)) * (i % (N // 2))
def extcl(ell_mat, cl):
    if len(cl) - 1 < ell_mat.ellmax:
        dl = np.zeros( ell_mat.ellmax + 1)
        dl[:len(cl)] = cl
    else:
        dl = cl
    return dl
def cl2ximat(ell_mat, cl):
    if len(cl) - 1 < ell_mat.ellmax:
        dl = np.zeros( ell_mat.ellmax + 1)
        dl[:len(cl)] = cl
    else:
        dl = cl
    return np.fft.irfft2(dl[ELLS])

def get_ellmat(lmin, lmax):
    lside = 2. * np.pi / lmin
    npix = int(lmax  / np.pi * lside)
    if npix % 2 == 1: npix += 1
    from lensit.ffs_covs.ell_mat import  ell_mat
    return ell_mat(None, (npix, npix), (lside, lside), cache=0)

CLS = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')
cls_grad = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_gradlensedCls.dat'))
cls_unl = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_lenspotentialCls.dat'))
ell_mat = get_ellmat(50, 5000)
print('lmin max %s %s '%(int(2. * np.pi /  ell_mat.lsides[0]), ell_mat.ellmax) + str(ell_mat.shape))
fals = tp.get_fal_sTP('PL', 1)[1]

Ft = np.zeros(max(ell_mat.ellmax + 1,  len(fals['tt'])))
Ft[:len(fals['tt'])] = fals['tt'].copy()
lmax_ftl = len(fals['tt']) - 1
print('lmax_ftl ' + str(lmax_ftl))
ELLS = ell_mat()
KX = ell_mat.get_kx_mat()
KY = ell_mat.get_ky_mat()
iKX =ell_mat.get_ikx_mat()
iKY =ell_mat.get_iky_mat()
CTT = extcl(ell_mat, cls_grad['tt'][:lmax_ftl+1])
FT = Ft[ELLS]
XIPP = cl2ximat(ell_mat, cls_unl['pp'][:4096])


def shiftF(F, npix, Laxis):
    # Shifts Fourier coefficients by npix values
    # Fourier rrft with
    #return np.fft.ifftshift(np.roll(np.fft.fftshift(F), npix, axis=Laxis))
    return np.roll(F, npix, axis=Laxis)

def get_N1pyfftw_XY(npixs, xory='x', ks='p', Laxis=0):
    # Defines grid:
    shape = ell_mat.shape
    nx = np.outer(np.ones(shape[0]),  Freq(np.arange(shape[1]), shape[1]))
    ny = np.outer(Freq(np.arange(shape[0]), shape[0]), np.ones(shape[1]))
    ls = np.int_(2 * np.pi * np.sqrt(nx ** 2 + ny ** 2) / ell_mat.lsides[0])
    print(np.min(ls), np.max(ls))
    FX = FY = FI = FJ = Ft[ls]
    ones = np.ones(shape, dtype=float)
    if ks == 'p':
        fresp = []
        fresp += [(CTT[ls] * (nx ** 2 + ny ** 2), ones)]
        fresp += [(CTT[ls] * nx, nx)]
        fresp += [(CTT[ls] * ny, ny)]

        for i in range(3):  # symzation
            fresp += [(fresp[i][1], fresp[i][0])]
    elif ks == 's':
        fresp =  [(ones, ones)]
    else:
        assert 0
    if xory == 'x':
        kA = [(ones, 1j * nx * CTT[ls])]  # input QE A
        kB = [(ones, 1j * nx * CTT[ls])]  # input QE B

    elif xory == 'y':
        kA = [(ones,  1j * ny * CTT[ls])]  # input QE A
        kB = [(ones,  1j * ny * CTT[ls])]  # input QE B
    elif xory == 'xy':
        kA = [(ones, 1j * nx * CTT[ls])]  # input QE A
        kB = [(ones, 1j * ny * CTT[ls])]  # input QE B
    elif xory == 'stt':
        kA = [(ones, ones)]
        kB = [(ones, ones)]
    elif xory == 'ftt':
        kA = [(ones, CTT[ls])]
        kA +=  [(CTT[ls], ones)]
        kB = [(ones, CTT[ls])]
        kB +=  [(CTT[ls], ones)]

    else:
        kA = fresp
        kB = fresp

    ifft2_12 = np.fft.ifft2
    ifft2_34 = np.fft.ifft2
    ifft2_43 = np.fft.ifft2

    #oupt = ifft2(inpt)
    #ifft2 = np.fft.irfft2
    n1_test = np.zeros(len(npixs), dtype=float)
    n1_test_im = np.zeros(len(npixs), dtype=float)
    n1_test1 = np.zeros(len(npixs), dtype=float)
    n1_test2 = np.zeros(len(npixs), dtype=float)
    assert XIPP.shape == shape
    for ip, npix in enumerate(npixs):
        term1 = np.zeros(shape, dtype=complex)
        term2 = np.zeros(shape, dtype=complex)
        for wXY in kA:
            for wIJ in kB:
                for fXI in fresp:
                    w1 = wXY[0] * fXI[0] * FX
                    w3 = wIJ[0] * fXI[1] * FI
                    for fYJ in fresp:
                        w2 = wXY[1] * fYJ[0] * FY
                        w4 = wIJ[1] * fYJ[1] * FJ
                        h12 = ifft2_12(w1 * shiftF(w2, -npix, Laxis).conj())
                        h34 = ifft2_34(w3 * shiftF(w4,  npix, Laxis).conj())
                        term1 += h12 * h34
                for fXJ in fresp:
                    w1 = wXY[0] * fXJ[0] * FX
                    w4 = wIJ[1] * fXJ[1] * FJ
                    for fYI in fresp:
                        #FIXME: under some cond. not necessary to redo h12
                        w2 = wXY[1] * fYI[1] * FY
                        w3 = wIJ[0] * fYI[0] * FI
                        h12 = ifft2_12(w1 * shiftF(w2, -npix, Laxis).conj())
                        h43 = ifft2_43(w4 * shiftF(w3,  npix, Laxis).conj())
                        term2 += h12 * h43
        reim1 = np.sum(XIPP * term1)
        reim2 = np.sum(XIPP * term2)

        n1_test[ip] = reim1.real + reim2.real
        n1_test_im[ip] = reim1.imag + reim2.real
        n1_test1[ip] = reim1.real
        n1_test2[ip] = reim2.real

    return n1_test, n1_test_im, n1_test1, n1_test2, FT, w1, w2, w3, w4