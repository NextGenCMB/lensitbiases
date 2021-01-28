import numpy as np
import pyfftw
import os
import plancklens
from plancklens import utils
from scipy.interpolate import UnivariateSpline as spl
from n1.scripts import test2plancklens_1002000 as tp

CLS = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')


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
def extcl(lmax, cl):
    if len(cl) - 1 < lmax:
        dl = np.zeros(lmax + 1)
        dl[:len(cl)] = cl
    else:
        dl = cl
    return dl

def get_ellmat(lmin, lmax):
    lside = 2. * np.pi / lmin
    npix = int(lmax  / np.pi * lside)
    if npix % 2 == 1: npix += 1
    from lensit.ffs_covs.ell_mat import  ell_mat
    return ell_mat(None, (npix, npix), (lside, lside), cache=0)


def get_fXY(k, ls):
    pass

def shiftF(F, npix, Laxis):
    # Shifts Fourier coefficients by npix values
    # Fourier rrft with
    #return np.fft.ifftshift(np.roll(np.fft.fftshift(F), npix, axis=Laxis))
    return np.roll(F, npix, axis=Laxis)

def get_N1pyfftw_XY(npixs, fals, cls_grad, xory='x', ks='p', Laxis=0,
                    lminbox=50, lmaxbox=5000, lminphi=0, lmaxphi=4096,
                    cpp=None, spline_result=False,
                    fftw=False, use_sym=True):
    # Defines grid:
    #FIXME: the way to get the l-L part assumes conj. prop. of the weights
    #FIXME: rffts when possible


    rFFT = False
    assert not rFFT, 'check carefully implications'

    lside = 2. * np.pi / lminbox
    npix = int(lmaxbox  / np.pi * lside)
    if npix % 2 == 1: npix += 1
    shape  = (npix, npix)
    rshape = (npix, npix // 2 + 1)
    fft_shape = rshape if rFFT else shape
    norm = 0.25 * (npix / lside) ** 4 #overall final normalization

    nx = np.outer(np.ones(shape[0]), Freq(np.arange(fft_shape[1]), shape[1]))
    ny = np.outer(Freq(np.arange(shape[0]), shape[0]), np.ones(fft_shape[1]))
    ls = np.int_(np.round(2 * np.pi * np.sqrt(nx ** 2 + ny ** 2) / lside))
    lmax_seen = ls[npix // 2, npix // 2]

    if fftw:
        def ifft(arr):
            inpt = pyfftw.byte_align(arr, dtype='complex128')
            #return pyfftw.interfaces.numpy_fft.irfft2(inpt) if rFFT else pyfftw.interfaces.numpy_fft.ifft2(inpt)
            outp = pyfftw.empty_aligned(shape, dtype='float64' if rFFT else 'complex128')
            fft_ob = pyfftw.FFTW(inpt, outp, axes=(0, 1), direction='FFTW_BACKWARD', flags=['FFTW_MEASURE'])
            fft_ob()
            return outp
    else:
        ifft = np.fft.irfft2 if rFFT else np.fft.ifft2


    print('lmin max %s %s ' % (int(2. * np.pi / lside), lmax_seen) + str(shape))
    CTT = extcl(lmax_seen, cls_grad['tt'])
    if cpp is None :
        cls_unl = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_lenspotentialCls.dat'))
        cpp = cls_unl['pp'][:lmaxphi + 1]

    cpp[lmaxphi + 1:] *= 0.
    cpp[:lminphi] *= 0.
    xipp = np.fft.irfft2(extcl(lmax_seen,cpp)[ls if rFFT else ls[:, 0:rshape[1]]])


    dL2did = lambda L: int(lside * L / (2 * np.pi))
    dnpix2L = lambda n: 2. * np.pi / lside * n

    ones = np.ones(fft_shape, dtype=float)
    if ks[0] == 'p':
        fresp = []
        fresp += [(CTT[ls] * (nx ** 2 + ny ** 2), ones)]
        fresp += [(CTT[ls] * nx, nx)]
        fresp += [(CTT[ls] * ny, ny)]
        for i in range(3):  # symzation
            fresp += [(fresp[i][1], fresp[i][0])]
        norm *= (2 * np.pi / lside) ** 2

    elif ks[0] == 's':
        fresp =  [(ones, ones)]
    elif ks[0] == 'f':
        fresp = [(ones, CTT[ls])]
        fresp +=  [(CTT[ls], ones)]
    else:
        assert 0
    if xory == 'x':
        kA = [ [1., 1j * nx * CTT[ls]] ]  # input QE A
        kB = [ [1., 1j * nx * CTT[ls]] ]  # input QE B

    elif xory == 'y':
        kA = [[1.,  1j * ny * CTT[ls]]]  # input QE A
        kB = [[1.,  1j * ny * CTT[ls]]]  # input QE B
    elif xory == 'xy':
        kA = [[1., 1j * nx * CTT[ls]]]  # input QE A
        kB = [[1., 1j * ny * CTT[ls]]]  # input QE B
    elif xory == 'stt':
        kA = [[1., 1.]]
        kB = [[1., 1.]]
    elif xory == 'ftt':
        kA = [[1., CTT[ls]]]
        kA +=  [[CTT[ls], 1.]]
        kB = [[1., CTT[ls]]]
        if use_sym: #FIXME: figure out better the symmetries that can be used
            norm *= 2.
            use_sym = False
        else:
            kB +=  [[CTT[ls], 1.]]

    else:
        print('using gl res')
        kA = fresp
        kB = fresp

    fresp2 = fresp
    if use_sym and len(fresp) > 1:
        norm *= 2
        print(len(fresp))
        fresp2 = fresp[::len(fresp) // 2]
    # Weigthing by filters:
    FX = FY = FI = FJ = extcl(lmax_seen, fals['tt'])[ls]
    for WXY in kA:
        WXY[0] *= FX
        WXY[1] *= FY
    for WIJ in kB:
        WIJ[0] *= FI
        WIJ[1] *= FJ

    n1_test = np.zeros(len(npixs), dtype=float)
    n1_test_im = np.zeros(len(npixs), dtype=float)
    n1_test1 = np.zeros(len(npixs), dtype=float)
    n1_test2 = np.zeros(len(npixs), dtype=float)

    for ip, npix in enumerate(npixs):
        term1 = np.zeros(shape, dtype=float if rFFT else complex)
        term2 = np.zeros(shape, dtype=float if rFFT else complex)
        for wXY in kA:
            for wIJ in kB:
                for fXI in fresp:
                    w1 = wXY[0] * fXI[0]
                    w3 = wIJ[0] * fXI[1]
                    for fYJ in fresp2:
                        w2 = wXY[1] * fYJ[0]
                        w4 = wIJ[1] * fYJ[1]
                        h12 = ifft(w1 * shiftF(w2, npix, Laxis).conj())
                        h34 = ifft(w3 * shiftF(w4, -npix, Laxis).conj())
                        term1 += h12 * h34
                for fXJ in fresp:
                    w1 = wXY[0] * fXJ[0]
                    w4 = wIJ[1] * fXJ[1]
                    for fYI in fresp2:
                        #FIXME: under some cond. not necessary to redo h12
                        w2 = wXY[1] * fYI[0]
                        w3 = wIJ[0] * fYI[1]
                        h12 = ifft(w1 * shiftF(w2, npix, Laxis).conj())
                        h43 = ifft(w4 * shiftF(w3,  -npix, Laxis).conj())
                        term2 += h12 * h43
        reim1 = np.sum(xipp * term1)
        reim2 = np.sum(xipp * term2)

        n1_test[ip] = reim1.real + reim2.real
        n1_test_im[ip] = reim1.imag + reim2.imag
        n1_test1[ip] = reim1.real
        n1_test2[ip] = reim2.real

    Lspix = dnpix2L(npixs) # L corresponding to the given deflections
    if not spline_result:
        return norm * n1_test, Lspix, n1_test1, n1_test2
    else:
        Lsout = np.arange(0, lmaxphi + 1)
        return spl(Lspix, norm * n1_test, ext='zeros', k=3, s=0)(Lsout), Lsout, n1_test1, n1_test2

