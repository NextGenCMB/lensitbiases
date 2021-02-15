import os
import numpy as np
from n1.utils_n1 import extcl, cls_dot
from n1.box import box
import pyfftw


class len_fft:
    def __init__(self, cls_unl, lminbox=50, lmaxbox=2500):
        lside = 2. * np.pi / lminbox
        npix = int(2 * lmaxbox / float(lminbox)) + 1
        if npix % 2 == 1: npix += 1

        # ===== instance with 2D flat-sky box info
        self.box = box(lside, npix)
        self.shape = self.box.shape

        # ==== Builds required spectra:
        # === Filter and cls array needed later on:
        cls_unl = {k: extcl(self.box.lmaxbox + lminbox, cls_unl[k]) for k in cls_unl.keys()}  # filtered maps spectra

        self.cunl_ls   = cls_dot([cls_unl])
        # === precalc of deflection corr fct:
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        ls = self.box.ls()
        xipp = np.array([np.fft.irfft2(extcl(self.box.lmaxbox, -cls_unl['pp'])[ls] * ny ** 2),
                         np.fft.irfft2(extcl(self.box.lmaxbox, -cls_unl['pp'])[ls] * nx * ny)])# 01 or 10

        xipp[0] -= xipp[0, 0, 0]
        xipp[1] -= xipp[1, 0, 0]
        self.xipp_m0 = xipp
        # === normalization (for lensing keys at least)
        norm = (self.box.shape[0] / self.box.lsides[0]) ** 2  # overall final normalization from rfft'ing
        norm *= (float(self.box.lminbox)) ** 4
        self.norm = norm

    def _ifft2(self, rm):
        outp = pyfftw.empty_aligned(self.box.shape, dtype='float64')
        inpt = pyfftw.empty_aligned(self.box.rshape, dtype='complex128')
        ifft2 = pyfftw.FFTW(inpt, outp, axes=(0, 1), direction='FFTW_BACKWARD', threads=int(os.environ.get('OMP_NUM_THREADS', 1)))
        return ifft2(pyfftw.byte_align(rm, dtype='complex128'))


    def _build_lenmunl_2d(self, job='TP', _pyfftw=True, der_axis=None):
        assert job in ['T', 'P', 'TP']
        assert der_axis in [None, 0, 1], der_axis
        ir2 = self._ifft2 if _pyfftw else np.fft.irfft2
        if job == 'T':
            Ss = ['T']
            specs = ['tt']
            XYs = ['TT']
            iSiT2i = {(0, 0):0}
        elif job == 'P':
            Ss = ['Q', 'U']
            specs = ['ee', 'bb']
            XYs = ['EE', 'BB']
            iSiT2i = {(0, 0):0, (0, 1):1,
                      (1, 0):1, (1, 1):2}
        elif job == 'TP':
            Ss = ['T', 'Q', 'U']
            specs = ['tt', 'te', 'ee', 'bb']
            XYs = ['TT', 'EE', 'BB', 'TE', 'ET']
            iSiT2i = {(0, 0):0, (0, 1):1, (0, 2):2,
                      (1, 0):1, (1, 1):3, (1, 2):4,
                      (2, 0):2, (2, 1):4, (2, 2):5}
        else:
            assert 0
        X2i = {'T': 0, 'E': 1, 'B': 2}
        nST = (len(Ss) * (len(Ss) + 1)) // 2
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        ls = self.box.ls()
        lenCST =  np.zeros((nST, self.box.shape[0], self.box.shape[1]), dtype=float)
        #=== Perturbatively lensed Stokes spectra
        for iS, S in enumerate(Ss):  # daig and off-diag
            for jT, T in enumerate(Ss[iS:]):
                K = np.zeros( (3,self.box.rshape[0], self.box.rshape[1]), dtype=complex) ## 00 11, 01
                for XY in XYs:  # TT, TE, ET, EE, BB
                    X,Y = XY
                    fac = self.box.X2S(S, X) * self.box.X2S(T, Y)
                    if np.any(fac):
                        i = X2i[X]; j = X2i[Y]
                        clST =  self.cunl_ls[i, j][ls] * fac
                        if der_axis is not None:
                            clST = clST * (1j * nx if der_axis else 1j * ny)
                        K[0]  -= clST * ny * ny
                        K[1]  -= clST * nx * nx
                        K[2]  -= clST * nx * ny
                iST = iSiT2i[(iS, jT + iS)]
                lenCST[iST] +=     ir2(K[0]) * self.xipp_m0[0] + ir2(K[1]) * self.xipp_m0[0].T
                lenCST[iST] += 2 * ir2(K[2]) * self.xipp_m0[1]
        lencCST = self.norm * (np.fft.rfft2(lenCST).real if der_axis is None else np.fft.rfft2(lenCST).imag)
        #=== Turns lensed Stokes spectra back to T E B:
        lencls = dict()
        for spec in specs:
            X, Y = spec.upper()
            lencls[spec] = np.zeros(self.box.rshape, dtype=float)
            for iS, S in enumerate(Ss):
                for iT, T in enumerate(Ss):
                    lencls[spec] += lencCST[iSiT2i[(iS, iT)]] * self.box.X2S(S, X) * self.box.X2S(T, Y)
        return lencls, specs


    def lensed_cls_2d(self, job='TP'):
        """Returns perturbatively lensed spectra for each and every mode of the 2d flat-sky box

            This calculates the lensed-unlensed part perturbatively, adding eventually the original unlensed part

            Args:
                job: 'T', 'P', or 'TP'(default)  for T-only, P-only or calculation of all spectra

            Returns:
                dict of 2d map of the lensed spectra, each of the 2d rfft shape


        """
        lencls, specs = self._build_lenmunl_2d(job=job)
        X2i = {'T': 0, 'E': 1, 'B': 2}
        ls = self.box.ls()
        for spec in specs:
            X, Y = spec.upper()
            lencls[spec] += self.cunl_ls[X2i[X], X2i[Y]][ls]
        return lencls

    def lensed_gradcls_2d(self, job='TP'):
        """Returns perturbatively lensed grad-spectra for each and every mode of the 2d flat-sky box

            This calculates the lensed-unlensed part perturbatively, adding eventually the original unlensed part

            Args:
                job: 'T', 'P', or 'TP' (default) for T-only, P-only or calculation of all spectra

            Returns:
                dict 2d map of the lensed grad-spectra, ech of the 2d rfft shape


        """
        lencls_0, specs = self._build_lenmunl_2d(job=job, der_axis=0)
        lencls_1, specs = self._build_lenmunl_2d(job=job, der_axis=1)
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        r2 = ny ** 2 + nx ** 2
        r2[0, 0] = 1. # to avoid dividing by zero
        X2i = {'T': 0, 'E': 1, 'B': 2}
        ls = self.box.ls()
        ret = dict()
        for spec in specs:
            X, Y = spec.upper()
            ret[spec] = self.cunl_ls[X2i[X], X2i[Y]][ls]  + (lencls_0[spec] * ny + lencls_1[spec] * nx) / r2
        return ret







