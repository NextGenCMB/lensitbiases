r"""This module contains rfft methods for calculation of lensed spectra and grad-spectra


"""
import os
import numpy as np
from n1.utils_n1 import extcl, cls_dot
from n1.box import box
import pyfftw


class len_fft:
    def __init__(self, cls_unl, cpp, lminbox=50, lmaxbox=2500, k2l=None):
        lside = 2. * np.pi / lminbox
        npix = int(2 * lmaxbox / float(lminbox)) + 1
        if npix % 2 == 1: npix += 1

        # === instance with 2D flat-sky box info
        self.box = box(lside, npix, k2l=k2l)
        self.shape = self.box.shape

        # === Filter and cls array needed later on:
        cls_unl = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cls_unl[k]) for k in cls_unl.keys()}  # filtered maps spectra

        self.cunl_ls   = cls_dot([cls_unl])
        # === precalc of deflection corr fct:
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        ls = self.box.ls()
        xipp = np.array([self._ifft2(extcl(self.box.lmaxbox, -cpp)[ls] * ny ** 2),
                         self._ifft2(extcl(self.box.lmaxbox, -cpp)[ls] * nx * ny)])# 01 or 10

        xipp[0] -= xipp[0, 0, 0]
        xipp[1] -= xipp[1, 0, 0]
        self.xipp_m0 = xipp
        # === normalization (for lensing keys at least)
        norm = (self.box.shape[0] / self.box.lsides[0]) ** 2  # overall final normalization from rfft'ing
        norm *= (float(self.box.lminbox)) ** 4
        self.norm = norm

    def _ifft2(self, rm):
        oshape = self.box.shape if rm.ndim == 2 else (rm.shape[0], self.box.shape[0], self.box.shape[1])
        inpt = pyfftw.empty_aligned(rm.shape, dtype='complex128')
        outp = pyfftw.empty_aligned(oshape, dtype='float64')
        ifft2 = pyfftw.FFTW(inpt, outp, axes=(-2, -1), direction='FFTW_BACKWARD', threads=int(os.environ.get('OMP_NUM_THREADS', 1)))
        return ifft2(pyfftw.byte_align(rm, dtype='complex128'))


    def _build_lenmunl_2d_highorder(self, nmax, job='TP', _pyfftw=True, der_axis=None):
        assert job in ['T', 'P', 'TP']
        assert der_axis in [None, 0, 1], der_axis
        assert 3 >= nmax >= 1, (nmax, 'higher orders not really tested and unnecessary')
        ir2 = self._ifft2 if _pyfftw else np.fft.irfft2
        if job == 'T':
            STs = ['TT']
            XYs = ['TT']
            specs = ['tt']
        elif job == 'P':
            STs =['QQ', 'QU', 'UU']
            specs = ['ee', 'bb']
            XYs = ['EE', 'BB']
        elif job == 'TP':
            STs =['TT', 'TQ', 'TU','QQ', 'QU', 'UU']
            specs = ['tt', 'te', 'ee', 'bb']
            XYs = ['TT', 'EE', 'BB', 'TE', 'ET']
        else:
            assert 0
        X2i = {'T': 0, 'E': 1, 'B': 2}
        lenCST = np.zeros((nmax + 1, len(STs), self.box.shape[0], self.box.shape[1]), dtype=float)
        unlCST = np.zeros((len(STs), self.box.rshape[0], self.box.rshape[1]), dtype=float)

        #=== Builds unlensed Stokes matrices to update later on
        ls = self.box.ls()
        for iST, ST in enumerate(STs):
            for XY in XYs:  # === TT, TE, ET, EE, BB at most
                unlCST[iST] += self.cunl_ls[X2i[XY[0]], X2i[XY[1]]][ls] * self.box.X2S(ST[0], XY[0]) * self.box.X2S(ST[1], XY[1])
        nyx = np.array(np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij'))
        if der_axis is not None:
            unlCST = unlCST * (1j * nyx[der_axis])
        # === Perturbatively lensed Stokes spectra
        xism = [self.xipp_m0[0], self.xipp_m0[1], self.xipp_m0[0].transpose()]
        aibi = [(0, 0, 1), (1, 1, 1), (0, 1, 2)]  # axis and mutliplicity count; (0 1) same as (1 0)
        for a1, b1, m1 in aibi * (nmax > 0):
            f1 = xism[a1 + b1]
            int1 = -nyx[a1] * nyx[b1]
            lenCST[0] += m1 * f1 * ir2(unlCST * int1)
            for a2, b2, m2 in aibi * (nmax > 1):
                f2 = xism[a2 + b2]
                int2 = -int1 * nyx[a2] * nyx[b2]
                lenCST[1] +=  (m1 * m2) * f1 * f2 * ir2(unlCST * int2)
                for a3, b3, m3 in aibi * (nmax > 2):
                    f3 = xism[a3 + b3]
                    int3 = -int2 * nyx[a3] * nyx[b3]
                    lenCST[2] += (m1 * m2 * m3) * f1 * f2 * f3 * ir2(unlCST * int3)
                    for a4, b4, m4 in aibi * (nmax > 3):
                        f4 = xism[a4 + b4]
                        int4 = -int3 * nyx[a4] * nyx[b4]
                        lenCST[3] += (m1 * m2 * m3 * m4) * f1 * f2 * f3 * f4 * ir2(unlCST * int4)
        factorial = [1, 1, 2, 6, 24, 120, 720]
        for n, lenCSTn in enumerate(lenCST): # Index 0 is order 1
            lenCSTn *= self.norm ** (n + 1) / factorial[n + 1]  # prefactor for each perturbative order
        #=== Turns lensed Stokes spectra back to T E B:
        lenCST_tot = np.fft.rfft2(np.sum(lenCST, axis=0)).real if der_axis is None else np.fft.rfft2(np.sum(lenCST, axis=0)).imag # norm already
        lencls_tot = dict()
        for spec in specs:
            X, Y = spec.upper()
            lencls_tot[spec] = np.zeros(self.box.rshape, dtype=float)
            for iST, (S, T) in enumerate(STs):  # daig and off-diag
                fac = (1 + (S != T)) *  self.box.X2S(S, X) * self.box.X2S(T, Y)
                if np.any(fac):
                    lencls_tot[spec] += lenCST_tot[iST] * fac
        return lencls_tot, specs


    def lensed_cls_2d(self, job='TP', nmax=1):
        """Returns perturbatively lensed spectra for each and every mode of the 2d flat-sky box

            This calculates the lensed-unlensed part perturbatively, adding eventually the original unlensed part

            Args:
                job: 'T', 'P', or 'TP'(default)  for T-only, P-only or calculation of all spectra
                nmax: perturbative order, defaults to 1 (linear in deflection power)

            Returns:
                dict of 2d map of the lensed spectra, each of the 2d rfft shape


        """
        lencls, specs = self._build_lenmunl_2d_highorder(nmax, job=job)
        X2i = {'T': 0, 'E': 1, 'B': 2}
        ls = self.box.ls()
        for spec in specs:
            X, Y = spec.upper()
            lencls[spec] += self.cunl_ls[X2i[X], X2i[Y]][ls]
        return lencls

    def lensed_gradcls_2d(self, job='TP', nmax=1):
        """Returns perturbatively lensed grad-spectra for each and every mode of the 2d flat-sky box

            This calculates the lensed-unlensed part perturbatively, adding eventually the original unlensed part

            Args:
                job: 'T', 'P', or 'TP' (default) for T-only, P-only or calculation of all spectra
                nmax: perturbative order, defaults to 1 (linear in deflection power)

            Returns:
                dict 2d map of the lensed grad-spectra, ech of the 2d rfft shape


        """
        lencls_0, specs = self._build_lenmunl_2d_highorder(nmax, job=job, der_axis=0)
        lencls_1, specs = self._build_lenmunl_2d_highorder(nmax, job=job, der_axis=1)
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        k2 = ny ** 2 + nx ** 2
        k2[0, 0] = 1. # to avoid dividing by zero
        X2i = {'T': 0, 'E': 1, 'B': 2}
        ls = self.box.ls()
        ret = dict()
        for spec in specs:
            X, Y = spec.upper()
            ret[spec] = self.cunl_ls[X2i[X], X2i[Y]][ls]  + (lencls_0[spec] * ny + lencls_1[spec] * nx) / k2
        return ret







