import os
import numpy as np
from n1.utils_n1 import extcl, cls_dot
from n1.box import box
import pyfftw

class n0_fft:
    def __init__(self, cls_ivfs, cls_w,  lminbox=50, lmaxbox=2500):
        lside = 2. * np.pi / lminbox
        npix = int(lmaxbox / np.pi * lside) + 1
        if npix % 2 == 1: npix += 1

        # ===== instance with 2D flat-sky box info
        self.box = box(lside, npix)
        self.shape = self.box.shape

        # ==== Builds required spectra:
        # === Filter and cls array needed later on:
        cls_ivfs = {k: extcl(self.box.lmaxbox + lminbox, cls_ivfs[k]) for k in cls_ivfs.keys()}  # filtered maps spectra
        cls_w = {k: extcl(self.box.lmaxbox + lminbox, cls_w[k]) for k in cls_w.keys()}  # estimator weights spectra

        self.K_ls = cls_dot([cls_ivfs])
        self.Kw_ls = cls_dot([self.K_ls, cls_w])
        self.wK_ls = cls_dot([cls_w, self.K_ls])
        self.wKw_ls = cls_dot([cls_w, self.Kw_ls])

        self._cos2p_sin2p = None

        # === normalization (for lensing keys at least)
        norm = (self.box.shape[0] / self.box.lsides[0]) ** 2  # overall final normalization from rfft'ing
        norm *= (float(self.box.lminbox)) ** 4
        self.norm = norm

    def _ifft2(self, rm):
        outp = pyfftw.empty_aligned(self.box.shape, dtype='float64')
        inpt = pyfftw.empty_aligned(self.box.rshape, dtype='complex128')
        ifft2 = pyfftw.FFTW(inpt, outp, axes=(0, 1), direction='FFTW_BACKWARD', threads=int(os.environ.get('OMP_NUM_THREADS', 1)))
        return ifft2(pyfftw.byte_align(rm, dtype='complex128'))

    def _X2S(self, S, X, rfft=True):
        """Matrix element sending X cmb mode to stokes flat-sky mode S


        """
        if S == 'T':  return 1. if X == 'T' else 0.
        if S == 'Q':
            if X == 'T': return 0.
            if self._cos2p_sin2p is None:
                self._cos2p_sin2p = self.box.cos2p_sin2p(rfft=rfft)
            sgn = 1 if X == 'E' else -1
            return sgn * self._cos2p_sin2p[0 if X == 'E' else 1]
        if S == 'U':
            if X == 'T': return 0.
            if self._cos2p_sin2p is None:
                self._cos2p_sin2p = self.box.cos2p_sin2p(rfft=rfft)
            return self._cos2p_sin2p[1 if X == 'E' else 0]
        assert 0


    def get_n0_2d(self, k, _pyfftw=True):
        """Returns unormalized QE noise for each 2d multipole on the flat-sky box

            Note:
                ON a square-periodic flat-sky box there can be tiny differences of N0(L) for same |L|

            No attempt is at optimization. see get_n0 method for much faster N0 array calculation

        """
        Fs = np.zeros((3, self.box.shape[0], self.box.shape[1]), dtype=float) # 00, 11 and 01 components
        X2i = {'T': 0, 'E': 1, 'B': 2}
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        ls = self.box.ls()

        Ss = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])
        ir2 = self._ifft2 if _pyfftw else np.fft.irfft2
        for XY in ['TT', 'EE', 'TE', 'ET', 'BB']:  # TT, TE, ET, EE, BB
            X,Y = XY
            i = X2i[X]
            j = X2i[Y]
            K      =       self.K_ls  [i, j][ls]
            wKw_00 =  -1 * self.wKw_ls[i, j][ls] * ny * ny
            wKw_11 =  -1 * self.wKw_ls[i, j][ls] * nx * nx
            wKw_01 =  -1 * self.wKw_ls[i, j][ls] * nx * ny

            Kw_0    = 1j * self.Kw_ls [i, j][ls] * ny
            Kw_1    = 1j * self.Kw_ls [i, j][ls] * nx
            wK_0   =  1j * self.Kw_ls [j, i][ls] * ny
            wK_1   =  1j * self.Kw_ls [j, i][ls] * nx
            for S in Ss:
                for T in Ss:
                    fac = self._X2S(S, X) * self._X2S(T, Y)
                    if np.any(fac):
                        Fs[0] +=     ir2(K * fac)  * ir2(wKw_00 * fac) + ir2(Kw_0 * fac) * ir2(wK_0 * fac)
                        Fs[1] +=     ir2(K * fac)  * ir2(wKw_11 * fac) + ir2(Kw_1 * fac) * ir2(wK_1 * fac)
                        Fs[2] +=     ir2(K * fac)  * ir2(wKw_01 * fac) + ir2(Kw_0 * fac) * ir2(wK_1 * fac)
        Fyy, Fxx, Fxy = np.fft.rfft2(Fs).real
        return - self.norm * (ny ** 2 * Fyy + nx ** 2 * Fxx + 2 * nx * ny * Fxy)

    def get_n0(self, k, _pyfftw=True):
        """Returns unormalized QE noise for multipole along an axis of the box

            This uses 1-dimensional rfft on a subset of the terms


        """
        Fxx = np.zeros(self.box.shape, dtype=float) # 00, 11 and 01 components
        X2i = {'T': 0, 'E': 1, 'B': 2}
        nx = self.box.nx_1d
        ls = self.box.ls()

        Ss = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])
        XYs = ['TT'] * (k in ['ptt', 'p']) + ['EE', 'BB'] * (k in ['p_p', 'p']) + ['ET', 'TE'] * (k == 'p')
        ir2 = self._ifft2 if _pyfftw else np.fft.irfft2
        for XY in XYs:
            X, Y = XY
            i = X2i[X]
            j = X2i[Y]
            K      = self.K_ls[i, j][ls]
            wKw_11 =  -1 * self.wKw_ls[i, j][ls] * nx ** 2
            Kw_1   =  1j * self.Kw_ls [i, j][ls] * nx
            wK_1   =  1j * self.Kw_ls [j, i][ls] * nx
            for i, S in enumerate(Ss):
                X2S = self._X2S(S, X)
                for T in Ss[i:] * (np.any(X2S)):
                    Y2T = self._X2S(T, Y)
                    if np.any(Y2T):
                        fac = X2S * Y2T
                        Fxx  +=  (1 + (S != T)) * ir2(K * fac)  * ir2(wKw_11 * fac) + ir2(Kw_1 * fac) * ir2(wK_1 * fac)
        # 1d fft method using only F11
        return -self.norm * self.box.nx_1d[:-1] ** 2 * np.sum(np.fft.rfft(Fxx).real, axis=0)[:-1], self.box.nx_1d[:-1] * self.box.lminbox





