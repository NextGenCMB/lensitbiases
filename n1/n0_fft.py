import os
import numpy as np
from n1.utils_n1 import extcl, cls_dot
from n1.box import box
import pyfftw


class nhl_fft:
    def __init__(self, cls_ivfs, cls_w, lminbox=50, lmaxbox=2500):
        lside = 2. * np.pi / lminbox
        npix = int(2 * lmaxbox / float(lminbox)) + 1
        if npix % 2 == 1: npix += 1

        # ===== instance with 2D flat-sky box info
        self.box = box(lside, npix)
        self.shape = self.box.shape

        # ==== Builds required spectra:
        # === Filter and cls array needed later on:
        cls_ivfs = {k: extcl(self.box.lmaxbox + lminbox, cls_ivfs[k]) for k in cls_ivfs.keys()}  # filtered maps spectra
        cls_w = {k: extcl(self.box.lmaxbox + lminbox, cls_w[k]) for k in cls_w.keys()}  # estimator weights spectra

        self.K_ls   = cls_dot([cls_ivfs])
        self.Kw_ls  = cls_dot([cls_ivfs, cls_w])
        self.wK_ls  = cls_dot([cls_w, cls_ivfs])
        self.wKw_ls = cls_dot([cls_w, cls_ivfs, cls_w])

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


    def get_nhl_2d(self, k, _pyfftw=True):
        """Returns unormalized QE noise for each 2d multipole on the flat-sky box

            Note:
                ON a square-periodic flat-sky box there can be tiny differences of N0(L) for same |L|

            No attempt is at optimization. see get_nhl method for much faster N0 array calculation

        """
        Fs = np.zeros((3, self.box.shape[0], self.box.shape[1]), dtype=float) # 00, 11 and 01 components
        X2i = {'T': 0, 'E': 1, 'B': 2}
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        ls = self.box.ls()

        ir2 = self._ifft2 if _pyfftw else np.fft.irfft2
        Ss = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])
        Ts = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])

        XYs = ['TT'] * (k in ['ptt', 'p']) + ['EE', 'BB'] * (k in ['p_p', 'p']) + ['ET', 'TE'] * (k == 'p')
        for S in Ss:
            for T in Ts:
                K = np.zeros(self.box.rshape, dtype=complex)
                wKw_11 = np.zeros(self.box.rshape, dtype=complex)
                wKw_00 = np.zeros(self.box.rshape, dtype=complex)
                wKw_01 = np.zeros(self.box.rshape, dtype=complex)
                wK_1 = np.zeros(self.box.rshape, dtype=complex)
                Kw_1 = np.zeros(self.box.rshape, dtype=complex)
                wK_0 = np.zeros(self.box.rshape, dtype=complex)
                Kw_0 = np.zeros(self.box.rshape, dtype=complex)
                for XY in XYs:  # TT, TE, ET, EE, BB
                    X,Y = XY
                    fac = self._X2S(S, X) * self._X2S(T, Y)
                    if np.any(fac):
                        i = X2i[X];j = X2i[Y]
                        K      +=       self.K_ls  [i, j][ls] * fac
                        wKw_00 +=  -1 * self.wKw_ls[i, j][ls] * ny * ny * fac
                        wKw_11 +=  -1 * self.wKw_ls[i, j][ls] * nx * nx * fac
                        wKw_01 +=  -1 * self.wKw_ls[i, j][ls] * nx * ny * fac

                        Kw_0   += 1j * self.Kw_ls [i, j][ls] * ny * fac
                        Kw_1   += 1j * self.Kw_ls [i, j][ls] * nx * fac
                        wK_0   +=  1j * self.Kw_ls [j, i][ls] * ny * fac
                        wK_1   +=  1j * self.Kw_ls [j, i][ls] * nx * fac

                Fs[0] +=     ir2(K)  * ir2(wKw_00) + ir2(Kw_0) * ir2(wK_0)
                Fs[1] +=     ir2(K)  * ir2(wKw_11) + ir2(Kw_1) * ir2(wK_1)
                Fs[2] +=     ir2(K)  * ir2(wKw_01) + ir2(Kw_0) * ir2(wK_1)
        Fyy, Fxx, Fxy = np.fft.rfft2(Fs).real
        n0_2d_gg = ny ** 2 * Fyy + nx ** 2 * Fxx + 2 * nx * ny * Fxy    # lensing gradient
        n0_2d_cc = nx ** 2 * Fyy + ny ** 2 * Fxx - 2 * nx * ny * Fxy    # lensing curl

        return - self.norm * np.array([n0_2d_gg, n0_2d_cc])

    def get_nhl(self, k, _pyfftw=True):
        """Returns unormalized-QE noise for multipole along an axis of the box

            Args:

                k: QE key. Here only 'ptt', 'p_p' and 'p' are supported, for TT, P-only and 'MV' estimators.
                _pyfftw: uses pfttw FFT's by default, falls back to numpy ffts if unset

            Note:

                Depending on the weight and filtered CMB spectra given as input to the instance the output
                can be made to to match the 'GMV' or 'SQE' estimator

            Note:

                This assumes (but does not for) that for all spectra :math:`C_\ell^{TB} = C_\ell^{EB} = 0`

            Returns:

                *Unormalized* QE Gaussian noise level, for the lensing gradient and lensing curl mode

            Note:

                To get the true :math:`N_L^{(0)}` this must be multiplied by the normalization (inverse response)
                applied to the estimator, often called :math:`A_L` or :math:`\frac{1}{\mathcal R_L}`

            This uses 1-dimensional rfft on a subset of the terms used for the 2D map


        """
        Fxx = np.zeros(self.box.shape, dtype=float) # 00, 11 and 01 components
        nx = self.box.nx_1d
        ls = self.box.ls()

        X2i = {'T': 0, 'E': 1, 'B': 2}
        Ss =  ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])
        Ts =  ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])
        XYs = ['TT'] * (k in ['ptt', 'p']) + ['EE', 'BB'] * (k in ['p_p', 'p']) + ['ET', 'TE'] * (k == 'p')
        ir2 = self._ifft2 if _pyfftw else np.fft.irfft2

        for S, T in zip(Ss, Ts):  # diag
            K = np.zeros(self.box.rshape, dtype=complex)
            wKw_11 = np.zeros(self.box.rshape, dtype=complex)
            wK_1 = np.zeros(self.box.rshape, dtype=complex)
            Kw_1 = np.zeros(self.box.rshape, dtype=complex)
            for X, Y in XYs:
                i = X2i[X]; j = X2i[Y]
                fac = self._X2S(S, X) * self._X2S(T, Y)
                if np.any(fac):
                    K      +=  self.K_ls  [i, j][ls] * fac
                    wKw_11 +=  self.wKw_ls[i, j][ls] * (-1 * (nx ** 2)) * fac
                    Kw_1   +=  self.Kw_ls [i, j][ls] * (1j * nx) * fac
                    wK_1   +=  self.Kw_ls [j, i][ls] * (1j * nx) * fac
            Fxx += (ir2(K) * ir2(wKw_11) + ir2(Kw_1) * ir2(wK_1))

        for i, S in enumerate(Ss):  # off-diag
            for T in Ts[i + 1:]:
                K = np.zeros(self.box.rshape, dtype=complex)
                wKw_11 = np.zeros(self.box.rshape, dtype=complex)
                wK_1 = np.zeros(self.box.rshape, dtype=complex)
                Kw_1 = np.zeros(self.box.rshape, dtype=complex)
                for X, Y in XYs:
                    i = X2i[X]
                    j = X2i[Y]
                    fac = self._X2S(S, X) * self._X2S(T, Y)
                    if np.any(fac):
                        K += self.K_ls[i, j][ls] * fac
                        wKw_11 += self.wKw_ls[i, j][ls] * (-1 * (nx ** 2)) * fac
                        Kw_1 += self.Kw_ls[i, j][ls] * (1j * nx) * fac
                        wK_1 += self.Kw_ls[j, i][ls] * (1j * nx) * fac
                Fxx += 2. * (ir2(K) * ir2(wKw_11) + ir2(Kw_1) * ir2(wK_1))

        # 1d fft mehod using only F11
        n0_gg = self.box.nx_1d ** 2 * np.sum(np.fft.rfft(Fxx, axis=1).real, axis=0)  # lensing gradient n0
        n0_cc = self.box.nx_1d ** 2 * np.sum(np.fft.rfft(Fxx, axis=0).real, axis=1)  # lensing curl n0
        return -self.norm * np.array([n0_gg, n0_cc]), np.abs(self.box.nx_1d) * self.box.lminbox






