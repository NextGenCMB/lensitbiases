import os
import numpy as np
from lensitbiases.utils_n1 import extcl, cls_dot
from lensitbiases.box import box
import pyfftw
import psutil
try:
    import numexpr
except ImportError:
    numexpr = None


class nhl_fft:
    def __init__(self, cls_ivfs, cls_w, lminbox=50, lmaxbox=2500, k2l=None, cls_w2=None):
        """

            Note:
                for a response calculation set cls_w to the QE qeweights cls, and cls_w2 to the sky response cls (lencls or gradcls typically),
                and ivfs to the filterting matrix B^t Cov^{-1} B  cls (fals)

        """
        lside = 2. * np.pi / lminbox
        npix = int(2 * lmaxbox / float(lminbox)) + 1
        if npix % 2 == 1: npix += 1

        # ===== instance with 2D flat-sky box info
        self.box = box(lside, npix, k2l=k2l)
        self.shape = self.box.shape

        # === Filter and cls array needed later on:

        cls_ivfs = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cls_ivfs[k]) for k in cls_ivfs.keys()}  # filtered maps spectra
        cls_w1 = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cls_w[k]) for k in cls_w.keys()}  # estimator weights spectra
        self._w1_is_w2 = cls_w2 is None
        if cls_w2 is None:
            cls_w2 = cls_w1
        else:
            cls_w2 = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cls_w2[k]) for k in cls_w2.keys()}  # second estimator weights spectra


        K_ls, Kw1_ls, w2K_ls, wKw_sym_ls, w2Kw1_ls = self._build_cl_ls(cls_ivfs, cls_w1, cls_w2)
        self.K_ls   = K_ls
        self.Kw1_ls  = Kw1_ls
        self.w2K_ls  = w2K_ls
        self.w2Kw1_ls = w2Kw1_ls
        self.wKw_sym_ls = wKw_sym_ls
        # We need the symmetric part only of this (there is a trace against symmetric K)

        self.cls_w1 = cls_w1
        self.cls_w2 = cls_w2
        self.cls_ivfs = cls_ivfs

        self._cos2p_sin2p = None
        self.nthreads = int(os.environ.get('OMP_NUM_THREADS', psutil.cpu_count(logical=False)) or 1)

        # === normalization (for lensing keys at least)
        norm = (self.box.shape[0] / self.box.lsides[0]) ** 2  # overall final normalization from rfft'ing
        norm *= (float(self.box.lminbox)) ** 4
        self.norm = norm

    @staticmethod
    def _build_cl_ls(cls_ivfs, cls_w1, cls_w2):
        K_ls   = cls_dot([cls_ivfs])
        Kw1_ls  = cls_dot([cls_ivfs, cls_w1])
        w2K_ls  = cls_dot([cls_w2, cls_ivfs])
        wKw_sym_ls = 0.5 * (cls_dot([cls_w1, cls_ivfs, cls_w2]) + cls_dot([cls_w2, cls_ivfs, cls_w1]))
        # We need the symmetric part only of this (there is a trace against symmetric K)
        w2Kw1_ls = cls_dot([cls_w2, cls_ivfs, cls_w1])
        return K_ls, Kw1_ls, w2K_ls, wKw_sym_ls, w2Kw1_ls


    def _ifft2(self, rm):
        oshape = self.box.shape if rm.ndim == 2 else (rm.shape[0], self.box.shape[0], self.box.shape[1])
        inpt = pyfftw.empty_aligned(rm.shape, dtype='complex128')
        outp = pyfftw.empty_aligned(oshape, dtype='float64')
        ifft2 = pyfftw.FFTW(inpt, outp, axes=(-2, -1), direction='FFTW_BACKWARD', threads=self.nthreads)
        return ifft2(pyfftw.byte_align(rm, dtype='complex128'))

    def _ifft2_cached(self, rm, out=None, normalise=True):
        """Same as _ifft2 but builds the FFTW plan only once per input shape

            If set, out (aligned, e.g. from pyfftw.empty_aligned) receives the result, which avoids a copy.
            normalise=False skips the 1/N normalization of the inverse FFT (N the number of pixels).

        """
        plans = self.__dict__.setdefault('_ifft2_plans', {})
        if rm.shape not in plans:
            oshape = self.box.shape if rm.ndim == 2 else (rm.shape[0], self.box.shape[0], self.box.shape[1])
            inpt = pyfftw.empty_aligned(rm.shape, dtype='complex128')
            outp = pyfftw.empty_aligned(oshape, dtype='float64')
            plans[rm.shape] = (pyfftw.FFTW(inpt, outp, axes=(-2, -1), direction='FFTW_BACKWARD', threads=self.nthreads), outp)
        plan, outp = plans[rm.shape]
        plan.input_array[:] = rm # c2r transforms overwrite their input, so never hand rm itself to FFTW
        if out is not None:
            return plan(output_array=out, normalise_idft=normalise)
        return plan(output_array=outp, normalise_idft=normalise).copy() # outp is reused by the plan



    def get_nhl_2d(self, k, _pyfftw=True, force_old=True):
        """Returns unormalized QE noise for each and every 2d multipole on the flat-sky box

            Note:
                On a square-periodic flat-sky box there can be tiny differences of N0(L) for same |L|

            No attempt is at optimization. see get_nhl method for much faster N0 array calculation

            Note:
                Delegates to _get_nhl_2d_symsafe(_fast) by default (see its docstring): the original
                formulation below is not symmetric in cls_w, cls_w2 in general, and is even wrong
                at matched weights (cls_w2 defaulting to cls_w1, self._w1_is_w2 True) for 'p'-type
                keys with a genuine T-E correlation (see the module's test). Pass force_old=True
                to get the original (unfixed) behavior back, e.g. to reproduce old results.

        """
        if not force_old:
            return self._get_nhl_2d_symsafe_fast(k, _pyfftw=_pyfftw)
        X2i = {'T': 0, 'E': 1, 'B': 2}
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        ls = self.box.ls()

        ir2 = self._ifft2 if _pyfftw else np.fft.irfft2
        Ss = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])
        Ts = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])

        XYs = ['TT'] * (k in ['ptt', 'p']) + ['EE', 'BB'] * (k in ['p_p', 'p']) + ['ET', 'TE'] * (k == 'p')
        Fs = np.zeros((3, self.box.shape[0], self.box.shape[1]), dtype=float) # 00, 11 and 01 components
        for i, S in enumerate(Ss):  # daig and off-diag
            for T in Ts[i:]:
                K      = np.zeros(self.box.rshape, dtype=complex)
                wKw_sym_11 = np.zeros(self.box.rshape, dtype=complex)
                wKw_sym_00 = np.zeros(self.box.rshape, dtype=complex)
                wKw_sym_01 = np.zeros(self.box.rshape, dtype=complex)
                w2K_1   = np.zeros(self.box.rshape, dtype=complex)
                Kw1_1   = np.zeros(self.box.rshape, dtype=complex)
                w2K_0   = np.zeros(self.box.rshape, dtype=complex)
                Kw1_0   = np.zeros(self.box.rshape, dtype=complex)
                for XY in XYs:  # TT, TE, ET, EE, BB for MV or SQE
                    X,Y = XY
                    fac = self.box.X2S(S, X) * self.box.X2S(T, Y)
                    if np.any(fac):
                        if S != T: fac *= np.sqrt(2.)# off-diagonal terms come with factor of 2
                        i = X2i[X]; j = X2i[Y]
                        K      +=       self.K_ls  [i, j][ls] * fac
                        wKw_sym_00 +=  -1 * self.wKw_sym_ls[i, j][ls] * ny * ny * fac
                        wKw_sym_11 +=  -1 * self.wKw_sym_ls[i, j][ls] * nx * nx * fac
                        wKw_sym_01 +=  -1 * self.wKw_sym_ls[i, j][ls] * nx * ny * fac

                        Kw1_0   +=  1j * self.Kw1_ls [i, j][ls] * ny * fac
                        Kw1_1   +=  1j * self.Kw1_ls [i, j][ls] * nx * fac
                        w2K_0   +=  1j * self.w2K_ls [i, j][ls] * ny * fac
                        w2K_1   +=  1j * self.w2K_ls [i, j][ls] * nx * fac
                ir2K = ir2(K)
                Fs[0] +=     ir2K  * ir2(wKw_sym_00) + ir2(Kw1_0) * ir2(w2K_0)
                Fs[1] +=     ir2K  * ir2(wKw_sym_11) + ir2(Kw1_1) * ir2(w2K_1)
                Fs[2] +=     ir2K  * ir2(wKw_sym_01) + ir2(Kw1_0) * ir2(w2K_1)
        Fyy, Fxx, Fxy = np.fft.rfft2(Fs).real
        n0_2d_gg = ny ** 2 * Fyy + nx ** 2 * Fxx + 2 * nx * ny * Fxy    # lensing gradient
        n0_2d_cc = nx ** 2 * Fyy + ny ** 2 * Fxx - 2 * nx * ny * Fxy    # lensing curl

        return - self.norm * np.array([n0_2d_gg, n0_2d_cc])


    def _get_nhl_2d_symsafe(self, k, _pyfftw=True):
        """Returns unormalized QE noise for each and every 2d multipole on the flat-sky box

            Note:
                On a square-periodic flat-sky box there can be tiny differences of N0(L) for same |L|

            No attempt is at optimization. see get_nhl method for much faster N0 array calculation

        """
        X2i = {'T': 0, 'E': 1, 'B': 2}
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        ls = self.box.ls()

        ir2 = self._ifft2 if _pyfftw else np.fft.irfft2
        Ss = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])
        Ts = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])

        XYs = ['TT'] * (k in ['ptt', 'p']) + ['EE', 'BB'] * (k in ['p_p', 'p']) + ['ET', 'TE'] * (k == 'p')
        Fs = np.zeros((4, self.box.shape[0], self.box.shape[1]), dtype=float) # 00, 11, 01 and 10 components
        for i, S in enumerate(Ss):  # daig and off-diag
            for T in Ts: # dropping symmetry in S-T for Cw1 != Cw2 and 'p'-type keys
                K      = np.zeros(self.box.rshape, dtype=complex)
                w2Kw1_11 = np.zeros(self.box.rshape, dtype=complex)
                w2Kw1_00 = np.zeros(self.box.rshape, dtype=complex)
                w2Kw1_01 = np.zeros(self.box.rshape, dtype=complex)
                w2Kw1_10 = np.zeros(self.box.rshape, dtype=complex)
                                
                w2K_1   = np.zeros(self.box.rshape, dtype=complex)
                Kw1_1   = np.zeros(self.box.rshape, dtype=complex)
                w2K_0   = np.zeros(self.box.rshape, dtype=complex)
                Kw1_0   = np.zeros(self.box.rshape, dtype=complex)
                for XY in XYs:  # TT, TE, ET, EE, BB for MV or SQE
                    X,Y = XY
                    fac = self.box.X2S(S, X) * self.box.X2S(T, Y)
                    if np.any(fac):
                        #if S != T: fac *= np.sqrt(2.)# off-diagonal terms come with factor of 2
                        i = X2i[X]; j = X2i[Y]
                        K      +=       self.K_ls  [i, j][ls] * fac
                        w2Kw1_00 +=  -1 * self.w2Kw1_ls[i, j][ls] * ny * ny * fac
                        w2Kw1_11 +=  -1 * self.w2Kw1_ls[i, j][ls] * nx * nx * fac
                        w2Kw1_01 +=  -1 * self.w2Kw1_ls[i, j][ls] * nx * ny * fac
                        #wKw_10 +=  -1 * self.w2Kw1_ls[i, j][ls] * ny * nx * fac: exact same

                        Kw1_0   +=  1j * self.Kw1_ls [i, j][ls] * ny * fac
                        Kw1_1   +=  1j * self.Kw1_ls [i, j][ls] * nx * fac
                        w2K_0   +=  1j * self.w2K_ls [i, j][ls] * ny * fac
                        w2K_1   +=  1j * self.w2K_ls [i, j][ls] * nx * fac
                ir2K = ir2(K)
                w2Kw1_10 = w2Kw1_01
                Fs[0] +=     ir2K  * ir2(w2Kw1_00) + ir2(Kw1_0) * ir2(w2K_0)
                Fs[1] +=     ir2K  * ir2(w2Kw1_11) + ir2(Kw1_1) * ir2(w2K_1)
                Fs[2] +=     ir2K  * ir2(w2Kw1_01) + ir2(Kw1_0) * ir2(w2K_1)
                Fs[3] +=     ir2K  * ir2(w2Kw1_10) + ir2(Kw1_1) * ir2(w2K_0)

        Fyy, Fxx, Fxy, Fyx = np.fft.rfft2(Fs).real
        n0_2d_gg = ny ** 2 * Fyy + nx ** 2 * Fxx + 2 * nx * ny * (Fxy + Fyx) * 0.5    # lensing gradient
        n0_2d_cc = nx ** 2 * Fyy + ny ** 2 * Fxx - 2 * nx * ny * (Fxy + Fyx) * 0.5    # lensing curl

        return - self.norm * np.array([n0_2d_gg, n0_2d_cc])

    def _get_nhl_2d_symsafe_fast(self, k, _pyfftw=True):
        """Same as _get_nhl_2d_symsafe, with each inverse FFT done only once (8 instead of 13 per S, T pair)

            The Fxy and Fyx components only enter through their sum and are accumulated together.
            The ls-lookups of the spectra are done once, and the ny, nx and 1j factors are applied
            once per (S, T) pair after summing over XY (all XY-sums are then real).
            With pyfftw, the inverse FFTs write into preallocated buffers, and Fs is updated in place.
            Uses numexpr for the elementwise operations if available.

        """
        X2i = {'T': 0, 'E': 1, 'B': 2}
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        ls = self.box.ls()

        if _pyfftw:
            bufs = [pyfftw.empty_aligned(self.box.shape, dtype='float64') for _ in range(8)]
            # unnormalized inverse FFTs: every Fs term is a product of two of them, rescaled once at the end
            ir2 = lambda rm, ib: self._ifft2_cached(rm, out=bufs[ib], normalise=False)
            r2 = pyfftw.builders.rfft2(pyfftw.empty_aligned((3, self.box.shape[0], self.box.shape[1]), dtype='float64'),
                                       axes=(-2, -1), threads=self.nthreads)
        else:
            ir2 = lambda rm, ib: np.fft.irfft2(rm)
            r2 = np.fft.rfft2
        Ss = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])
        Ts = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])

        XYs = ['TT'] * (k in ['ptt', 'p']) + ['EE', 'BB'] * (k in ['p_p', 'p']) + ['ET', 'TE'] * (k == 'p')
        # spectra on the 2d box, (K, Kw1, w2K, w2Kw1) for each XY
        cls2d = {XY: [cl[X2i[XY[0]], X2i[XY[1]]][ls] for cl in [self.K_ls, self.Kw1_ls, self.w2K_ls, self.w2Kw1_ls]] for XY in XYs}
        nyny, nxnx, nxny = ny * ny, nx * nx, nx * ny
        Fs = np.zeros((3, self.box.shape[0], self.box.shape[1]), dtype=float) # 00, 11 and 01 + 10 components
        tmp = np.empty(self.box.shape, dtype=float)
        def addprods(F, *abs): # F += a0 * b0 + a1 * b1 + ... without temporaries
            if numexpr is not None:
                ab = {'F': F, **{'a%s' % i: x for i, x in enumerate(abs)}}
                expr = 'F + ' + ' + '.join('a%s * a%s' % (i, i + 1) for i in range(0, len(abs), 2))
                numexpr.evaluate(expr, local_dict=ab, out=F)
            else:
                for i in range(0, len(abs), 2):
                    np.multiply(abs[i], abs[i + 1], out=tmp)
                    F += tmp
        def addfac(F, cl, fac): # F += cl * fac
            if numexpr is not None:
                numexpr.evaluate('F + cl * fac', local_dict={'F': F, 'cl': cl, 'fac': fac}, out=F)
            else:
                F += cl * fac
        for S in Ss:  # daig and off-diag
            for T in Ts: # dropping symmetry in S-T for Cw1 != Cw2 and 'p'-type keys
                K, Kw1, w2K, w2Kw1 = np.zeros((4, self.box.rshape[0], self.box.rshape[1]), dtype=float)
                for XY in XYs:  # TT, TE, ET, EE, BB for MV or SQE
                    X,Y = XY
                    fac = self.box.X2S(S, X) * self.box.X2S(T, Y)
                    if np.any(fac):
                        K_XY, Kw1_XY, w2K_XY, w2Kw1_XY = cls2d[XY]
                        addfac(K, K_XY, fac)
                        addfac(Kw1, Kw1_XY, fac)
                        addfac(w2K, w2K_XY, fac)
                        addfac(w2Kw1, w2Kw1_XY, fac)
                ir2K = ir2(K, 0)
                ir2Kw1_0, ir2Kw1_1 = ir2(1j * ny * Kw1, 1), ir2(1j * nx * Kw1, 2)
                ir2w2K_0, ir2w2K_1 = ir2(1j * ny * w2K, 3), ir2(1j * nx * w2K, 4)
                addprods(Fs[0], ir2K, ir2(-nyny * w2Kw1, 5), ir2Kw1_0, ir2w2K_0)
                addprods(Fs[1], ir2K, ir2(-nxnx * w2Kw1, 5), ir2Kw1_1, ir2w2K_1)
                addprods(Fs[2], ir2K, ir2(-2 * nxny * w2Kw1, 5), ir2Kw1_0, ir2w2K_1, ir2Kw1_1, ir2w2K_0)

        if _pyfftw:
            Fs *= 1. / np.prod(self.box.shape) ** 2
        Fyy, Fxx, Fxy_p_Fyx = r2(Fs).real
        n0_2d_gg = nyny * Fyy + nxnx * Fxx + nxny * Fxy_p_Fyx    # lensing gradient
        n0_2d_cc = nxnx * Fyy + nyny * Fxx - nxny * Fxy_p_Fyx    # lensing curl

        return - self.norm * np.array([n0_2d_gg, n0_2d_cc])


    def get_nhl_ds_2d(self, k, cls_ivfs_dd, _pyfftw=True):
        """Returns unormalized QE noise for each and every 2d multipole on the flat-sky box

            This returns the 'ds' unnormalized expectation (where data spectra do not match sims spectra)
            See e.g. Planck papers. For d~s, twice the output is ~ N0

            ds ~ 1/2 (\bar X S_WF + \bar S X_WF)

            Note:
                On a square-periodic flat-sky box there can be tiny differences of N0(L) for same |L|

            No attempt is at optimization. see get_nhl method for much faster N0 array calculation



        """
        X2i = {'T': 0, 'E': 1, 'B': 2}
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        ls = self.box.ls()

        ir2 = self._ifft2 if _pyfftw else np.fft.irfft2
        Ss = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])
        Ts = ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])

        XYs = ['TT'] * (k in ['ptt', 'p']) + ['EE', 'BB'] * (k in ['p_p', 'p']) + ['ET', 'TE'] * (k == 'p')
        Fs = np.zeros((4, self.box.shape[0], self.box.shape[1]), dtype=float) # 00, 11 and 01 10 components
        Ks_ls, Ksw_ls, wKs_ls, wKsw_sym_ls, _ = self._build_cl_ls(self.cls_ivfs, self.cls_w1, self.cls_w2)
        Kd_ls, Kdw_ls, wKd_ls, wKdw_sym_ls, _ = self._build_cl_ls(cls_ivfs_dd,   self.cls_w1, self.cls_w2)

        for i, S in enumerate(Ss):  # daig and off-diag
            for T in Ts: # not certain about syms in all cases, dropping this
                Kd      = np.zeros(self.box.rshape, dtype=complex)
                Ks      = np.zeros(self.box.rshape, dtype=complex)
                wKdw_sym_11 = np.zeros(self.box.rshape, dtype=complex)
                wKdw_sym_00 = np.zeros(self.box.rshape, dtype=complex)
                wKdw_sym_01 = np.zeros(self.box.rshape, dtype=complex)
                wKsw_sym_11 = np.zeros(self.box.rshape, dtype=complex)
                wKsw_sym_00 = np.zeros(self.box.rshape, dtype=complex)
                wKsw_sym_01 = np.zeros(self.box.rshape, dtype=complex)
                wKd_1   = np.zeros(self.box.rshape, dtype=complex)
                Kdw_1   = np.zeros(self.box.rshape, dtype=complex)
                wKd_0   = np.zeros(self.box.rshape, dtype=complex)
                Kdw_0   = np.zeros(self.box.rshape, dtype=complex)
                wKs_1   = np.zeros(self.box.rshape, dtype=complex)
                Ksw_1   = np.zeros(self.box.rshape, dtype=complex)
                wKs_0   = np.zeros(self.box.rshape, dtype=complex)
                Ksw_0   = np.zeros(self.box.rshape, dtype=complex)
                for XY in XYs:  # TT, TE, ET, EE, BB for MV or SQE
                    X,Y = XY
                    fac = self.box.X2S(S, X) * self.box.X2S(T, Y)
                    if np.any(fac):
                        #if S != T: fac *= np.sqrt(2.)# off-diagonal terms come with factor of 2
                        i = X2i[X]; j = X2i[Y]
                        Kd      +=       Kd_ls  [i, j][ls] * fac
                        wKdw_sym_00 +=  -1 * wKdw_sym_ls[i, j][ls] * ny * ny * fac
                        wKdw_sym_11 +=  -1 * wKdw_sym_ls[i, j][ls] * nx * nx * fac
                        wKdw_sym_01 +=  -1 * wKdw_sym_ls[i, j][ls] * nx * ny * fac

                        Ks      +=       Ks_ls  [i, j][ls] * fac
                        wKsw_sym_00 +=  -1 * wKsw_sym_ls[i, j][ls] * ny * ny * fac
                        wKsw_sym_11 +=  -1 * wKsw_sym_ls[i, j][ls] * nx * nx * fac
                        wKsw_sym_01 +=  -1 * wKsw_sym_ls[i, j][ls] * nx * ny * fac

                        Kdw_0   +=  1j * Kdw_ls [i, j][ls] * ny * fac
                        Kdw_1   +=  1j * Kdw_ls [i, j][ls] * nx * fac
                        wKd_0   +=  1j * wKd_ls [i, j][ls] * ny * fac
                        wKd_1   +=  1j * wKd_ls [i, j][ls] * nx * fac

                        Ksw_0   +=  1j * Ksw_ls [i, j][ls] * ny * fac
                        Ksw_1   +=  1j * Ksw_ls [i, j][ls] * nx * fac
                        wKs_0   +=  1j * wKs_ls [i, j][ls] * ny * fac
                        wKs_1   +=  1j * wKs_ls [i, j][ls] * nx * fac

                ir2Kd = ir2(Kd)
                ir2Ks = ir2(Ks)

                Fs[0] +=     ir2Kd  * ir2(wKsw_sym_00) + ir2(Kdw_0) * ir2(wKs_0)
                Fs[0] +=     ir2Ks  * ir2(wKdw_sym_00) + ir2(Ksw_0) * ir2(wKd_0)

                Fs[1] +=     ir2Kd  * ir2(wKsw_sym_11) + ir2(Kdw_1) * ir2(wKs_1)
                Fs[1] +=     ir2Ks  * ir2(wKdw_sym_11) + ir2(Ksw_1) * ir2(wKd_1)

                Fs[2] += ir2Kd * ir2(wKsw_sym_01) + ir2(Kdw_0) * ir2(wKs_1)
                Fs[2] += ir2Ks * ir2(wKdw_sym_01) + ir2(Ksw_0) * ir2(wKd_1)

                Fs[3] += ir2Kd * ir2(wKsw_sym_01) + ir2(Kdw_1) * ir2(wKs_0)
                Fs[3] += ir2Ks * ir2(wKdw_sym_01) + ir2(Ksw_1) * ir2(wKd_0)

        Fyy, Fxx, Fxy, Fyx = np.fft.rfft2(Fs).real
        n0_2d_gg = ny ** 2 * Fyy + nx ** 2 * Fxx + nx * ny * (Fxy + Fyx)    # lensing gradient
        n0_2d_cc = nx ** 2 * Fyy + ny ** 2 * Fxx - nx * ny * (Fxy + Fyx)    # lensing curl

        return - 0.25 * self.norm * np.array([n0_2d_gg, n0_2d_cc])


    def get_nhl(self, k, _pyfftw=True):
        """Returns unormalized-QE noise for multipole along an axis of the box

            Args:

                k: QE key. Here only 'ptt', 'p_p' and 'p' are supported, for TT, P-only and 'MV' estimators.
                _pyfftw: uses pfttw FFT's by default, falls back to numpy ffts if unset

            Note:

                Depending on the weight and filtered CMB spectra given as input to the instance the output
                matches the 'GMV' or 'SQE' estimator

            Note:

                This assumes (but does not for) that for all spectra :math:`C_\ell^{TB} = C_\ell^{EB} = 0`

            Returns:

                *Unormalized* QE Gaussian noise level, for the lensing gradient and lensing curl mode

            Note:

                To get the true :math:`N_L^{(0)}` this must be multiplied by the normalization (inverse response)
                applied to the estimator, often called :math:`A_L` or :math:`\frac{1}{\mathcal R_L}`

            This uses 1-dimensional rfft on a subset of the terms used for the 2D map


        """

        X2i = {'T': 0, 'E': 1, 'B': 2}
        Ss =  ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])
        Ts =  ['T'] * (k in ['ptt', 'p']) + ['Q', 'U'] * (k in ['p_p', 'p'])
        XYs = ['TT'] * (k in ['ptt', 'p']) + ['EE', 'BB'] * (k in ['p_p', 'p']) + ['ET', 'TE'] * (k == 'p')
        ir2 = self._ifft2 if _pyfftw else np.fft.irfft2

        nx = self.box.nx_1d
        ls = self.box.ls()
        Fxx = np.zeros(self.box.shape, dtype=float) # 00, 11 and 01 components
        for i, S in enumerate(Ss):  # off-diag
            for T in Ts[i:]:
                K = np.zeros(self.box.rshape, dtype=complex)
                wKw_sym_11 = np.zeros(self.box.rshape, dtype=complex)
                w2K_1 = np.zeros(self.box.rshape, dtype=complex)
                Kw1_1 = np.zeros(self.box.rshape, dtype=complex)
                for X, Y in XYs:
                    i = X2i[X]; j = X2i[Y]
                    fac = self.box.X2S(S, X) * self.box.X2S(T, Y) # off-diagonal terms come with factor of 2
                    if np.any(fac):
                        if S != T:
                            fac *= np.sqrt(2.)
                        K      += self.K_ls  [i, j][ls] * fac
                        wKw_sym_11 += self.wKw_sym_ls[i, j][ls] * (-1 * (nx ** 2)) * fac
                        Kw1_1   += self.Kw1_ls [i, j][ls] * (1j * nx) * fac
                        w2K_1   += self.w2K_ls [i, j][ls] * (1j * nx) * fac
                Fxx += (ir2(K) * ir2(wKw_sym_11) + ir2(Kw1_1) * ir2(w2K_1))

        # 1d fft method using only F11
        n0_gg = self.box.nx_1d ** 2 * np.sum(np.fft.rfft(Fxx, axis=1).real, axis=0)  # lensing gradient n0
        n0_cc = self.box.nx_1d ** 2 * np.sum(np.fft.rfft(Fxx, axis=0).real, axis=1)  # lensing curl n0
        return -self.norm * np.array([n0_gg, n0_cc]), np.abs(self.box.nx_1d) * self.box.lminbox






