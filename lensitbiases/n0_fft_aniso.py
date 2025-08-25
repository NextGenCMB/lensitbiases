import os
import numpy as np
from lensitbiases.utils_n1 import extcl, cls_dot
from lensitbiases.box import box
import pyfftw


class nhl_fft:
    def __init__(self, cls_noise, cls_noise_filt, cls_w, transf_cl, lminbox=50, lmaxbox=2500, lx_cut=0, 
                 iso_filt=False, _iso_dat=False, _response=False, k2l=None, cls_w2=None):
        """
         
         More flexible that anisotropic noise (along one direction, e.g. SPT-3G) and lx-cuts

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
        # beam-deconvolved noise
        cls_noise = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cls_noise[k]) for k in cls_noise.keys()}  # white-alike noise spectra
        cls_noise_filt = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cls_noise_filt[k]) for k in cls_noise_filt.keys()}  # white-alike noise spectra

        cls_w1 = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cls_w[k]) for k in cls_w.keys()}  # estimator weights spectra
        bl = extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, transf_cl)  # beam transfer functions
        if cls_w2 is None:
            cls_w2 = cls_w1
        else:
            cls_w2 = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cls_w2[k]) for k in cls_w2.keys()}  # second estimator weights spectra


        #K_ls, Kw1_ls, w2K_ls, wKw_sym_ls = self._build_cl_ls(cls_ivfs, cls_w1, cls_w2)
        #self.K_ls   = K_ls
        #self.Kw1_ls  = Kw1_ls
        #self.w2K_ls  = w2K_ls
        #self.wKw_sym_ls = wKw_sym_ls
        # We need the symmetric part only of this (there is a trace against symmetric K)

        self.cls_w1 = cls_dot([cls_w1])
        self.cls_w2 = cls_dot([cls_w2]) if cls_w2 is not cls_w1 else self.cls_w1
        self.cls_noise = cls_dot([cls_noise])
        self.cls_noise_filt = cls_dot([cls_noise_filt])
        self.cls_cmb = cls_dot([cls_w1]) # FIXME
        self.bl = cls_dot([bl])

        self._cos2p_sin2p = None

        # === normalization (for lensing keys at least)
        norm = (self.box.shape[0] / self.box.lsides[0]) ** 2  # overall final normalization from rfft'ing
        norm *= (float(self.box.lminbox)) ** 4
        self.norm = norm

        self.lx_cut = lx_cut

        self.iso_filt = iso_filt
        self._response = _response
        self._lmax_E = None
        self._lmax_B = None
        self._iso_dat = _iso_dat
    #@staticmethod
    #def _build_cl_ls(cls_ivfs, cls_w1, cls_w2):
    #    K_ls   = cls_dot([cls_ivfs])
    #    Kw1_ls  = cls_dot([cls_ivfs, cls_w1])
    #    w2K_ls  = cls_dot([cls_w2, cls_ivfs])
    #    wKw_sym_ls = 0.5 * (cls_dot([cls_w1, cls_ivfs, cls_w2]) + cls_dot([cls_w2, cls_ivfs, cls_w1]))
    #    # We need the symmetric part only of this (there is a trace against symmetric K)
    #    return K_ls, Kw1_ls, w2K_ls, wKw_sym_ls

    def _lx_int(self):
        return np.int_(np.abs(self.box.lx()))
    
    def _build_K(self, i, j, _response_mode=False):   
        # FIXME: this is wrong for TE etc !!
        # Here cls_w1 is the same as cl_cmb
        # For response calc, just 1/cls_filt
        ret = np.zeros(self.box.rshape, dtype=complex)
        if i != j:
            assert np.all(self.cls_cmb[i, j] == 0)
            return ret
        cmb = (self.cls_cmb[i, j]*self.bl ** 2)[self.box.ls()]
        cmb_cov =  cmb + self.cls_noise[i, j][np.int_(np.abs(self.box.lx())) if not self._iso_dat else self.box.ls()]
        cmb_filt = cmb + self.cls_noise_filt[i, j][self.box.ls() if self.iso_filt else np.int_(np.abs(self.box.lx()))]
        if self._response or _response_mode:
            ret[np.where(cmb != 0)] = 1. / cmb_filt[np.where(cmb != 0)]
        else:
            ret[np.where(cmb != 0)] = cmb_cov[np.where(cmb != 0)] / (cmb_filt[np.where(cmb != 0)] ** 2)
        ret *= (np.abs(self.box.lx()) >= self.lx_cut) * self.bl[self.box.ls()] ** 2
        if i == 1 and j == 1 and self._lmax_E is not None:
            ret *= (self.box.ls() <= self._lmax_E)
        if i == 2 and j == 2 and self._lmax_B is not None:
            ret *= (self.box.ls() <= self._lmax_B)
        return ret
    
    def _build_Kw1(self, i, j):
        ret = np.zeros(self.box.rshape, dtype=complex)
        for k in range(3):
            if np.any(self.cls_w1[k, j]) and np.any(self.cls_cmb[i, k]):
                ret += self._build_K(i, k) * self.cls_w1[k, j][self.box.ls()] 
        return ret
            
    def _build_w2K(self, i, j):
        ret = np.zeros(self.box.rshape, dtype=complex)
        for k in range(3):
            if np.any(self.cls_w2[i, k]) and np.any(self.cls_cmb[k, j]):
                ret += self._build_K(k, j) * self.cls_w2[i, k][self.box.ls()] 
        return ret

    def _build_wKw_sym(self, i, j):
        assert self.cls_w1 is self.cls_w2
        ret = np.zeros(self.box.rshape, dtype=complex)
        for k1 in range(3):
            for k2 in range(3):
             if np.any(self.cls_w1[i, k1]) and np.any(self.cls_w2[k2, j]) and np.any(self.cls_cmb[k1, k2]):
                ret += self._build_K(k1, k2) * (self.cls_w1[i, k1] * self.cls_w2[k2, j])[self.box.ls()]
        return ret

    def _ifft2(self, rm):
        oshape = self.box.shape if rm.ndim == 2 else (rm.shape[0], self.box.shape[0], self.box.shape[1])
        inpt = pyfftw.empty_aligned(rm.shape, dtype='complex128')
        outp = pyfftw.empty_aligned(oshape, dtype='float64')
        ifft2 = pyfftw.FFTW(inpt, outp, axes=(-2, -1), direction='FFTW_BACKWARD', threads=int(os.environ.get('OMP_NUM_THREADS', 1)))
        return ifft2(pyfftw.byte_align(rm, dtype='complex128'))


    def get_response_2d(self, k, _pyfftw=True):
        return self.get_nhl_2d(k, _pyfftw=_pyfftw, _response_mode=True)

    def get_nhl_2d(self, k, _pyfftw=True, _response_mode=False):
        """Returns unormalized QE noise for each and every 2d multipole on the flat-sky box

            Note:
                On a square-periodic flat-sky box there can be tiny differences of N0(L) for same |L|

            No attempt is at optimization. see get_nhl method for much faster N0 array calculation

        """
        if _response_mode:
            self._response = True
        X2i = {'T': 0, 'E': 1, 'B': 2}
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')

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
                        K      +=       self._build_K(i, j) * fac
                        wKw_sym_00 +=  -1 * self._build_wKw_sym(i, j) * ny * ny * fac
                        wKw_sym_11 +=  -1 * self._build_wKw_sym(i, j) * nx * nx * fac
                        wKw_sym_01 +=  -1 * self._build_wKw_sym(i, j) * nx * ny * fac

                        Kw1_0   +=  1j * self._build_Kw1(i, j) * ny * fac
                        Kw1_1   +=  1j * self._build_Kw1(i, j) * nx * fac
                        w2K_0   +=  1j * self._build_w2K(i, j) * ny * fac
                        w2K_1   +=  1j * self._build_w2K(i, j) * nx * fac
                ir2K = ir2(K)
                Fs[0] +=     ir2K  * ir2(wKw_sym_00) + ir2(Kw1_0) * ir2(w2K_0)
                Fs[1] +=     ir2K  * ir2(wKw_sym_11) + ir2(Kw1_1) * ir2(w2K_1)
                Fs[2] +=     ir2K  * ir2(wKw_sym_01) + ir2(Kw1_0) * ir2(w2K_1)
        Fyy, Fxx, Fxy = np.fft.rfft2(Fs).real
        n0_2d_gg = ny ** 2 * Fyy + nx ** 2 * Fxx + 2 * nx * ny * Fxy    # lensing gradient
        n0_2d_cc = nx ** 2 * Fyy + ny ** 2 * Fxx - 2 * nx * ny * Fxy    # lensing curl      
        self._response = False      
        return - self.norm * np.array([n0_2d_gg, n0_2d_cc])