import os
import numpy as np
from lensitbiases.utils_n1 import extcl, cls_dot
from lensitbiases.box import box, rectangle
from lenspyx.cachers import cacher_mem, cacher_none
import pyfftw
import psutil
from ducc0.fft import good_size
def cli(arr):
    ret = np.zeros_like(arr)
    ret[np.where(arr != 0.)] = 1. / arr[np.where(arr != 0.)]
    return ret

class nhl_fft:
    def __init__(self, cls_noise, cls_noise_filt, cls_w, transf_cl, 
                 y_extent_deg=1.85, lminbox=50, lmaxbox=2500, lx_cut=0, lx_lp=0, y2x_axis_ratio=1.,
                 iso_filt=False, _iso_dat=False, _response=False, k2l=None, cls_w2=None, _Kcache=None, verbose=False):
        """
         
        More flexible lensing responses and biases calculator allowing anisotropic noise (along one direction, e.g. SPT-3G) and lx-cuts

            y2x_box_ratio: uses a rectangular box with y/x = y2x_box_ratio
        
        """
        transf_cl = np.atleast_2d(transf_cl)
        nchannels = transf_cl.shape[0]

        if isinstance(cls_noise, dict):
            cls_noise = [cls_noise] * nchannels
        if isinstance(cls_noise_filt, dict):
            cls_noise_filt = [cls_noise_filt] * nchannels


        lside = 2. * np.pi / lminbox
        npix_x = int(2 * lmaxbox / float(lminbox)) + 1
        npix_x += npix_x%2

        # Use rectangular box ? relevant if lx-cuts etc
        if y2x_axis_ratio != 1.:
            npix_y = int(npix_x * y2x_axis_ratio)
            npix_y += npix_y%2
            self.box = rectangle((npix_y*(lside / npix_x), lside), (npix_y, npix_x), k2l=k2l)

        else:
        # ===== instance with 2D flat-sky box info
            self.box = box(lside, npix_x, k2l=k2l)
            self.shape = self.box.shape

        # === Filter and cls array needed later on:
        # beam-deconvolved noise
        cls_noise = [{k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cl[k]) for k in cl.keys()} for cl in cls_noise]  # white-alike noise spectra
        cls_noise_filt = [{k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cl[k]) for k in cl.keys()} for cl in cls_noise_filt]  # white-alike noise spectra

        # distinguishes 2 cases: either cl array or 2D rfft map from elsewhere (e.g. for iterative calculations)
        cls_w_1d = False
        for spec in cls_w:
            assert cls_w[spec].ndim == 1 or cls_w[spec].shape == self.box.rshape
            if cls_w[spec].ndim == 1:
                cls_w_1d = True

        if cls_w_1d:
            cls_w1 = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cls_w[k]) if cls_w[k].ndim == 1 else cls_w[k] for k in cls_w.keys()}  # estimator weights spectra
        else:
            cls_w1 = cls_w
        bl = [extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, tcl) for tcl in transf_cl]  # beam transfer functions
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
        self.cls_w_1d = cls_w_1d

        self.cls_w1 = cls_dot([cls_w1]) if cls_w_1d else cls_w1
        self.cls_w2 = cls_dot([cls_w2]) if cls_w2 is not cls_w1 else self.cls_w1
        self.cls_cmb = cls_dot([cls_w1]) if cls_w_1d else cls_w1 # FIXME

        # First dim for these is the channel
        self.cls_noise = np.array([cls_dot([cl]) for cl in cls_noise])
        self.cls_noise_filt = np.array([cls_dot([cl]) for cl in cls_noise_filt])
        self.bl = np.array([cls_dot([tbl]) for tbl in bl])

        self._cos2p_sin2p = None

        # === normalization (for lensing keys at least)
        norm = np.prod(np.array(self.box.shape) /np.array(self.box.lsides))  # overall final normalization from rfft'ing
        norm *= (float(self.box.lminbox_x) * float(self.box.lminbox_y)) ** 2
        self.norm = norm

        self.lx_cut = lx_cut
        self.lx_lp = lx_lp

        self.iso_filt = iso_filt
        self._response = _response
        self._lmax_T = None
        self._lmax_E = None
        self._lmax_B = None
        self._iso_dat = _iso_dat

        lcell_amin = (self.box.lsides[0] / self.box.shape[0]) / np.pi * 180 * 60
        y_extent = max(int(y_extent_deg * 60 / lcell_amin), 0)
        if verbose:
            print('grid points y_extent:', y_extent )
        self.y_extent = y_extent

        self.nchannels = nchannels
        self._multifreq = True

        self._Kcache = cacher_mem() if _Kcache is None else _Kcache

    def plot_rfft(self, rfftm, kmin=None,title='',**imshow_kwargs):
        import pylab as pl
        kmin = kmin or rfftm.shape[1]
        pl.figure()
        image = pl.imshow( (rfftm )[:kmin, 0:kmin], origin='lower', **imshow_kwargs)
        pl.ylabel(r'$\ell_y / %.0f$'%self.box.lminbox_y)
        pl.xlabel(r'$\ell_x / %.0f$'%self.box.lminbox_x)
        pl.title(title)
        return image
    def _mk_window(self, typ=None):
        if typ == None:
            typ == 'box'
        w = np.where(np.abs(self.box.ny_1d) <= self.y_extent, 1., 0.)
        w2 = np.fft.irfft(np.fft.rfft(w) ** 2)
        return w2/w2[0]
    def _noise_mat(self, cl, y_extent=None, _isofilt=False):
        """Noise covariance matrix resulting from a bunch of scans each with extent y-extent 
        
            correlation fct is the triangle function along y times the underlying isotropic process
        
        """
        if not np.any(cl):
            return np.zeros(self.box.rshape, dtype=float)
        if _isofilt:
            return cl[self.box.ls()]
        if y_extent is None:
            y_extent = self.y_extent
        xi = self._ifft2(cl[self.box.ls()])
        return np.fft.rfft2((xi.T * self._mk_window()).T).real

    def _lx_int(self):
        return np.int_(np.abs(self.box.lx()))
    
    def pcl(self, channel, i, j):
        noise_mat = self._noise_mat(self.cls_noise[channel, i, j]) * (np.abs(self.box.lx()) > self.lx_cut)
        return (self.box.sum_in_l(noise_mat) / self.box.mode_counts())

    def _lowpassfunc(self, lx):
        lx_lpi = 1./self.lx_lp if self.lx_lp > 0 else 0.
        return np.exp(- (lx *lx_lpi) ** 6)
        
    def _build_K(self, i, j, _response_mode=False, _multifreq=False):   
        # Here cls_w1 is the same as cl_cmb
        # For response calc, just 1/cls_filt
        resp = (self._response or _response_mode)
        fn = '%s%s%s_%s_%s_%s'%('r'*resp,min(i, j), max(j, i), self._lmax_T, self._lmax_E, self._lmax_B)
        if self._Kcache.is_cached(fn):
            return self._Kcache.load(fn)
        ret = np.zeros(self.box.rshape, dtype=complex)
        if i != j:
            assert np.all(self._cls_w(i, j, 3) == 0)
            # FIXME: this is wrong for TE etc !!
            return ret
        if _multifreq or self._multifreq:
            assert self.bl.ndim == 2 and self.cls_noise.ndim == 4 and self.cls_noise_filt.ndim == 4, (self.bl.ndim, self.cls_noise.ndim, self.cls_noise_filt)
            # Here using Ci - Ci(Ci + Ni)^{-1}Ci
            cmbi = cli(self._cls_w(i, j, 3))
            # Build noise, summing over frequencies when relevant
            Ni_f = cli(self._noise_mat(self.cls_noise_filt[0, i, j], _isofilt=self.iso_filt) ) * (self.bl[0] ** 2)[self.box.ls()]
            for cha in range(1, self.nchannels):
                Ni_f += cli(self._noise_mat(self.cls_noise_filt[cha, i, j], _isofilt=self.iso_filt) ) * (self.bl[cha] ** 2)[self.box.ls()]
            Kf = cmbi - cmbi * cli(cmbi + Ni_f) * cmbi
            if resp:
                ret[:] = Kf
            else:
                # FIXME: assuming no noise covariance between channels for now
                ret[:] = Kf * self._cls_w(i, j, 3) * Kf
                lowpass2 = self._lowpassfunc(self.box.lx()) ** 2
                Ni_fdf = lowpass2 * cli(self._noise_mat(self.cls_noise_filt[0, i, j], _isofilt=self.iso_filt) ) ** 2 * self._noise_mat(self.cls_noise[0, i, j], _isofilt=self._iso_dat) * (self.bl[0] ** 2)[self.box.ls()]
                for cha in range(1, self.nchannels):
                    Ni_fdf += lowpass2 * cli(self._noise_mat(self.cls_noise_filt[cha, i, j], _isofilt=self.iso_filt) ) ** 2 * self._noise_mat(self.cls_noise[cha, i, j], _isofilt=self._iso_dat) * (self.bl[cha] ** 2)[self.box.ls()]
                ret += cmbi * cli(cmbi + Ni_f) * Ni_fdf * cli(cmbi + Ni_f) * cmbi
        else:
            assert 0
            #cmb = (self.cls_cmb[i, j]*self.bl ** 2)[self.box.ls()]
            #cmb_filt = cmb + self._noise_mat(self.cls_noise_filt[i, j])
            #if self._response or _response_mode:
            #    ret[np.where(cmb != 0)] = 1. / cmb_filt[np.where(cmb != 0)]
            #else:
            #    cmb_cov  = cmb + self._noise_mat(self.cls_noise[i, j])
            #    ret[np.where(cmb != 0)] = cmb_cov[np.where(cmb != 0)] / (cmb_filt[np.where(cmb != 0)] ** 2)
            #ret *= (self.bl ** 2)[self.box.ls()]
        ret *= (np.abs(self.box.lx()) >= self.lx_cut) 
        if i == 0 and j == 0 and self._lmax_T is not None:
            ret *= (self.box.ls() <= self._lmax_T)
        if i == 1 and j == 1 and self._lmax_E is not None:
            ret *= (self.box.ls() <= self._lmax_E)
        if i == 2 and j == 2 and self._lmax_B is not None:
            ret *= (self.box.ls() <= self._lmax_B)
        if self.lx_lp > 0: # Low-pass filter
            ret *= np.exp(- (self.box.lx() / self.lx_lp) ** 6)
        self._Kcache.cache(fn, ret)
        return ret
    
    def _cls_w(self, i, j, one_or_two_or_three=1):
        assert one_or_two_or_three in [1, 2, 3]
        cls = {1:self.cls_w1, 2:self.cls_w2, 3:self.cls_cmb}[one_or_two_or_three]
        if self.cls_w_1d:
            return cls[i, j][self.box.ls()]
        else:
            x, y= {0:'t',1:'e',2:'b'}[i], {0:'t',1:'e',2:'b'}[j]
            return cls.get(x+y, cls.get(y+x, [0.]))
        
    def _build_Kw1(self, i, j):
        ret = np.zeros(self.box.rshape, dtype=complex)
        for k in range(3):
            if np.any(self._cls_w(k, j, 1)) and np.any(self._cls_w(i, k, 3)):
                ret += self._build_K(i, k) * self._cls_w(k, j, 1)
        return ret
            
    def _build_w2K(self, i, j):
        ret = np.zeros(self.box.rshape, dtype=complex)
        for k in range(3):
            if np.any(self._cls_w(i, k, 2)) and np.any(self._cls_w(k, j, 3)):
                ret += self._build_K(k, j) * self._cls_w(i, k, 2) 
        return ret

    def _build_wKw_sym(self, i, j):
        assert self.cls_w1 is self.cls_w2
        ret = np.zeros(self.box.rshape, dtype=complex)
        for k1 in range(3):
            for k2 in range(3):
             if np.any(self._cls_w(i, k1, 1)) and np.any(self._cls_w(k2, j, 2)) and np.any(self._cls_w(k1, k2, 3)):
                ret += self._build_K(k1, k2) * (self._cls_w(i, k1, 1) * self._cls_w(k2, j, 2))
        return ret

    def _ifft2(self, rm):
        oshape = self.box.shape if rm.ndim == 2 else (rm.shape[0], self.box.shape[0], self.box.shape[1])
        inpt = pyfftw.empty_aligned(rm.shape, dtype='complex128')
        outp = pyfftw.empty_aligned(oshape, dtype='float64')
        ifft2 = pyfftw.FFTW(inpt, outp, axes=(-2, -1), direction='FFTW_BACKWARD', threads=int(os.environ.get('OMP_NUM_THREADS', psutil.cpu_count(logical=False))))
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