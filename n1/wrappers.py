import os
import numpy as np
import n1
from n1.utils_n1 import camb_clfile, get_ivf_cls, cli, dls2cls,cls2dls, enumerate_progress
from n1 import n1_fft, n0_fft, len_fft
from scipy.interpolate import UnivariateSpline as spl
default_cls= os.path.join(os.path.abspath(os.path.dirname(n1.__file__)), 'data', 'cls', 'FFP10_wdipole_')


class cmbconf:
    def __init__(self, k, jt_TP, nlevt, nlevp, transf, lmin, lmax,
                 cls_cmblen=None, cls_cmbresponse=None, cls_cmbfilt=None, cls_qeweight=None, cls_cmblunl=None):
        assert k in ['p', 'ptt', 'p_p']
        self.nlevt = nlevt
        self.nlevp = nlevp
        self.transf = transf
        self.lmin = lmin
        self.lmax = lmax

        if cls_cmblen is None:
            cls_cmblen = camb_clfile(default_cls + 'lensedCls.dat')
        if cls_cmbresponse is None:
            cls_cmbresponse = camb_clfile(default_cls + 'gradlensedCls.dat')
        if cls_cmbfilt is None:
            cls_cmbfilt = camb_clfile(default_cls + 'lensedCls.dat')
        if cls_qeweight is None:
            cls_qeweight = camb_clfile(default_cls + 'gradlensedCls.dat')
        if cls_cmblunl is None:
            cls_cmblunl =  camb_clfile(default_cls + 'lenspotentialCls.dat')

        ivfs_cls, fals = get_ivf_cls(cls_cmblen, cls_cmbfilt, self.lmin, self.lmax, self.nlevt, self.nlevp, self.nlevt, self.nlevp,
                             self.transf, jt_tp=jt_TP)
        if k == 'ptt':
            fals['ee'] *= 0.
            fals['bb'] *= 0.
            ivfs_cls['ee'] *= 0.
            ivfs_cls['bb'] *= 0.
        if k == 'p_p':
            fals['tt'] *= 0.
            ivfs_cls['tt'] *= 0.
            ivfs_cls['te'] *= 0.
        if k in ['ptt', 'p_p']:
            cls_qeweight['te'] *= 0.


        self.cls_unl = cls_cmblunl
        self.cls_w = cls_qeweight
        self.cls_f = cls_cmbresponse
        self.fals = fals
        self.ivfs_cls = ivfs_cls

        self.k = k
        self.jt_TP = jt_TP

    def get_N0(self):
        pass

    def get_N0_iterative_2d(self, itermax, lminbox=14, lmaxbox=7160):
        print("*** Warning:: Using perturbative lensed Cls and weights on full 2d-boxes, and weights equal to lensed Cls")
        assert self.k in ['p_p', 'p', 'ptt'], self.k
        assert itermax >= 0, itermax
        lmax_qlm = 2 * self.lmax
        N0s = []
        N0 = np.inf
        for irr, it in enumerate_progress(range(itermax + 1)):
            cpp = np.copy(self.cls_unl['pp'])
            clwf = 0. if it == 0 else cpp[:lmax_qlm + 1] * cli(cpp[:lmax_qlm + 1] + N0[:lmax_qlm + 1])
            cpp[:lmax_qlm + 1] *= (1. - clwf)
            lib_len = len_fft.len_fft(self.cls_unl, cpp, lminbox=lminbox, lmaxbox=lmaxbox)
            cls_plen_2d =  lib_len.lensed_cls_2d()
            # bin it
            cls_plen = {k: lib_len.box.sum_in_l(cls_plen_2d[k]) * cli(lib_len.box.mode_counts() * 1.) for k in cls_plen_2d.keys()}
            ivfs_cls, fals = get_ivf_cls(cls_plen, cls_plen, self.lmin, self.lmax, self.nlevt, self.nlevp,  self.nlevt, self.nlevp, self.transf,
                                         jt_tp=self.jt_TP)
            cls_w = {k: np.copy(cls_plen[k]) for k in cls_plen.keys()}
            cls_f = cls_w
            if self.k == 'ptt':
                fals['ee'] *= 0.
                fals['bb'] *= 0.
                ivfs_cls['ee'] *= 0.
                ivfs_cls['bb'] *= 0.
            if self.k == 'p_p':
                fals['tt'] *= 0.
                ivfs_cls['tt'] *= 0.
                ivfs_cls['te'] *= 0.
            if self.k in ['ptt', 'p_p']:
                cls_w['te'] *= 0.
            nhllib = n0_fft.nhl_fft(ivfs_cls, cls_w,  lminbox=lminbox, lmaxbox=lmaxbox)
            # NB: could possibly use 1d and spline to get all modes in the box
            n_gg, n_cc = nhllib.get_nhl_2d(self.k)
            r_gg, r_cc = n0_fft.nhl_fft(fals, cls_f,  lminbox=lminbox, lmaxbox=lmaxbox).get_nhl_2d(self.k) # Could try to check when n = r
            N0 =  lib_len.box.sum_in_l(n_gg)  * cli(lib_len.box.mode_counts() * 1.)
            N0 *= cli(  (lib_len.box.sum_in_l(r_gg)  * cli(lib_len.box.mode_counts() * 1.) ) ** 2 )
            N0s.append(N0[:lmax_qlm+1])
            ls, = np.where(lib_len.box.mode_counts()[:lmax_qlm+1])
        return np.array(N0s), (cls_plen, cpp), ls

    def get_N1(self, Ls, lminbox=50, lmaxbox=2500):
        n1s = self.get_n1(Ls, lminbox=lminbox, lmaxbox=lmaxbox)
        (Rgg, Rcc), ls = self.get_response()
        n1s /= (spl(ls * 1., Rgg, k=2, s=0, ext='zeros')(Ls * 1.)) ** 2
        return n1s

    def get_n0(self):
        lib = n0_fft.nhl_fft(self.ivfs_cls, self.cls_w, lminbox=14, lmaxbox=7160)
        return lib.get_nhl(self.k)

    def get_n1(self, Ls, lminbox=50, lmaxbox=2500):
        lib = n1_fft.n1_fft(self.fals, self.cls_w, self.cls_f, self.cls_unl['pp'], lminbox=lminbox, lmaxbox=lmaxbox)
        return np.array([lib.get_n1(self.k, L, do_n1mat=False) for L in Ls])

    def get_response(self, lminbox=14, lmaxbox=7160):
        lib = n0_fft.nhl_fft(self.fals, self.cls_f, lminbox=lminbox, lmaxbox=lmaxbox)
        return lib.get_nhl(self.k)

    def get_response_2d(self, lminbox=14, lmaxbox=7160):
        lib = n0_fft.nhl_fft(self.fals, self.cls_f, lminbox=lminbox, lmaxbox=lmaxbox)
        return lib.get_nhl_2d(self.k)