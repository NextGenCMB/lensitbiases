import os
import numpy as np
import lensitbiases
from lensitbiases.utils_n1 import camb_clfile, get_ivf_cls, cli, enumerate_progress
from lensitbiases import n1_fft, n0_fft, len_fft
from scipy.interpolate import UnivariateSpline as spl
import lensit as li

default_cls= os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(li.__file__))), 'inputs', 'cls', 'fiducial_flatsky_')

#FIXME: allow response for different cls_w and cls_f!

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
            if 'te' in fals.keys():
                fals['te'] *= 0.
        if k in ['ptt', 'p_p']:
            cls_qeweight['te'] *= 0.


        self.cls_unl = cls_cmblunl
        self.cls_w = cls_qeweight
        self.cls_f = cls_cmbresponse
        self.fals = fals
        self.ivfs_cls = ivfs_cls

        self.k = k
        self.jt_TP = jt_TP

        self.k2l = 'lensit'

    def get_N0(self):
        pass

    def get_N0_iterative_2d(self, itermax, nmax, lminbox=14.179630807244129, lmaxbox=7258,
                            filt_is_unl=False, f_is_len=False, w_is_unl=False, wN1inWF=False):
        if not f_is_len:
            print("*** Warning:: Using order %s perturbative lensed Cls and weights on full 2d-boxes, "
                  " and order %s perturbative responses gradien Cls"%(nmax, nmax))
        else:
            print("*** Warning:: Using order %s perturbative lensed Cls and weights on full 2d-boxes, "
                  " and taking response Cls same as weight same as lensed Cls"%(nmax))
        assert self.k in ['p_p', 'p', 'ptt'], self.k
        assert itermax >= 0, itermax
        lmax_qlm = 2 * self.lmax
        N0s = []
        N1s = []
        N0pN1 = np.inf
        for irr, it in enumerate_progress(range(itermax + 1)):
            cpp = np.copy(self.cls_unl['pp'])
            clwf = 0. if it == 0 else cpp[:lmax_qlm + 1] * cli(cpp[:lmax_qlm + 1] + N0pN1[:lmax_qlm + 1])
            cpp[:lmax_qlm + 1] *= (1. - clwf)
            lib_len = len_fft.len_fft(self.cls_unl, cpp, lminbox=lminbox, lmaxbox=lmaxbox, k2l=self.k2l)
            cls_plen_2d =  lib_len.lensed_cls_2d(nmax=nmax)
            # bin it
            cls_plen = {k: lib_len.box.sum_in_l(cls_plen_2d[k]) * cli(lib_len.box.mode_counts() * 1.) for k in cls_plen_2d.keys()}
            cls_filt = cls_plen if not filt_is_unl else self.cls_unl
            ivfs_cls, fals = get_ivf_cls(cls_plen, cls_filt, self.lmin, self.lmax, self.nlevt, self.nlevp,  self.nlevt, self.nlevp, self.transf,
                                         jt_tp=self.jt_TP)
            if not f_is_len:
                cls_f_2d = lib_len.lensed_gradcls_2d(nmax=nmax) # response cls
                cls_f = {k: lib_len.box.sum_in_l(cls_f_2d[k]) * cli(lib_len.box.mode_counts() * 1.) for k in cls_f_2d.keys()}
            else:
                cls_f = cls_plen
            cls_w = cls_f if not w_is_unl else {k:np.copy(self.cls_unl[k]) for k in self.cls_unl.keys()}
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
            nhllib = n0_fft.nhl_fft(ivfs_cls, cls_w,  lminbox=lminbox, lmaxbox=lmaxbox, k2l=self.k2l)
            # NB: could possibly use 1d and spline to get all modes in the box
            n_gg, n_cc = nhllib.get_nhl_2d(self.k)
            r_gg, r_cc = n0_fft.nhl_fft(fals, cls_f,  lminbox=lminbox, lmaxbox=lmaxbox, k2l=self.k2l, cls_w2=cls_w).get_nhl_2d(self.k) # Could try to check when n = r
            N0 =  lib_len.box.sum_in_l(n_gg)  * cli(lib_len.box.mode_counts() * 1.)
            N0 *= cli(  (lib_len.box.sum_in_l(r_gg)  * cli(lib_len.box.mode_counts() * 1.) ) ** 2 )
            N0s.append(N0[:lmax_qlm+1])
            if wN1inWF:
                # need to spline the cls for the N1 calc:
                ls, = np.where(lib_len.box.mode_counts()[:self.lmax + 1] > 0)
                fals_spl  = {k: spl(ls, fals[k][ls], k=2, s=0, ext='zeros')(np.arange(self.lmax + 1) * 1.) for k in fals.keys()}
                cls_w_spl = {k: spl(ls, cls_w[k][ls], k=2, s=0, ext='zeros')(np.arange(self.lmax + 1) * 1.) for k in cls_w.keys()}
                cls_f_spl = {k: spl(ls, cls_f[k][ls], k=2, s=0, ext='zeros')(np.arange(self.lmax + 1) * 1.) for k in cls_f.keys()}
                #This one spline probably not needed
                cpp_spl = spl(ls, cpp[ls], k=2, s=0, ext='zeros')(np.arange(self.lmax + 1) * 1.)

                libn1 = n1_fft.n1_fft(fals_spl, cls_w_spl, cls_f_spl, cpp_spl, lminbox=lminbox,  lmaxbox=lmaxbox, k2l=self.k2l)
                Ls = np.linspace(10, lmax_qlm, 50)
                n1 =  np.array([libn1.get_n1(self.k, L, do_n1mat=False) for L in Ls])
                n1_spl = spl(Ls, n1, s=0, k=3, ext='zeros')(np.arange(len(N0)) * 1.)
                N1 = n1_spl * cli(  (lib_len.box.sum_in_l(r_gg)  * cli(lib_len.box.mode_counts() * 1.) ) ** 2 )
                N1s.append(N1[:lmax_qlm + 1])
                N0pN1 = N0 + N1
            else:
                N0pN1 = N0
            ls, = np.where(lib_len.box.mode_counts()[:lmax_qlm+1])
        if not wN1inWF:
            return np.array(N0s), (cls_plen, cpp), ls
        return np.array(N0s), np.array(N1s), (cls_plen, cpp), ls

    def get_N1(self, Ls, lminbox=50, lmaxbox=2500):
        n1s = self.get_n1(Ls, lminbox=lminbox, lmaxbox=lmaxbox)
        (Rgg, Rcc), ls = self.get_response()
        n1s /= (spl(ls * 1., Rgg, k=2, s=0, ext='zeros')(Ls * 1.)) ** 2
        return n1s

    def get_n0(self, lminbox=14.179630807244129, lmaxbox=7258):
        lib = n0_fft.nhl_fft(self.ivfs_cls, self.cls_w, lminbox=lminbox, lmaxbox=lmaxbox, k2l=self.k2l)
        return lib.get_nhl(self.k)

    def get_n0_2d(self, lminbox=14.179630807244129, lmaxbox=7258):
        lib = n0_fft.nhl_fft(self.ivfs_cls, self.cls_w, lminbox=lminbox, lmaxbox=lmaxbox, k2l=self.k2l)
        return lib.get_nhl_2d(self.k)


    def get_n1(self, Ls, lminbox=50, lmaxbox=2500):
        lib = n1_fft.n1_fft(self.fals, self.cls_w, self.cls_f, self.cls_unl['pp'], lminbox=lminbox, lmaxbox=lmaxbox, k2l=self.k2l)
        return np.array([lib.get_n1(self.k, L, do_n1mat=False) for L in Ls])

    def get_response(self, lminbox=14.179630807244129, lmaxbox=7258):
        lib = n0_fft.nhl_fft(self.fals, self.cls_f, lminbox=lminbox, lmaxbox=lmaxbox, k2l=self.k2l, cls_w2=self.cls_w)
        return lib.get_nhl(self.k)

    def get_response_2d(self, lminbox=14.179630807244129, lmaxbox=7258):
        lib = n0_fft.nhl_fft(self.fals, self.cls_f, lminbox=lminbox, lmaxbox=lmaxbox, k2l=self.k2l, cls_w2=self.cls_w)
        return lib.get_nhl_2d(self.k)