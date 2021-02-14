import os
import numpy as np
import n1
from n1.utils_n1 import camb_clfile, get_ivf_cls, cli, dls2cls,cls2dls
from n1 import n1_fft, n0_fft
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

    def get_N0(self):
        pass

    def get_N0_iterative(self, itermax):
        assert self.k in ['p_p', 'p', 'ptt'], self.k
        try:
            from camb.correlations import lensed_cls
        except ImportError:
            assert 0, "could not import camb.correlations.lensed_cls"

        lmax_qlm = 2 * self.lmax
        lmin_ivf = max(self.lmin, 1)
        transfi2 = 1./ self.transf ** 2
        llp2 = np.arange(lmax_qlm + 1, dtype=float) ** 2 * np.arange(1, lmax_qlm + 2, dtype=float) ** 2 / (2. * np.pi)
        N0s = []
        N0 = np.inf
        cls_unl = self.cls_unl
        for irr, it in enumerate(range(itermax + 1)):
            dls_unl, cldd = cls2dls(cls_unl)
            clwf = 0. if it == 0 else cldd[:lmax_qlm + 1] * cli(cldd[:lmax_qlm + 1] + llp2 * N0[:lmax_qlm + 1])
            cldd[:lmax_qlm + 1] *= (1. - clwf)
            cls_plen = dls2cls(lensed_cls(dls_unl, cldd))
            cls_ivfs = {}
            assert 0
            #FIXME: build here cls_ivfs and new grad cls
            #fals = cls_ivfs
            #n_gg = n0_fft.n0_fft(ivfs_cls, cls_w,  lminbox=14, lmaxbox=7160).get_n0(self.k)[0][0]
            #r_gg = n0_fft.n0_fft(fals, cls_f,  lminbox=14, lmaxbox=7160).get_n0(self.k)[0][0]
            #N0 = n_gg * cli(r_gg ** 2)
            #N0s.append(N0)
        return np.array(N0s)

    def get_N1(self, Ls, lminbox=50, lmaxbox=2500):
        n1s = self.get_n1(Ls, lminbox=lminbox, lmaxbox=lmaxbox)
        (Rgg, Rcc), ls = self.get_response()
        n1s /= (spl(ls * 1., Rgg, k=2, s=0, ext='zeros')(Ls * 1.)) ** 2
        return n1s

    def get_n0(self):
        lib = n0_fft.n0_fft(self.ivfs_cls, self.cls_w,  lminbox=14, lmaxbox=7160)
        return lib.get_n0(self.k)

    def get_n1(self, Ls, lminbox=50, lmaxbox=2500):
        lib = n1_fft.n1_fft(self.fals, self.cls_w, self.cls_f, self.cls_unl['pp'], lminbox=lminbox, lmaxbox=lmaxbox)
        return np.array([lib.get_n1(self.k, L, do_n1mat=False) for L in Ls])

    def get_response(self):
        lib = n0_fft.n0_fft(self.fals, self.cls_f,  lminbox=14, lmaxbox=7160)
        return lib.get_n0(self.k)