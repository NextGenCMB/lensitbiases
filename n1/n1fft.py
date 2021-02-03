import numpy as np

from n1 import  n1_utils
from n1.n1_utils import extcl


class n1_ptt:
    def __init__(self, fals, cls_weight, cls_grad, cpp, lminbox=50, lmaxbox=2500):
        """Looks like this works fine...

            lmin-lmax sort of tuned for Planck 2018 percent-level converged N1 calculation on 100 < L < 800

            Args:
                fals: dictionary with map filters
                cls_weight: spectra dict. for the QE estimator weights
                cls_grad: spectra dict. for the QE estimators and response
                cpp: anisotropy source spectrum
                lminbox: minimal multipole present in the 2D box
                lmaxbox: maximal multipole (along an axis) present in the 2D box


        """
        lside = 2. * np.pi / lminbox
        npix = int(lmaxbox / np.pi * lside) + 1
        if npix % 2 == 1: npix += 1

        #===== instance with 2D flat-sky box info
        self.box = n1_utils.box(lside, npix)

        # === precalc of deflection corr fct:
        nx = np.outer(np.ones(self.box.shape[0]), self.box.nx_1d)
        ny = np.outer(self.box.ny_1d, np.ones(self.box.rshape[1]))
        ls = self.box.ls()

        self.xipp_00 = np.fft.irfft2(extcl(self.box.lmaxbox, -cpp)[ls] * ny ** 2)  # 00
        self.xipp_01 = np.fft.irfft2(extcl(self.box.lmaxbox, -cpp)[ls] * nx * ny)  # 01 or 10
        del nx, ny, ls

        # === Filter and cls array needed later on:
        self.F     = extcl(self.box.lmaxbox + lminbox, fals['tt'])
        self.ctt_f = extcl(self.box.lmaxbox + lminbox, cls_grad['tt'])       # responses spectra
        self.ctt_w = extcl(self.box.lmaxbox + lminbox, cls_weight['tt'])    # estimator weights spectra


        # === normalization (for tt keys at least)
        norm = (self.box.shape[0] / self.box.lsides[0]) ** 4  # overall final normalization from rfft'ing
        norm *= (float(self.box.lminbox)) ** 8
        # :always 2 powers in xi_ab, 4 powers of ik_x or ik_y in XY and IJ weights, and two add. powers matching xi_ab's from the responses
        self.norm = norm

        self._wTT = None


    def _get_shifted_lylx_sym(self, L):
        """Shifts frequencies in both directions

            Returns:
                frequency maps k_y - L/root(2), k_x - L/root(2)

        """
        # simplest :
        #dL = L / np.sqrt(2.) / self.box.lminbox
        #ly = np.outer(self.box.ny_1d - dL  , np.ones(len(self.box.nx_1d)))
        #lx = np.outer(np.ones(len(self.box.ny_1d)), self.box.nx_1d - dL)
        # This way preserves periodicity of the shifts and is the way to go for periodic boxes:
        npix = self.box.shape[0]
        # new 1d frequencies of q - L, respecting box periodicity:
        ns_y = (npix // 2 + self.box.ny_1d - (L / np.sqrt(2.) / self.box.lminbox)) % npix - npix // 2
        ly = np.outer(ns_y, np.ones(self.box.rshape[1]))
        lx = np.outer(np.ones(self.box.rshape[0]), ns_y[:self.box.rshape[1]])
        return ly, lx


    def build_key(self,k, L):
        """Symmetrized TT QE func for L + q, L- q param.

            This has the real fftws conjugacy property

            2 C_{L + q} (L + q) L + 2 C_{L - q} (L - q) L

        """
        if k == 'ptt':
            if self._wTT is None:
                qml = np.array(self._get_shifted_lylx_sym( L * 0.5))  # this is q - L/2
                qpl = np.array(self._get_shifted_lylx_sym(-L * 0.5))  # this is q + L/2
                ls_m_sqd = qml[0] ** 2 + qml[1] ** 2
                ls_p_sqd = qpl[0] ** 2 + qpl[1] ** 2
                ls_m = self.box.rsqd2l(ls_m_sqd)
                ls_p = self.box.rsqd2l(ls_p_sqd)
                #TODO: add curl version here as well
                w = - (self.ctt_w[ls_p] * ls_p_sqd + self.ctt_w[ls_m] * ls_m_sqd)
                w += (self.ctt_w[ls_p] + self.ctt_w[ls_m]) * (qpl[0] * qml[0] + qpl[1] * qml[1])
                w *= self.F[ls_m] * self.F[ls_p]
                self._wTT = (w, qml, qpl, ls_m, ls_p)
            return self._wTT
        else:
            assert 0, 'no recipe to build ' + k

    def get_hf_w(self, L, pi, pj, ders_i, ders_j):
        r"""Position-space weight functions entering the TT N1

            Note:
                all of these functions have tuned-in version implemented below

        """
        assert len(ders_i) == pi
        assert len(ders_j) == pj

        wtt, qml, qpl, ls_m, ls_p = self.build_key('ptt', L)
        w = wtt  * 1j ** (pi + pj)
        if pi > 0:
            w *= (self.ctt_f ** pi)[ls_p]
            for deri in ders_i:
                w *= qpl[deri]
        if pj > 0:
            w *= (self.ctt_f ** pj)[ls_m]
            for derj in ders_j:
                w *= qml[derj]
        return np.fft.irfft2(w)

    def _get_hf_00_w(self, L):
        r"""Tuned version of hf_00_{}

        """
        wtt, qml, qpl, ls_m, ls_p = self.build_key('ptt', L)
        return np.fft.irfft2(wtt)

    def _get_hf_11_10_w(self, L):
        r"""Tuned version of real part of hf_11_{xy} (itself real)

        """
        wtt, qml, qpl, ls_m, ls_p = self.build_key('ptt', L)
        return np.fft.irfft2(-0.5 * wtt * self.ctt_f[ls_p] * self.ctt_f[ls_m] * (qml[1] * qpl[0] + qml[0] * qpl[1]))

    def _get_hf_11_aa_w(self, L, a=0):
        r"""Tuned version of real part of hf_11_{yy} (itself real)

        """
        wtt, qml, qpl, ls_m, ls_p = self.build_key('ptt', L)
        return np.fft.irfft2(-wtt * self.ctt_f[ls_p] * self.ctt_f[ls_m] * (qml[a] * qpl[a]))

    def _get_hf_10_a_w(self, L, a=0):
        r"""Tuned version of real and imaginary part of hf_10_{a} using only rffts

        """
        wtt, qml, qpl, ls_m, ls_p = self.build_key('ptt', L)
        w = wtt  * 0.5j
        facp = self.ctt_f[ls_p] * qpl[a]
        facm = self.ctt_f[ls_m] * qml[a]
        return np.fft.irfft2(w * (facp + facm)), np.fft.irfft2(-1j * w * (facp - facm))

    def destroy_key(self, k):
        if k == 'ptt':
            self._wTT = None
        else:
            assert 0, 'no recipe to destroy ' + k

    def get_n1(self, L, do_n1mat=False):
        r"""N1 lensing gradient-induced lensing bias, for lensing gradient or curl bias

            Args:
                L: multipole L of :math:`N^{(1)}_L`
                do_n1mat: if set, returns the array :math:`N^{(1)}_{LL'}`

            Returns:
                gradient-induced, lensing gradient or curl bias :math:`N_L^{(1)}`
                N1 matrix row if requested

            Note:
                This keeps the QE functions intact and absords the l_phi's factor in Cphiphi_L in the integrals

        """

        L = float(L)
        # --- precalc of some of the rfft'ed maps:)
        h_00 = self._get_hf_00_w(L)
        re_h_10_y, im_h_10_y = self._get_hf_10_a_w(L, a=0)
        h_11_yy = self._get_hf_11_aa_w(L, a=0)
        # the other is the transpose

        term1 = h_11_yy * h_00 + (re_h_10_y ** 2 - im_h_10_y ** 2)
        term2 = self._get_hf_11_10_w(L) * h_00 + (re_h_10_y * re_h_10_y.T - im_h_10_y * im_h_10_y.T)
        n1 = 2. * np.sum(self.xipp_00 * term1 + self.xipp_01 * term2)
        self.destroy_key('ptt')

        if do_n1mat:
            nx = np.outer(np.ones(self.box.shape[0]), self.box.nx_1d)
            ny = np.outer(self.box.ny_1d, np.ones(self.box.rshape[1]))
            f1 = np.fft.rfft2(term1) * ny ** 2
            f2 = np.fft.rfft2(term2) * nx * ny
            n1_mat = -2. / np.prod(self.box.shape) * self.box.sum_in_l(f1.real + f2.real)
            return self.norm * n1, self.norm  * n1_mat
        return self.norm * n1
