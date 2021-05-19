r"""rFFT N1 and N1 matrix main module

    FFT-based N1 calculations (~ in O(ms) time per lensing multipole for Planck-like config)

    This uses 5 rfft's of moderate size per L for TT, 20 for PP, 45 for MV (SQE or GMV)

    Calculation of the N1 matrix comes for all cases with just 2 additional rfft's


    Note:

        This assumes (but never checks for) :math:`C_\ell^{TB} = C_\ell^{EB} = 0` throughout

        (If otherwise required this can adapted with little extra work)

    Note:

        Default behavior is to use pyfftw rffts with OMP_NUM_THREADS threads, 1 if not set


"""
import numpy as np
from lensitbiases.utils_n1 import extcl, cls_dot, prepare_cls
from lensitbiases.box import box
import os
import pyfftw

def get_n1(k, Ls, jt_TP, do_n1mat=True, lminbox=50, lmaxbox=2500):
    """Example to show how to get N1 for a set of key and cls, using the Planck defaults FFP10 spectra and config

        Args:
            k: Quadratic estimator key ('p': MV (SQE with jt_TP set to False, or GMV with jt_TP set to True)
                                        (see https://arxiv.org/abs/2101.12193 for the differences between these estimators)
                                        'ptt': TT-only QE
                                        'p_p': Pol-only QE
            Ls: list of multipole wanted
            jt_TP: uses joint temperature and polarization filtering if set, separate if not
            do_n1mat: returns N1 matrix as well if set
            lminbox: minimum multipole of the 2D box used
            lmaxbox: maximum multipole of the 2D box used along an axis
                    (often this can be kept fairly small since high lensing multipoles contribute very little)


        Note:

            In all cases the calculation of the N1 matrix only uses two additional rfft's, thus coming basically at the same cost

        Returns:

            if do_n1mat is set:
                N1, N1mat and ls, where N1 is the bias array, N1mat the matrix N1_{LL'} and ls the multipole present in the 2D box.
                The N1 matrix is zero whenever L' is not in ls
                It holds:

                    :math:`\sum_{L'} N1_{LL'} C^{pp}_L' = N^{(0)}_{L}`

                where :math:`C^{pp}_L` is the anisotropy source spectrum

            if do_n1mat is not set:
                M1 array for all L in Ls

    """
    assert k in ['p', 'p_p', 'ptt'], k
    # --- loads example filter, QE weights, response CMB spectra and anisotropy source spectrum
    ivfs_cls, fals, cls_weights, cls_grad, cpp = prepare_cls(k, jt_TP=jt_TP)
    # --- instantiation
    n1lib = n1_fft(fals, cls_weights, cls_grad, cpp, lminbox=lminbox, lmaxbox=lmaxbox)
    # --- computing the biases with or without the n1 matrix:
    if not do_n1mat:
        return np.array([n1lib.get_n1(k, L, do_n1mat=False) for L in Ls])
    else:
        n1mat = np.zeros((len(Ls), n1lib.box.lmaxbox + 1))
        n1 = np.zeros(len(Ls))
        for i, L in enumerate(Ls):
            tn1, tn1mat = n1lib.get_n1(k, L, do_n1mat=True)
            n1mat[i] = tn1mat
            n1[i] = tn1
        ls, = np.where(n1lib.box.mode_counts() > 0)
        return n1, n1mat, ls



class n1_fft:
    def __init__(self, fals, cls_w, cls_f, cpp, lminbox=50, lmaxbox=2500, k2l=None):
        r""" N1 calculator main module

            Args:
                  fals: dict of inverse-variance filtering cls
                  cls_w: Quadratic estimator CMB spectra used in QE(mapping the inverse-variance filtered maps to the Wiener-filtered CMB)
                        (for optimal QE the grad-lensed Cls, but for practical purposes lensed cls are close to this as well)
                  cls_f: CMB spectra entering the response function of the CMB to lensing
                        (in principle always the grad-lensed Cls, but for practical purposes the lensed cls are very close to this as well)
                  lminbox: a 2d flat-sky will be constructed with this as minimum multipole
                  lmaxbox: a 2d flat-sky will be constructed with this as maximum multipole along an axis

            There are presumably still possible speed-ups in the small-box regime where the ffts does not actually dominate the cost

        """

        lside = 2. * np.pi / lminbox
        npix = int(lmaxbox / np.pi * lside) + 1
        if npix % 2 == 1: npix += 1

        #===== instance with 2D flat-sky box info
        self.box = box(lside, npix, k2l=k2l)
        self.shape = self.box.shape


        self._cos2p_sin2p = None
        self._cos2p_sin2p_v1 = None
        #==== Builds required spectra:
        # === Filter and cls array needed later on:
        fals   = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, fals[k] ) for k in fals.keys()}   # Filtering matrix
        cls_f  = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cls_f[k]) for k in cls_f.keys()}  # responses spectra
        cls_w  = {k: extcl(self.box.lmaxbox + int(self.box.lminbox) + 1, cls_w[k]) for k in cls_w.keys()}  # estimator weights spectra


        self.F_ls = cls_dot([fals])
        self.wF_ls = cls_dot([cls_w, fals])
        self.Fw_ls = cls_dot([fals, cls_w]) # idem, but transpose

        self.fF_ls = cls_dot([cls_f, fals])
        self.Ff_ls = cls_dot([fals,  cls_f]) # idem, but transpose

        self.fFw_ls = cls_dot([cls_f, fals, cls_w])
        self.wFf_ls = cls_dot([cls_w, fals, cls_f])# idem, but transpose


        # === precalc of deflection corr fct:
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        ls = self.box.ls()
        self.xipp = np.array([np.fft.irfft2(extcl(self.box.lmaxbox, -cpp)[ls] * ny ** 2),
                              np.fft.irfft2(extcl(self.box.lmaxbox, -cpp)[ls] * nx * ny)])# 01 or 10
        del nx, ny, ls

        # === normalization (for lensing keys at least)
        norm = (self.box.shape[0] / self.box.lsides[0]) ** 4  # overall final normalization from rfft'ing
        norm *= (float(self.box.lminbox)) ** 8
        # :always 2 powers in xi_ab, 4 powers of ik_x or ik_y in XY and IJ weights, and two add. powers matching xi_ab's from the responses
        self.norm = norm

    def _irfft2(self, rm):
        oshape = self.box.shape if rm.ndim == 2 else (rm.shape[0], self.box.shape[0], self.box.shape[1])
        inpt = pyfftw.empty_aligned(rm.shape, dtype='complex128')
        outp = pyfftw.empty_aligned(oshape, dtype='float64')
        ifft2 = pyfftw.FFTW(inpt, outp, axes=(-2, -1), direction='FFTW_BACKWARD', threads=int(os.environ.get('OMP_NUM_THREADS', 1)))
        return ifft2(pyfftw.byte_align(rm, dtype='complex128'))


    @staticmethod
    def _get_cos2p_sin2p(ls):
        """Returns the cosines and sines of twice the angle of the map of vectors

            Note:
                This assumes no vector is zero

        """
        ly, lx = ls
        r_sqd = lx ** 2 + ly ** 2
        return 2 * lx ** 2 / r_sqd - 1., 2 * lx * ly / r_sqd

    def _get_shifted_lylx_sym(self, L, rfft=True):
        """Shifts frequencies in both directions

            Returns:
                frequency maps k_y - L/root(2), k_x - L/root(2)

        """
        npix = self.box.shape[0]
        # new 1d frequencies of q - L, respecting box periodicity:
        ns_y = (npix // 2 + self.box.ny_1d - (L / np.sqrt(2.) / self.box.lminbox)) % npix - npix // 2
        return np.meshgrid(ns_y, ns_y[:self.box.rshape[1]] if rfft else ns_y, indexing='ij')


    def _build_key(self, k, L, rfft=False):
        self.l1s =  np.array(self._get_shifted_lylx_sym(-L * 0.5, rfft=rfft))  # this is q + L/2
        self.l2s = -np.array(self._get_shifted_lylx_sym (L * 0.5, rfft=rfft))  # this is -(q - L/2) = L/2 - q
        self.l1_int = self.box.rsqd2l(np.sum(self.l1s ** 2, axis=0))
        self.l2_int = self.box.rsqd2l(np.sum(self.l2s ** 2, axis=0))
        # TODO: to get curl key only change should be these two lines here:
        # For curl estimator must replace i (Lx ly) by i (-Ly Lx)
        # In principle we have l1 + l2 = (L/root(2), L/root(2)) but lets keep it explicit
        if k[0] == 'p':
            self.Ll1 = np.sum((self.l1s + self.l2s) * self.l1s, axis=0)
            self.Ll2 = np.sum((self.l1s + self.l2s) * self.l2s, axis=0)
        elif k[0] == 'x':
            self.Ll1 = -(self.l1s + self.l2s)[1] * self.l1s[0] + (self.l1s + self.l2s)[0] * self.l1s[1]
            self.Ll2 = -(self.l1s + self.l2s)[1] * self.l2s[0] + (self.l1s + self.l2s)[0] * self.l2s[1]
        elif k[0] == 'f':
            self.Ll1 = 1.
            self.Ll2 = 1.
        else:
            assert 0, 'dont know what to do for QE key ' + k + ', implement this.'
        if k in ['p_p', 'p', 'x_p', 'x']:
            l1s, l2s = (self.l1s, self.l2s)
            r1sqd_r2sqd = np.sum(l1s ** 2, axis=0) * np.sum(l2s ** 2, axis=0)
            dotp = np.sum(l1s * l2s, axis=0)
            cos2p = 2 * dotp ** 2 / r1sqd_r2sqd - 1.
            sin2p = 2 * dotp * (l2s[1] * l1s[0] - l2s[0] * l1s[1]) / r1sqd_r2sqd
            self._cos2p_sin2p = (cos2p, sin2p)
            self._cos2p_sin2p_v1 = {1: self._get_cos2p_sin2p(l1s), 2: self._get_cos2p_sin2p(l2s)}


    def _destroy_key(self, k):
        self.l1s = None
        self.l2s = None
        self.l1_int = None
        self.l2_int = None
        self._cos2p_sin2p_v1 = None
        self._cos2p_sin2p = None
        self.Ll1 = None
        self.Ll2 = None

    def cos2p_sin2p_2v(self):
        """Returns the cosines and sines of twice the angle between the two maps of vectors

            Note:
                This assumes no vector is zero

                cos2p is same as  c1 c2 + s1 s2
                sin2p is same as -s1 c2 + c1 s2

        """
        if self._cos2p_sin2p is None:
            r1sqd_r2sqd = np.sum(self.l1s ** 2, axis=0) * np.sum(self.l2s ** 2, axis=0)
            dotp = np.sum(self.l1s * self.l2s, axis=0)
            cos2p = 2 * dotp ** 2 / r1sqd_r2sqd - 1.
            sin2p = 2 * dotp * (self.l2s[1] * self.l1s[0] -self.l2s[0] * self.l1s[1]) / r1sqd_r2sqd
            self._cos2p_sin2p = (cos2p, sin2p)
        return self._cos2p_sin2p

    def _X2S(self, S, X, vec1or2):
        """Matrix element sending X cmb mode to stokes flat-sky mode S


        """
        if S == 'T':
            return 1. if X == 'T' else 0.
        if S == 'Q':
            if X == 'T':
                return 0.
            if self._cos2p_sin2p_v1 is None:
                self._cos2p_sin2p_v1 = {1:  self._cos2p_sin2p(self.l1s), 2: self._cos2p_sin2p(self.l2s)}
            sgn = 1 if X == 'E' else -1
            return sgn * self._cos2p_sin2p_v1[vec1or2][0 if X == 'E' else 1]
        if S == 'U':
            if X == 'T':
                return 0.
            if self._cos2p_sin2p_v1 is None:
                self._cos2p_sin2p_v1 = {1: self._cos2p_sin2p(self.l1s), 2: self._cos2p_sin2p(self.l2s)}
            return self._cos2p_sin2p_v1[vec1or2][1 if X == 'E' else 0]
        assert 0

    def _X2Y(self, Y, X):
        """Matrix element

                (R^tR)^{YX}_{l1,l2} = \sum_{S in T,Q,U} R^{S Y}_{l1} R^{S X}_{l2}

        """
        assert X in ['T', 'E', 'B'] and Y in ['T', 'E', 'B'], (X, Y)

        if X == 'T' or Y == 'T':
            return 1. if X == Y else 0.
        if X == Y:
            return self.cos2p_sin2p_2v()[0] # same as c1 * c1 + s1 * s2 = cos2p_12
        return (1 if X == 'B' else -1) * self.cos2p_sin2p_2v()[1]# same as -+ (c1 s2 - s1 c2) = -+ sin_2p_12:


    def _W_SS_diag(self, S):
        """ Builds all W_SS terms needed for a diagonal piece

        """
        s = self.l1_int.shape

        X2i = {'T': 0, 'E': 1, 'B': 2}
        W1_SS = np.zeros(s, dtype=float)
        W2_SS = np.zeros(s, dtype=float)
        W1_SS_0z = np.zeros(s, dtype=float)
        W2_SS_0z = np.zeros(s, dtype=float)
        W1_SS_z0 = np.zeros(s, dtype=float)
        W2_SS_z0 = np.zeros(s, dtype=float)
        W1_SS_00 = np.zeros(s, dtype=float)
        W2_SS_00 = np.zeros(s, dtype=float)
        W1_SS_01 = np.zeros(s, dtype=float)
        W2_SS_01 = np.zeros(s, dtype=float)

        for X in ['T'] if S == 'T' else ['E', 'B']:
            # RSX_1 = self._X2S(S, X, 1)
            # for Y in ['E', 'B']: # Assuming no EB and BE, Y must be X
            # S XY XY' S   ->
            for Y in ['B'] if X == 'B' else ['T', 'E']:  # could restrict this is sep-TP configuration
                cl_XY_1 = self.Fw_ls[X2i[X], X2i[Y]]
                cl_XY_2 = self.F_ls[X2i[X], X2i[Y]]
                cl_XY_1_0 = self.fFw_ls[X2i[X], X2i[Y]]
                cl_XY_2_0 = self.fF_ls[X2i[X], X2i[Y]]
                for Xp in ['T'] if Y == 'T' else ['E', 'B']:
                    RtR_YXp = self._X2Y(Y, Xp)
                    # for Yp in ['E', 'B']: # Assuming no EB and BE, Yp must be Xp
                    for Yp in ['B'] * (S != 'T') if Xp == 'B' else (['T'] if S == 'T' else ['E']):
                        cl_XpYp_1 = self.F_ls[X2i[Xp], X2i[Yp]]
                        cl_XpYp_2 = self.wF_ls[X2i[Xp], X2i[Yp]]
                        cl_XpYp_1_0 = self.Ff_ls[X2i[Xp], X2i[Yp]]
                        cl_XpYp_2_0 = self.wFf_ls[X2i[Xp], X2i[Yp]]

                        toSS_RtR = self._X2S(S, X, 1) * self._X2S(S, Yp, 2) * RtR_YXp
                        # terms without any response weight:
                        W1_SS += toSS_RtR * cl_XY_1[self.l1_int] * cl_XpYp_1[self.l2_int]
                        W2_SS += toSS_RtR * cl_XY_2[self.l1_int] * cl_XpYp_2[self.l2_int]
                        # ==== terms with one derivatives (need both re and im)
                        W1_SS_0z += toSS_RtR * cl_XY_1_0[self.l1_int] * cl_XpYp_1[self.l2_int] * self.l1s[0]
                        W2_SS_0z += toSS_RtR * cl_XY_2_0[self.l1_int] * cl_XpYp_2[self.l2_int] * self.l1s[0]
                        W1_SS_z0 += toSS_RtR * cl_XY_1[self.l1_int] * cl_XpYp_1_0[self.l2_int] * self.l2s[0]
                        W2_SS_z0 += toSS_RtR * cl_XY_2[self.l1_int] * cl_XpYp_2_0[self.l2_int] * self.l2s[0]
                        # ==== terms with two derivatives : 00 (real)
                        term1 = toSS_RtR * cl_XY_1_0[self.l1_int] * cl_XpYp_1_0[self.l2_int]
                        term2 = toSS_RtR * cl_XY_2_0[self.l1_int] * cl_XpYp_2_0[self.l2_int]
                        W1_SS_00 += term1 * self.l1s[0] * self.l2s[0]
                        W2_SS_00 += term2 * self.l1s[0] * self.l2s[0]
                        # ==== terms with two derivatives : 01 (cmplx, need the real part)
                        W1_SS_01 += term1 * (self.l1s[0] * self.l2s[1] + self.l1s[1] * self.l2s[0])
                        W2_SS_01 += term2 * (self.l1s[0] * self.l2s[1] + self.l1s[1] * self.l2s[0])


        for W in [W1_SS, W1_SS_00, W1_SS_z0, W1_SS_0z, W1_SS_01]:
            W *= self.Ll1
        for W in [W2_SS, W2_SS_00, W2_SS_z0, W2_SS_0z, W2_SS_01]:
            W *= self.Ll2
        SS = (W1_SS + W2_SS).astype(complex)
        SS_00 = (-1) * (W1_SS_00 + W2_SS_00).astype(complex)
        SS_01_re = (-1 * 0.5) *(W1_SS_01 + W2_SS_01).astype(complex)
        SS_0z_re =  (   0.5 * 1j) * ( (W1_SS_0z + W2_SS_0z) -  (W1_SS_z0 + W2_SS_z0))  # SS,(0, ), real part
        SS_0z_im =  (   0.5 * 1.) * ( (W1_SS_0z + W2_SS_0z) +  (W1_SS_z0 + W2_SS_z0))  # SS,(0, ), im part
        return SS, SS_00, SS_01_re, SS_0z_re, SS_0z_im

    def _W_TS_odiag(self, T, S, verbose=False):
        """Builds all W_TS terms needed for one of the diagonal (TQ, TU, UQ) terms

            Needed are:

                TS (re and im), TS(0,) (re and im) TS(,0) (re and im)
                TS(0, 0) (re and im), TS(0, 1) (re and im)

        """
        assert S != T, (S, T)
        s = self.l1_int.shape
        W1 = np.zeros((4, s[0], s[1]), dtype=float)
        W2 = np.zeros((4, s[0], s[1]), dtype=float)

        W1_10 = np.zeros(s, dtype=float)
        W2_10 = np.zeros(s, dtype=float)
        W1_01 = np.zeros(s, dtype=float)
        W2_01 = np.zeros(s, dtype=float)
        # We will combine these two at the end to produce the real and im parts of the (0, ) terms
        W1_TS_0z = np.zeros(s, dtype=float)
        W2_TS_0z = np.zeros(s, dtype=float)
        W1_ST_z0 = np.zeros(s, dtype=float)
        W2_ST_z0 = np.zeros(s, dtype=float)
        W1_TS_z0 = np.zeros(s, dtype=float)
        W2_TS_z0 = np.zeros(s, dtype=float)
        W1_ST_0z = np.zeros(s, dtype=float)
        W2_ST_0z = np.zeros(s, dtype=float)


        X2i = {'T': 0, 'E': 1, 'B': 2}

        for X in ['T', 'E', 'B']:
            # RSX_1 = self._X2S(S, X, 1)
            # for Y in ['E', 'B']: # Assuming no EB and BE, Y must be X
            for Y in ['B'] if X == 'B' else ['T', 'E']: # could restrict this is sep-TP configuration
                cl_XY_1 = self.Fw_ls[X2i[X], X2i[Y]]
                cl_XY_2 = self.F_ls[X2i[X], X2i[Y]]
                cl_XY_1_0 = self.fFw_ls[X2i[X], X2i[Y]]
                cl_XY_2_0 = self.fF_ls[X2i[X], X2i[Y]]
                for Xp in ['T'] if Y == 'T' else ['E', 'B']:
                    RtR_YXp = self._X2Y(Y, Xp)
                    # for Yp in ['E', 'B']: # Assuming no EB and BE, Yp must be Xp
                    for Yp in ['B'] if Xp == 'B' else ['T', 'E']:
                        cl_XpYp_1 = self.F_ls[X2i[Xp], X2i[Yp]]
                        cl_XpYp_2 = self.wF_ls[X2i[Xp], X2i[Yp]]
                        cl_XpYp_1_0 = self.Ff_ls[X2i[Xp], X2i[Yp]]
                        cl_XpYp_2_0 = self.wFf_ls[X2i[Xp], X2i[Yp]]

                        toTS = self._X2S(T, X, 1) * self._X2S(S, Yp, 2)
                        toST = self._X2S(S, X, 1) * self._X2S(T, Yp, 2)
                        TSpST = toTS + toST
                        TSmST = toTS - toST
                        # terms without any response weight:
                        term1 = RtR_YXp * cl_XY_1[self.l1_int] * cl_XpYp_1[self.l2_int]
                        term2 = RtR_YXp * cl_XY_2[self.l1_int] * cl_XpYp_2[self.l2_int]

                        if verbose:
                            print('term1 ' + X + Y + ' ' + Xp + Yp + ' empty! ' * (not np.any(term1)))
                        if verbose:
                            print('term2 ' + X + Y + ' ' + Xp + Yp + ' empty! ' * (not np.any(term1)))

                        W1[0] +=   term1 * TSpST  # real part of TS
                        W2[0] +=   term2 * TSpST
                        W1[1] +=   term1 * TSmST  # im. part of TS
                        W2[1] +=   term2 * TSmST
                        # ==== terms with two derivatives
                        term1 = RtR_YXp * cl_XY_1_0[self.l1_int] * cl_XpYp_1_0[self.l2_int] * self.l1s[0] * self.l2s[0]
                        term2 = RtR_YXp * cl_XY_2_0[self.l1_int] * cl_XpYp_2_0[self.l2_int] * self.l1s[0] * self.l2s[0]

                        W1[2] +=  term1 * TSpST  # real part of TS_00
                        W2[2] +=  term2 * TSpST  #
                        W1[3] +=  term1 * TSmST  # imag part of TS_00
                        W2[3] +=  term2 * TSmST  #

                        # ==== terms with two derivatives (1 0) and (0 1) (with swapped QU - UQ)
                        term1 = RtR_YXp * cl_XY_1_0[self.l1_int] * cl_XpYp_1_0[self.l2_int]
                        term2 = RtR_YXp * cl_XY_2_0[self.l1_int] * cl_XpYp_2_0[self.l2_int]
                        W1_01 += toTS * term1 * self.l1s[0] * self.l2s[1]
                        W2_01 += toTS * term2 * self.l1s[0] * self.l2s[1]
                        W1_10 += toST * term1 * self.l1s[1] * self.l2s[0]
                        W2_10 += toST * term2 * self.l1s[1] * self.l2s[0]

                        # ==== terms with one derivatives
                        W1_TS_0z += toTS * RtR_YXp * cl_XY_1_0[self.l1_int] * cl_XpYp_1[self.l2_int] * self.l1s[0]
                        W2_TS_0z += toTS * RtR_YXp * cl_XY_2_0[self.l1_int] * cl_XpYp_2[self.l2_int] * self.l1s[0]
                        W1_ST_z0 += toST * RtR_YXp * cl_XY_1[self.l1_int] * cl_XpYp_1_0[self.l2_int] * self.l2s[0]
                        W2_ST_z0 += toST * RtR_YXp * cl_XY_2[self.l1_int] * cl_XpYp_2_0[self.l2_int] * self.l2s[0]
                        W1_TS_z0 += toTS * RtR_YXp * cl_XY_1[self.l1_int] * cl_XpYp_1_0[self.l2_int] * self.l2s[0]
                        W2_TS_z0 += toTS * RtR_YXp * cl_XY_2[self.l1_int] * cl_XpYp_2_0[self.l2_int] * self.l2s[0]
                        W1_ST_0z += toST * RtR_YXp * cl_XY_1_0[self.l1_int] * cl_XpYp_1[self.l2_int] * self.l1s[0]
                        W2_ST_0z += toST * RtR_YXp * cl_XY_2_0[self.l1_int] * cl_XpYp_2[self.l2_int] * self.l1s[0]

        for W in [W1, W1_01, W1_10, W1_TS_0z, W1_TS_z0, W1_ST_z0, W1_ST_0z]:
            W *= self.Ll1
        for W in [W2, W2_01, W2_10, W2_TS_0z, W2_TS_z0, W2_ST_z0, W2_ST_0z]:
            W *= self.Ll2
        #i_sign = 1j ** (ders_1 is not None) * 1j ** (ders_2 is not None)
        W_re =  0.5  * (W1[0] + W2[0]).astype(complex)
        W_im = -0.5j * (W1[1] + W2[1])
        W00_re = - 0.5  * (W1[2] + W2[2]).astype(complex)
        W00_im = + 0.5j * (W1[3] + W2[3])
        W_0z_re =  (   0.5 * 1j) * ( (W1_TS_0z + W2_TS_0z) -  (W1_ST_z0 + W2_ST_z0))  # TS,(0, ), real part
        W_0z_im =  (   0.5 * 1.) * ( (W1_TS_0z + W2_TS_0z) +  (W1_ST_z0 + W2_ST_z0))  # TS,(0, ), im part
        W_z0_re =  (   0.5 * 1j) * ( (W1_TS_z0 + W2_TS_z0) -  (W1_ST_0z + W2_ST_0z))  # TS,(,0), real part
        W_z0_im =  (   0.5 * 1.) * ( (W1_TS_z0 + W2_TS_z0) +  (W1_ST_0z + W2_ST_0z))  # TS,(,0), im part
        # W01 has re QQ, re UU, re QU im QU
        W_01_re = (-1 * 0.5 ) * ( (W1_01 + W2_01) + (W1_10 + W2_10) ).astype(complex) # Re and Im of TS
        W_01_im = (+1 * 0.5j) * ( (W1_01 + W2_01) - (W1_10 + W2_10) )

        return W_re, W_im, W00_re, W00_im, W_01_re, W_01_im, W_0z_re, W_0z_im, W_z0_re, W_z0_im

    def _W_ST_Pol(self, verbose=False):
        """Same as _W_ST but returns all Stokes weights in one go

            Returns: QQ, UU , QU and UQ

        """
        #TODO: extend to the case of joint-TP
        s = self.l1_int.shape
        W1_zz = np.zeros((4, s[0], s[1]), dtype=float)  # QQ, UU  QU re, QU im
        W2_zz = np.zeros((4, s[0], s[1]), dtype=float)  # terms of QE with Cl weight on l1 leg

        W1_00 = np.zeros((4, s[0], s[1]), dtype=float)  # QQ_00, UU_00, QU_00 re and im
        W2_00 = np.zeros((4, s[0], s[1]), dtype=float)

        W1_01 = np.zeros((4, s[0], s[1]), dtype=float)  # for QQ_01 real party, UU_01 real part, QU_01 real and im
        W2_01 = np.zeros((4, s[0], s[1]), dtype=float)
        W1_10 = np.zeros((4, s[0], s[1]), dtype=float)  # for QQ_01 real party, UU_01 real part, QU_01 real and im
        W2_10 = np.zeros((4, s[0], s[1]), dtype=float)

        # We will combine these two at the end to produce the real and im parts of the (0, ) terms
        W1_0z = np.zeros((4, s[0], s[1]), dtype=float)  # QQ_0z, UU_0z, QU_0z UQ_0z
        W2_0z = np.zeros((4, s[0], s[1]), dtype=float)
        W1_z0 = np.zeros((4, s[0], s[1]), dtype=float)  # QQ_0z, UU_0z, UQ_0z QU_0z
        W2_z0 = np.zeros((4, s[0], s[1]), dtype=float)   # QQ_0z, UU_0z, UQ_0z QU_0z


        X2i = {'T': 0, 'E': 1, 'B': 2}
        for X in ['E', 'B']:
            #RSX_1 = self._X2S(S, X, 1)
            #for Y in ['E', 'B']: # Assuming no EB and BE, Y must be X
            Y = X
            cl_XY_1 = self.Fw_ls[X2i[X], X2i[Y]]
            cl_XY_2 = self.F_ls[X2i[X], X2i[Y]]
            cl_XY_1_0 = self.fFw_ls[X2i[X], X2i[Y]]
            cl_XY_2_0 = self.fF_ls[X2i[X], X2i[Y]]
            for Xp in ['E', 'B']:
                RtR_YXp = self._X2Y(Y, Xp)
                #for Yp in ['E', 'B']: # Assuming no EB and BE, Yp must be Xp
                Yp = Xp
                cl_XpYp_1 = self.F_ls[X2i[Xp], X2i[Yp]]
                cl_XpYp_2 = self.wF_ls[X2i[Xp], X2i[Yp]]
                cl_XpYp_1_0 = self.Ff_ls[X2i[Xp], X2i[Yp]]
                cl_XpYp_2_0 = self.wFf_ls[X2i[Xp], X2i[Yp]]

                toQQ = self._X2S('Q', X, 1) * self._X2S('Q', Yp, 2)
                toUU = self._X2S('U', X, 1) * self._X2S('U', Yp, 2)
                toQU = self._X2S('Q', X, 1) * self._X2S('U', Yp, 2)
                toUQ = self._X2S('U', X, 1) * self._X2S('Q', Yp, 2)



                # terms without any response weight:
                term1 = RtR_YXp * cl_XY_1[self.l1_int] * cl_XpYp_1[self.l2_int]
                term2 = RtR_YXp * cl_XY_2[self.l1_int] * cl_XpYp_2[self.l2_int]

                W1_zz[0] += toQQ * term1
                W1_zz[1] += toUU * term1
                W1_zz[2] += (toQU + toUQ) * term1    # 2  the real part of QU
                W1_zz[3] += (toQU - toUQ) * term1    # 2j the imag part of QU

                W2_zz[0] += toQQ * term2
                W2_zz[1] += toUU * term2
                W2_zz[2] += (toQU + toUQ) * term2
                W2_zz[3] += (toQU - toUQ) * term2

                # ==== terms with two derivatives
                term1 = RtR_YXp * cl_XY_1_0[self.l1_int] * cl_XpYp_1_0[self.l2_int] * self.l1s[0] * self.l2s[0]
                term2 = RtR_YXp * cl_XY_2_0[self.l1_int] * cl_XpYp_2_0[self.l2_int] * self.l1s[0] * self.l2s[0]

                W1_00[0] += toQQ * term1
                W1_00[1] += toUU * term1
                W1_00[2] += (toQU + toUQ) * term1    # 2  the real part of QU
                W1_00[3] += (toQU - toUQ) * term1    # 2j the imag part of QU

                W2_00[0] += toQQ * term2
                W2_00[1] += toUU * term2
                W2_00[2] += (toQU + toUQ) * term2
                W2_00[3] += (toQU - toUQ) * term2

                # ==== terms with two derivatives (1 0) and (0 1) (with swapped QU - UQ)
                term1 = RtR_YXp * cl_XY_1_0[self.l1_int] * cl_XpYp_1_0[self.l2_int] * self.l1s[0] * self.l2s[1]
                term2 = RtR_YXp * cl_XY_2_0[self.l1_int] * cl_XpYp_2_0[self.l2_int] * self.l1s[0] * self.l2s[1]

                W1_01[0] += toQQ * term1
                W1_01[1] += toUU * term1
                W1_01[2] += toQU * term1
                #W1_01[3] += toUQ * term1

                W2_01[0] += toQQ * term2
                W2_01[1] += toUU * term2
                W2_01[2] += toQU * term2
                #W2_01[3] += toUQ * term2

                term1 = RtR_YXp * cl_XY_1_0[self.l1_int] * cl_XpYp_1_0[self.l2_int] * self.l1s[1] * self.l2s[0]
                term2 = RtR_YXp * cl_XY_2_0[self.l1_int] * cl_XpYp_2_0[self.l2_int] * self.l1s[1] * self.l2s[0]

                W1_10[0] += toQQ * term1
                W1_10[1] += toUU * term1
                W1_10[2] += toUQ * term1 # (UQ <> UQ)
                #W1_10[3] += toQU * term1 # (UQ <> UQ)

                W2_10[0] += toQQ * term2
                W2_10[1] += toUU * term2
                W2_10[2] += toUQ * term2 # (UQ <> UQ)
                #W2_10[3] += toQU * term2 # (UQ <> UQ)

                # ==== terms with one derivatives
                term1 = RtR_YXp * cl_XY_1_0[self.l1_int] * cl_XpYp_1[self.l2_int] * self.l1s[0]
                term2 = RtR_YXp * cl_XY_2_0[self.l1_int] * cl_XpYp_2[self.l2_int] * self.l1s[0]
                W1_0z[0] += toQQ * term1
                W1_0z[1] += toUU * term1
                W1_0z[2] += toQU * term1
                W1_0z[3] += toUQ * term1

                W2_0z[0] += toQQ * term2
                W2_0z[1] += toUU * term2
                W2_0z[2] += toQU * term2
                W2_0z[3] += toUQ * term2
                # ==== We swap QU and UQ indices to make easier the combination to real and imaginary part later
                term1 = RtR_YXp * cl_XY_1[self.l1_int] * cl_XpYp_1_0[self.l2_int] * self.l2s[0]
                term2 = RtR_YXp * cl_XY_2[self.l1_int] * cl_XpYp_2_0[self.l2_int] * self.l2s[0]

                W1_z0[0] += toQQ * term1
                W1_z0[1] += toUU * term1
                W1_z0[2] += toUQ * term1 # QU <-> UQ
                W1_z0[3] += toQU * term1 # QU <-> UQ

                W2_z0[0] += toQQ * term2
                W2_z0[1] += toUU * term2
                W2_z0[2] += toUQ * term2 # QU <-> UQ
                W2_z0[3] += toQU * term2 # QU <-> UQ

                if verbose:
                    print('term1 ' + X + Y + ' ' + Xp + Yp + ' empty! ' * (not np.any(term1)))
                if verbose:
                    print('term2 ' + X + Y + ' ' + Xp + Yp + ' empty! ' * (not np.any(term1)))
        for W in [W1_zz, W1_0z, W1_z0, W1_00, W1_10, W1_01]:
            W *= self.Ll1
        for W in [W2_zz, W2_0z, W2_z0, W2_00, W2_10, W2_01]:
            W *= self.Ll2
        #i_sign = 1j ** (ders_1 is not None) * 1j ** (ders_2 is not None)
        W_zz = (W1_zz + W2_zz).astype(complex)
        W_zz[2] *=  0.5
        W_zz[3] *= -0.5j

        W_00 = (W1_00 + W2_00).astype(complex)
        W_00[0] *= (-1)     # (1j) ** 2 factor
        W_00[1] *= (-1)
        W_00[2] *= (-1 *  0.5 )
        W_00[3] *= (-1 * -0.5j)

        W_0_re =  (   0.5 * 1j) * ( (W1_0z + W2_0z) -  (W1_z0 + W2_z0))  # 1j from deriv
        W_0_im =  (   0.5 * 1.) * ( (W1_0z + W2_0z) +  (W1_z0 + W2_z0))  # 1j from deriv

        # W01 has re QQ, re UU, re QU im QU
        W_01 =    (-1 * 0.5) * ( (W1_01 + W2_01) + (W1_10 + W2_10) ).astype(complex) # real part of QQ and UU, and Re and Im of QU
        W_01[3] = (+1 * 0.5j ) * ((W1_01[2] + W2_01[2]) - (W1_10[2] + W2_10[2]) )

        return W_zz, W_00, W_0_re, W_0_im, W_01


    def _W_ST(self, S, T, ders_1=None, ders_2=None, verbose=False):
        """Stokes QE weight function for a pair of Stokes parameter

            Args:
                S: QE weight function first leg Stokes parameter (T, Q or U)
                T: QE weight function second leg Stokes parameter (T, Q or U)
                ders_1: 0 or 1 for the response deflection axis on first leg, or None is absent
                ders_2: 0 or 1 for the response deflection axis on second leg, or None is absent
                verbose: some printout if set

            Note:
                Here l1, l2 are intended to be L/2 + q, L/2 - q.
                The window function for lensing gradient is prop. to (l1 + l2) l1 C_l1 + (l1 <-> l2)


        """
        assert S in ['T', 'Q', 'U'] and T in ['T', 'Q', 'U']
        assert ders_1 in [None, 0, 1], ders_1  # axis of derivative and Cl factor on first leg if relevant
        assert ders_2 in [None, 0, 1], ders_2  # axis of derivative and Cl factor on first leg if relevant
        s = self.l1_int.shape
        W1 = np.zeros(s, dtype=float)  # terms of QE with Cl weight on l1 leg
        W2 = np.zeros(s, dtype=float)  # terms of QE with Cl weight on l2 leg
        Xs = ['T', 'E', 'B']
        X2i = {'T' : 0, 'E' : 1, 'B' : 2}
        for X in ['T'] if S == 'T' else ['E', 'B']:
            RSX_1 = self._X2S(S, X, 1)
            for Y in Xs:
                if ders_1 is not None:
                    cl_XY_1 = self.fFw_ls[X2i[X], X2i[Y]][self.l1_int] * (self.l1s[ders_1]) # wFf transpose..
                    cl_XY_2 = self.fF_ls[ X2i[X], X2i[Y]][self.l1_int] * (self.l1s[ders_1])
                else:
                    cl_XY_1 = self.Fw_ls[X2i[X], X2i[Y]][self.l1_int]
                    cl_XY_2 = self.F_ls[ X2i[X], X2i[Y]][self.l1_int]
                for Xp in (['T'] if Y == 'T' else ['E', 'B']):
                    RtR_YXp = self._X2Y(Y, Xp)
                    for Yp in ['T'] if T == 'T' else ['E', 'B']:
                        if ders_2 is not None:
                            cl_XpYp_1 = self.Ff_ls[ X2i[Xp], X2i[Yp]][self.l2_int] * self.l2s[ders_2]
                            cl_XpYp_2 = self.wFf_ls[X2i[Xp], X2i[Yp]][self.l2_int] * self.l2s[ders_2]

                        else:
                            cl_XpYp_1 =  self.F_ls[ X2i[Xp], X2i[Yp]][self.l2_int]
                            cl_XpYp_2 =  self.wF_ls[X2i[Xp], X2i[Yp]][self.l2_int]

                        RTYp_2 = self._X2S(T, Yp, 2)
                        term1 = (RSX_1 * RTYp_2 * RtR_YXp) * cl_XY_1 * cl_XpYp_1
                        term2 = (RSX_1 * RTYp_2 * RtR_YXp) * cl_XY_2 * cl_XpYp_2

                        W1 += term1
                        W2 += term2

                        if verbose and np.any(term1):
                            print('term1 ' + X + Y + ' ' + Xp + Yp)
                        if verbose and np.any(term2):
                            print('term2 ' + X + Y + ' ' + Xp + Yp)


        W1 *= self.Ll1
        W2 *= self.Ll2
        i_sign = 1j  ** (ders_1 is not None) * 1j ** (ders_2 is not None)
        return i_sign* (W1 + W2)#, W1, W2

    def _get_n1_TS(self, T, S, verbose=False):
        """Factors of xi_00 and xi_11 in N1 for T != S """
        assert T != S
        W_re, W_im, W00_re, W00_im, W_01_re, W_01_im, W_0z_re, W_0z_im, W_z0_re, W_z0_im = self._irfft2(np.array(self._W_TS_odiag(T, S, verbose=verbose)))
        # UQ_{0,z} = - QU_{z, 0}^dagger
        sgn_Q = 1 if 'Q' in [T, S] else -1  # symmetry taking W_(1,) to W_(0,) takes a (-1)^{ (S = Q) + (T = Q)}
        term_00 = 4. * ( (W00_re  * W_re         + W00_im * W_im) + (W_0z_re * W_z0_re      + W_0z_im * W_z0_im))
        term_01 = 4. * ( (W_01_re * W_re         + W_01_im * W_im) + sgn_Q * (W_0z_re * (-W_z0_re.T) - W_0z_im * W_z0_im.T))
        return np.array([term_00, term_01])

    def _get_n1_SS(self, S):
        """Factors of xi_00 and xi_11 in N1 for T == S """
        SS, SS_00, SS_01_re, SS_0z_re, SS_0z_im = self._irfft2(np.array(self._W_SS_diag(S)))
        term_00 =  2 * (SS * SS_00    - (SS_0z_re ** 2         - SS_0z_im ** 2))
        term_01 =  2 * (SS * SS_01_re - (SS_0z_re * SS_0z_re.T - SS_0z_im * SS_0z_im.T))
        return np.array([term_00, term_01])

    def get_n1(self, k, L, do_n1mat=True, _optimize=2, _pyfftw=True):
        L = float(L)
        if not _pyfftw: self._irfft2 = np.fft.irfft2
        _rfft = True
        n1_mat = None
        #if _optimize == 2:
        #     _optimize = 1 if k == 'p_p' else 2
        if _optimize==0:  #--- raw, slower version serving as test case
            Xs = ['T', 'Q', 'U']
            self._build_key(k, L, rfft=False)
            n1 = 0.
            for a in [0, 1]:
                for b in [0, 1]:
                    term1 = 0j
                    term2 = 0j
                    for T in Xs:
                        for S in Xs:
                            term1 += np.fft.ifft2(self._W_ST(T, S, ders_1=a, ders_2=b)) * np.fft.ifft2(self._W_ST(S, T))
                            term2 += np.fft.ifft2(self._W_ST(T, S, ders_1=a)) * np.fft.ifft2(self._W_ST(S, T, ders_1=b))
                    xipp = self.xipp[a + b] if (a + b) != 2 else self.xipp[0].T
                    n1 += np.sum(xipp * (term1 - term2).real)

        elif _optimize == 1:
            assert k in ['p_p', 'x_p']
            # is a bit faster but assumes sep_TP
            # 20 rfft's instead of 5 for T.
            # For small boxes though the building of the weights can be more than the FFT's
            self._build_key(k, L, rfft=_rfft)
            ift = np.fft.irfft2 if _rfft else np.fft.ifft2

            W_zz, W_00, W_0_re, W_0_im, W_01 = ift(np.array(self._W_ST_Pol()))
            QQ, UU, QU_re, QU_im = W_zz
            QQ00, UU00, QU00_re, QU00_im = W_00
            QQ01_re, UU01_re, QU01_re, QU01_im = W_01
            QQ0_re, UU0_re, QU0_re, UQ0_re = W_0_re
            QQ0_im, UU0_im, QU0_im, UQ0_im = W_0_im

            n1_QQ  = np.sum(self.xipp[0] * (QQ * QQ00 - (QQ0_re ** 2 - QQ0_im ** 2)))
            n1_QQ += np.sum(self.xipp[1] * (QQ * QQ01_re - (QQ0_re * QQ0_re.T - QQ0_im * QQ0_im.T )))

            n1_UU  = np.sum(self.xipp[0] * (UU * UU00 - (UU0_re ** 2 - UU0_im ** 2)))
            n1_UU += np.sum(self.xipp[1] * (UU * UU01_re - (UU0_re * UU0_re.T - UU0_im * UU0_im.T )))

            n1_QU  =  np.sum(self.xipp[0] * (QU00_re * QU_re + QU00_im * QU_im))
            n1_QU +=  np.sum(self.xipp[1] * (QU01_re * QU_re + QU01_im * QU_im))
            n1_QU -=  np.sum(self.xipp[0] * (QU0_re * UQ0_re - QU0_im * UQ0_im))
            n1_QU +=  np.sum(self.xipp[1] * (QU0_re * UQ0_re.T - QU0_im * UQ0_im.T))

            n1 = 2 * (n1_QQ + n1_UU) + 4 * n1_QU

        elif _optimize == 2:
            # This assumes nothing except C^{EB} == C^{TB} = 0

            self._build_key(k, L, rfft=_rfft)
            if k in ['p', 'x']:
                terms  = self._get_n1_SS('Q') + self._get_n1_SS('U') + self._get_n1_SS('T')
                terms += self._get_n1_TS('Q', 'U') + self._get_n1_TS('T', 'Q') + self._get_n1_TS('T', 'U')

            elif k in ['ptt', 'xtt']:
                terms = self._get_n1_SS('T')
            elif k in ['p_p', 'x_p']:
                terms  = self._get_n1_SS('Q') + self._get_n1_SS('U') + self._get_n1_TS('Q', 'U')
            else:
                assert 0, k + 'not implemented'
            n1 =  np.sum(self.xipp * terms)
            if do_n1mat:
                ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
                f1 = np.fft.rfft2(terms[0]) * ny ** 2
                f2 = np.fft.rfft2(terms[1]) * nx * ny
                n1_mat = - self.box.sum_in_l(f1.real + f2.real) / np.prod(self.box.shape)
            else:
                n1_mat = 0.
        else:
            n1 = 0
            assert 0
        self._destroy_key(k)
        return -self.norm * n1 if not do_n1mat else (-self.norm * n1, - self.norm * n1_mat)
