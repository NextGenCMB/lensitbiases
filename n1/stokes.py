r"""One must multiply W^ST W^{SpTp}   (XY IJ in original notation) by f^{SSp}_{l1 l1'} f^{TTp}_{l2 l2'}
    This gives rise to terms like
      (W^{ST} * il1 C_l1^{SSp}  il2 C_l2^{TTp} ) * (W^{SpTp}                                   )
    + (W^{ST} * il1 C_l1^{SSp}                 ) * (W^{SpTp}                   il2  C_l2^{TTp} )
    + (W^{ST} *                 il2 C_l2^{TTp} ) * (W^{SpTp}  il1  C_l1 ^{TTp}                  )
    + (W^{ST} *                                ) * (W^{SpTp}) il1  C_l1 ^{SSp} il2  C_l2^{TTp} )

    Each Cl^{SSp} contains a factor R^{SX} R^{Sp Y} C^{XY}, the sum over S is a delta in the TEB fields, giving

   [  W_{pL}^{ST, 1_a 2_b}(r) * W_{mL}^{ST,        }(r)
    + W_{pL}^{ST, 1_a    }(r) * W_{mL}^{ST,     2_b}(r)
    + W_{pL}^{ST,     2_b}(r) * W_{mL}^{ST, 1_a    }(r)
    + W_{pL}^{ST,        }(r) * W_{mL}^{ST, 1_a 2_b}(r)] * xi_ab(r)

    A trace is intended on the Stokes indices. A factor i l_a C_l^{XY} comes in the relevant place,
    forming e.g. (C^w C^{f,-1}C^w)^{XY} terms


    Symmetries:
        W_{-L}^{ST, 0   0} = W_{+L}^{TS,  0  0}


    Reality conditions:
        W_{L}^{SS, 0, 0} OK, W_{L}^{QU, 0, 0} no

"""
import numpy as np
from n1.n1_utils import extcl, box

def _cldict2arr(cls_dict):
    lmaxp1 = np.max([len(cl) for cl in cls_dict.values()])
    ret = np.zeros((3, 3, lmaxp1), dtype=float)
    for i, x in enumerate(['t', 'e', 'b']):
        for j, y in enumerate(['t', 'e', 'b']):
            ret[i, j] =  extcl(lmaxp1 - 1, cls_dict.get(x + y, cls_dict.get(y + x, np.array([0.]))))
    return ret

def cls_dot(cls_list):
    """T E B spectral matrices product

        Args:
            list of dict cls spectral matrices to multiply (given as dictionaries or (3, 3, lmax + 1) arrays

        Returns:
            (3, 3, lmax + 1) array where 0, 1, 2 stands for T E B


    """
    if  len(cls_list) == 1:
        return _cldict2arr(cls_list[0]) if isinstance(cls_list[0], dict) else cls_list[0]
    cls = cls_dot(cls_list[1:])
    cls_0 =  _cldict2arr(cls_list[0]) if isinstance(cls_list[0], dict) else cls_list[0]
    ret = np.zeros_like(cls_0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ret[i, j] += cls_0[i, k] * cls[k, j]
    return ret



class stokes:
    def __init__(self, fals, cls_w, cls_f, cpp, lminbox=50, lmaxbox=2500):

        lside = 2. * np.pi / lminbox
        npix = int(lmaxbox / np.pi * lside) + 1
        if npix % 2 == 1: npix += 1

        #===== instance with 2D flat-sky box info
        self.box = box(lside, npix)
        self.shape = self.box.shape


        self._cos2p_sin2p = None
        self._cos2p_sin2p_v1 = None
        #==== Builds required spectra:
        # === Filter and cls array needed later on:
        fals   = {k: extcl(self.box.lmaxbox + lminbox, fals[k]) for k in fals.keys()}
        cls_f  = {k: extcl(self.box.lmaxbox + lminbox, cls_f[k]) for k in cls_f.keys()}    # responses spectra
        cls_w  = {k: extcl(self.box.lmaxbox + lminbox, cls_w[k]) for k in cls_w.keys()}   # estimator weights spectra


        self.F_ls = cls_dot([fals])
        self.wF_ls = cls_dot([cls_w, fals])
        self.Fw_ls = cls_dot([fals, cls_w])

        self.fF_ls = cls_dot([cls_f, fals])
        self.Ff_ls = cls_dot([fals,  cls_f])

        self.fFw_ls = cls_dot([cls_f, fals, cls_w])
        self.wFf_ls = cls_dot([cls_w, fals, cls_f])


        # === precalc of deflection corr fct:
        ny, nx = np.meshgrid(self.box.ny_1d, self.box.nx_1d, indexing='ij')
        ls = self.box.ls()

        self.xipp = {0 : np.fft.irfft2(extcl(self.box.lmaxbox, -cpp)[ls] * ny ** 2),  # 00, 11.T
                     1 : np.fft.irfft2(extcl(self.box.lmaxbox, -cpp)[ls] * nx * ny) } # 01 or 10

        del nx, ny, ls

        # === normalization (for tt keys at least)
        norm = (self.box.shape[0] / self.box.lsides[0]) ** 4  # overall final normalization from rfft'ing
        norm *= (float(self.box.lminbox)) ** 8
        # :always 2 powers in xi_ab, 4 powers of ik_x or ik_y in XY and IJ weights, and two add. powers matching xi_ab's from the responses
        self.norm = norm


    @staticmethod
    def cos2p_sin2p(ls):
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

        if k in ['p_p', 'p']:
            l1s, l2s = (self.l1s, self.l2s)
            r1sqd_r2sqd = np.sum(l1s ** 2, axis=0) * np.sum(l2s ** 2, axis=0)
            dotp = np.sum(l1s * l2s, axis=0)
            cos2p = 2 * dotp ** 2 / r1sqd_r2sqd - 1.
            sin2p = 2 * dotp * (l2s[1] * l1s[0] - l2s[0] * l1s[1]) / r1sqd_r2sqd
            self._cos2p_sin2p = (cos2p, sin2p)
            self._cos2p_sin2p_v1 = {1: self.cos2p_sin2p(l1s), 2: self.cos2p_sin2p(l2s)}


    def _destroy_key(self, k):
        self._cos2p_sin2p_v1 = None
        self._cos2p_sin2p = None

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

    def X2S(self, S, X, vec1or2):
        """Matrix element sending X cmb mode to stokes flat-sky mode S


        """
        if S == 'T':
            return 1. if X == 'T' else 0.
        if S == 'Q':
            if X == 'T':
                return 0.
            if self._cos2p_sin2p_v1 is None:
                self._cos2p_sin2p_v1 = {1:  self.cos2p_sin2p(self.l1s), 2: self.cos2p_sin2p(self.l2s)}
            sgn = 1 if X == 'E' else -1
            return sgn * self._cos2p_sin2p_v1[vec1or2][0 if X == 'E' else 1]
        if S == 'U':
            if X == 'T':
                return 0.
            if self._cos2p_sin2p_v1 is None:
                self._cos2p_sin2p_v1 = {1: self.cos2p_sin2p(self.l1s), 2: self.cos2p_sin2p(self.l2s)}
            return self._cos2p_sin2p_v1[vec1or2][1 if X == 'E' else 0]
        assert 0

    def X2Y(self, Y, X):
        """Matrix element

                (R^tR)^{YX}_{l1,l2} = \sum_{S in T,Q,U} R^{S Y}_{l1} R^{S X}_{l2}

        """
        assert X in ['T', 'E', 'B'] and Y in ['T', 'E', 'B'], (X, Y)

        if X == 'T' or Y == 'T':
            return 1. if X == Y else 0.
        if X == Y:
            return self.cos2p_sin2p_2v()[0] # same as c1 * c1 + s1 * s2 = cos2p_12
        return (1 if X == 'B' else -1) * self.cos2p_sin2p_2v()[1]# same as -+ (c1 s2 - s1 c2) = -+ sin_2p_12:


    def W_ST_Pol(self, verbose=False):
        """Same as W_ST but returns all Stokes weights in one go

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
            #RSX_1 = self.X2S(S, X, 1)
            #for Y in ['E', 'B']: # Assuming no EB and BE, Y must be X
            Y = X
            cl_XY_1 = self.Fw_ls[X2i[X], X2i[Y]]
            cl_XY_2 = self.F_ls[X2i[X], X2i[Y]]
            cl_XY_1_0 = self.fFw_ls[X2i[X], X2i[Y]]
            cl_XY_2_0 = self.fF_ls[X2i[X], X2i[Y]]
            for Xp in ['E', 'B']:
                RtR_YXp = self.X2Y(Y, Xp)
                #for Yp in ['E', 'B']: # Assuming no EB and BE, Yp must be Xp
                Yp = Xp
                cl_XpYp_1 = self.F_ls[X2i[Xp], X2i[Yp]]
                cl_XpYp_2 = self.wF_ls[X2i[Xp], X2i[Yp]]
                cl_XpYp_1_0 = self.Ff_ls[X2i[Xp], X2i[Yp]]
                cl_XpYp_2_0 = self.wFf_ls[X2i[Xp], X2i[Yp]]

                toQQ = self.X2S('Q', X, 1) * self.X2S('Q', Yp, 2)
                toUU = self.X2S('U', X, 1) * self.X2S('U', Yp, 2)
                toQU = self.X2S('Q', X, 1) * self.X2S('U', Yp, 2)
                toUQ = self.X2S('U', X, 1) * self.X2S('Q', Yp, 2)



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

                # ==== terms with two derivatives (1 0) and (0 1) (with swapped QU - UQ) #FIXME: could save some calc here..
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

        Ll1 = np.sum((self.l1s + self.l2s) * self.l1s, axis=0)
        Ll2 = np.sum((self.l1s + self.l2s) * self.l2s, axis=0)
        for W in [W1_zz, W1_0z, W1_z0, W1_00, W1_10, W1_01]:
            W *= Ll1
        for W in [W2_zz, W2_0z, W2_z0, W2_00, W2_10, W2_01]:
            W *= Ll2
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


    def W_ST(self, S, T,  ders_1=None, ders_2=None, verbose=False, Ponly=False):
        """Stokes QE weight function for Stokes parameter S T

            Note:
                here l1, l2 are intended to be q + L/2, q - L/2 where the W function originally comes from l1, l2 = L - l1
                the window function then is prop. to (l1- l2) l1 C_l1 + sym.

            QQ and UU have rfft's symmetries. QU (same as UQ) do not

        """
        assert S in ['T', 'Q', 'U'] and T in ['T', 'Q', 'U']
        assert ders_1 in [None, 0, 1], ders_1  # axis of derivative and Cl factor on first leg if relevant
        assert ders_2 in [None, 0, 1], ders_2  # axis of derivative and Cl factor on first leg if relevant
        s = self.l1_int.shape
        W1 = np.zeros(s, dtype=float)  # terms of QE with Cl weight on l1 leg
        W2 = np.zeros(s, dtype=float)  # terms of QE with Cl weight on l2 leg
        Xs = ['T'] * (not Ponly) + ['E', 'B']
        X2i = {'T' : 0, 'E' : 1, 'B' : 2}
        for X in ['T'] if S == 'T' else ['E', 'B']:
            RSX_1 = self.X2S(S, X, 1)
            for Y in Xs:
                if ders_1 is not None:
                    cl_XY_1 = self.fFw_ls[X2i[X], X2i[Y]][self.l1_int] * (self.l1s[ders_1]) # wFf transpose..
                    cl_XY_2 = self.fF_ls[ X2i[X], X2i[Y]][self.l1_int] * (self.l1s[ders_1])
                else:
                    cl_XY_1 = self.Fw_ls[X2i[X], X2i[Y]][self.l1_int]
                    cl_XY_2 = self.F_ls[ X2i[X], X2i[Y]][self.l1_int]
                for Xp in (['T'] if Y == 'T' else ['E', 'B']):
                    RtR_YXp = self.X2Y(Y, Xp)
                    for Yp in ['T'] if T == 'T' else ['E', 'B']:
                        if ders_2 is not None:
                            cl_XpYp_1 = self.Ff_ls[ X2i[Xp], X2i[Yp]][self.l2_int] * (self.l2s[ders_2])
                            cl_XpYp_2 = self.wFf_ls[X2i[Xp], X2i[Yp]][self.l2_int] * (self.l2s[ders_2])

                        else:
                            cl_XpYp_1 =  self.F_ls[ X2i[Xp], X2i[Yp]][self.l2_int]
                            cl_XpYp_2 =  self.wF_ls[X2i[Xp], X2i[Yp]][self.l2_int]

                        RTYp_2 = self.X2S(T, Yp, 2)

                        # Check ordering of X Y Xp Yp: ( XY or YX, XpYp or YpXp ?)
                        term1 = (RSX_1 * RTYp_2 * RtR_YXp) * cl_XY_1 * cl_XpYp_1
                        term2 = (RSX_1 * RTYp_2 * RtR_YXp) * cl_XY_2 * cl_XpYp_2

                        W1 += term1
                        W2 += term2

                        if verbose and np.any(term1):
                            print('term1 ' + X + Y + ' ' + Xp + Yp)
                        if verbose and np.any(term2):
                            print('term2 ' + X + Y + ' ' + Xp + Yp)


        W1 *= np.sum( (self.l1s + self.l2s) * self.l1s, axis=0)
        W2 *= np.sum( (self.l2s + self.l1s) * self.l2s, axis=0)
        i_sign = 1j  ** (ders_1 is not None) * 1j ** (ders_2 is not None)
        return i_sign* (W1 + W2)#, W1, W2

    def W_spin(self, s1, s2, ders_1=None, ders_2=None):
        #FIXME: hack
        WQ = 0j
        WU = 0j
        if s1 in [-2, 2]:
            WQ += ( self.W_ST('Q', 'Q', ders_1=ders_1, ders_2=ders_2) + 1j * s1 / abs(s1) * self.W_ST('U', 'Q', ders_1=ders_1, ders_2=ders_2) )
            WU += ( self.W_ST('Q', 'U', ders_1=ders_1, ders_2=ders_2) + 1j * s1 / abs(s1) * self.W_ST('U', 'U', ders_1=ders_1, ders_2=ders_2) )
        return WQ +  1j * s2 / abs(s2) * WU


    def get_n1(self, k, L, optimize=2):
        L = float(L)
        _rfft = True
        # --- precalc of the rfft'ed maps:

        Xs = []
        if k in ['ptt', 'p']: Xs += ['T']
        if k in ['p_p', 'p']: Xs += ['Q', 'U']

        #--- raw version which matches TT perfectly
        if not optimize:
            self._build_key(k, L, rfft=False)
            n1 = 0.
            n1_QQ = 0.
            n1_UU = 0.
            n1_QU = 0.
            for a in [0, 1]:
                for b in [0, 1]:
                    term1 = 0j
                    term2 = 0j
                    for T in Xs:
                        for S in Xs:
                            term1 += np.fft.ifft2(self.W_ST(T, S, ders_1=a, ders_2=b)) *  np.fft.ifft2(self.W_ST(S, T))
                            term2 += np.fft.ifft2(self.W_ST(T, S, ders_1=a)) * np.fft.ifft2(self.W_ST(S, T, ders_1=b))
                    xipp = self.xipp[a + b] if (a + b) != 2 else self.xipp[0].T
                    n1 += np.sum(xipp * (term1 - term2))

        elif optimize == 2:
            # 20 rfft's instead of 5 for T.
            # For small boxes though the building of the weights can be more than the FFT's
            self._build_key(k, L, rfft=_rfft)
            ift = np.fft.irfft2 if _rfft else np.fft.ifft2

            W_zz, W_00, W_0_re, W_0_im, W_01 = ift(np.array(self.W_ST_Pol()))
            QQ, UU, QU_re, QU_im = W_zz
            QQ00, UU00, QU00_re, QU00_im = W_00
            QQ01_re, UU01_re, QU01_re, QU01_im = W_01
            QQ0_re, UU0_re, QU0_re, UQ0_re = W_0_re
            QQ0_im, UU0_im, QU0_im, UQ0_im = W_0_im

            n1_QQ  = 2  * np.sum(self.xipp[0] * (QQ * QQ00 - (QQ0_re ** 2 - QQ0_im ** 2)))
            n1_QQ += 2. * np.sum(self.xipp[1] * (QQ * QQ01_re - (QQ0_re * QQ0_re.T - QQ0_im * QQ0_im.T )))

            n1_UU  = 2  * np.sum(self.xipp[0] * (UU * UU00 - (UU0_re ** 2 - UU0_im ** 2)))
            n1_UU += 2. * np.sum(self.xipp[1] * (UU * UU01_re - (UU0_re * UU0_re.T - UU0_im * UU0_im.T )))

            n1_QU  =  4 * np.sum(self.xipp[0] * (QU00_re * QU_re + QU00_im * QU_im))
            n1_QU +=  4 * np.sum(self.xipp[1] * (QU01_re * QU_re + QU01_im * QU_im))
            n1_QU -=  4 * np.sum(self.xipp[0] * (QU0_re * UQ0_re - QU0_im * UQ0_im))
            n1_QU +=  4 * np.sum(self.xipp[1] * (QU0_re * UQ0_re.T - QU0_im * UQ0_im.T))

            n1 = n1_QQ + n1_UU + n1_QU

        self._destroy_key(k)
        return -self.norm * n1, -self.norm * n1_QQ, -self.norm * n1_UU, -self.norm * n1_QU
