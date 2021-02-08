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
from n1.n1_utils import extcl

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
    def __init__(self, box, fals, cls_w, cls_f):

        self.box = box
        self.shape = box.shape


        self._cos2p_sin2p = None

        # Builds required spectra:
        self.F_ls = cls_dot([fals])
        self.wF_ls = cls_dot([cls_w, fals])

        self.fF_ls = cls_dot([cls_f, fals])
        self.Ff_ls = cls_dot([fals,  cls_f])

        self.fwF_ls = cls_dot([cls_f, cls_w, fals])
        self.wFf_ls = cls_dot([cls_w, fals,  cls_f])

    @staticmethod
    def cos2p_sin2p(ly, lx):
        """Returns the cosines and sines of twice the angle of the map of vectors

            Note:
                This assumes no vector is zero

        """
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
        l1s = np.array(self._get_shifted_lylx_sym(-L * 0.5, rfft=rfft))  # this is q + L/2
        l2s = np.array(self._get_shifted_lylx_sym (L * 0.5, rfft=rfft))  # this is q - L/2
        self.l1s = l1s
        self.l2s = l2s
        self.l1_int = self.box.rsqd2l(np.sum(l1s ** 2, axis=0))
        self.l2_int = self.box.rsqd2l(np.sum(l2s ** 2, axis=0))
        self.cos2p_sin2p_2v()

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

    def X2S(self, S, X, vec_1or2):
        """Matrix element sending X cmb mode to stokes flat-sky mode S


        """
        assert S in ['T', 'Q', 'U'], S
        assert X in ['T', 'E', 'B'], X
        assert vec_1or2 in [1, 2], vec_1or2

        ly, lx = self.l1s if vec_1or2 == 1 else self.l2s
        if S == 'T':
            return 1. if X == 'T' else 0.
        if S == 'Q':
            if X == 'E': return self.cos2p_sin2p(ly, lx)[0]
            if X == 'B': return -self.cos2p_sin2p(ly, lx)[1]
            return 0.
        if S == 'U':
            if X == 'E': return self.cos2p_sin2p(ly, lx)[1]
            if X == 'B': return self.cos2p_sin2p(ly, lx)[0]
            return 0.
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

    def W_ST(self, S, T,  ders_1=None, ders_2=None):
        """Stokes QE weight function for Stokes parameter S T

            Note:
                here l1, l2 are intended to be q + L/2, q - L/2 where the W function originally comes from l1, l2 = L - l1
                the window function then is prop. to (l1- l2) l1 C_l1 + sym.

            QQ and UU have rfft's symmetries. QU (same as UQ) do not

        """
        assert S in ['T', 'Q', 'U'] and T in ['T', 'Q', 'U']
        assert ders_1 in [None, 0, 1], ders_1  # axis of derivative and Cl factor on first leg if relevant
        assert ders_2 in [None, 0, 1], ders_2  # axis of derivative and Cl factor on first leg if relevant

        # (qpl[0] * qml[0] + qpl[1] * qml[1])
        W1 = np.zeros(self.shape, dtype=float)  # terms of QE with Cl weight on l1 leg
        W2 = np.zeros(self.shape, dtype=float)  # terms of QE with Cl weight on l2 leg

        X2i = {'T' : 0, 'E' : 1, 'B' : 2}
        for X in ['T'] if S == 'T' else ['E', 'B']:
            RSX_1 = self.X2S(S, X, 1)
            for Y in ['T', 'E', 'B']:
                if ders_1 is not None:
                    cl_XY_1 = self.fwF_ls[X2i[X], X2i[Y]][self.l1_int] * (self.l1s[ders_1])
                    cl_XY_2 = self.fF_ls[ X2i[X], X2i[Y]][self.l1_int] * (self.l1s[ders_1])
                else:
                    cl_XY_1 = self.wF_ls[X2i[X], X2i[Y]][self.l1_int]
                    cl_XY_2 = self.F_ls[ X2i[X], X2i[Y]][self.l1_int]
                for Xp in (['T'] if Y == 'T' else ['E', 'B']):
                    RtR_YXp = self.X2Y(Y, Xp)
                    for Yp in ['T'] if T == 'T' else ['E', 'B']:
                        if ders_2 is not None:
                            cl_XpYp_1 = self.wF_ls[ X2i[Xp], X2i[Yp]][self.l2_int] * (self.l2s[ders_2])
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

                        if np.any(term1):
                            print('term1 ' + X + Y + ' ' + Xp + Yp)
                        if np.any(term2):
                            print('term2 ' + X + Y + ' ' + Xp + Yp)


        W1 *= np.sum( (self.l1s - self.l2s) * self.l1s, axis=0)
        W2 *= np.sum( (self.l2s - self.l1s) * self.l2s, axis=0)

        i_sign = 1j  ** ( (ders_1 is not None) + (ders_2 is not None) )
        W1 = i_sign * W1
        W2 = i_sign * W2

        return W1 + W2, W1, W2