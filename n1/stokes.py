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
        cls_f = {k: extcl(self.box.lmaxbox + lminbox, cls_f[k]) for k in cls_f.keys()}    # responses spectra
        cls_w = {k: extcl(self.box.lmaxbox + lminbox, cls_w[k]) for k in cls_w.keys()}   # estimator weights spectra


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

        self.xipp = np.zeros((2, 2, self.shape[0], self.shape[1]))
        self.xipp[0,0] = np.fft.irfft2(extcl(self.box.lmaxbox, -cpp)[ls] * ny ** 2)  # 00
        self.xipp[1,0]=  np.fft.irfft2(extcl(self.box.lmaxbox, -cpp)[ls] * nx * ny)  # 01 or 10
        self.xipp[0,1] = np.fft.irfft2(extcl(self.box.lmaxbox, -cpp)[ls] * nx * ny)  # 01 or 10
        self.xipp[1,1] = np.fft.irfft2(extcl(self.box.lmaxbox, -cpp)[ls] * nx ** 2)  # 01 or 10

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
        l1s =  np.array(self._get_shifted_lylx_sym(-L * 0.5, rfft=rfft))  # this is q + L/2
        l2s = -np.array(self._get_shifted_lylx_sym (L * 0.5, rfft=rfft))  # this is -(q - L/2) = L/2 - q
        self.l1s = l1s
        self.l2s = l2s
        self.l1_int = self.box.rsqd2l(np.sum(l1s ** 2, axis=0))
        self.l2_int = self.box.rsqd2l(np.sum(l2s ** 2, axis=0))
        self.cos2p_sin2p_2v()
        #if k in ['p_p', 'p']:
        #    self.X2S('Q', 'E', 1)

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

    def W_ST_Pol(self, ders_1=None, ders_2=None, verbose=False, rfft=False):
        """Same as W_ST but returns all Stokes weights in one go

            Returns: QQ, UU , QU and UQ

        """
        W1 = np.zeros((4, self.shape[0], self.shape[1]), dtype=float)  # terms of QE with Cl weight on l1 leg
        W2 = np.zeros((4, self.shape[0], self.shape[1]), dtype=float)  # terms of QE with Cl weight on l1 leg
        X2i = {'T': 0, 'E': 1, 'B': 2}
        for X in ['E', 'B']:
            #RSX_1 = self.X2S(S, X, 1)
            #for Y in ['E', 'B']: # Assuming no EB and BE, Y must be X
            Y = X
            if ders_1 is not None:
                cl_XY_1 = self.fFw_ls[X2i[X], X2i[Y]][self.l1_int] * (self.l1s[ders_1])  # wFf transpose..
                cl_XY_2 = self.fF_ls[X2i[X], X2i[Y]][self.l1_int] * (self.l1s[ders_1])
            else:
                cl_XY_1 = self.Fw_ls[X2i[X], X2i[Y]][self.l1_int]
                cl_XY_2 = self.F_ls[X2i[X], X2i[Y]][self.l1_int]
            for Xp in ['E', 'B']:
                RtR_YXp = self.X2Y(Y, Xp)
                #for Yp in ['E', 'B']: # Assuming no EB and BE, Yp must be Xp
                Yp = Xp
                if ders_2 is not None:
                    cl_XpYp_1 = self.Ff_ls[X2i[Xp], X2i[Yp]][self.l2_int] * (self.l2s[ders_2])
                    cl_XpYp_2 = self.wFf_ls[X2i[Xp], X2i[Yp]][self.l2_int] * (self.l2s[ders_2])

                else:
                    cl_XpYp_1 = self.F_ls[X2i[Xp], X2i[Yp]][self.l2_int]
                    cl_XpYp_2 = self.wF_ls[X2i[Xp], X2i[Yp]][self.l2_int]

                #RTYp_2 = self.X2S(T, Yp, 2)

                # Check ordering of X Y Xp Yp: ( XY or YX, XpYp or YpXp ?)
                term1 = RtR_YXp * cl_XY_1 * cl_XpYp_1
                term2 = RtR_YXp * cl_XY_2 * cl_XpYp_2

                W1[0] += self.X2S('Q', X, 1)  * self.X2S('Q', Yp, 2) * term1
                W1[1] += self.X2S('U', X, 1)  * self.X2S('U', Yp, 2) * term1
                W1[2] += self.X2S('Q', X, 1)  * self.X2S('U', Yp, 2) * term1
                W1[3] += self.X2S('U', X, 1)  * self.X2S('Q', Yp, 2) * term1

                W2[0] += self.X2S('Q', X, 1)  * self.X2S('Q', Yp, 2) * term2
                W2[1] += self.X2S('U', X, 1)  * self.X2S('U', Yp, 2) * term2
                W2[2] += self.X2S('Q', X, 1)  * self.X2S('U', Yp, 2) * term2
                W2[3] += self.X2S('U', X, 1)  * self.X2S('Q', Yp, 2) * term2
                if verbose and np.any(term1):
                    print('term1 ' + X + Y + ' ' + Xp + Yp)
                if verbose and np.any(term2):
                    print('term2 ' + X + Y + ' ' + Xp + Yp)

        W1 *= np.sum((self.l1s + self.l2s) * self.l1s, axis=0)
        W2 *= np.sum((self.l2s + self.l1s) * self.l2s, axis=0)
        i_sign = 1j ** (ders_1 is not None) * 1j ** (ders_2 is not None)
        return i_sign * (W1 + W2)  # , W1, W2

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

        W1 = np.zeros(self.shape, dtype=float)  # terms of QE with Cl weight on l1 leg
        W2 = np.zeros(self.shape, dtype=float)  # terms of QE with Cl weight on l2 leg
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

    def get_n1(self, k, L, optimize=False):
        L = float(L)

        self._build_key(k, L, rfft=False)
        # --- precalc of the rfft'ed maps:

        Xs = []
        if k in ['ptt', 'p']: Xs += ['T']
        if k in ['p_p', 'p']: Xs += ['Q', 'U']

        #--- raw version which matches TT perfectly
        if not optimize:
            n1 = 0.
            n1_term2 = 0.
            for a in [0, 1]:
                for b in [0, 1]:
                    term1 = 0j
                    term2 = 0j
                    for T in Xs:
                        for S in Xs:
                            term1 += np.fft.ifft2(self.W_ST(T, S, ders_1=a, ders_2=b)) *  np.fft.ifft2(self.W_ST(S, T))
                            term2 += np.fft.ifft2(self.W_ST(T, S, ders_1=a)) * np.fft.ifft2(self.W_ST(S, T, ders_1=b))
                    n1 += np.sum(self.xipp[a, b] * (term1 - term2))
                    n1_term2 += np.sum(self.xipp[a, b] *  (- term2))
        else:
            i2 = np.fft.ifft2
            ir2  = lambda m: np.fft.irfft2(m[:, :self.box.rshape[1]])
            if k == 'p_p':
                n1 = 0.
                n1_term2 = 0.

                # 4 rffts and 9 ffts -> 22 rffts...  -> 5.5 times larger than T-only

                QQ =  ir2(self.W_ST('Q', 'Q', Ponly=True))
                UU =  ir2(self.W_ST('U', 'U', Ponly=True))
                UQ =  i2(self.W_ST('U', 'Q', Ponly=True))

                QQ_00_QQ = ir2(self.W_ST('Q', 'Q', Ponly=True, ders_1=0, ders_2=0)) * QQ
                UU_00_UU = ir2(self.W_ST('U', 'U', Ponly=True, ders_1=0, ders_2=0)) * UU
                QU_00_UQ = i2( self.W_ST('Q', 'U', Ponly=True, ders_1=0, ders_2=0)) * UQ
                #QU_11_UQ = i2(self.W_ST('Q', 'U', ders_1=1, ders_2=1)) * i2(self.W_ST('U', 'Q')) # real part is transpose of above

                n1 += 2 * np.sum(self.xipp[0, 0] * (QQ_00_QQ + UU_00_UU + 2 * QU_00_UQ.real))

                QQ_01_QQ = i2(self.W_ST('Q', 'Q', Ponly=True, ders_1=0, ders_2=1)) * QQ
                UU_01_UU = i2(self.W_ST('U', 'U', Ponly=True, ders_1=0, ders_2=1)) * UU
                QU_01_UQ = i2(self.W_ST('Q', 'U', Ponly=True, ders_1=0, ders_2=1)) * UQ
                n1 += 2 * np.sum(self.xipp[0, 1] * (QQ_01_QQ.real + UU_01_UU.real + 2 * QU_01_UQ.real))

                #n1 += 4 * np.sum(self.xipp[0, 0] * QU_00_UQ.real)
                #n1 += 2 * np.sum(self.xipp[1, 1] * QU_11_UQ.real)

                # term2, fancy: seems to work so far
                QQ0, UU0, QU0, UQ0 = np.fft.ifft2(self.W_ST_Pol(ders_1=0))
                term2_00 = (QQ0 ** 2).real + (UU0 ** 2).real + 2 * (QU0 * UQ0).real
                term2_01 = (QQ0 * QQ0.T).real + (UU0 * UU0.T).real - 2 * (QU0 * UQ0.T).real

                n1_term2 -= 2. * np.sum(self.xipp[0, 1] * term2_01)
                n1_term2 -= 2. * np.sum(self.xipp[0, 0] * term2_00)
                n1 += n1_term2
        self._destroy_key(k)
        return -self.norm * n1, -self.norm * n1_term2
