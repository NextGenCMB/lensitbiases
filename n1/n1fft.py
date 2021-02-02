import numpy as np
import pyfftw
import os
import plancklens
from plancklens import utils
from scipy.interpolate import UnivariateSpline as spl
from n1.scripts import test2plancklens_1002000 as tp
from n1 import n1 as n1f
CLS = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')


def Freq(i, N):
    """
     Outputs the absolute integers frequencies [0,1,...,N/2,N/2-1,...,1]
     in numpy fft convention as integer i runs from 0 to N-1.
     Inputs can be numpy arrays e.g. i (i1,i2,i3) with N (N1,N2,N3)
                                  or i (i1,i2,...) with N
     Both inputs must be integers.
     All entries of N must be even.
    """
    assert (np.all(N % 2 == 0)), "This routine only for even numbers of points"
    return i - 2 * (i >= (N // 2)) * (i % (N // 2))


def extcl(lmax, cl):
    if len(cl) - 1 < lmax:
        dl = np.zeros(lmax + 1)
        dl[:len(cl)] = cl
    else:
        dl = cl
    return dl

def get_ellmat(lmin, lmax):
    lside = 2. * np.pi / lmin
    npix = int(lmax  / np.pi * lside)
    if npix % 2 == 1: npix += 1
    from lensit.ffs_covs.ell_mat import  ell_mat
    return ell_mat(None, (npix, npix), (lside, lside), cache=0)


def get_fXY(k, ls):
    pass

def shiftF(F, npix, Laxis):
    # Shifts Fourier coefficients by npix values
    # Fourier rrft with
    #return np.fft.ifftshift(np.roll(np.fft.fftshift(F), npix, axis=Laxis))
    return np.roll(F, npix, axis=Laxis)



def get_N1pyfftw_XY(npixs, fals, cls_grad, xory='x', ks='p', Laxis=0,
                    lminbox=50, lmaxbox=5000, lminphi=0, lmaxphi=4096,
                    cpp=None, spline_result=False,
                    fftw=False, use_sym=True, rFFT=False):
    """This works very fine for simple weight functions but could speed up lensing parts etc"""
    # Defines grid:
    #FIXME: the way to get the l-L part assumes conj. prop. of the weights
    #FIXME: rffts when possible


    #rFFT = False
    #assert not rFFT, 'check carefully implications'

    lside = 2. * np.pi / lminbox
    npix = int(lmaxbox  / np.pi * lside)
    if npix % 2 == 1: npix += 1
    shape  = (npix, npix)
    rshape = (npix, npix // 2 + 1)
    fft_shape = rshape if rFFT else shape
    norm = 0.25 * (npix / lside) ** 4 #overall final normalization

    #=== frequencies
    nx = Freq(np.arange(fft_shape[1]), shape[1]) # unsigned FFT frequencies
    ny = Freq(np.arange(shape[0]), shape[0])
    nx[shape[1] // 2:] *= -1
    ny[shape[0] // 2:] *= -1
    nx = np.outer(np.ones(shape[0]), nx)
    ny = np.outer(ny, np.ones(fft_shape[1]))
    ls = np.int_(np.round(2 * np.pi * np.sqrt(nx ** 2 + ny ** 2) / lside))
    lmax_seen = ls[npix // 2, npix // 2]

    if fftw:
        def ifft(arr):
            inpt = pyfftw.byte_align(arr, dtype='complex128')
            #return pyfftw.interfaces.numpy_fft.irfft2(inpt) if rFFT else pyfftw.interfaces.numpy_fft.ifft2(inpt)
            outp = pyfftw.empty_aligned(shape, dtype='float64' if rFFT else 'complex128')
            fft_ob = pyfftw.FFTW(inpt, outp, axes=(0, 1), direction='FFTW_BACKWARD', flags=['FFTW_MEASURE'])
            fft_ob()
            return outp
    else:
        ifft = np.fft.irfft2 if rFFT else np.fft.ifft2


    print('lmin max %s %s ' % (int(2. * np.pi / lside), lmax_seen) + str(shape))
    CTT = extcl(lmax_seen, cls_grad['tt'])
    if cpp is None :
        cls_unl = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_lenspotentialCls.dat'))
        cpp = cls_unl['pp'][:lmaxphi + 1]

    cpp[lmaxphi + 1:] *= 0.
    cpp[:lminphi] *= 0.
    xipp = np.fft.irfft2(extcl(lmax_seen,cpp)[ls if rFFT else ls[:, 0:rshape[1]]])


    dL2did = lambda L: int(lside * L / (2 * np.pi))
    dnpix2L = lambda n: 2. * np.pi / lside * n

    ones = np.ones(fft_shape, dtype=float)
    if ks[0] == 'p':
        fresp = []
        fresp += [(CTT[ls] * (nx ** 2 + ny ** 2), ones)]
        fresp += [(-CTT[ls] * 1j * nx, 1j * nx)]
        fresp += [(-CTT[ls] * 1j * ny, 1j * ny)]
        for i in range(3):  # symzation
            fresp += [(fresp[i][1], fresp[i][0])]
        norm *= lminbox ** 4  # 4 powers of n

    elif ks[0] == 's':
        fresp =  [(ones, ones)]
    elif ks[0] == 'f':
        fresp = [(ones, CTT[ls])]
        if use_sym:
            norm *= 2.
            use_sym = False
        else:
            fresp +=  [(CTT[ls], ones)]
    else:
        assert 0
    if xory == 'xx':
        kA = [ (1., 1j * nx * CTT[ls]) ]  # input QE A #TODO: check no need to symmetrize here
        kB = kA  # input QE B
        norm *= 4. * lminbox ** 2
    elif xory == 'yy':
        kA = [(1.,  1j * ny * CTT[ls])]  # input QE A #TODO: check no need to symmetrize here
        kB = kA
        norm *= 4 * lminbox ** 2
    elif xory == 'xy':
        kA = [[1., 1j * nx * CTT[ls]]]  # input QE A
        kB = [[1., 1j * ny * CTT[ls]]]  # input QE B
        norm *= 4 * lminbox ** 2
    elif xory == 'stt':
        kA = [[1., 1.]]
        kB = [[1., 1.]]
    elif xory == 'ftt':
        kA = [[1., CTT[ls]]]
        kA +=  [[CTT[ls], 1.]]
        kB = [[1., CTT[ls]]]
        if use_sym: #FIXME: figure out better the symmetries that can be used
            norm *= 2.
            use_sym = False
        else:
            kB +=  [[CTT[ls], 1.]]

    else:
        kA  = [(CTT[ls] * (nx ** 2 + ny ** 2), 1.)]
        kA += [(-CTT[ls] * 1j * nx, 1j * nx)]
        kA += [(-CTT[ls] * 1j * ny, 1j * ny)]
        for i in range(3):  # symzation
            kA += [(kA[i][1], kA[i][0])]
        if use_sym:
            kB = kA[:len(kA) // 2]
            norm *= 2.
            use_sym = False
        else:
            kB = kA
        norm *= lminbox ** 4 # 4 powers of n (2 for kA, 2 for kB)

    fresp2 = fresp
    if use_sym and len(fresp) > 1:
        norm *= 2.
        fresp2 = fresp[:len(fresp) // 2]
        use_sym = False
    # Weigthing by filters:
    FX = FY = FI = FJ = extcl(lmax_seen, fals['tt'])[ls]
    print("expected number of XY-IJ terms = %s"%(len(fresp) * len(fresp2) * len(kA) * len(kB)))
    n1_test = np.zeros(len(npixs), dtype=float)
    n1_test_im = np.zeros(len(npixs), dtype=float)
    n1_test1 = np.zeros(len(npixs), dtype=float)
    n1_test2 = np.zeros(len(npixs), dtype=float)

    for ip, npix in enumerate(npixs):
        term1 = np.zeros(shape, dtype=float if rFFT else complex)
        term2 = np.zeros(shape, dtype=float if rFFT else complex)
        for wXY in kA:
            for wIJ in kB:
                for fXI in fresp:
                    w1 = wXY[0] * fXI[0] * FX
                    w3 = wIJ[0] * fXI[1] * FI
                    for fYJ in fresp2:
                        w2 = wXY[1] * fYJ[0]* FY
                        w4 = wIJ[1] * fYJ[1]* FJ
                        h12 = ifft(w1 * shiftF(w2, npix, Laxis).conj())
                        h34 = ifft(w3 * shiftF(w4, -npix, Laxis).conj())
                        term1 += h12 * h34
                for fXJ in fresp:
                    w1 = wXY[0] * fXJ[0]* FX
                    w4 = wIJ[1] * fXJ[1]* FJ
                    for fYI in fresp2:
                        #FIXME: under some cond. not necessary to redo h12
                        w2 = wXY[1] * fYI[0] * FY
                        w3 = wIJ[0] * fYI[1] * FI
                        h12 = ifft(w1 * shiftF(w2, npix, Laxis).conj())
                        h43 = ifft(w4 * shiftF(w3,  -npix, Laxis).conj())
                        term2 += h12 * h43
        reim1 = np.sum(xipp * term1)
        reim2 = np.sum(xipp * term2)

        n1_test[ip] = reim1.real + reim2.real
        n1_test_im[ip] = reim1.imag + reim2.imag
        n1_test1[ip] = reim1.real
        n1_test2[ip] = reim2.real

    Lspix = dnpix2L(npixs) # L corresponding to the given deflections
    if not spline_result:
        return norm * n1_test, Lspix, n1_test1, n1_test2
    else:
        Lsout = np.arange(0, lmaxphi + 1)
        return spl(Lspix, norm * n1_test, ext='zeros', k=3, s=0)(Lsout), Lsout, term1, term2


class n1_ptt:
    def __init__(self, fals, cls_grad, lminbox=100, lmaxbox=5000, lminphi=0, lmaxphi=2500, cpp=None, rFFT=False):
        """Looks like this works fine...

        """
        lside = 2. * np.pi / lminbox
        npix = int(lmaxbox / np.pi * lside)
        if npix % 2 == 1: npix += 1
        shape = (npix, npix)
        rshape = (npix, npix // 2 + 1)
        fft_shape = rshape if rFFT else shape

        # === frequencies
        nx = Freq(np.arange(fft_shape[1]), shape[1])  # unsigned FFT frequencies
        ny = Freq(np.arange(shape[0]), shape[0])
        nx[shape[1] // 2:] *= -1
        ny[shape[0] // 2:] *= -1
        self.nx_1d = nx.copy()
        self.ny_1d = ny.copy()

        nx = np.outer(np.ones(shape[0]), nx)
        ny = np.outer(ny, np.ones(fft_shape[1]))
        self.lminbox = lminbox

        ls = self.freq2ls(nx, ny)
        lmax_seen = ls[npix // 2, npix // 2]

        self.shape = shape
        self._wTT_pL = None
        self._wTT_mL = None
        self._wTT = None

        if cpp is None:
            cls_unl = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_lenspotentialCls.dat'))
            cpp = cls_unl['pp'][:lmaxphi + 1]

        cpp[lmaxphi + 1:] *= 0.
        cpp[:lminphi] *= 0.
        rfft_sli = slice(0, rshape[1])
        self.xiab = np.zeros((3, shape[0], shape[1]), dtype=float)
        self.xiab[0] = np.fft.irfft2(extcl(lmax_seen, cpp)[ls[:, rfft_sli]] * (1j * ny[:, rfft_sli]) ** 2)  # 00
        self.xiab[2] = np.fft.irfft2(extcl(lmax_seen, cpp)[ls[:, rfft_sli]] * (1j * nx[:, rfft_sli]) ** 2)  # 11
        self.xiab[1] = np.fft.irfft2(
            extcl(lmax_seen, cpp)[ls[:, rfft_sli]] * (1j * ny[:, rfft_sli]) * (1j * nx[:, rfft_sli]))  # 01 or 10

        lmax_spec = 2 * lmax_seen# + lminbox
        self.F = extcl(lmax_spec , fals['tt'])  # + 1 because of shifted freq. boxes
        self.ctt = extcl(lmax_spec, cls_grad['tt'])

        #self.ctt_mat = extcl(lmax_spec , cls_grad['tt'])[ls]
        #self.Ft_mat = extcl(lmax_spec, fals['tt'])[ls]

        self.ls = ls

        self.ns = np.array([ny, nx])
        self.lside = lside

        norm = (npix / self.lside) ** 4  # overall final normalization
        norm *= (float(lminbox)) ** 8  # always 2 powers in xi_ab, 4 powers of ik_x or ik_y in g's, and final rescaling by L ** 2
        self.norm = -1 * norm

        # shifted L stuff
        self.shifted_p_nynx = np.empty_like(self.ns)
        self.shifted_p_ls = np.empty_like(self.ls)
        self.shifted_m_nynx = np.empty_like(self.ns)
        self.shifted_m_ls = np.empty_like(self.ls)
    @staticmethod
    def shiftF(F, npix, Laxis):
        # Shifts Fourier coefficients by npix values
        # Fourier rrft with
        # return np.fft.ifftshift(np.roll(np.fft.fftshift(F), npix, axis=Laxis))
        return np.roll(F, npix, axis=Laxis)


    def get_shifted_lylx(self, L, Laxis):
        """Builds frequency maps shifted by L

            Returns:
                frequency maps k_y - L_y, k_x - L_x   (l_1 - L) = -l_2

        """
        nLs = L / self.lminbox
        iL, dL = (int(np.round(nLs)), nLs - int(np.round(nLs)))
        fL = np.roll(self.ns[Laxis], iL, axis=Laxis)  # rolling with positive L is taking l - L
        fLp = self.ns[0] if Laxis else self.ns[1]
        return (fLp, fL - dL) if Laxis else (fL -dL, fLp)

    def get_shifted_lylx_sym(self, L):
        """Shifts by L/root(2) in both directions

            Returns:
                frequency maps k_y - L/root(2), k_x - L/root(2)

        """
        ly = np.outer(self.ny_1d -  L / np.sqrt(2.) / self.lminbox, np.ones(len(self.ny_1d)))
        return ly, ly.transpose()

        #nLs = L / np.sqrt(2.) / self.lminbox
        #iL, dL = (int(np.round(nLs)), nLs - int(np.round(nLs)))
        #ly = np.outer(np.roll(self.ny_1d, iL) -dL, np.ones(self.shape[1]))
        #return ly, np.copy(ly.transpose())


    @staticmethod
    def flip(F):
        """F(-r)


        """
        ret = np.empty_like(F)
        npix0, npix1 = ret.shape
        sli = slice(npix1, 0, -1)
        for i in range(npix0):
            ret[i, 1:] = F[-i, sli]
        ret[0, 0] = F[0, 0]
        ret[1:, 0] = F[slice(npix0, 0, -1), 0]
        return ret

    def freq2ls(self, nx, ny):
        """Maps flat-sky frequencies onto integer multipoles

        """
        return np.int_(np.round(self.lminbox * (np.sqrt(nx ** 2 + ny ** 2))))

    def get_g(self, dlpix, pi, pj, ders_i, ders_j, Laxis=0):
        """It must hold

            g_{+L}^ij_{a b , c d ...} = (-1)^{i+j} g^{ji}_{-L} (r) e^{i L \cdot r}

        """
        assert len(ders_i) == pi  # one derivative per power  (e.g. [0, 0] for 2 derivatives on y-axis)
        assert len(ders_j) == pj

        wi = (self.F * self.ctt ** pi)[self.ls] * 1j ** pi  # l leg
        wj = (self.F * self.ctt ** pj)[self.ls] * 1j ** pj  # (L - l) leg
        for der_i in ders_i:
            wi *= self.ns[der_i]
        for der_j in ders_j:
            wj *= self.ns[der_j]

        return np.fft.ifft2(wi * self.shiftF(wj, dlpix, Laxis).conj())

    def get_gf(self, L, pi, pj, ders_i, ders_j, Laxis=0):
        """

            This is formally \int_{l1, l2} W^{TT}(l1, l2) * e^{i l_1 r}  ( i Cl_1 l_1)_(ders_i)( i Cl_2 l_2)_(ders_j)
                                where l2 is L - l1

        """

        ml2_yx = self.shifted_p_nynx  if L > 0 else  self.shifted_m_nynx # this is -l2
        l2s = self.shifted_p_ls if L > 0 else self.shifted_m_ls

        w = -(self.ns[1] * ml2_yx[1] + self.ns[0] * ml2_yx[0]) * ( self.ctt[l2s] + self.ctt[self.ls])
        w += self.ctt[self.ls] * ( self.ns[1] ** 2 + self.ns[0] ** 2)
        w += self.ctt[l2s] * ( ml2_yx[0] ** 2 + ml2_yx[1] ** 2)
        w *= self.F[self.ls] * self.F[l2s]
        #FIXME: these calcs up to here could be done only once per L

        if pi > 0:
            w *= (self.ctt ** pi)[self.ls]
            for deri in ders_i:
                w *= self.ns[deri]
        if pj > 0:
            w *= (self.ctt ** pj)[l2s]
            for derj in ders_j:
                w *= ml2_yx[derj]
        return np.fft.ifft2(w)

    def _set_wtt(self, L, Laxis=0):
        ml2_yx = self.get_shifted_lylx(L, Laxis=Laxis)  # if L > 0 else  self.shifted_m_nynx # this is -l2
        l2s = self.freq2ls(ml2_yx[1], ml2_yx[0])
        # ml2_yx = self.shifted_m_nynx  # this is -l2
        # l2s = self.shifted_m_ls

        w = -(self.ns[1] * ml2_yx[1] + self.ns[0] * ml2_yx[0]) * (self.ctt[l2s] + self.ctt[self.ls])
        w += self.ctt[self.ls] * (self.ns[1] ** 2 + self.ns[0] ** 2)
        w += self.ctt[l2s] * (ml2_yx[0] ** 2 + ml2_yx[1] ** 2)
        w *= self.F[self.ls] * self.F[l2s]
        if L > 0:
            self._wTT_pL = (w, l2s, ml2_yx)
        else:
            self._wTT_mL = (w, l2s, ml2_yx)


    def _get_wtt_sym(self, L, Laxis=2, rfft=False):
        """Symmetrized TT QE func for L + q, L- q param.

            This has the real fftws conjugacy property

            2 C_{L + q} (L + q) L + 2 C_{L - q} (L - q) L

        """
        if self._wTT is None:
            if Laxis == 2:
                qml = np.array(self.get_shifted_lylx_sym( L * 0.5))  # this is q - L/2
                qpl = np.array(self.get_shifted_lylx_sym(-L * 0.5))  # this is q + L/2
                #nLs = 0.5 * L / np.sqrt(2.) / self.lminbox
                #iL, dL = (int(np.round(nLs)), nLs - int(np.round(nLs)))
                #ly = np.outer(np.roll(self.ny_1d, iL) - dL, np.ones(self.shape[1]))
                #qml = np.array([ly, ly.transpose()]) # this is q - L/2
                #ly = np.outer(np.roll(self.ny_1d, -iL) + dL, np.ones(self.shape[1]))
                #qpl = np.array([ly, ly.transpose()]) # this is q + L/2

            else:
                qml = np.array(self.get_shifted_lylx( L * 0.5, Laxis)) # this is q - L/2
                qpl = np.array(self.get_shifted_lylx(-L * 0.5, Laxis)) # this is q + L/2

            ls_m_sqd = qml[0] ** 2 + qml[1] ** 2
            ls_p_sqd = qpl[0] ** 2 + qpl[1] ** 2
            ls_m = np.int_(np.round(self.lminbox * np.sqrt(ls_m_sqd)))
            ls_p = np.int_(np.round(self.lminbox * np.sqrt(ls_p_sqd)))
            w = - (self.ctt[ls_p] * ls_p_sqd +  self.ctt[ls_m] * ls_m_sqd)
            w +=  (self.ctt[ls_p] + self.ctt[ls_m]) * (qpl[0] * qml[0] + qpl[1] * qml[1])
            w *= self.F[ls_m] * self.F[ls_p]
            sli = slice(0, self.shape[1] // 2 + 1)

            self._wTT = ( w, qml, qpl, ls_m, ls_p)
            self._wTT_rfft = (w[:, sli], qml[:, :, sli], qpl[:,:, sli], ls_m[:, sli], ls_p[:, sli])
        return self._wTT_rfft if rfft else self._wTT


    def get_hf_w(self, L, pi, pj, ders_i, ders_j, Laxis=0, rfft=False):
        r""" Similar as gf bth with phase factor extracted (param. in terms of q + L/2, q - L/2


        """
        assert len(ders_i) == pi
        assert len(ders_j) == pj

        wtt, qml, qpl, ls_m, ls_p = self._get_wtt_sym(L, Laxis=Laxis, rfft=rfft)
        w = wtt  * 1j ** (pi + pj)
        if pi > 0:
            w *= (self.ctt ** pi)[ls_p]
            for deri in ders_i:
                w *= qpl[deri]
        if pj > 0:
            w *= (self.ctt ** pj)[ls_m]
            for derj in ders_j:
                w *= qml[derj]
        return np.fft.irfft2(w) if rfft else np.fft.ifft2(w)

    def _get_hf_11_10_w(self, L, Laxis=0):
        r"""Tuned version of real part of hf_11_{xy}

        """
        wtt, qml, qpl, ls_m, ls_p = self._get_wtt_sym(L, Laxis=Laxis, rfft=True)
        w = -0.5 * wtt * self.ctt[ls_p] * self.ctt[ls_m] * (qml[1] * qpl[0] + qml[0] * qpl[1])
        return np.fft.irfft2(w)


    def get_gf_w(self, L, pi, pj, ders_i, ders_j, Laxis=0):
        """

            This is formally \int_{l1, l2} W^{TT}(l1, l2) * e^{i l_1 r}  ( i Cl_1 l_1)_(ders_i)( i Cl_2 l_2)_(ders_j)
                                where l2 is L - l1

        """
        if L > 0:
            if self._wTT_pL is None:
                self._set_wtt(L, Laxis=Laxis)
            _w, l2s, ml2_yx = self._wTT_pL
        else:
            if self._wTT_mL is None:
                self._set_wtt(L, Laxis=Laxis)
            _w, l2s, ml2_yx = self._wTT_mL
        w = np.copy(_w)
        if pi > 0:
            w *= (self.ctt ** pi)[self.ls]
            for deri in ders_i:
                w *= self.ns[deri]
        if pj > 0:
            w *= (self.ctt ** pj)[l2s]
            for derj in ders_j:
                w *= ml2_yx[derj]
        return np.fft.ifft2(w)

    def get_g_anyL(self, L, pi, pj, ders_i, ders_j, Laxis=0):
        """L here

        """
        assert len(ders_i) == pi  # one derivative per power  (e.g. [0, 0] for 2 derivatives on y-axis)
        assert len(ders_j) == pj

        wi = (self.F * self.ctt ** pi)[self.ls] * 1j ** pi  # l leg
        for der_i in ders_i:
            wi *= self.ns[der_i]

        nynx = self.shifted_p_nynx if L > 0 else self.shifted_m_nynx
        ls = self.shifted_p_ls if L > 0 else self.shifted_m_ls
        wj = (self.F * self.ctt ** pj)[ls] * 1j ** pj  # (L - l) leg
        for der_j in ders_j:
            wj *= nynx[der_j]
        return np.fft.ifft2(wi * wj.conj())

    def _get_n1(self, dl, gorc='g'):
        r"""N1 lensing bias without using symmetry tricks

            Note:
                here the input argument is in pixel units

        """
        c = d = 0 if gorc == 'g' else 1
        ret_re = 0.
        for a in [0, 1]:
            for b in [0, 1]:
                term1  = self.get_g(dl, 2, 1, [a, c], [b]) * self.get_g(-dl, 1, 0, [d],    [])
                term1 += self.get_g(dl, 2, 0, [a, c], [])  * self.get_g(-dl, 1, 1, [d],    [b])
                term1 += self.get_g(dl, 1, 1, [c],    [b]) * self.get_g(-dl, 2, 0, [d, a], [])
                term1 += self.get_g(dl, 1, 0, [c],    [])  * self.get_g(-dl, 2, 1, [d, a], [b])

                ret_re += np.sum(self.xiab[a + b] * term1.real)

                term2 =  self.get_g(dl, 2, 1, [a, c], [b]) * self.get_g(-dl, 0, 1,  [],  [d])
                term2 += self.get_g(dl, 2, 0, [a, c], []) * self.get_g(-dl, 0, 2,   [],  [b, d])
                term2 += self.get_g(dl, 1, 1, [c],    [b]) * self.get_g(-dl, 1, 1, [a],  [d])
                term2 += self.get_g(dl, 1, 0, [c],    []) * self.get_g(-dl, 1, 2,  [a],  [d, b])

                ret_re += np.sum(self.xiab[a + b] * term2.real)

        return self.norm * ret_re * dl ** 2

    def get_n1(self, L, Laxis=0):
        r"""N1 lensing gradient-induced lensing bias, for lensing gradient or curl bias

            Args:
                L: multipole L of :math:`N^{(1)}_L`
                Laxis(optional, 0 or 1): Axis towards which L is pointing. (For testing purposes, result should be independent of this)

            Returns:
                gradient-induced, lensing gradient or curl bias :math:`N_L^{(1)}`


        """
        L = float(L)
        #--- Builds shifted frequency grids for feeding into the FFT's

        self.shifted_p_nynx = self.get_shifted_lylx(L, Laxis)
        self.shifted_m_nynx = self.get_shifted_lylx(-L, Laxis)
        self.shifted_p_ls = self.freq2ls(self.shifted_p_nynx[1], self.shifted_p_nynx[0])
        self.shifted_m_ls = self.freq2ls(self.shifted_m_nynx[1], self.shifted_m_nynx[0])

        #--- precalc of fft'ed maps:(probably still some redundancy here with the g01's)
        gf_00_mL_zz =  self.get_gf(-L, 0, 0, [], [])
        gf_10_pL_az = [self.get_gf( L, 1, 0, [a], [ ]) for a in [0, 1] ]
        gf_01_mL_za = [self.get_gf(-L, 0, 1, [ ], [a]) for a in [0, 1] ] # in principle this is the just above but with -r e^iLr
        #gf_01_mL_za = [self.flip(f) for f in gf_10_pL_az]
        #FIXME: this looks right up to details:
        #eiLr  = np.exp(- 2j * np.pi / self.shape[0] * self.ns[Laxis] * ( L / self.lminbox ) )
        #gf_01_mL_za = [eiLr * f for f in gf_10_pL_az] # in principle this is the just above but with -r

        #--- Loop over cartesian deflection field components
        n1 = 0.
        for a in [0, 1]:
            for b in [0, 1]:
                term =  (self.get_gf(L, 1, 1, [a], [b]) * gf_00_mL_zz).real
                term += (gf_10_pL_az[a] * gf_01_mL_za[b]).real
                n1 += np.sum(self.xiab[a + b] * term)

        return self.norm * n1

    def get_n1_devel(self, L, Laxis=0, rfft=True):
        r"""N1 lensing gradient-induced lensing bias, for lensing gradient or curl bias

            Args:
                L: multipole L of :math:`N^{(1)}_L`
                Laxis(optional, 0 or 1): Axis towards which L is pointing. (For testing purposes, result should be independent of this)

            Returns:
                gradient-induced, lensing gradient or curl bias :math:`N_L^{(1)}`

            Note:
                This keeps the QE functions intact and absords the l_phi's factor in Cphiphi_L in the integrals

        """
        # --- Builds shifted frequency grids for feeding into the FFT's
        L = float(L)

        # --- precalc of some of the fft'ed maps:(probably still some redundancy here with the g11's)
        self._wTT = None
        if Laxis >= 2:
            h_00 = self.get_hf_w(L, 0, 0, [], [], Laxis=Laxis, rfft=rfft)
            h_10_y = self.get_hf_w(L, 1, 0, [0], [], Laxis=Laxis,  rfft=False)
            h_11_yy = self.get_hf_w(L, 1, 1, [0], [0], Laxis=Laxis, rfft=rfft)
            re10 = h_10_y.real
            im10 = h_10_y.imag
            # the other is the transpose

            term1 = h_11_yy * h_00 + (re10 ** 2 - im10 ** 2)
            term2 = self._get_hf_11_10_w(L, Laxis=Laxis) * h_00 + (re10 * re10.T - im10 * im10.T)
            n1 = 2. * np.sum(self.xiab[0 + 0] * term1 + self.xiab[1 + 0] * term2)
        else:

            h_00 = self.get_hf_w(L, 0, 0, [], [], Laxis=Laxis, rfft=rfft)
            h_10_a = [self.get_hf_w(L, 1, 0, [a], [], Laxis=Laxis) for a in [0, 1]]
            # h_01_a = [self.get_hf_w(-L, 0, 1, [], [a]) for a in [0, 1]]

            # gf_01_mL_za = [self.flip(f) for f in gf_10_pL_az]
            # eiLr  = np.exp(-2j * np.pi / self.shape[0] * self.ns[Laxis] * ( L / self.lminbox ) )
            # gf_01_mL_za = [eiLr * f for f in gf_10_pL_az] # in principle this is the just above but with -r

            # --- Loop over cartesian deflection field components
            ## 00:
            term = self.get_hf_w(L, 1, 1, [0], [0], Laxis=Laxis, rfft=rfft) * h_00 + (h_10_a[0] ** 2).real
            n1 = np.sum(self.xiab[0 + 0] * term)
            ## 11:
            term = self.get_hf_w(L, 1, 1, [1], [1], Laxis=Laxis, rfft=rfft) * h_00 + (h_10_a[1] ** 2).real
            n1 += np.sum(self.xiab[1 + 1] * term)
            # 01:
            term = self.get_hf_w(L, 1, 1, [0], [1], Laxis=Laxis, rfft=False).real * h_00 + (h_10_a[1] * h_10_a[0])
            n1 += 2. * np.sum(self.xiab[1 + 0] * term)
        # 10:
        #term = self.get_hf_w(L, 1, 1, [1], [0]).real  * h_00 + (h_10_a[1] * h_10_a[0]).real
        #n1 += np.sum(self.xiab[1 + 0] * term)

        #for a in [0, 1]:
        #    for b in [0, 1]:
        #        term = (self.get_hf_w(L, 1, 1, [a], [b]) * h_00).real
        #        term += (h_10_a[a] * h_10_a[b]).real
        #        n1 += np.sum(self.xiab[a + b] * term)
        # FIXME: make this less bg prone
        self._wTT = None
        return self.norm * n1

    def get_n1_devel1(self, L, Laxis=0):
        r"""N1 lensing gradient-induced lensing bias, for lensing gradient or curl bias

            Args:
                L: multipole L of :math:`N^{(1)}_L`
                Laxis(optional, 0 or 1): Axis towards which L is pointing. (For testing purposes, result should be independent of this)

            Returns:
                gradient-induced, lensing gradient or curl bias :math:`N_L^{(1)}`

            Note:
                This keeps the QE functions intact and absords the l_phi's factor in Cphiphi_L in the integrals

        """
        # --- Builds shifted frequency grids for feeding into the FFT's
        L = float(L)

        # --- precalc of some of the fft'ed maps:(probably still some redundancy here with the g11's)
        self._wTT_pL = None
        self._wTT_mL = None
        gf_00_mL_zz = self.get_gf_w(-L, 0, 0, [], [])
        gf_10_pL_az = [self.get_gf_w(L, 1, 0, [a], []) for a in [0, 1]]
        gf_01_mL_za = [self.get_gf_w(-L, 0, 1, [], [a]) for a in [0, 1]]  # in principle this is the just above but with -r e^iLr
        # gf_01_mL_za = [self.flip(f) for f in gf_10_pL_az]
        # FIXME: this looks right up to details:
        #eiLr  = np.exp(-2j * np.pi / self.shape[0] * self.ns[Laxis] * ( L / self.lminbox ) )
        #gf_01_mL_za = [eiLr * f for f in gf_10_pL_az] # in principle this is the just above but with -r

        # --- Loop over cartesian deflection field components
        n1 = 0.
        for a in [0, 1]:
            for b in [0, 1]:
                term = (self.get_gf_w(L, 1, 1, [a], [b]) * gf_00_mL_zz).real
                term += (gf_10_pL_az[a] * gf_01_mL_za[b]).real
                n1 += np.sum(self.xiab[a + b] * term)
        # FIXME: make this less bg prone
        self._wTT_pL = None
        self._wTT_mL = None

        return self.norm * n1

    def get_n1_correctbutold(self, L, gorc='g', Laxis=0):
        r"""N1 lensing gradient-induced lensing bias, for lensing gradient or curl bias

            Args:
                L: multipole L of :math:`N^{(1)}_L`
                gorc: return gradient or curl N^{(1)}
                Laxis(optional, 0 or 1): Axis towards which L is pointing. (For testing purposes, result should be independent of this)

            Returns:
                gradient-induced, lensing gradient or curl bias :math:`N_L^{(1)}`


        """
        c = d = (Laxis if gorc == 'g' else (1 - Laxis)) # only change to do for curl
        n1 = 0.
        dl = L

        #--- Builds shifted frequency grids for feeding into the FFT's
        self.shifted_p_nynx = self.get_shifted_lylx( L, Laxis)
        self.shifted_m_nynx = self.get_shifted_lylx(-L, Laxis)
        self.shifted_p_ls = self.freq2ls(self.shifted_p_nynx[1], self.shifted_p_nynx[0])
        self.shifted_m_ls = self.freq2ls(self.shifted_m_nynx[1], self.shifted_m_nynx[0])

        #--- precalc of fft'ed maps:(probably still some redundancy here with the g11's)
        twice_gm_10_d_z_plus_gm_01_z_d = 2. * (self.get_g_anyL(-dl, 1, 0, [d], [], Laxis=Laxis) + self.get_g_anyL(-dl, 0, 1, [], [d], Laxis=Laxis))
        gp_20_ac_z = [self.get_g_anyL(dl, 2, 0, [a, c], [], Laxis=Laxis) for a in [0, 1]]
        twice_gm_11_d_b_plus_gm_02_z_bd =  [2 * self.get_g_anyL(-dl, 1, 1, [d], [b], Laxis=Laxis) + self.get_g_anyL(-dl, 0, 2, [], [b, d], Laxis=Laxis) for b in [0, 1]]
        gp_11_c_b =  [self.get_g_anyL(dl, 1, 1, [c], [b], Laxis=Laxis) for b in [0, 1]]
        gm_11_a_d =  [self.get_g_anyL(-dl, 1, 1, [a], [d], Laxis=Laxis) for a in [0, 1]]

        #--- Loop over cartesian deflection field components
        for a in [0, 1]:
            for b in [0, 1]:
                term =  (self.get_g_anyL(dl, 2, 1, [a, c], [b], Laxis=Laxis) * twice_gm_10_d_z_plus_gm_01_z_d).real
                term += (gp_20_ac_z[a] * twice_gm_11_d_b_plus_gm_02_z_bd [b]).real
                term += (gp_11_c_b[b] * gm_11_a_d[a]).real
                n1 += np.sum(self.xiab[a + b] * term)

        return self.norm * n1 * (dl /self.lminbox) ** 2


def get_n1f(L, kA, ks, lminphi, lmaxphi, cls_grad, cpp=None, dL=30, exp='PL', lps=None):
    """Call to f90 4-dimensional integral code """
    kB = kA
    fals = tp.get_fal_sTP(exp, 1)[1]
    ftlA = felA =  fblA = ftlB = felB = fblB = np.copy(fals['tt'])
    lminA = 100
    lminB = 100
    if cpp is None:
        cpp = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_lenspotentialCls.dat'))['pp']
    cpp = cpp[:lmaxphi + 1]
    cpp[:lminphi] *= 0.
    if lps is None:
        lps = n1f.library_n1._default_lps_grid(lmaxphi)
    cltt = clttw = cls_grad['tt']
    clte = cltew = cls_grad['te']
    clee = cleew = cls_grad['ee']
    return n1f._calc_n1L_sTP(L, cpp, kA, kB, ks, cltt, clte, clee, clttw, cltew, cleew,
                          ftlA, felA, fblA, ftlB, felB, fblB, lminA, lminB, dL, lps)