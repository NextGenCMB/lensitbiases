import numpy as np

def cli(cl):
    ret = np.zeros_like(cl, dtype=float)
    ret[np.where(cl != 0)] = 1./ cl[np.where(cl != 0)]
    return ret

def extcl(lmax, cl):
    if len(cl) - 1 < lmax:
        dl = np.zeros(lmax + 1)
        dl[:len(cl)] = cl
    else:
        dl = cl[:lmax+1]
    return dl

def freqs(i, N):
    """Outputs the absolute integers frequencies [0,1,...,N/2,N/2-1,...,1]
         in numpy fft convention as integer i runs from 0 to N-1.
         Inputs can be numpy arrays e.g. i (i1,i2,i3) with N (N1,N2,N3)
                                      or i (i1,i2,...) with N
         Both inputs must be integers.
         All entries of N must be even.
    """
    assert (np.all(N % 2 == 0)), "This routine only for even numbers of points"
    return i - 2 * (i >= (N // 2)) * (i % (N // 2))

def rfft2_reals(shape):
    """Pure reals modes in 2d rfft according to patch specifics

    """
    N0, N1 = shape
    fx = [0]
    fy = [0]
    if N1 % 2 == 0:
        fx.append(0)
        fy.append(N1 // 2)
    if N0 % 2 == 0:
        fx.append(N0 // 2)
        fy.append(0)
    if N1 % 2 == 0 and N0 % 2 == 0:
        fx.append(N0 // 2)
        fy.append(N1 // 2)
    return np.array(fx), np.array(fy)

def get_fal(jt_tp=False):
    from plancklens.patchy import patchy
    from plancklens import utils
    import healpy as hp
    import os
    lmax_ivf = 2048
    lmin_ivf = 100
    nlevt = 35.
    nlevp = 55.
    beam = 6.5
    cls_len = utils.camb_clfile(os.path.join('../../plancklens/plancklens/data/cls/', 'FFP10_wdipole_lensedCls.dat'))
    transf = hp.gauss_beam(beam / 60 / 180 * np.pi, lmax=lmax_ivf)
    ivcl, fal = patchy.get_ivf_cls(cls_len, cls_len, lmin_ivf, lmax_ivf, nlevt, nlevp, nlevt, nlevp, transf,
                                   jt_tp=jt_tp)
    return ivcl, fal, lmax_ivf

class box:
    def __init__(self, lside, npix):
        assert npix % 2 == 0, npix

        shape = (npix, npix)
        rshape = (npix, npix // 2 + 1)

        # === frequencies
        nx = freqs(np.arange(rshape[1]), shape[1])  # unsigned FFT frequencies
        ny = freqs(np.arange(shape[0]), shape[0])
        nx[shape[1] // 2:] *= -1
        ny[shape[0] // 2:] *= -1

        self.nx_1d = nx
        self.ny_1d = ny

        self.shape = shape
        self.rshape = rshape
        self.lsides = (lside, lside)

        # mini and maxi multipole in box
        self.lminbox = self.rsqd2l(1.)
        self.lmaxbox = self.rsqd2l(nx[npix//2] ** 2 + ny[npix//2] ** 2)

        self._ellcounts = None


    def _get_lcounts(self):
        if self._ellcounts is None:
            ls = self.ls()
            shape = self.shape
            rshape = self.rshape
            counts = np.bincount(ls[:, 1:rshape[1] - 1].flatten(), minlength=self.lmaxbox + 1)
            s_counts = np.bincount(ls[0:shape[0] // 2 + 1, [-1, 0]].flatten(),  minlength=self.lmaxbox + 1)
            counts[0:len(s_counts)] += s_counts
            self._ellcounts = counts
        return self._ellcounts

    def rsqd2l(self, r2):
        return np.int_(np.round((2. * np.pi / self.lsides[0]) * (np.sqrt(r2))))

    def ls(self):
        n2y, n2x = np.meshgrid(self.ny_1d ** 2, self.nx_1d ** 2, indexing='ij')
        return self.rsqd2l(n2y + n2x)

    def cos2p_sin2p(self, rfft=True):
        """Returns the cosines and sines of twice the polar angle

        """
        s = self.rshape if rfft else self.shape
        k2y, k2x = np.meshgrid(self.ny_1d ** 2, (self.nx_1d ** 2 if rfft else self.ny_1d ** 2), indexing='ij')
        k2 = k2y + k2x
        cos2p = np.ones(s, dtype=float)
        cos2p[1:] = 2 * k2x[1:] / k2[1:] - 1.
        sin2p = np.zeros(s, dtype=float)
        sin2p[1:] = np.outer(2 * self.ny_1d[1:], self.nx_1d if rfft else self.ny_1d) / k2[1:]
        return cos2p, sin2p

    def mode_counts(self):
        """Mode number counts on the flat-sky patch.

            Goes roughly like :math:`2\ell + 1` times the sky faction

        """
        nl = 2 * self._get_lcounts()
        lx, ly = rfft2_reals(self.shape)
        for l in np.array(self.rsqd2l(lx ** 2 + ly ** 2), dtype=int):
            nl[l] -= 1
        return nl

    def sum_in_l(self, weights):
        assert weights.shape == self.rshape, (weights.shape, self.rshape)
        shape = self.shape
        rshape = self.rshape
        ls = self.ls()
        cl = np.bincount(ls[:, 1:rshape[1] - 1].flatten(), weights=weights[:, 1:rshape[1] - 1].flatten(), minlength=self.lmaxbox+ 1)
        cl += np.bincount(ls[0:shape[0] // 2 + 1, [-1, 0]].flatten(), weights=weights[0:shape[0] // 2 + 1, [-1, 0]].flatten(), minlength=self.lmaxbox + 1)
        return cl * self.mode_counts() * cli(self._get_lcounts())

    def map2cl(self,m, lmax=None):
        assert m.shape == self.shape, (m.shape, self.shape)
        if lmax is None: lmax = self.lmaxbox
        norm =  np.prod(self.lsides) / float(np.prod(self.shape)) ** 2
        return norm * self.sum_in_l(np.abs(np.fft.rfft2(m)) ** 2)[:lmax+1] * cli(self.mode_counts())