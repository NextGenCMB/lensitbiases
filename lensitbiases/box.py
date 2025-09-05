r"""Module handling a 2D flat-sky mode structure


"""
import numpy as np
from lensitbiases.utils_box import freqs, rfft2_reals
from lensitbiases.utils_n1 import cli

class box:
    def __init__(self, lside, npix, k2l=None):
        shape  = (npix, npix)
        rshape = (npix, npix // 2 + 1)

        # === frequencies
        nx = freqs(np.arange(rshape[1]), shape[1], signed=True)  # unsigned FFT frequencies
        ny = freqs(np.arange(shape[0]), shape[0], signed=True)

        self.nx_1d = nx
        self.ny_1d = ny

        self.shape = shape
        self.rshape = rshape
        self.lsides = (lside, lside)

        # 2d frequency to multipole scheme
        self.k2l = k2l

        # mini and maxi multipole in box
        self.lminbox = 2 * np.pi / lside # This is used for normalizations etc
        self.lmaxbox = self.rsqd2l(nx[npix//2] ** 2 + ny[npix//2] ** 2)

        self._ellcounts = None
        self._cos2p_sin2p = None

    def plot_rfft(self, rfftm, kmin=None,title='',**imshow_kwargs):
        import pylab as pl
        kmin = kmin or rfftm.shape[1]
        pl.figure()
        image = pl.imshow( (rfftm )[:kmin, 0:kmin], **imshow_kwargs)
        pl.ylabel(r'$\ell_y / %.0f$'%self.lminbox)
        pl.xlabel(r'$\ell_x / %.0f$'%self.lminbox)
        pl.title(title)
        return image

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
        if self.k2l is None: # default
            return np.int_(np.round(2. * np.pi / self.lsides[0] * np.sqrt(r2)))
        elif self.k2l in ['lensit']:
            k = (2. * np.pi / self.lsides[0]) * np.sqrt(r2)
            return np.uint16(np.round(k - 0.5) + 0.5 * ((k - 0.5) < 0))
        else:
            assert 0, self.k2l + ' not implemented'

    def ls(self, rfft=True):
        n2y, n2x = np.meshgrid(self.ny_1d ** 2, (self.nx_1d if rfft else self.ny_1d) ** 2, indexing='ij')
        return self.rsqd2l(n2y + n2x)
    
    def lx(self, rfft=True):
        _, n2x = np.meshgrid(self.ny_1d, (self.nx_1d if rfft else self.ny_1d), indexing='ij')
        return n2x * (2. * np.pi / self.lsides[1])

    def ly(self, rfft=True):
        n2y, _ = np.meshgrid(self.ny_1d, (self.nx_1d if rfft else self.ny_1d), indexing='ij')
        return n2y * (2. * np.pi / self.lsides[0])

    def triangles_count(self, lx_ly_cond=lambda lx,ly: np.ones(lx.shape, dtype=bool)):
        """Returns number of pairs of vector l1, l2 such l1 + l2 = L for each vector L, including cuts on l1 and l2

                The normalization is such that the output is 1 everywhere when there are no cuts at all
        
        """
        return np.fft.fft2(np.fft.irfft2(np.where(lx_ly_cond(self.lx(), self.ly()), 1., 0)) ** 2).real
    
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

    def X2S(self, S, X, rfft=True):
        """Matrix element sending X (T, E or B) cmb mode to stokes flat-sky  Stokes S (T, Q or U)


        """
        if S == 'T':  return 1. if X == 'T' else 0.
        if S == 'Q':
            if X == 'T': return 0.
            if self._cos2p_sin2p is None:
                self._cos2p_sin2p = self.cos2p_sin2p(rfft=rfft)
            sgn = 1 if X == 'E' else -1
            return sgn * self._cos2p_sin2p[0 if X == 'E' else 1]
        if S == 'U':
            if X == 'T': return 0.
            if self._cos2p_sin2p is None:
                self._cos2p_sin2p = self.cos2p_sin2p(rfft=rfft)
            return self._cos2p_sin2p[1 if X == 'E' else 0]
        assert 0


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
        assert weights.shape in (self.rshape, self.shape), (weights.shape, self.rshape,self.shape)
        if weights.shape == self.rshape:
            shape = self.shape
            rshape = self.rshape
            ls = self.ls()
            cl = np.bincount(ls[:, 1:rshape[1] - 1].flatten(), weights=weights[:, 1:rshape[1] - 1].flatten(), minlength=self.lmaxbox+ 1)
            cl += np.bincount(ls[0:shape[0] // 2 + 1, [-1, 0]].flatten(), weights=weights[0:shape[0] // 2 + 1, [-1, 0]].flatten(), minlength=self.lmaxbox + 1)
            return cl * self.mode_counts() * cli(self._get_lcounts())
        elif weights.shape == self.shape:
            cl = np.bincount(self.ls(rfft=False).flatten(), weights=weights.flatten(), minlength=self.lmaxbox+ 1)
            return cl

    def map2cl(self,m, lmax=None):
        assert m.shape == self.shape, (m.shape, self.shape)
        if lmax is None: lmax = self.lmaxbox
        norm =  np.prod(self.lsides) / float(np.prod(self.shape)) ** 2
        return norm * self.sum_in_l(np.abs(np.fft.rfft2(m)) ** 2)[:lmax+1] * cli(self.mode_counts())