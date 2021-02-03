import numpy as np

def cli(cl):
    ret = np.zeros_like(cl, dtype=float)
    ret[np.where(cl != 0)] = 1./ cl[np.where(cl != 0)]
    return ret

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
        self.lminbox = 2. * np.pi / lside
        self.lmaxbox = self._rsqd2l(nx[npix//2] ** 2 + ny[npix//2] ** 2)

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

    def _rsqd2l(self, r2):
        return np.int_(np.round(self.lminbox * (np.sqrt(r2))))

    def ls(self):
        r2 = np.outer(self.ny_1d ** 2, np.ones(len(self.nx_1d)))
        r2 += np.outer(np.ones(len(self.ny_1d)), self.nx_1d ** 2)
        return self._rsqd2l(r2)


    def mode_counts(self):
        """Mode number counts on the flat-sky patch.

            Goes roughly like :math:`2\ell + 1` times the sky faction

        """
        nl = 2 * self._get_lcounts()
        lx, ly = rfft2_reals(self.shape)
        for l in np.array(self._rsqd2l(lx ** 2 + ly ** 2), dtype=int):
            nl[l] -= 1
        return nl

    def bin_in_l(self, weights):
        assert weights.shape == self.rshape, (weights.shape, self.rshape)
        shape = self.shape
        rshape = self.rshape
        ls = self.ls()
        cl = np.bincount(ls[:, 1:rshape[1] - 1].flatten(), weights=weights[:, 1:rshape[1] - 1].flatten(), minlength=self.lmaxbox+ 1)
        cl += np.bincount(ls[0:shape[0] // 2 + 1, [-1, 0]].flatten(), weights=weights[0:shape[0] // 2 + 1, [-1, 0]].flatten(), minlength=self.lmaxbox + 1)
        return cl * cli(self._get_lcounts())

    def map2cl(self,m, lmax=None):
        assert m.shape == self.shape, (m.shape, self.shape)
        norm =  np.prod(self.lsides) / float(np.prod(self.shape)) ** 2
        return norm * self.bin_in_l(np.abs(np.fft.rfft2(m)) ** 2)[:(lmax or self.lmaxbox) + 1]
