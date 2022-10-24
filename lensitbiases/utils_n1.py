import sys, os
import numpy as np
import lensitbiases
import time

def prepare_cls(k, jt_TP=False):
    path =  os.path.join(os.path.abspath(os.path.dirname(lensitbiases.__file__)), 'data', 'cls')
    cls_grad = camb_clfile(os.path.join(path, 'FFP10_wdipole_gradlensedCls.dat'))
    cls_unl = camb_clfile(os.path.join(path, 'FFP10_wdipole_lenspotentialCls.dat'))
    cls_weights = camb_clfile(os.path.join(path, 'FFP10_wdipole_gradlensedCls.dat'))
    ivfs_cls, fals = get_fal(jt_tp=jt_TP)[:2]
    if k == 'ptt':
        fals['ee'] *= 0.
        fals['bb'] *= 0.
    if k == 'p_p':
        fals['tt'] *= 0.
    if k in ['ptt', 'p_p']:
        cls_weights['te'] *= 0.
    return ivfs_cls, fals, cls_weights, cls_grad, cls_unl['pp']


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

def get_fal(clscmb_filt=None, clscmb_dat=None, lmin_ivf=100, lmax_ivf=2048, nlevt=35., nlevp=55., beam=6.5, jt_tp=False):
    """Loads default filtering and inverse-variance filtered spectral matrices for test cases

        Args:
            clscmb_filt: CMB spectra used in the inverse-variance filtering (defaults to Planck FFP10 lensed spectra)
            clscmb_dat: CMB spectra in the data maps (defaults to clscmb_filt)
            lmin_ivf: minimum CMB multipole fed into the QE
            lmax_ivf: maximum CMB multipole fed into the QE
            nlevt: noise level in temperature in uKarcmin
            nlevp: noise level in polarizatiion in uKarcmin
            beam: beam fwhm width in arcmin
            jt_tp: True of joint temperature-polarisation filtering, False if not

        Returns:
            spectral matrix of the inverse-variance filtered CMB, filtering matrix and lmax_ivf

    """
    if clscmb_filt is None:
        path = os.path.abspath(os.path.dirname(lensitbiases.__file__))
        clscmb_filt = camb_clfile(os.path.join(path, 'data','cls', 'FFP10_wdipole_lensedCls.dat'))
    if clscmb_dat is None:
        clscmb_dat = clscmb_filt
    l = np.arange(lmax_ivf + 1, dtype=int)
    transf = np.exp(-l * (l + 1.) * (beam / 60 / 180 * np.pi/ 2.3548200450309493) ** 2 * 0.5)
    ivcl, fal = get_ivf_cls(clscmb_dat, clscmb_filt, lmin_ivf, lmax_ivf, nlevt, nlevp, nlevt, nlevp, transf,  jt_tp=jt_tp)
    return ivcl, fal, lmax_ivf


def _get_fal(a, cl_len, nlev, transf, lmin, lmax):
    """Simple diagonal isotropic filter

    """
    fal = cli(cl_len.get(a + a)[:lmax + 1] + (nlev / 60. / 180. * np.pi) ** 2 / transf[:lmax + 1] ** 2)
    fal[:lmin] *= 0.
    return fal


def get_ivf_cls(cls_cmb_dat, cls_cmb_filt, lmin, lmax, nlevt_f, nlevp_f, nlevt_m, nlevp_m, transf, jt_tp=False):
    """inverse filtered spectra (spectra of Cov^-1 X) for CMB inverse-variance filtering


        Args:
            cls_cmb_dat: dict of cmb cls of the data maps
            cls_cmb_filt: dict of cmb cls used in the filtering matrix
            lmin: minimum multipole considered
            lmax: maximum multipole considered
            nlevt_f: fiducial temperature noise level used in the filtering in uK-amin
            nlevp_f: fiducial polarization noise level used in the filtering in uK-amin
            nlevt_m: temperature noise level of the data in uK-amin
            nlevp_m: polarization noise level of the data in uK-amin
            transf: CMB transfer function
            jt_tp: if set joint temperature-polarization filtering is performed. If not they are filtered independently

        Returns:
            dict of inverse-variance filtered maps spectra (for N0 calcs.)
            dict of filtering matrix spectra (for response calcs. This has no dependence on the data parts of the inputs)


    """
    ivf_cls = {}
    if not jt_tp:
        filt_cls_i = {}
        for a in ['t']:
            ivf_cls[a + a] = _get_fal(a, cls_cmb_filt, nlevt_f, transf, lmin, lmax) ** 2 * cli(_get_fal(a, cls_cmb_dat, nlevt_m, transf, 0, lmax))
            filt_cls_i[a + a] = _get_fal(a, cls_cmb_filt, nlevt_f, transf, lmin, lmax)
        for a in ['e', 'b']:
            ivf_cls[a + a] = _get_fal(a, cls_cmb_filt, nlevp_f, transf, lmin, lmax) ** 2 * cli(_get_fal(a, cls_cmb_dat, nlevp_m, transf, 0, lmax))
            filt_cls_i[a + a] = _get_fal(a, cls_cmb_filt, nlevp_f, transf, lmin, lmax)
        ivf_cls['te'] = cls_cmb_dat['te'][:lmax + 1] * _get_fal('e', cls_cmb_filt, nlevp_f, transf, lmin, lmax) * _get_fal('t', cls_cmb_filt,  nlevt_f, transf,   lmin, lmax)
        return ivf_cls, filt_cls_i
    else:
        filt_cls = np.zeros((3, 3, lmax + 1), dtype=float)
        dat_cls = np.zeros((3, 3, lmax + 1), dtype=float)
        filt_cls[0, 0] = cli(_get_fal('t', cls_cmb_filt, nlevt_f, transf, lmin, lmax))
        filt_cls[1, 1] = cli(_get_fal('e', cls_cmb_filt, nlevp_f, transf, lmin, lmax))
        filt_cls[2, 2] = cli(_get_fal('b', cls_cmb_filt, nlevp_f, transf, lmin, lmax))
        filt_cls[0, 1, lmin:] = cls_cmb_filt['te'][lmin:lmax + 1]
        filt_cls[1, 0, lmin:] = cls_cmb_filt['te'][lmin:lmax + 1]
        dat_cls[0, 0] = cli(_get_fal('t', cls_cmb_dat, nlevt_m, transf, lmin, lmax))
        dat_cls[1, 1] = cli(_get_fal('e', cls_cmb_dat, nlevp_m, transf, lmin, lmax))
        dat_cls[2, 2] = cli(_get_fal('b', cls_cmb_dat, nlevp_m, transf, lmin, lmax))
        dat_cls[0, 1, lmin:] = cls_cmb_dat['te'][lmin:lmax + 1]
        dat_cls[1, 0, lmin:] = cls_cmb_dat['te'][lmin:lmax + 1]
        filt_cls_i = np.linalg.pinv(filt_cls.swapaxes(0, 2)).swapaxes(0, 2)
        return _cls_dot(filt_cls_i, dat_cls, lmin, lmax), \
               {'tt':filt_cls_i[0,0], 'ee':filt_cls_i[1, 1], 'bb':filt_cls_i[2, 2], 'te':filt_cls_i[0, 1]}

def _cls_dot(cls_fidi, cls_dat, lmin, lmax):
    zro = np.zeros(lmax + 1, dtype=float)
    ret = {'tt':zro.copy(), 'te':zro.copy(), 'ee':zro.copy(), 'bb':zro.copy()}
    for i in range(3):
        for j in range(3):
            ret['tt'] += cls_fidi[0, i] * cls_fidi[0, j] * cls_dat[i, j]
            ret['te'] += cls_fidi[0, i] * cls_fidi[1, j] * cls_dat[i, j]
            ret['ee'] += cls_fidi[1, i] * cls_fidi[1, j] * cls_dat[i, j]
            ret['bb'] += cls_fidi[2, i] * cls_fidi[2, j] * cls_dat[i, j]
    for cl in ret.values():
        cl[:lmin] *= 0
    return ret


def camb_clfile(fname, lmax=None):
    """CAMB spectra (lenspotentialCls, lensedCls or tensCls types) returned as a dict of numpy arrays.

    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.

    """
    cols = np.loadtxt(fname).transpose()
    ell = cols[0].astype(np.int64)
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)
    cls = {k : np.zeros(lmax + 1, dtype=float) for k in ['tt', 'ee', 'bb', 'te']}
    w = ell * (ell + 1) / (2. * np.pi)  # weights in output file
    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k][ell[idc]] = cols[i + 1][idc] / w[idc]
    if len(cols) > 5:
        wpp = lambda ell : ell ** 2 * (ell + 1) ** 2 / (2. * np.pi)
        wptpe = lambda ell : np.sqrt(ell.astype(float) ** 3 * (ell + 1.) ** 3) / (2. * np.pi)
        for i, k in enumerate(['pp', 'pt', 'pe']):
            cls[k] = np.zeros(lmax + 1, dtype=float)
        cls['pp'][ell[idc]] = cols[5][idc] / wpp(ell[idc])
        cls['pt'][ell[idc]] = cols[6][idc] / wptpe(ell[idc])
        cls['pe'][ell[idc]] = cols[7][idc] / wptpe(ell[idc])
    return cls

def cls2dls(cls):
    """Turns cls dict. into camb cl array format"""
    keys = ['tt', 'ee', 'bb', 'te']
    lmax = np.max([len(cl) for cl in cls.values()]) - 1
    dls = np.zeros((lmax + 1, 4), dtype=float)
    refac = np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float) / (2. * np.pi)
    for i, k in enumerate(keys):
        cl = cls.get(k, np.zeros(lmax + 1, dtype=float))
        sli = slice(0, min(len(cl), lmax + 1))
        dls[sli, i] = cl[sli] * refac[sli]
    cldd = np.copy(cls.get('pp', None))
    if cldd is not None:
        cldd *= np.arange(len(cldd)) ** 2 * np.arange(1, len(cldd) + 1, dtype=float) ** 2 /  (2. * np.pi)
    return dls, cldd

def dls2cls(dls):
    """Inverse operation to cls2dls"""
    assert dls.shape[1] == 4
    lmax = dls.shape[0] - 1
    cls = {}
    refac = 2. * np.pi * cli( np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float))
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k] = dls[:, i] * refac
    return cls

def enumerate_progress(lst, label=''):
    """Simple progress bar.

    """
    t0 = time.time()
    ni = len(lst)
    for i, v in enumerate(lst):
        yield i, v
        ppct = int(100. * (i - 1) / ni)
        cpct = int(100. * (i + 0) / ni)
        if cpct > ppct:
            dt = time.time() - t0
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            sys.stdout.write("\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " +
                             label + " " + int(10. * cpct / 100) * "-" + "> " + ("%02d" % cpct) + r"%")
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
