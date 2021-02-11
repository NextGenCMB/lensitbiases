import os
import numpy as np
import n1


def prepare_cls(k, jt_TP=False):
    from plancklens import utils
    path =  os.path.join(os.path.abspath(os.path.dirname(n1.__file__)),'data', 'cls')
    cls_grad = utils.camb_clfile(os.path.join(path, 'FFP10_wdipole_gradlensedCls.dat'))
    cls_unl = utils.camb_clfile(os.path.join(path, 'FFP10_wdipole_lenspotentialCls.dat'))
    cls_weights = utils.camb_clfile(os.path.join(path, 'FFP10_wdipole_gradlensedCls.dat'))
    fals = get_fal(jt_tp=jt_TP)[1]
    if k == 'ptt':
        fals['ee'] *= 0.
        fals['bb'] *= 0.
    if k == 'p_p':
        fals['tt'] *= 0.
    if k in ['ptt', 'p_p']:
        cls_weights['te'] *= 0.
    return fals, cls_weights, cls_grad, cls_unl['pp']


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

def get_fal(jt_tp=False):
    """Loads default filtering and inverse-variance filtered spectral matrices for test cases

        Args:
            jt_tp: True of joint temperature-polarisation filtering, False if not


    """
    from plancklens.patchy import patchy
    from plancklens import utils
    import healpy as hp
    lmax_ivf = 2048
    lmin_ivf = 100
    nlevt = 35.
    nlevp = 55.
    beam = 6.5
    path = os.path.abspath(os.path.dirname(n1.__file__))
    cls_len = utils.camb_clfile(os.path.join(path, 'data','cls', 'FFP10_wdipole_lensedCls.dat'))
    transf = hp.gauss_beam(beam / 60 / 180 * np.pi, lmax=lmax_ivf)
    ivcl, fal = patchy.get_ivf_cls(cls_len, cls_len, lmin_ivf, lmax_ivf, nlevt, nlevp, nlevt, nlevp, transf,
                                   jt_tp=jt_tp)
    return ivcl, fal, lmax_ivf