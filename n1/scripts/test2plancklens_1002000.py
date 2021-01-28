"""
  N1 calculation for Abishek and Yacine paper

"""
import os
import argparse
import plancklens
import healpy as hp
import numpy as np
from plancklens.patchy import patchy
from plancklens import utils
from plancklens.n1 import n1 as n1_plancklens
from n1 import n1
import time
from plancklens.helpers import mpi

CLS = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')
N1_DIR = os.path.join(os.environ['ONED'],'N1', 'temp', 'test_N1s_100_2048')
N1_DIR_PL = os.path.join(os.environ['ONED'],'N1', 'temp', 'GMV_N1s_100_2048')

LMAXP = 2500
LMAXOUT = 2048

cls_grad = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_gradlensedCls.dat'))
cls_grad['bb'] *= 0.
n1_dd = n1.library_n1(N1_DIR, cls_grad['tt'], cls_grad['te'], cls_grad['ee'], lmaxphi=LMAXP)
n1_dd_pl = n1_plancklens.library_n1(N1_DIR_PL, cls_grad['tt'], cls_grad['te'], cls_grad['ee'], lmaxphi=LMAXP)


def get_fal_sTP(exp, jt_tp=False):
    if exp in ['SO']:
        lmax_ivf=  3000
        lmin_ivf = 100
        nlevt = 5.
        nlevp = 5. * np.sqrt(2.)
        beam = 5.
    elif exp in ['S4']:
        lmax_ivf = 3000
        lmin_ivf = 100
        nlevt = 1.
        nlevp = 1. * np.sqrt(2.)
        beam = 1.
    elif exp in ['PL']:
        lmax_ivf = 3000
        lmin_ivf = 100
        nlevt = 35.
        nlevp = 60.
        beam = 5.
    else:
        assert 0
    cls_len = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_lensedCls.dat'))
    transf = hp.gauss_beam(beam / 60 / 180 * np.pi, lmax=lmax_ivf)
    ivcl, fal = patchy.get_ivf_cls(cls_len, cls_len, lmin_ivf, lmax_ivf, nlevt, nlevp, nlevt,  nlevp, transf, jt_tp=jt_tp)
    return ivcl, fal, lmax_ivf

def get_N0s(exp, qe_key1='p', qe_key2='p'):
    fn_njtp = N1_DIR_PL + '/nhls_jtp' + exp + '_' + qe_key1 + '_' + qe_key2 + '.dat'
    fn_nstp = N1_DIR_PL + '/nhls_stp' + exp + '_' + qe_key1 + '_' + qe_key2 + '.dat'

    fn_Rjtp = N1_DIR_PL + '/Rs_jtp' + exp + '_' + qe_key1 + '.dat'
    fn_Rstp = N1_DIR_PL + '/Rs_stp' + exp + '_' + qe_key1 + '.dat'
    if np.any([not os.path.exists(fn) for fn in [fn_Rstp, fn_Rjtp, fn_nstp, fn_njtp]]):
        ivcl_sTP, fal_sTP, lmax_ivf = get_fal_sTP(exp, jt_tp=False)
        ivcl_jTP, fal_jTP, lmax_ivf = get_fal_sTP(exp, jt_tp=True)
        from plancklens import nhl, qresp
        cl_weight = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_gradlensedCls.dat'))
        cl_weight['bb'] *= 0.
        cl_grad = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_gradlensedCls.dat'))
        cl_grad['bb'] *= 0.

        nhls_jTP = np.array(nhl.get_nhl(qe_key1, qe_key2, cl_weight, ivcl_jTP, lmax_ivf, lmax_ivf, lmax_out=LMAXOUT))
        nhls_sTP = np.array(nhl.get_nhl(qe_key1, qe_key2, cl_weight, ivcl_sTP, lmax_ivf, lmax_ivf, lmax_out=LMAXOUT))

        assert qe_key1 == qe_key2
        Rs_jTP = np.array(qresp.get_response(qe_key1, lmax_ivf, 'p', cl_weight, cl_grad, fal_jTP, lmax_qlm=LMAXOUT))
        Rs_sTP = np.array(qresp.get_response(qe_key1, lmax_ivf, 'p', cl_weight, cl_grad, fal_sTP, lmax_qlm=LMAXOUT))

        np.savetxt(fn_nstp, nhls_sTP.transpose())
        np.savetxt(fn_njtp, nhls_jTP.transpose())
        np.savetxt(fn_Rstp, Rs_sTP.transpose())
        np.savetxt(fn_Rjtp, Rs_jTP.transpose())

    Rs_sTP = np.loadtxt(fn_Rstp).transpose()
    Rs_jTP = np.loadtxt(fn_Rjtp).transpose()
    nhls_sTP = np.loadtxt(fn_nstp).transpose()
    nhls_jTP = np.loadtxt(fn_njtp).transpose()

    N0_jTP =  nhls_jTP[0] * utils.cli(Rs_jTP[0]) ** 2
    N0_sTP =  nhls_sTP[0] * utils.cli(Rs_sTP[0]) ** 2
    N0_jTP_x =  nhls_jTP[1] * utils.cli(Rs_jTP[1]) ** 2
    N0_sTP_x =  nhls_sTP[1] * utils.cli(Rs_sTP[1]) ** 2
    return N0_sTP, N0_jTP, Rs_sTP, Rs_jTP, N0_sTP_x, N0_jTP_x

def get_N1_SQE(exp, k1='p', k2='p', ks='p'):
    fals = get_fal_sTP(exp)[1]
    ftlA = fals['tt']
    felA = fals['ee']
    fblA = fals['bb']
    cpp = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_lenspotentialCls.dat'))['pp']
    return n1_dd.get_n1(k1, ks, cpp, ftlA, felA, fblA, Lmax=LMAXOUT, kB=k2)

def get_N1_GMV(exp, k1='p', k2='p'):
    fals = get_fal_sTP(exp, jt_tp=True)[1]
    cpp = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_lenspotentialCls.dat'))['pp']
    return n1_dd.get_n1_jtp(k1, 'p', cpp, fals, Lmax=LMAXOUT, kB=k2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Planck 2018 QE calculation example')
    parser.add_argument('-k1k2s', dest='k1k2s', action='store', nargs='+', default=['ptt ptt', 'ptt pte', 'ptt ptb', 'ptt pee', 'ptt peb', 'pte pte', 'pte ptb', 'pte pee', 'pte peb',
                 'ptb ptb', 'ptb pee', 'ptb peb', 'pee pee', 'pee peb', 'peb peb'])
    parser.add_argument('-v', dest='version', action='store', nargs=1, type=int, default=1)
    args = parser.parse_args()

    if args.version[0] == 2:
        print("Switching module version to " + str(args.version[0]))
        from n1 import n1f_v2
        n1.n1f = n1f_v2
    L = 400
    for exp in ['PL']:
        fals = get_fal_sTP(exp, jt_tp=False)[1]
        ftlA = fals['tt']
        felA = fals['ee']
        fblA = fals['bb']
        ftlB = fals['tt']
        felB = fals['ee']
        fblB = fals['bb']

        cpp = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_lenspotentialCls.dat'))['pp']
        cls_grad = utils.camb_clfile(os.path.join(CLS, 'FFP10_wdipole_gradlensedCls.dat'))
        cls_grad['bb'] *= 0.
        kA = 'pee'
        kB = 'ptb'
        cltt = cls_grad['tt'];clte = cls_grad['te'];clee = cls_grad['ee']
        clttw = cls_grad['tt'];cltew = cls_grad['te'];cleew = cls_grad['ee']
        lminA = 30
        lminB = 30
        k_ind = 'p'
        dL = 10
        lmaxphi = LMAXP
        lps =  [1] + list(range(2, 111, 10))
        lps += list(range(lps[-1] + 30, 580, 30))
        lps += list(range(lps[-1] + 100, lmaxphi // 2, 100))
        lps += list(range(lps[-1] + 300, lmaxphi, 300))
        if lps[-1] != lmaxphi: lps.append(lmaxphi)
        lps = np.array(lps)
        devs = []
        for k1k2 in args.k1k2s:
            Nt = 1
            kA, kB = k1k2.split(' ')
            t0 = time.time()
            for i in range(Nt):
                this_n1 = n1._calc_n1L_sTP(L, cpp, kA, kB, k_ind, cltt, clte, clee, clttw, cltew, cleew,
                              ftlA, felA, fblA, ftlB, felB, fblB, lminA, lminB, dL, lps)
            ti = time.time() - t0
            print('%s %s N1: %.3f sec per trial'%(kA, kB, ti /Nt))
            t0 = time.time()
            for i in range(Nt):
                pl_n1 = n1_plancklens._calc_n1L_sTP(L, cpp, kA, kB, k_ind, cltt, clte, clee, clttw, cltew, cleew,
                              ftlA, felA, fblA, ftlB, felB, fblB, lminA, lminB, dL, lps)
            ti = time.time() - t0
            print('%s %s N1: %.3f sec per trial (plancklens)'%(kA, kB, ti /Nt))
            print('rel dev %.5e, val %.5e'%(this_n1/pl_n1-1, pl_n1))
            devs.append(this_n1/pl_n1-1)
        print('*********')
        print('maxreldev %.3e'%np.max(np.abs(devs)))
