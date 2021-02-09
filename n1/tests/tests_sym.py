import numpy as np
from n1 import stokes,n1fft, n1devel, n1_utils
from plancklens import utils
import os


def prepare_cls(k, jt_TP=False):
    cls_grad = utils.camb_clfile(os.path.join(n1devel.CLS, 'FFP10_wdipole_gradlensedCls.dat'))
    cls_unl = utils.camb_clfile(os.path.join(n1devel.CLS, 'FFP10_wdipole_lenspotentialCls.dat'))
    cls_weights = utils.camb_clfile(os.path.join(n1devel.CLS, 'FFP10_wdipole_gradlensedCls.dat'))
    fals = n1_utils.get_fal(jt_tp=jt_TP)[1]

    if k == 'ptt':
        fals['ee'] *= 0.
        fals['bb'] *= 0.
    if k == 'p_p':
        fals['tt'] *= 0.
    if k in ['ptt', 'p_p']:
        cls_weights['te'] *= 0.
        cls_grad['te'] *= 0.
    return fals, cls_weights, cls_grad, cls_unl['pp']


def test_transpose(k, L):
    """This tests:

        W_L^{ST, (1 0)}_{0} = W_L^{ST, (1 0)}_{1}.transpose() * (-1) ** (S == Q) * (-1) ** (T == Q)

        Note:
            This symmetry conditions relies on no gradient-curl spectra (TB, EB) present

    """


    Xs = []
    if k in ['ptt', 'p']: Xs += ['T']
    if k in ['p_p', 'p']: Xs += ['Q', 'U']
    for jt_TP in [False] if k in ['ptt', 'p_p'] else [False, True]:
        fals, cls_weights, cls_grad, cpp = prepare_cls(k, jt_TP=jt_TP)
        slib = stokes.stokes(fals, cls_weights, cls_grad, cpp)
        slib._build_key('p_p', L)
        for S in Xs:
            for T in Xs:
                WST_0 = np.fft.ifft2(slib.W_ST(T, S, verbose=False, ders_1=0))
                WST_1 = np.fft.ifft2(slib.W_ST(T, S, verbose=False, ders_1=1))
                sgn = (-1) ** (S == 'Q') *  (-1) ** (T == 'Q')
                if not np.allclose(WST_1.transpose(),sgn * WST_0):
                    print("NOK " + S + T + '_0 = ' +'%2s'%str(sgn) + S + T + '_1' + '  jTP' *jt_TP)
                else:
                    print(" OK " + S + T + '_0 = ' +'%2s'%str(sgn) + S + T + '_1' + '  jTP' *jt_TP)
        if k in ['p_p', 'p']:
            WQU_00= np.fft.ifft2(slib.W_ST('Q', 'U'))
            WUQ_00= np.fft.ifft2(slib.W_ST('U', 'Q'))

            real = WQU_00.real - WUQ_00.real
            imag = WQU_00.imag - WUQ_00.transpose().imag

            if np.allclose(real, np.zeros_like(real)):
                print(" OK   QU_00 = UQ_00 for real part " + '  jTP' * jt_TP)
            else:
                print("NOK   QU_00 = UQ_00 for real part " + '  jTP' * jt_TP)

            if np.allclose(imag, np.zeros_like(real)):
                print(" OK   QU_00 = UQ_00.T for imag part " +  '  jTP' * jt_TP)
            else:
                print("NOK   QU_00 = UQ_00.T for imag part " + '  jTP' * jt_TP)



def test_symmetries():
    """This tests:

        - W_{-L}^{ST} = W_{+L}^{TS} for all T, S and MV weights
        - W_{-L}^{ST, (0, 1)} = (-1) W_{+L}^{TS, (1, 0)} for all T, S and MV weights
    """
    from n1 import n1fft, n1devel, n1_utils
    import os
    from plancklens import utils

    cls_grad = utils.camb_clfile(os.path.join(n1devel.CLS, 'FFP10_wdipole_gradlensedCls.dat'))
    cls_unl = utils.camb_clfile(os.path.join(n1devel.CLS, 'FFP10_wdipole_lenspotentialCls.dat'))

    cls_weights = utils.camb_clfile(os.path.join(n1devel.CLS, 'FFP10_wdipole_gradlensedCls.dat'))
    L = 299.
    for jt_TP in [False, True]:
        fal = n1_utils.get_fal(jt_tp=jt_TP)[1]
        lib = n1fft.n1_ptt(fal, cls_weights, cls_grad, cls_unl['pp'], lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p',  L)
                WST = slib.W_ST(T, S, verbose=False)

                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib.W_ST(S, T, verbose=False)
                print('%.3e'%np.max(np.abs(WTS - WST)) , S, T, np.any(WST), 'jt_TP:',jt_TP)

    print("# *** with 1 derivative: **** ")
    for jt_TP in [False, True]:
        sgn = -1
        fal = n1_utils.get_fal(jt_tp=jt_TP)[1]
        lib = n1fft.n1_ptt(fal, cls_weights, cls_grad, cls_unl['pp'], lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib.W_ST(T, S, ders_1=0, verbose=False)

                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib.W_ST(S, T, ders_2=0, verbose=False)
                print('(, 0) %.3e' % np.max(np.abs(WTS -sgn * WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)
    for jt_TP in [False, True]:
        sgn = -1
        fal = n1_utils.get_fal(jt_tp=jt_TP)[1]
        lib = n1fft.n1_ptt(fal, cls_weights, cls_grad, cls_unl['pp'], lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib.W_ST(T, S, ders_1=1, verbose=False)

                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib.W_ST(S, T, ders_2=1, verbose=False)
                print('(, 1) %.3e' % np.max(np.abs(WTS - sgn * WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)

    print("# *** with 2 derivatives: **** ")
    for jt_TP in [False, True]:
        fal = n1_utils.get_fal(jt_tp=jt_TP)[1]
        lib = n1fft.n1_ptt(fal, cls_weights, cls_grad, cls_unl['pp'], lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib.W_ST(T, S, ders_1=1, ders_2=0, verbose=False)

                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib.W_ST(S, T, ders_1=0, ders_2=1, verbose=False)
                print('(0, 1) %.3e' % np.max(np.abs(WTS - WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)
    for jt_TP in [False, True]:
        fal = n1_utils.get_fal(jt_tp=jt_TP)[1]
        lib = n1fft.n1_ptt(fal, cls_weights, cls_grad, cls_unl['pp'], lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib.W_ST(T, S, ders_1=1, ders_2=1, verbose=False)

                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib.W_ST(S, T, ders_1=1, ders_2=1, verbose=False)
                print('(1, 1) %.3e' % np.max(np.abs(WTS - WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)
    for jt_TP in [False, True]:
        fal = n1_utils.get_fal(jt_tp=jt_TP)[1]
        lib = n1fft.n1_ptt(fal, cls_weights, cls_grad, cls_unl['pp'], lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib.W_ST(T, S, ders_1=0, ders_2=1, verbose=False)

                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib.W_ST(S, T, ders_1=1, ders_2=0, verbose=False)
                print('(1, 0) %.3e' % np.max(np.abs(WTS - WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)
    for jt_TP in [False, True]:
        fal = n1_utils.get_fal(jt_tp=jt_TP)[1]
        lib = n1fft.n1_ptt(fal, cls_weights, cls_grad, cls_unl['pp'], lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib.W_ST(T, S, ders_1=0, ders_2=0, verbose=False)

                slib = stokes.stokes(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib.W_ST(S, T, ders_1=0, ders_2=0, verbose=False)
                print('(0, 0) %.3e' % np.max(np.abs(WTS - WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)