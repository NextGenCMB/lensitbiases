import numpy as np
from n1 import stokes, n1fft

def test_transpose():
    """This tests:

        W_L^{ST, (1 0)}_{0} = W_L^{ST, (1 0)}_{1}.transpose()
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
                WST_0 = np.fft.ifft2(slib.W_ST(T, S, verbose=False, ders_1=0))
                WST_1 = np.fft.ifft2(slib.W_ST(T, S, verbose=False, ders_1=1))
                sgn = (-1) ** (S == 'Q') *  (-1) ** (T == 'Q')
                if not np.allclose(WST_1.transpose(),sgn * WST_0):
                    print("NOK " + S + T  +  '  jTP' *jt_TP)
                else:
                    print(" OK " + S + T  +  '  jTP' *jt_TP)


def test_symmetries():
    """This tests:

        - W_{-L}^{ST} = W_{+L}^{TS} for all T, S and MV weights
        - W_{-L}^{ST, (0, 1)} = -W_{+L}^{TS, (1, 0)} for all T, S and MV weights
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