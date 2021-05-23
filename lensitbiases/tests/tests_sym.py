import numpy as np
from lensitbiases.utils_n1 import prepare_cls
from lensitbiases.n1_fft import n1_fft


def test_transpose(k, L):
    """This tests:

        W_L^{ST, (1 0)}_{0} = W_L^{ST, (1 0)}_{1}.transpose() *  parity * (-1) ** (S == Q) * (-1) ** (T == Q)

        Note:
            This symmetry conditions relies on no gradient-curl spectra (TB, EB) present

    """


    Xs = []
    if k in ['ptt', 'p', 'xtt', 'x']: Xs += ['T']
    if k in ['p_p', 'p', 'x_p', 'x']: Xs += ['Q', 'U']
    for jt_TP in [False] if k in ['ptt', 'p_p', 'xtt', 'x_p'] else [False, True]:
        ivfs_cls, fals, cls_weights, cls_grad, cpp = prepare_cls(k, jt_TP=jt_TP)
        slib = n1_fft(fals, cls_weights, cls_grad, cpp)
        slib._build_key(k, L)
        for S in Xs:  # Tests W^{ST, (1,)} = sgn * W^{ST, (0,)}.transpose()
            for T in Xs:
                for d0,d1 in [(0, None), (0, 0), (0, 1)]:
                    # Tests W^{ST, (1,)} = sgn * W^{ST, (0,)}.transpose()
                    # Tests W^{ST, (1,1)} = sgn * W^{ST, (0,0)}.transpose()
                    WST_0 = np.fft.ifft2(slib._W_ST(T, S, verbose=False, ders_1=d0, ders_2=d1))
                    WST_1 = np.fft.ifft2(slib._W_ST(T, S, verbose=False, ders_1=1-d0, ders_2=None if d1 is None else 1-d1))
                    sgn = (-1) ** (S == 'Q') *  (-1) ** (T == 'Q') * slib.xy_sym
                    if not np.allclose(WST_1.transpose(),sgn * WST_0):
                        print(k + ": NOK " + S + T + '_0 = ' +'%2s x '%str(sgn) + S + T + '_1' + '  jTP' *jt_TP)
                        return False
                    else:
                        print(k + ":  OK " + S + T + '_0 = ' +'%2s x '%str(sgn) + S + T + '_1' + '  jTP' *jt_TP)

        for ST in ['QU', 'TQ', 'TU', 'UQ', 'QT', 'UT', 'TT', 'UU', 'QQ']:
            S, T = ST
            for a in [None]:
                WST= np.fft.ifft2(slib._W_ST(S, T))
                WTS= np.fft.ifft2(slib._W_ST(T, S))
                sgn = - (-1) ** (S == 'Q') *  (-1) ** (T == 'Q') * slib.xy_sym
                real = WST.real - WTS.real
                imag = WST.imag - sgn * WTS.transpose().imag

                if np.allclose(real, np.zeros_like(real)):
                    print(k + ":  OK   %s_00 = %s_00 for real part "%(S+T, T+S) + '  jTP' * jt_TP)
                else:
                    print(k + ": NOK   %s_00 = %s_00 for real part "%(S+T, T+S) + '  jTP' * jt_TP)
                    return False
                if np.allclose(imag, np.zeros_like(real)):
                    print(k+":  OK   %s_00 = parity * %s_00.T for imag part "%(S+T, T+S) +  '  jTP' * jt_TP)
                else:
                    print(k+": NOK   %s_00 = parity * %s_00.T for imag part "%(S+T, T+S) + '  jTP' * jt_TP)
                    return False
    return True
assert(test_transpose('p', 100))
assert(test_transpose('x', 100))
assert(test_transpose('p', 101))
assert(test_transpose('x', 101))
print("YEAH")
def rfft_map_building_pol():
    k = 'p_p'
    jt_TP = False
    fals, cls_weights, cls_grad, cpp = prepare_cls(k, jt_TP=jt_TP)
    slib = n1_fft(fals, cls_weights, cls_grad, cpp)
    L = 200.
    rfft = True
    slib._build_key('p_p', L, rfft=rfft)
    ift2 =  np.fft.ifft2 if not rfft else  np.fft.irfft2

    W_zz, W_00, W_0_re, W_0_im, W_01 = ift2(np.array(slib._W_ST_Pol()))
    slib._build_key('p_p', L, rfft=False)
    # ====== tests W_zz
    QQ, UU, QU_re, QU_im = W_zz
    print(np.max(np.abs(QQ - np.fft.ifft2(slib._W_ST('Q', 'Q')))),
          np.max(np.abs(UU - np.fft.ifft2(slib._W_ST('U', 'U')))),
          np.max(np.abs(QU_re - np.fft.ifft2(slib._W_ST('Q', 'U')).real)),
          np.max(np.abs(QU_im - np.fft.ifft2(slib._W_ST('Q', 'U')).imag)))
    # ====== tests W_00
    QQ00, UU00, QU00_re, QU00_im = W_00
    print(np.max(np.abs(QQ00 - np.fft.ifft2(slib._W_ST('Q', 'Q', ders_1=0, ders_2=0)).real)),
          np.max(np.abs(UU00 - np.fft.ifft2(slib._W_ST('U', 'U', ders_1=0, ders_2=0)))),
          np.max(np.abs(QU00_re - np.fft.ifft2(slib._W_ST('Q', 'U', ders_1=0, ders_2=0)).real)),
          np.max(np.abs(QU00_im - np.fft.ifft2(slib._W_ST('Q', 'U', ders_1=0, ders_2=0)).imag)))

    # ====== tests W_01
    QQ01_re, UU01_re, QU01_re, QU01_im = W_01
    print(np.max(np.abs(QQ01_re - np.fft.ifft2(slib._W_ST('Q', 'Q', ders_1=0, ders_2=1)).real)),
          np.max(np.abs(UU01_re - np.fft.ifft2(slib._W_ST('U', 'U', ders_1=0, ders_2=1)).real)),
          np.max(np.abs(QU01_re - np.fft.ifft2(slib._W_ST('Q', 'U', ders_1=0, ders_2=1)).real)),
          np.max(np.abs(QU01_im - np.fft.ifft2(slib._W_ST('Q', 'U', ders_1=0, ders_2=1)).imag)))
    #===== tests W_0_re
    QQ0_re, UU0_re, QU0_re, UQ0_re = W_0_re
    print(np.max(np.abs(QQ0_re - np.fft.ifft2(slib._W_ST('Q', 'Q', ders_1=0, ders_2=None)).real)),
          np.max(np.abs(UU0_re - np.fft.ifft2(slib._W_ST('U', 'U', ders_1=0, ders_2=None)).real)),
          np.max(np.abs(QU0_re - np.fft.ifft2(slib._W_ST('Q', 'U', ders_1=0, ders_2=None)).real)),
          np.max(np.abs(UQ0_re - np.fft.ifft2(slib._W_ST('U', 'Q', ders_1=0, ders_2=None)).real)))

    #===== tests W_0_im
    QQ0_im, UU0_im, QU0_im, UQ0_im = W_0_im
    print(np.max(np.abs(QQ0_im - np.fft.ifft2(slib._W_ST('Q', 'Q', ders_1=0, ders_2=None)).imag)),
          np.max(np.abs(UU0_im - np.fft.ifft2(slib._W_ST('U', 'U', ders_1=0, ders_2=None)).imag)),
          np.max(np.abs(QU0_im - np.fft.ifft2(slib._W_ST('Q', 'U', ders_1=0, ders_2=None)).imag)),
          np.max(np.abs(UQ0_im - np.fft.ifft2(slib._W_ST('U', 'Q', ders_1=0, ders_2=None)).imag)))

def test_symmetries():
    """This tests:

        - W_{-L}^{ST} = W_{+L}^{TS} for all T, S and all weights (because of changing l sign is same as swapping l1 and l2)
        - W_{-L}^{ST, (0, 1)} = (-1) W_{+L}^{TS, (1, 0)} for all T, S and MV weights
    """
    from lensitbiases import _n1_ptt

    L = 299.
    for jt_TP in [False, True]:
        fal, cls_weights, cls_grad, cpp = prepare_cls('p', jt_TP=jt_TP)
        lib = _n1_ptt.n1_ptt(fal, cls_weights, cls_grad, cpp, lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p',  L)
                WST = slib._W_ST(T, S, verbose=False)

                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib._W_ST(S, T, verbose=False)
                print('%.3e'%np.max(np.abs(WTS - WST)) , S, T, np.any(WST), 'jt_TP:',jt_TP)

    print("# *** with 1 derivative: **** ")
    for jt_TP in [False, True]:
        sgn = -1
        fal, cls_weights, cls_grad, cpp = prepare_cls('p', jt_TP=jt_TP)
        lib = _n1_ptt.n1_ptt(fal, cls_weights, cls_grad, cpp, lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib._W_ST(T, S, ders_1=0, verbose=False)

                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib._W_ST(S, T, ders_2=0, verbose=False)
                print('(, 0) %.3e' % np.max(np.abs(WTS -sgn * WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)
    for jt_TP in [False, True]:
        sgn = -1
        fal, cls_weights, cls_grad, cpp = prepare_cls('p', jt_TP=jt_TP)
        lib = _n1_ptt.n1_ptt(fal, cls_weights, cls_grad, cpp, lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib._W_ST(T, S, ders_1=1, verbose=False)

                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib._W_ST(S, T, ders_2=1, verbose=False)
                print('(, 1) %.3e' % np.max(np.abs(WTS - sgn * WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)

    print("# *** with 2 derivatives: **** ")
    for jt_TP in [False, True]:
        fal, cls_weights, cls_grad, cpp = prepare_cls('p', jt_TP=jt_TP)
        lib = _n1_ptt.n1_ptt(fal, cls_weights, cls_grad, cpp, lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib._W_ST(T, S, ders_1=1, ders_2=0, verbose=False)

                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib._W_ST(S, T, ders_1=0, ders_2=1, verbose=False)
                print('(0, 1) %.3e' % np.max(np.abs(WTS - WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)
    for jt_TP in [False, True]:
        fal, cls_weights, cls_grad, cpp = prepare_cls('p', jt_TP=jt_TP)
        lib = _n1_ptt.n1_ptt(fal, cls_weights, cls_grad, cpp, lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib._W_ST(T, S, ders_1=1, ders_2=1, verbose=False)

                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib._W_ST(S, T, ders_1=1, ders_2=1, verbose=False)
                print('(1, 1) %.3e' % np.max(np.abs(WTS - WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)
    for jt_TP in [False, True]:
        fal, cls_weights, cls_grad, cpp = prepare_cls('p', jt_TP=jt_TP)
        lib = _n1_ptt.n1_ptt(fal, cls_weights, cls_grad, cpp, lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib._W_ST(T, S, ders_1=0, ders_2=1, verbose=False)

                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib._W_ST(S, T, ders_1=1, ders_2=0, verbose=False)
                print('(1, 0) %.3e' % np.max(np.abs(WTS - WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)
    for jt_TP in [False, True]:
        fal, cls_weights, cls_grad, cpp = prepare_cls('p', jt_TP=jt_TP)
        lib = _n1_ptt.n1_ptt(fal, cls_weights, cls_grad, cpp, lminbox=50, lmaxbox=2500)
        for S in ['T', 'Q', 'U']:
            for T in ['T', 'Q', 'U']:
                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', L)
                WST = slib._W_ST(T, S, ders_1=0, ders_2=0, verbose=False)

                slib = n1_fft(lib.box, lib.Fls, lib.cls_w, lib.cls_f)
                slib._build_key('p_p', -L)
                WTS = slib._W_ST(S, T, ders_1=0, ders_2=0, verbose=False)
                print('(0, 0) %.3e' % np.max(np.abs(WTS - WST)), S, T, np.any(WST), 'jt_TP:', jt_TP)