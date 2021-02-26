from os import path as op

def get_default_cls():
    from lensitbiases.utils_n1 import camb_clfile
    fn = op.join(op.dirname(op.abspath(__file__)), 'data', 'cls')
    cls_len  = camb_clfile(op.join(fn, 'FFP10_wdipole_lensedCls.dat'))
    cls_unl  = camb_clfile(op.join(fn, 'FFP10_wdipole_lenspotentialCls.dat'))
    cls_grad = camb_clfile(op.join(fn, 'FFP10_wdipole_gradlensedCls.dat'))
    return cls_unl, cls_len, cls_grad
