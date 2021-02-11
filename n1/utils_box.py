import numpy as np

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