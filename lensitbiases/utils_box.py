import numpy as np

def lminlmax2npix(lminbox, lmaxbox):
    return  np.int_(2 * lmaxbox / lminbox + 1)

def freqs(i, n, signed=False):
    """Outputs the absolute integers frequencies [0,1,...,N/2,N/2-1,...,1]
         in numpy fft convention as integer i runs from 0 to N-1.
         Inputs can be numpy arrays e.g. i (i1,i2,i3) with N (N1,N2,N3)
                                      or i (i1,i2,...) with N
         Both inputs must be integers.
         All entries of N must be even.


    """
    #assert (np.all(N % 2 == 0)), "This routine only for even numbers of points"
    #return i - 2 * (i >= (N // 2)) * (i % (N // 2))
    N = (n - 1) // 2 + 1
    p1 = np.arange(0, N, dtype=int)
    results = np.empty(n, int)
    results[:N] = p1
    p2 = np.arange(-(n // 2), 0, dtype=int)
    results[N:] = p2
    return results[i] if signed else np.abs(results[i])

def rfft2_reals(shape):
    """Pure reals modes in 2d rfft according to patch specifics

    """
    N0, N1 = shape
    f0 = [0]
    f1 = [0]
    if N1 % 2 == 0:
        f0.append(0)
        f1.append(N1 // 2)
    if N0 % 2 == 0:
        f0.append(N0 // 2)
        f1.append(0)
    if N1 % 2 == 0 and N0 % 2 == 0:
        f0.append(N0 // 2)
        f1.append(N1 // 2)
    return np.array(f0), np.array(f1)


def lowprimes(n:np.ndarray or int):
    """Finds approximations of integer array n from above built exclusively of low prime numbers 2,3,5.

        Python but still ok here for reasonable n

     """
    if np.isscalar(n):
        n = [n]
        scal = True
    else:
        scal = False
    # --- first builds all candidates powers of low primes, sorted
    nmax = 2 ** int(np.ceil(np.log2(np.max(n)))) # at the worst power of two larger will do
    grid = [0]
    n2 = 1
    while n2 <= nmax:
        n3 = 1
        while n2 * n3 <= nmax:
            n_ = n2 * n3
            while n_ <= nmax:
                grid.append(n_)
                n_ *= 5
            n3 *= 3
        n2 *= 2
    grid = np.sort(grid)
    # --- then loop over them to find the smallest larger integer
    unique_ns = np.unique(np.sort(n))
    nuniq = len(unique_ns)
    sols = {}
    i_unsolved = 0
    for n_ in grid:
        while n_ >= unique_ns[i_unsolved]:
            sols[unique_ns[i_unsolved]] = n_
            i_unsolved += 1
            if i_unsolved >= nuniq:
                return sols[n[0]] if scal else np.array([sols[i] for i in n])
    assert 0