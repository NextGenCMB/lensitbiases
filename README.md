# rFFT N1 (and N0's)
rFFT-based N1 lensing bias calculation and tests

* rFFT-based N1 and N1 matrix calculations (~ in O(ms) time per lensing multipole for Planck-like config, which allows on-the-fly evaluation of the bias)
  
* 5 rFFT's of moderate size per L for N1 TT, 20 for PP, 45 for MV or GMV

installation:

    * pip install -e ./ [--user]

# Notes, TODOs, etc
* dN1/dcl TT, EE, BB tested for Planck and works below the percent for <= 400 and ~% for <= 2048, probably due to interpolation scheme
* case of l1 or l2 being zero still to fix
* better freq to multipole mapping?
