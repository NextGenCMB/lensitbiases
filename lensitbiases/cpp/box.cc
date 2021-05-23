#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>
#include <cmath>
#include <complex.h>
#include <fftw3.h>

namespace py = pybind11;

using a_c_c = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;
using a_d_c = py::array_t<double, py::array::c_style | py::array::forcecast>;

class Square
  {
  private:
    double lside;  // physical extent in radians
    int npix;
    int nk1; // number of frequencies in first dimension
    int nk2; // number of frequencies in second dimension

    std::vector<int> k1; // signed fft frequencies along first axis
    std::vector<int> k2; // signed fft frequencies along second axis
    
  public:
    Square(const double lside_, const int npix_) {
      lside = lside_; 
      npix = npix_;
      nk1 = npix; 
      nk2 = npix / 2 + 1;
      for (int i2=0; i2<nk2; i2++) {
        k2.push_back(i2); }
      
    }
           
    double lmin(){return 2 * M_PI / lside;}

    a_c_c cos2p(){  // cos 2phi for pixel i in rfft map (cos 2p = 2 cos^2 p - 1 =  (ix ** 2 - iy ** 2) / (ix ** 2 + iy ** 2)) 
      a_c_c c2p_(nk1 * nk2);
      int i1_mid = nk1 / 2 + 1;
      auto c2p = c2p_.mutable_unchecked<1>();
      for (int i=0; i<=i1_mid; ++i){
        c2p[i] = 1.;
      }
      for (int i=i1_mid + 1; i<nk1; ++i){
        c2p[i] = -1.;
      }
      int j1 = 0; // signed fft frequencies in first dimension

      for (int i2=0; i2<nk2; ++i2){
        j1 = 0;
        for (int i1=1; i1<=i1_mid; ++i1){
            j1 += 1;
            c2p[i2 * nk1 + i1] = (double)(j1 * j1 - i2 * i2) / (j1 * j1 + i2 * i2);
        }
        j1 = (npix%2==0)? -j1 : -j1 - 1;
        for (int i1=i1_mid + 1; i1<nk1; ++i1){
            j1 += 1;
            c2p[i2 * nk1 + i1] = (double)( j1 * j1 - i2 * i2) / (j1 * j1 + i2 * i2);
        }
      }
      return c2p_;
      /*
      for (int ix = 0; ix < nkx / 2 + 1, ++ix){
        for (int iy = 0; iy < nky, ++iy){
          
        }
      }
      // something like fr= lambda i, N :  i - 2 * (i > (N // 2)) * (i % ( (N+1) // 2))      
      // FIXME: sign of frequencies irrelevant but take out the N/2 term...
      int ix = i / nky;
      int iy = i % nky;
      ix -= 2 * (ix > (nkx / 2)) * (ix % ( (nkx+1) /  2));
      iy -= 2 * (iy > (nky / 2)) * (iy % ( (nky+1) /  2));
      return static_cast<double>(iy * iy - ix * ix) / (iy * iy + ix * ix);*/
    }

};



/* binders */

using namespace pybind11;
PYBIND11_MODULE(cpp_box, m) {
  m.attr("__version__") = "0.1.0";

  m.doc() = R"pbdoc(
  Tests of C++ flat-sky tools
  )pbdoc";

  py::class_<Square>(m ,"Square")
    .def(py::init<const double, const int>())
    .def("lmin", &Square::lmin)
    .def("cos2p", &Square::cos2p);
}
