#include <numeric>
#include <iostream>
#include "pybind11/pybind11.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor/xindex_view.hpp"
#include "xtensor-python/pyarray.hpp"


#define ARR xt::xarray<double>
#define CARR const xt::xarray<double>
#define PYARR xt::pyarray<double>

const double ivc =  1/299792.458;


auto cc(CARR& w1, CARR& f1, CARR& w2, CARR& f2, CARR& shift){
  ARR new_w1(w1);
  auto dif = ARR::from_shape(shift.shape());
  std::adjacent_difference(shift.begin(), shift.end(), dif.begin());
  auto slope = xt::diff(f2) / xt::diff(w2);
  auto intinds = xt::xarray<int>::from_shape(w1.shape());
  auto r_coefs = xt::xarray<double>::from_shape(shift.shape());
  for(size_t ind = 0; ind < shift.size(); ++ind){
    int tmpid = w2.size()-1;
    new_w1 += dif[ind]*w1*ivc;
    auto arg = new_w1 >= *w2.cbegin() && new_w1 <= *w2.crbegin();
    auto sel_w1 = xt::filter(new_w1, arg);
    auto sel_f1 = xt::filter(f1, arg);
    auto sel_inds = xt::filter(intinds, arg);
    for(int itrw = sel_w1.size()-1; itrw >= 0; --itrw){
      while( w2[tmpid] > sel_w1[itrw]) --tmpid;
      sel_inds[itrw] = tmpid;
    }
    auto sel_w2 = xt::index_view(w2, sel_inds);
    auto sel_f2 = xt::index_view(f2, sel_inds);
    auto sel_slope = xt::index_view(slope, sel_inds);
    auto delta_w = sel_w2 - sel_w1;
    auto int_vals2 = sel_f2 + delta_w * sel_slope;
    double r_coef = xt::sum(int_vals2 * sel_f1)();
    r_coefs[ind] = r_coef;
  }
  return r_coefs;
}


auto iccf(PYARR& w1, PYARR& f1, PYARR& w2, PYARR& f2, PYARR& shift){
  std::cout << "norm flux1" << std::endl;
  auto mean1 = xt::mean(f1);
  auto invstd1 = 1/xt::stddev(f1);
  std::cout << "norm flux2" << std::endl;
  auto mean2 = xt::mean(f2);
  auto invstd2 = 1/xt::stddev(f2);
  auto new_f1 = (f1-mean1)*invstd1;
  auto new_f2 = (f2-mean2)*invstd2;
  std::cout << "run cc" << std::endl;
  return xt::xarray<double>({1.0, 2.0, 3.0});
  return cc(w1, new_f1, w2, new_f2, shift);
}


int main(){
    xt::xarray<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    xt::xarray<double> arr2
      {5.0, 6.0, 7.0};

    auto bad = arr2 * arr2;
    auto res = xt::stddev(arr2);
    std::cout << "data = " << bad << std::endl;

    // xt::xarray<double> res = xt::view(arr1, 1) + arr2;

    xt::xarray<double> arr{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
    // correlation_coefficient(arr, arr, arr, arr);

    std::cout << res << std::endl;

    return 0;
}

PYBIND11_MODULE(mytest, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";

    m.def("iccf", iccf, "Sum the sines of the input values");
}