#include <iostream>
#include <numeric>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
// #include "xtensor/xview.hpp"
#include "xtensor/xindex_view.hpp"
// #include "xtensor/xtensor.hpp"


#define ARR xt::xarray<double>
#define CARR const xt::xarray<double>
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

auto correlation_coefficient(ARR & w1, CARR & f1, CARR & t2, CARR & f2){
    auto arg = w1 < 7;
    auto arg2 = xt::argwhere(w1 < 7 && w1 > 2);
    xt::xarray<int> inds {1, 3, 5};
    // auto newline = xt::filter(w1, inds);
    auto newline = xt::index_view(w1, arg2);
    auto dif = xt::xarray<double>::from_shape(w1.shape());
    xt::xarray<double> tmp(w1);
    tmp[1] = 3.1415;
    std::adjacent_difference(w1.begin(), w1.end(), dif.begin());

    // auto dif = xt::zeros(w1.shape());
    // auto dif = xt::diff(t1); 
    // newline += 100;
    std::cout << w1 << w1.size() << std::endl;
    std::cout << tmp << std::endl;
    std::cout << dif << dif.size() << std::endl;
    std::cout << arg << std::endl;
    // std::cout << arg2 << std::endl;
    std::cout << *w1.rbegin() << std::endl;
    std::cout << "newline" << newline << std::endl;
    std::cout << newline * newline << std::endl;
    double test = xt::sum(newline * newline)();
    std::cout << test << std::endl;
    return w1+t2;
}


auto iccf(CARR & t1, CARR & f1, CARR & t2, CARR & f2){
    return t1+t2;
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
    correlation_coefficient(arr, arr, arr, arr);

    std::cout << res << std::endl;

    return 0;
}