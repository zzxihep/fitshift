#include <cmath>
#include <numeric>
#include <iostream>
#include <vector>
#include <deque>
// #include <assert.h>
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor/xindex_view.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor/xadapt.hpp"


#define ARR xt::xarray<double>
#define CARR const xt::xarray<double>
#define PYARR xt::pyarray<double>

const double ivc =  1/299792.458;


template<typename IT>
inline double mean(IT from, IT aflast){
  double val = 0.0;
  size_t count = 0;
  for(auto itr = from; itr != aflast; ++itr){
    val += *itr;
    ++count;
  }
  return val / count;
}

template<typename IT>
inline double stddev(IT from, IT aflast){
  double mn = mean(from, aflast);
  double val = 0;
  size_t count = 0;
  for(auto itr = from; itr != aflast; ++itr){
    val += (*itr-mn) * (*itr-mn);
  }
  return sqrt(val/count);
}

template <typename IT1, typename IT2, typename IT3>
inline void multiply(IT1 xfirst, IT1 xaflast, IT2 y, IT3 result){
    while( xfirst != xaflast )
        *(result++) = *(xfirst++) * *(y++);
}

template <typename IT>
inline double sum(IT xfirst, IT xaflast){
    double result = 0;
    while( xfirst != xaflast)
        result += *(xfirst++);
    return result;
}

typedef std::vector<double> VEC ;

std::vector<double> new_cc(const VEC& w1, const VEC& f1, const VEC& w2, const VEC& f2, const VEC& shift){
    VEC new_w1(w1);
    VEC dif(shift.size());
    std::adjacent_difference(shift.begin(), shift.end(), dif.begin());
    std::deque<double> new_w2(w2.begin(), w2.end());
    std::deque<double> new_f2(f2.begin(), f2.end());
    new_w2.push_front(-1.0e50);
    new_w2.push_back(1.0e50);
    new_f2.push_front(new_f2.front());
    new_f2.push_back(new_f2.back());
    std::vector<double> step_w2(new_w2.size()-1);
    std::vector<double> step_f2(new_f2.size()-1);
    std::adjacent_difference(new_w2.begin(), new_w2.end(), step_w2.begin());
    std::adjacent_difference(new_f2.begin(), new_f2.end(), step_f2.begin());
    std::vector<double> slope(step_f2.size());
    for(size_t ind = 0; ind < step_f2.size(); ++ind)
        slope[ind] = step_f2[ind] / step_w2[ind];
    std::vector<double> result(shift.size());
    std::vector<double> int_f2(new_w1.size());
    // std::cout << "Flag 4" << std::endl;
    for(size_t ind = 0; ind < shift.size(); ++ind){
        for ( size_t aa = 0; aa < new_w1.size(); ++aa)
            new_w1[aa] += dif[ind] * w1[aa] * ivc;
        size_t tmpid = new_w2.size()-1;
        for(size_t bb = new_w1.size()-1; bb != 0; --bb){
            while (new_w2[tmpid] > new_w1[bb]) --tmpid;
            double deltaw = new_w1[bb] - new_w2[tmpid];
            double slop = slope[tmpid];
            double basef = new_f2[tmpid];
            int_f2[bb] = basef + slop * deltaw;
        }
        auto itfrom = new_w1.begin();
        auto itend = new_w1.rbegin();
        while(*itfrom < w2.front()) ++itfrom;
        while(*itend > w2.back()) ++itend;
        auto shift1 = itfrom - new_w1.begin();
        auto shift2 = itend - new_w1.rbegin();
        // assert(shift1 >= 0);
        // assert(shift2 >= 0);
        auto arrfrom = f1.begin()+shift1;
        auto arrend = f1.end()-shift2;
        auto length = arrend - arrfrom;
        multiply(arrfrom, arrend, int_f2.begin(), int_f2.begin());
        double r = sum(int_f2.begin(), int_f2.begin()+length);
        r /= length;
        r = sqrt(r);
        result[ind] = r;
    }
    return result;
}


// auto cc(CARR& w1, CARR& f1, CARR& w2, CARR& f2, CARR& shift){
//   ARR new_w1(w1);
//   auto dif = ARR::from_shape(shift.shape());
//   std::adjacent_difference(shift.begin(), shift.end(), dif.begin());
//   std::cout << "dif = " << dif << std::endl;
//   auto slope = xt::diff(f2) / xt::diff(w2);
//   auto intinds = xt::xarray<int>::from_shape(w1.shape());
//   auto r_coefs = xt::xarray<double>::from_shape(shift.shape());
//   for(size_t ind = 0; ind < shift.size(); ++ind){
//     int tmpid = w2.size()-1;
//     new_w1 += dif[ind]*w1*ivc;
//     std::cout << *new_w1.begin() << "  " << *new_w1.rbegin() <<  std::endl;
//     auto arg = new_w1 >= *w2.begin() && new_w1 <= *w2.rbegin();
//     ARR sel_w1 = xt::filter(new_w1, arg);
//     ARR sel_f1 = xt::filter(f1, arg);
//     auto sel_inds = xt::filter(intinds, arg);
//     for(int itrw = sel_w1.size()-1; itrw >= 0; --itrw){
//       while( w2[tmpid] > sel_w1[itrw]) --tmpid;
//       sel_inds[itrw] = tmpid;
//     }
//     ARR sel_w2 = xt::index_view(w2, sel_inds);
//     ARR sel_f2 = xt::index_view(f2, sel_inds);
//     ARR sel_slope = xt::index_view(slope, sel_inds);
//     auto delta_w = sel_w2 - sel_w1;
//     auto int_vals2 = sel_f2 + delta_w * sel_slope;
//     auto mean1 = xt::mean(sel_f1)();
//     auto std1 = xt::stddev(sel_f1)();
//     auto mean2 = xt::mean(int_vals2)();
//     auto std2 = xt::stddev(int_vals2)();
//     double r_coef = xt::sum((int_vals2-mean2) * (sel_f1-mean1))() /
//                     int_vals2.size() / std1 / std2;
//     // double r_coef = xt::sum((int_vals2) * (sel_f1))() /
//     //                 int_vals2.size();
//     r_coefs[ind] = r_coef;
//   }
//   return r_coefs;
// }


auto iccf(PYARR& w1, PYARR& f1, PYARR& w2, PYARR& f2, PYARR& shift){
  std::cout << "norm flux1" << std::endl;
  auto mean1 = xt::mean(f1)();
  auto invstd1 = 1/xt::stddev(f1)();
  std::cout << "norm flux2" << std::endl;
  auto mean2 = xt::mean(f2)();
  auto invstd2 = 1.0/xt::stddev(f2)();
  auto new_f1 = (f1-mean1)*invstd1;
  std::cout << "invstd2 = " << invstd2 << std::endl;
  std::cout << "mean2 = " << mean2 << std::endl;
  std::cout << "f2.size() = " << f2.size() << std::endl;
  auto new_f2 = (f2-mean2)*invstd2;
  std::cout << "Flag1" << std::endl;
  std::vector<double> vw1(w1.begin(), w1.end());
  std::cout << "Flag2" << std::endl;
  std::vector<double> vf1(new_f1.begin(), new_f1.end());
  std::vector<double> vw2(w2.begin(), w2.end());
  std::vector<double> vf2(new_f2.begin(), new_f2.end());
  std::vector<double> vshift(shift.begin(), shift.end());
  std::cout << "run cc" << std::endl;
  auto result1 = new_cc(vw1, vf1, vw2, vf2, vshift);
  std::vector<double> vshift2(vshift);
  for(auto & val : vshift2) val = -val;
  auto result2 = new_cc(vw2, vf2, vw1, vf1, vshift2);
  for(size_t ind = 0; ind < result1.size(); ++ind)
    result1[ind] = (result1[ind] + result2[ind]) * 0.5;
  return result1;
}

//
// int main(){
//     xt::xarray<double> arr1
//       {{1.0, 2.0, 3.0},
//        {2.0, 5.0, 7.0},
//        {2.0, 5.0, 7.0}};
//
//     xt::xarray<double> arr2
//       {5.0, 6.0, 7.0};
//
//     auto bad = arr2 * arr2;
//     auto res = xt::stddev(arr2);
//     std::cout << "data = " << bad << std::endl;
//
//     // xt::xarray<double> res = xt::view(arr1, 1) + arr2;
//
//     xt::xarray<double> arr{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
//     // correlation_coefficient(arr, arr, arr, arr);
//
//     std::cout << res << std::endl;
//
//     return 0;
// }

PYBIND11_MODULE(mytest, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";

    m.def("iccf", iccf, "Sum the sines of the input values");
}
