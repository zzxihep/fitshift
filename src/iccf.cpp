#include <cmath>
#include <numeric>
// #include <iostream>
#include <vector>
#include <deque>
// #include "pybind11/pybind11.h"
#include <pybind11/stl.h>
#include <gsl/gsl_statistics_double.h>


typedef std::vector<double> VEC ;

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
inline double stddev(IT from, IT aflast, double mean){
  double val = 0;
  size_t count = 0;
  for(auto itr = from; itr != aflast; ++itr){
    val += (*itr-mean) * (*itr-mean);
    ++count;
  }
  return sqrt(val/count);
}

VEC new_cc(const VEC& w1, const VEC& f1, const VEC& w2, const VEC& f2, const VEC& shift){
    VEC new_w1(w1);
    VEC dif(shift.size());
    std::adjacent_difference(shift.begin(), shift.end(), dif.begin());
    std::deque<double> new_w2(w2.begin(), w2.end());
    std::deque<double> new_f2(f2.begin(), f2.end());
    new_w2.push_front(-1.0e50);
    new_w2.push_back(1.0e50);
    new_f2.push_front(new_f2.front());
    new_f2.push_back(new_f2.back());
    VEC step_w2(new_w2.size()-1);
    VEC step_f2(new_f2.size()-1);
    std::adjacent_difference(new_w2.begin(), new_w2.end(), step_w2.begin());
    std::adjacent_difference(new_f2.begin(), new_f2.end(), step_f2.begin());
    VEC slope(step_f2.size());
    for(size_t ind = 0; ind < step_f2.size(); ++ind)
        slope[ind] = step_f2[ind] / step_w2[ind];
    VEC result(shift.size());
    VEC int_f2(new_w1.size());
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
        auto arrfrom = f1.begin()+shift1;
        auto arrend = f1.end()-shift2;
        auto brrfrom = int_f2.begin()+shift1;
        auto length = arrend - arrfrom;
        double r = std::inner_product(arrfrom, arrend, brrfrom, 0.0);
        // double r = 0;
        // while(arrfrom != arrend) r += *arrfrom++ * *brrfrom++;
        r /= length;
        result[ind] = r;
    }
    return result;
}


auto iccf( const VEC& w1, const VEC& f1, const VEC& w2, const VEC& f2, const VEC& shift){
  double mean1 = mean(f1.begin(), f1.end());
  // double mean1 = gsl_stats_mean(f1.begin(), 1, f1.size());
  double invstd1 = 1/stddev(f1.begin(), f1.end(), mean1);
  auto mean2 = mean(f2.begin(), f2.end());
  auto invstd2 = 1.0/stddev(f2.begin(), f2.end(), mean2);
  VEC new_f1(f1);
  for(auto & val : new_f1) val = (val - mean1) * invstd1;
  VEC new_f2(f2);
  for(auto & val : new_f2) val = (val - mean2) * invstd2;
  auto result1 = new_cc(w1, new_f1, w2, new_f2, shift);
  VEC shift2(shift);
  for(auto & val : shift2) val = -val;
  auto result2 = new_cc(w2, new_f2, w1, new_f1, shift2);
  for(size_t ind = 0; ind < result1.size(); ++ind)
    result1[ind] = (result1[ind] + result2[ind]) * 0.5;
  return result1;
}

PYBIND11_MODULE(libccf, m)
{
    // xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";

    m.def("iccf", iccf, "Sum the sines of the input values");
}
