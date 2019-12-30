#include <cmath>
#include <numeric>
#include <utility>
#include <vector>
#include <iostream>
#include <iomanip>
#include <gsl/gsl_sf_legendre.h>
#include "convol.h"


// Please ensure -1 < arrx < 1
ARR legendre_poly( const ARR & arrx, const ARR & arrpar){
    ARR arrresult(arrx.size());
    for ( auto & val : arrresult) val = 0;
    double * buff = new double[arrpar.size()];
    double order = arrpar.size() - 1;
    for(size_t ind = 0; ind < arrx.size(); ++ind){
        gsl_sf_legendre_Pl_array(order, arrx[ind], buff);
        double temp = 0;
        for ( size_t nn = 0; nn < arrpar.size(); ++nn)
            temp += arrpar[nn] * buff[nn];
        arrresult[ind] = temp;
    }
    delete [] buff;
    return arrresult;
}

ARR poly(const ARR & arrx, const ARR & arrpar){
    ARR arrret(arrx.size());
    for ( auto & val : arrret) val = 0;
    for ( size_t ind = 0; ind < arrpar.size(); ++ind){
        double par = arrpar[ind];
        if ( par != 0)
            for ( size_t j = 0; j < arrret.size(); ++j){
                double comp = par * pow(arrx[j], ind);
                arrret[j] += comp;
            }
    }
    return arrret;
}


// Dlambda = a0 + a1*lam + a2*lam^2 + a3*lam3 ...
// return lambda + Dlambda
ARR map_wave(const ARR & wave, const ARR & map_par){
    ARR new_wave = poly(wave, map_par);
    for ( size_t ind = 0; ind < new_wave.size(); ++ind)
        new_wave[ind] += wave[ind];
    return new_wave;
}

inline double gaussian(double x, double sigma, double x0){
    // return 1/sqrt(2*M_PI)/sigma * exp(-(x-x0)*(x-x0)/(2*sigma*sigma));
    return exp(-(x-x0)*(x-x0)/(2*sigma*sigma));
}

ARR gaussian(const ARR & arrx, double sigma, double x0){
    ARR arrret(arrx.size());
    for ( size_t ind=0; ind<arrx.size(); ++ind){
        double x = arrx[ind];
        arrret[ind] = gaussian(x, sigma, x0);
    }
    return arrret;
}

std::pair<int, int> gaussian2(const ARR & arrx, double sigma, double x0, ARR & result, size_t index_ref, double threshold_ratio){
    // double x_ref = arrx[index_ref];
    double val_ref = gaussian(x0, sigma, x0);
    double valtmp = val_ref;
    double threshold = val_ref * threshold_ratio;
    int indl = index_ref;
    result[indl] = valtmp;
    while ( valtmp > threshold) {
        --indl;
        if ( indl < 0) {indl = 0; break;}
        valtmp = gaussian(arrx[indl], sigma, x0);
        result[indl] = valtmp;
    };
    int indr = index_ref + 1;
    valtmp = val_ref;
    while (indr<arrx.size() && valtmp>threshold){
        valtmp = gaussian(arrx[indr], sigma, x0);
        result[indr++] = valtmp;
    }
    return std::make_pair(indl, indr);
}


ARR get_edge(const ARR & wave){
    ARR interval(wave.size());
    std::adjacent_difference(wave.begin(), wave.end(), interval.begin());
    interval[0] = interval[1];
    for(auto & val : interval)
        val *= 0.5;
    ARR edge_out(wave.size()+1);
    for(size_t ind = 0; ind < wave.size(); ++ind)
        edge_out[ind] = wave[ind] - interval[ind];
    edge_out.back() = wave.back() + interval.back();
    return edge_out;
}


// a gaussian filter for spectrum, the sigma of gaussians can be different in
// different wave, the sigma is defined as sigma = par0 + par1*wave + par2*wave^2 ...
// return the fluxes after smooth
ARR gauss_filter(const ARR & wave, const ARR & flux, const ARR & arrpar){
    ARR arrsigma = poly(wave, arrpar);
    ARR gauss_profile(wave.size());
    ARR new_flux(wave.size());
    for( auto & val : new_flux) val = 0;
    ARR arredge = get_edge(wave);
    ARR arrwidth;
    for( size_t d = 0; d < arredge.size()-1; ++d)
        arrwidth.push_back(arredge[d+1]-arredge[d]);
    for( size_t ind = 0; ind < wave.size(); ++ind){
        double sigma = arrsigma[ind];
        double w0 = wave[ind];
        double mf = flux[ind] * arrwidth[ind];
        auto indlr = gaussian2(wave, sigma, w0, gauss_profile, ind, 1.0e-5);
        const int indl = indlr.first;
        const int indr = indlr.second;
        double area = 0;
        for ( size_t j =indl; j < indr; ++j)
            area += arrwidth[j] * gauss_profile[j];
        double inv_area = 1.0/area;
        for ( size_t j = indl; j < indr; ++j){
            new_flux[j] += mf * (gauss_profile[j] * inv_area);
            gauss_profile[j] = 0;
        }
    }
    return new_flux;
}


int main(){
    std::cout << gaussian(0, 1, 0) << std::endl;
    std::cout << gaussian(-1, 1, 0) << std::endl;
    std::cout << gaussian(1, 1, 0) << std::endl;
    ARR arrtest = {-1, 0, 1};
    ARR result = gaussian(arrtest, 1, 0);
    for ( auto val : result)
        std::cout << val << std::endl;
    size_t arrsize = 8;
    double arrpoly[arrsize];
    for ( size_t ind = 0; ind < arrsize; ++ind) arrpoly[ind] = 0;
    gsl_sf_legendre_Pl_array(5, 0.5, arrpoly);
    for(size_t ind = 0; ind < arrsize; ++ind)
        std::cout << arrpoly[ind] << "  ";
    std::cout << std::endl;
    return 0;
}
