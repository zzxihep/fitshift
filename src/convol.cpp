#include <vector>
#include <cmath>
#include <iostream>
#include "convol.h"

#define ARR std::vector<double>

ARR convol(const ARR & wave, const ARR & flux, const ARR & par){
    ARR newflux(flux.size());
    return newflux;
}

ARR poly(const ARR & arrx, const ARR & arrpar){
    ARR arrret(arrx.size());
    for ( auto & val : arrret) val = 0;
    for ( size_t ind = 0; ind < arrpar.size(); ++ind){
        double par = arrpar[ind];
        if ( par != 0)
            for ( size_t j = 0; j < arrret.size(); ++j){
                double comp = par * pow(arrx[j], j);
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

double gaussian(double x, double sigma, double x0){
    return 1/sqrt(2*M_PI) * exp(-(x-x0)*(x-x0)/(2*sigma*sigma));
}

ARR gaussian(const ARR & arrx, double sigma, double x0){
    ARR arrret(arrx.size());
    for ( size_t ind=0; ind<arrx.size(); ++ind){
        double x = arrx[ind];
        arrret[ind] = 1/sqrt(2*M_PI) * exp(-(x-x0)*(x-x0)/(2*sigma*sigma));
    }
    return arrret;
}

// a gaussian filter for spectrum, the sigma of gaussians can be different in
// different wave, the sigma is defined as sigma = par0 + par1*wave + par2*wave^2 ...
// return the fluxes after smooth
ARR gauss_filter(const ARR & wave, const ARR & flux, const ARR & arrpar){
    ARR arrsigma = poly(wave, arrpar);
    ARR new_flux(wave.size());
    for( auto & val : new_flux) val = 0;
    for( size_t ind = 0; ind < wave.size(); ++ind){
        double sigma = arrsigma[ind];
        double w = wave[ind];
        double f = flux[ind];
        ARR gauss_profile = gaussian(wave, sigma, w);
        for ( size_t j = 0; j < wave.size(); ++j)
            new_flux[j] += f * gauss_profile[j];
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
    return 0;
}