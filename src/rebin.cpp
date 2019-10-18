#include<vector>
#include<numeric>
#include<iostream>
#include "rebin.h"

#define DARR std::vector<double>


DARR get_edge(const DARR & wave){
    DARR interval(wave.size());
    std::adjacent_difference(wave.begin(), wave.end(), interval.begin());
    interval[0] = interval[1];
    for(auto & val : interval)
        val *= 0.5;
    DARR edge_out(wave.size()+1);
    for(size_t ind = 0; ind < wave.size(); ++ind)
        edge_out[ind] = wave[ind] - interval[ind];
    edge_out.back() = wave.back() + interval.back();
    return edge_out;
}

struct wtf {
    public:
    int type;
    double wave;
    double flux;
    bool operator < (const wtf & b) const{
        return wave < b.wave;
    }
};


std::vector<wtf> ssort(const std::vector<wtf> & arr1, const std::vector<wtf> & arr2){
    std::vector<wtf> result(arr1.size()+arr2.size());
    size_t ind = 0;
    auto itr1 = arr1.begin();
    auto itr2 = arr2.begin();
    while(itr1 != arr1.end() && itr2 != arr2.end())
        if ( (*itr1) < (*itr2))
            result[ind++] = *(itr1++);
        else
            result[ind++] = *(itr2++);
    while ( itr1 != arr1.end())
        result[ind++] = *(itr1++);
    while (itr2 != arr2.end())
        result[ind++] = *(itr2++);
    return result;
}


DARR rebin(const DARR & wave, const DARR & flux, const DARR & new_wave){
    auto edge1 = get_edge(wave);
    auto edge2 = get_edge(new_wave);
    std::vector<wtf> group1(edge1.size());
    std::vector<wtf> group2(edge2.size());
    for (size_t ind = 0; ind < wave.size(); ++ind)
        group1[ind] = {0, edge1[ind], flux[ind]};
    group1.back() = {0, edge1.back(), 0.0}; // the last edge flux should be 0
    for ( size_t ind = 0; ind < edge2.size(); ++ind)
        group2[ind] = {1, edge2[ind], 0.0};
    auto sorted_data = ssort(group1, group2);
    for ( size_t ind = 1; ind < sorted_data.size(); ++ind)
        if ( sorted_data[ind].type == 1)
            sorted_data[ind].flux = sorted_data[ind-1].flux;
    DARR new_flux;
    double tmpflux = 0, w1 = 0, w2 = 0;
    size_t ind = 0;
    while ( sorted_data[ind].type != 1) ++ind;
    w1 = sorted_data[ind].wave;
    while(++ind != sorted_data.size()){
        double length = sorted_data[ind].wave - sorted_data[ind-1].wave;
        tmpflux += length * sorted_data[ind-1].flux;
        if ( sorted_data[ind].type == 1){
            w2 = sorted_data[ind].wave;
            double realflux = tmpflux / (w2 - w1);
            new_flux.push_back(realflux);
            w1 = sorted_data[ind].wave;
            tmpflux = 0;
        }
    }
    return new_flux;
}


int main(){
    DARR wave = {1, 2, 3, 5, 9.0};
    auto edge = get_edge(wave);
    for ( auto v : wave)
        std::cout << v << "  ";
    std::cout << std::endl;
    for ( auto & v : edge)
        std::cout << v << "  ";
    std::cout << std::endl;
    DARR wave2 = {2, 3, 5, 6};
    rebin(wave, wave, wave2);
    return 0;
}