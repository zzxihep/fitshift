#ifndef __CONVOL_H
#define __CONVOL_H

#include <vector>
#define ARR std::vector<double>

ARR map_wave(const ARR & wave, const ARR & map_par);
ARR gauss_filter(const ARR & wave, const ARR & flux, const ARR & arrpar);

#endif