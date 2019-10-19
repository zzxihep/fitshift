#ifndef __CONVOL_H
#define __CONVOL_H

#include <vector>
#define ARR std::vector<double>

ARR poly(const ARR & arrx, const ARR & arrpar);
ARR map_wave(const ARR & wave, const ARR & map_par);
ARR gauss_filter(const ARR & wave, const ARR & flux, const ARR & arrpar);

#endif