#include <map>
#include <vector>
#include "rebin.h"
#include "convol.h"

typedef std::vector<double> DARR;
typedef (const std::vector<double>) conDARR;


class Template{
    DARR wave;
    DARR flux;
    DARR err;
public:
    explicit Template();
    Template(conDARR & wave, conDARR & flux, conDARR & err);
    Template(conDARR & wave, conDARR & flux);
    DARR get_scale(conDARR & wave, conDARR & par);
    DARR gauss_filter(conDARR & par);
    DARR shift_wave(conDARR & par);
    DARR get_spectrum(conDARR & scale_par, conDARR & sigma_par, conDARR & shift_par);
};

Template::Template(conDARR & wave, conDARR & flux):
