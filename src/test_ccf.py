#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import specio
import libccf


def main():
    # spec = specio.Spectrum('data/spec-4961-55719-0378.fits')
    spec = specio.Spectrum('/home/zzx/workspace/data/xiamen/P200-Hale_spec/spec2/fawftbred0027.fits')

    shift = np.arange(-300, 300, 1.0)
    print(len(shift))
    result = libccf.iccf_spec(spec.wave, spec.flux, spec.wave,
                         spec.flux, shift)
    # print(type(result))
    # print(result)
    plt.plot(shift, result)
    plt.show()
    # print(spec.err)
    peak, cent = libccf.iccf_mc(spec.wave, spec.flux_unit, spec.err_unit,
                                      spec.wave, spec.flux_unit, spec.err_unit,
                                      shift, 1000);
    print(len(peak))
    print(len(cent))
    # plt.hist(peak, bins=50)
    plt.hist(cent, bins=50)
    plt.show()


if __name__ == '__main__':
    main()
