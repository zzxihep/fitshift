#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import specio
import libccf


def main():
    spec = specio.Spectrum('data/spec-4961-55719-0378.fits')
    # spec = specio.Spectrum('/home/zzx/workspace/data/xiamen/P200-Hale_spec/spec2/fawftbred0027.fits')

    shift = np.arange(-3000, 3000, 10)
    print(len(shift))
    result = libccf.iccf(spec.wave, spec.flux, spec.wave,
                         spec.flux, shift)
    # print(type(result))
    # print(result)
    plt.plot(shift, result)
    # plt.plot(spec.wave, result)
    plt.show()


if __name__ == '__main__':
    main()
