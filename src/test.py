#!/usr/bin/env python2


import numpy as np
import matplotlib.pyplot as plt
import convol


def main():
    data = np.loadtxt('spec-2127-53859-0085.txt')
    wave = data[:, 0]
    flux = data[:, 1]
    unit = np.median(flux)
    scaleflux = flux
    plt.plot(wave, scaleflux)
    scalewave = 1
    wave = wave / scalewave

    # sigmapar = [1.0e1, 0]
    sigmapar = [0, 1.0e-3]
    newflux = convol.gauss_filter(wave, scaleflux, sigmapar)

    plt.plot(wave*scalewave, newflux, color='red')

    plt.figure()
    testflux = np.zeros(wave.shape)
    # print(len(testflux))
    testflux[500] = 1.0
    testflux[3200] = 1.0
    # sigmapar = [0, 1.0e-3, 1.0e-7]
    sigmapar = [0, 8.0e-3]
    stestflux = convol.gauss_filter(wave, testflux, sigmapar)
    plt.plot(wave, stestflux)

    plt.show()


if __name__ == "__main__":
    main()