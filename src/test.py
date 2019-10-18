#!/usr/bin/env python2


import numpy as np
import matplotlib.pyplot as plt
import convol


def test_map_wave():
    plt.figure()
    wave = np.arange(4000, 9000, 2.0)
    new_wave = convol.map_wave(wave, [10, 1, 0.1])
    dwave = new_wave - wave
    plt.plot(dwave)


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
    print(wave[0])
    print(wave[-1])
    nwave = np.arange(len(wave))+3814
    # sigmapar = [0, 1.0e-3, 1.0e-7]
    sigmapar = [0, 8.0e-3]
    stestflux = convol.gauss_filter(nwave, testflux, sigmapar)
    plt.plot(nwave, stestflux)
    stestflux2 = convol.gauss_filter(wave, testflux, sigmapar)
    plt.plot(wave, stestflux2)

    test_map_wave()
    plt.show()


if __name__ == "__main__":
    main()