#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import convol
import specio
import rebin


def test_map_wave():
    plt.figure()
    wave = np.arange(4000, 9000, 2.0)
    new_wave = convol.map_wave(wave, [10, 1, 0.1])
    dwave = new_wave - wave
    plt.plot(dwave)


def test_rebin_err(wave, flux, err):
    begin, end = wave[0], wave[-1]
    length = len(wave)
    new_wave = np.linspace(begin, end, length/5)
    # new_wave = wave
    new_flux = np.array(rebin.rebin(wave, flux, new_wave))
    print(new_wave)
    print(len(new_wave))
    print(len(new_flux))
    new_err = np.array(rebin.rebin_err(wave, err, new_wave))
    plt.plot(wave, flux)
    plt.plot(new_wave, new_flux)

    y1 = flux-err
    y2 = flux+err

    ny1 = new_flux-new_err
    ny2 = new_flux+new_err

    plt.fill_between(wave, y1, y2, color='C0', alpha=0.3)
    plt.plot(wave, y1, color='C0', alpha=0.3)
    plt.plot(wave, y2, color='C0', alpha=0.3)
    plt.fill_between(new_wave, ny1, ny2, color='C1', alpha=0.2)
    plt.plot(new_wave, ny1, color='C1', alpha=0.3)
    plt.plot(new_wave, ny2, color='C1', alpha=0.3)
    plt.show()


def test_conv(wave, flux):
    unit = np.median(flux)
    scaleflux = flux
    plt.plot(wave, scaleflux)
    scalewave = 1
    wave = wave / scalewave

    # sigmapar = [1.0e1, 0]
    sigmapar = [0, 1.0e-3]
    newflux = convol.gauss_filter(wave, scaleflux, sigmapar)

    plt.plot(wave*scalewave, newflux, color='red')
    # plt.show()

    plt.figure()
    testflux = np.zeros(wave.shape)
    # print(len(testflux))
    testflux[500] = 1.0
    testflux[3200] = 1.0
    print(wave[0])
    print(wave[-1])
    nwave = np.arange(len(wave), dtype=np.float64)+3814
    # sigmapar = [0, 1.0e-3, 1.0e-7]
    sigmapar = [0, 8.0e-3]
    stestflux = convol.gauss_filter(nwave, testflux, sigmapar)
    plt.plot(nwave, stestflux)
    stestflux2 = convol.gauss_filter(wave, testflux, sigmapar)
    plt.plot(wave, stestflux2)

    test_map_wave()
    plt.show()


def main():
    wave, flux, err = specio.read_sdss('data/spec-2127-53859-0085.fits')
    test_rebin_err(wave, flux, err)
    test_conv(wave, flux)


if __name__ == "__main__":
    main()