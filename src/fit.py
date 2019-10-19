#!/usr/bin/env python

import math
import numpy as np
from astropy.io import fits
import lmfit
import matplotlib.pyplot as plt
import rebin
import convol


def read_template(fname):
    fit = fits.open(fname)
    data = fit[1].data
    wave = (10**data['LogLam']).astype(np.float64)
    flux = data['Flux'].astype(np.float64)
    err = data['PropErr'].astype(np.float64)
    return wave, flux, err


def read_target(fname):
    fit = fits.open(fname)
    data = fit[0].data
    head = fit[0].header
    size = head['NAXIS1']
    begin = head['CRVAL1']
    step = head['CD1_1']
    wave = np.arange(size)*step + begin
    flux = data[0, 0, :].astype(np.float64)
    err = data[3, 0, :].astype(np.float64)
    return wave, flux, err


def read_sdss(fname):
    fit = fits.open(fname)
    data = fit[1].data
    wave = (np.power(10, data['loglam'])).astype(np.float64)
    flux = (data['flux'] * 1.0e-17).astype(np.float64)
    err = (data['ivar'] * 1.0e-17).astype(np.float64)
    return wave, flux, err


class model:
    def __init__(self, tmpname):
        wave, flux, err = read_template(tmpname)
        self.wave = wave
        self.flux = flux
        self.err = err
        self.wshift = -(wave[0]+wave[-1])/2
        self.wscale = 2/(wave[-1]-wave[0])
        typicalflux = np.median(self.flux)
        exponent = math.floor(math.log10(typicalflux))
        self.unit = 10**exponent
        self.flux = self.flux / self.unit

    def trans_wave(self, wave):
        return (wave + self.wshift) * self.wscale

    def get_scale(self, wave, par):
        tmpwave = self.trans_wave(wave)
        return convol.poly(tmpwave, par)

    def convol_spectrum(self, wave, par):
        tmpwave = self.trans_wave(wave)
        return convol.gauss_filter(tmpwave, self.flux, par)

    def get_wave(self, parwave):
        return convol.map_wave(self.wave, parwave)

    def get_spectrum(self, wave, shift1, sigma1, scale0, scale1, scale2, scale3, scale4, scale5):
        new_wave = self.get_wave([0, shift1])
        new_flux = self.convol_spectrum(new_wave, [0, sigma1])
        par_scale = [scale0, scale1, scale2, scale3, scale4, scale5]
        scale = self.get_scale(new_wave, par_scale)
        flux_aftscale = new_flux * scale
        outflux = rebin.rebin(new_wave, flux_aftscale, wave)
        return outflux


def main():
    tmpname = 'data/F5_-1.0_Dwarf.fits'
    ftargetname = 'data/spec-4961-55719-0378.fits'
    print('read template file')
    wt, ft, et = read_template(tmpname)
    print('read target file')
    wo, fo, eo = read_sdss(ftargetname)
    newfluxo = rebin.rebin(wo, fo, wt)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.plot(wt, ft)
    ax2.plot(wt, newfluxo)
    plt.show()


if __name__ == "__main__":
    main()