#!/usr/bin/env python

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