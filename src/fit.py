#!/usr/bin/env python

import math
import numpy as np
from astropy.io import fits
from lmfit import minimize, Parameters
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
    err = (1/data['ivar'] * 1.0e-17).astype(np.float64)
    # ax1 = plt.subplot(211)
    # ax2 = plt.subplot(212, sharex=ax1)
    # ax1.plot(wave, flux)
    # ax2.plot(wave, err)
    # plt.show()
    return wave, flux, err


class Model:
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
        return np.array(convol.poly(tmpwave, par))

    def convol_spectrum(self, wave, par):
        # tmpwave = self.trans_wave(wave)
        return np.array(convol.gauss_filter(wave, self.flux, par))

    def get_wave(self, parwave):
        # print(parwave)
        return np.array(convol.map_wave(self.wave, parwave))

    def get_spectrum(self, wave, arrshift, arrsigma, arrscale):
        # print(shift1)
        new_wave = self.get_wave(arrshift)
        new_flux = self.convol_spectrum(new_wave, arrsigma)
        # print(len(new_flux))
        # print(new_flux.shape)
        scale = self.get_scale(new_wave, arrscale)
        # print(len(scale))
        # print(scale.shape)
        flux_aftscale = new_flux * scale
        outflux = np.array(rebin.rebin(new_wave, flux_aftscale, wave))
        return outflux


def get_residual(tmpname):
    template = Model(tmpname)
    def get_spec_model(pars, x):
        shift0 = pars['shift0'].value
        shift1 = pars['shift1'].value
        sigma1 = pars['sigma'].value
        scale0 = pars['scale0'].value
        scale1 = pars['scale1'].value
        scale2 = pars['scale2'].value
        scale3 = pars['scale3'].value
        scale4 = pars['scale4'].value
        scale5 = pars['scale5'].value
        shift = [shift0, shift1]
        sigma = [0.0, sigma1]
        scale = [scale0, scale1, scale2, scale3, scale4, scale5]
        print(shift)
        print(sigma)
        print(scale)
        spec_mod = template.get_spectrum(x, shift, sigma, scale)
        return spec_mod
    def residual(pars, x, data=None, eps=None):
        spec_mod = get_spec_model(pars, x)
        return (data - spec_mod) / eps
    return residual, get_spec_model


def get_scale_pars(pars):
    keywordlst = []
    valuelst = []
    for keyword in pars:
        if 'scale' in keyword:
            keywordlst.append(keyword)
            valuelst.append(pars[keyword].value)
    keywordlst = np.array(keywordlst)
    valuelst = np.array(valuelst)
    arg = np.argsort(keywordlst)
    return valuelst[arg]


def set_scale_pars(pars, order, valuelst=None):
    if valuelst is None:
        _value = np.zeros(order+1)
        _value[0] = 1
    else:
        _value = valuelst
    for ind in range(order+1):
        scalename = 'scale'+str(ind)
        pars.add(scalename, value=_value[ind])


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
    # ax2.plot(wo, fo)
    ax2.errorbar(wo, fo, yerr=eo)
    tempalte = Model('data/F5_-1.0_Dwarf.fits')
    flux_fromtemp = tempalte.get_spectrum(wo, [0], [0, 1.0e-4], [1., 3.0e-1, -2.0e-1])
    print('template.unit')
    print(tempalte.unit)
    flux_fromtemp = flux_fromtemp * tempalte.unit
    ax1.plot(wo, flux_fromtemp)
    plt.show()
    residual, get_spec = get_residual('data/F5_-1.0_Dwarf.fits')
    params = Parameters()
    params.add('shift0', value=1.1568794774016442)
    params.add('shift1', value=-0.0007594668175056121)
    params.add('sigma', value=0.00016558594418925043, min=1.0e-8)
    scalevalst = [4.543402040007523, -0.20454792267985503, -0.2391637452260473,
                  0.2190777818642178, -0.09965310075298969, -0.1255319879292037]
    set_scale_pars(params, 5, valuelst=scalevalst)

    arg = np.where((wo>3660) & (wo<10170))
    new_wo = wo[arg]
    new_fo = fo[arg]
    new_eo = eo[arg]
    unit = 10**math.floor(math.log10(np.median(new_fo)))
    new_fo = new_fo / unit
    new_eo = new_eo / unit * 2.0
    print(unit)

    out = minimize(residual, params, args=(new_wo, new_fo, new_eo))
    out_parms = out.params
    print(out_parms)
    spec_fit = get_spec(out_parms, new_wo)
    # plt.errorbar(new_wo, new_fo, yerr=new_eo)
    plt.plot(new_wo, new_fo)
    plt.plot(new_wo, spec_fit)
    # plt.show()
    mymodel = Model('data/F5_-1.0_Dwarf.fits')
    scalepar = get_scale_pars(out_parms)
    myscale = mymodel.get_scale(new_wo, scalepar)
    plt.figure()
    plt.plot(new_wo, myscale)
    print(scalepar)
    plt.show()


if __name__ == "__main__":
    main()
