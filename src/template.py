#!/usr/bin/env python

# import os
# import time
import math
import numpy as np
# from astropy.io import fits
from astropy.constants import c
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from lmfit.printfuncs import report_fit
import corner
import rebin
import convol
import specio
import func


class Model:
    def __init__(self, tmpname):
        self.filename = tmpname
        wave, flux, err = specio.read_template(tmpname)
        self.wave = wave
        self.flux = flux
        self.err = err
        self.wshift = -(wave[0]+wave[-1])/2
        self.wscale = 1.98/(wave[-1]-wave[0])
        typicalflux = np.median(self.flux)
        exponent = math.floor(math.log10(typicalflux))
        self.unit = 10**exponent
        self.flux = self.flux / self.unit

    def reset_wave_zoom(self, wave):
        self.wshift = -(wave[0]+wave[-1])/2
        self.wscale = 1.99/(wave[-1]-wave[0])

    def trans_wave(self, wave):
        return (wave + self.wshift) * self.wscale

    def get_scale(self, wave, par):
        tmpwave = self.trans_wave(wave)
        return np.array(convol.poly(tmpwave, par))

    def get_legendre_scale(self, wave, par):
        tmpwave = self.trans_wave(wave)
        return np.array(convol.legendre_poly(tmpwave, par))

    def convol_spectrum(self, wave, par):
        # tmpwave = self.trans_wave(wave)
        if par is None:
            return self.flux
        return np.array(convol.gauss_filter(wave, self.flux, par))

    def get_wave(self, parwave):
        return np.array(convol.map_wave(self.wave, parwave))

    def get_spectrum_pre(self, wave, arrshift, arrsigma, arrscale):
        new_wave = self.get_wave(arrshift)
        new_flux = self.convol_spectrum(new_wave, arrsigma)
        # scale = self.get_scale(new_wave, arrscale)
        scale = self.get_legendre_scale(wave, arrscale)
        flux_rebin = np.array(rebin.rebin(new_wave, new_flux, wave))
        flux_aftscale = flux_rebin * scale
        return flux_aftscale

    def set_lmpar_name(self, parscale, parsigma, parshift):
        self.lmscale_name = [None] * (max(parscale)+1)
        for key in parscale:
            self.lmscale_name[key] = parscale[key]
        self.lmsigma_name = [None] * (max(parsigma)+1)
        for key in parsigma:
            self.lmsigma_name[key] = parsigma[key]
        self.lmshift_name = [None] * (max(parshift)+1)
        for key in parshift:
            self.lmshift_name[key] = parshift[key]

    def get_scale_par(self, pars):
        parscale = []
        for parname in self.lmscale_name:
            if parname is None:
                parscale.append(0.0)
            else:
                parscale.append(pars[parname].value)
        return parscale

    def get_sigma_par(self, pars):
        parsigma = []
        for parname in self.lmsigma_name:
            if parname is None:
                parsigma.append(0.0)
            else:
                parsigma.append(pars[parname].value)
        return parsigma

    def get_shift_par(self, pars):
        parshift = []
        for parname in self.lmshift_name:
            if parname is None:
                parshift.append(0.0)
            else:
                parshift.append(pars[parname].value)
        return parshift

    def get_spectrum(self, pars, wave):
        try:
            parscale = self.get_scale_par(pars)
            parsigma = self.get_sigma_par(pars)
            parshift = self.get_shift_par(pars)
        except AttributeError as err:
            print(err)
            print('May be you forget to execute set_lmpar_name function')
            parscale, parsigma, parshift = get_pars_name(pars)
            self.set_lmpar_name(parscale, parsigma, parshift)
            return self.get_spectrum(pars, wave)
        return self.get_spectrum_pre(wave, parshift, parsigma, parscale)

    def residual(self, pars, x, data, eps=None):
        spec = self.get_spectrum(pars, x)
        if eps is not None:
            return (data - spec) / eps
        else:
            return data-spec


def get_pars_name(pars):
    """
    return dictionary of parscale, parsigma, parshift
    """
    pardata = {'scale': [], 'sigma': [], 'shift': []}
    for key in pars:
        flag = key[:5]
        order = int(key[5:])
        pardata[flag].append([order, key])
    dicscale = dict(pardata['scale'])
    dicsigma = dict(pardata['sigma'])
    dicshift = dict(pardata['shift'])
    return dicscale, dicsigma, dicshift


def set_pars(pars, prefix, order, valuelst=None, minlst=None, maxlst=None):
    """
    add par to pars, the parname is named as prefix+str(order)
    order: list(like [0, 1, 2, 3]) or int (the orderlst will be regarded as
    [0, 1, 2,..., order]),
    return a dictionary(keyword:order, value:parname)
    """
    if isinstance(order, int):
        numlst = range(order+1)
    else:
        numlst = order
    if valuelst is None:
        valuelst = [None] * len(numlst)
    if minlst is None:
        minlst = [-float('inf')] * len(numlst)
    if maxlst is None:
        maxlst = [float('inf')] * len(numlst)
    keywordlst = []
    for ind, num in enumerate(numlst):
        key = prefix+str(num)
        pars.add(key, value=valuelst[ind], min=minlst[ind], max=maxlst[ind])
        keywordlst.append([num, key])
    return dict(keywordlst)


def fit(template, wave, flux, err, params=None, show=False, isprint=False):
    if params is None:
        params = Parameters()
        # set_pars(params, 'shift', [0, 1], valuelst=[1.16, -0.00076])
        shiftparname = set_pars(params, 'shift', [1], valuelst=[0.0])
        sigmaparname = set_pars(params, 'sigma', [1], valuelst=[0.001],
                                minlst=[1.0e-8])
        scalevalst = [4.5, -0.2, -0.24, 0.22, -0.1, -0.13]
        scaleparname = set_pars(params, 'scale', 5, valuelst=scalevalst)
        template.set_lmpar_name(scaleparname, sigmaparname, shiftparname)
    template.reset_wave_zoom(wave)
    # start = time.process_time()
    # print(start)
    out = minimize(template.residual, params, args=(wave, flux, err),
                   method='leastsq')
    # end = time.process_time()
    # print("Time used:", end-start)
    if isprint:
        report_fit(out)
    if show:
        spec_fit = template.get_spectrum(out.params, wave)
        lower = flux - err
        upper = flux + err
        plt.figure()
        plt.fill_between(wave, lower, upper, alpha=0.3, color='grey')
        plt.plot(wave, flux)
        plt.plot(wave, spec_fit)
        plt.show()
    return out


def fit2(template, spec1, spec2, mask=None, params=None, isshow=False,
         isprint=False):
    """
    Some spectrum are observed by multichannel instrument. So there are
    multiple spectral files corresponding to one observation. This function try
    to fit the two part of the spectrum simultaneously using 2 scale curves,
    1 shift, 2 sigma pars. the mask window format should be like this
    [[l1, r1], [l2, r2], ...]
    """
    temp1 = template
    fname = template.filename
    temp2 = Model(fname)
    pars = Parameters()
    ascale_valst = [3.12, 0.284, -0.023, 0.2, -0.0086, 0.127]
    ascalepar = set_pars(pars, prefix='a_scale', order=5,
                         valuelst=ascale_valst)
    bscale_valst = [3.9, 0.16, -0.019, -0.03, 0.055, -0.02]
    bscalepar = set_pars(pars, prefix='b_scale', order=5,
                         valuelst=bscale_valst)
    asigmapar = set_pars(pars, prefix='a_sigma', order=[1], valuelst=[1.0e-4],
                         minlst=[1.0e-8])
    bsigmapar = set_pars(pars, prefix='b_sigma', order=[1], valuelst=[1.0e-4],
                         minlst=[1.0e-8])
    shiftpar = set_pars(pars, prefix='shift', order=[1], valuelst=[-2.7713e-04])
    temp1.set_lmpar_name(ascalepar, asigmapar, shiftpar)
    temp2.set_lmpar_name(bscalepar, bsigmapar, shiftpar)
    temp1.reset_wave_zoom(spec1.wave)
    temp2.reset_wave_zoom(spec2.wave)
    wave = np.append(spec1.wave, spec2.wave)
    flux = np.append(spec1.flux_unit, spec2.flux_unit)
    err = np.append(spec1.err_unit, spec2.err_unit)
    arg_mask = func.mask(wave, mask)

    def residual(pars, x, data, eps=None):
        residual1 = temp1.residual(pars, spec1.wave, spec1.flux_unit, eps=spec1.err_unit)
        residual2 = temp2.residual(pars, spec2.wave, spec2.flux_unit, eps=spec2.err_unit)
        all_res = np.append(residual1, residual2)
        all_res[arg_mask] = 1.0
        return all_res
    out = minimize(residual, pars, args=(wave, flux, err))
    report_fit(out)
    scale1_par = temp1.get_scale_par(out.params)
    scale1 = temp1.get_scale(spec1.wave, scale1_par)
    scale2_par = temp2.get_scale_par(out.params)
    scale2 = temp2.get_scale(spec2.wave, scale2_par)
    shift_par = temp1.get_shift_par(out.params)
    temp_new_wave = temp1.get_wave(shift_par)
    plt.plot(temp_new_wave, temp1.flux)
    plt.plot(spec1.wave, spec1.flux_unit / scale1)
    plt.plot(spec2.wave, spec2.flux_unit / scale2)
    # plt.plot(wave[arg_mask], flux[arg_mask], color='red')
    plt.show()
    return out


def test():
    tempname = 'data/templates/F0_+0.5_Dwarf.fits'
    templat = Model(tempname)
    spec1 = specio.Spectrum(objname1)
    spec2 = specio.Spectrum(objname2)
    maskwindow = [
        [6270.0, 6320.0],
        [6860.0, 6970.0],
        [7150.0, 7340.0]
    ]
    fit2(templat, spec1, spec2, mask=maskwindow)


def main():
    tmpname = 'data/templates/F0_+0.5_Dwarf.fits'
    model = Model(tmpname)

    # ftargetname = 'data/spec-4961-55719-0378.fits'
    ftargetname = '/home/zzx/workspace/data/xiamen/P200-Hale_spec/blue/reduce_second/specdir/fawftbblue0073.fits'
    # new_wo, new_fo, new_eo = specio.read_sdss(ftargetname, lw=3660, rw=10170)
    new_wo, new_fo, new_eo = specio.read_iraf(ftargetname)
    unit = func.get_unit(new_fo)
    new_fo = new_fo / unit
    new_eo = new_eo / unit
    # new_wo, new_fo, new_eo = specio.read_iraf(targetname)
    result = fit(model, new_wo, new_fo, new_eo, show=True, isprint=True)

    residual = model.residual

    out_parms = result.params
    # spec_fit = model.get_spectrum(out_parms, new_wo)
    scalepar = model.get_scale_par(out_parms)
    parshift = model.get_shift_par(out_parms)
    print(parshift[1])
    velocity = str(parshift[1]*c.to('km/s'))
    print('shift = '+velocity)
    myscale = model.get_scale(new_wo, scalepar)

    plt.figure()
    plt.plot(new_wo, myscale)
    plt.show()

    emcee_params = result.params.copy()
    # emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(1.0e-20),
    #                  max=np.log(1.0e20))
    result_emcee = minimize(residual, params=emcee_params,
                            args=(new_wo, new_fo, new_eo), method='emcee',
                            nan_policy='omit', steps=1000)
    report_fit(result_emcee)
    plt.plot(new_wo, new_fo)
    # plt.plot(new_wo, spec_fit)
    spec_emcee = model.get_spectrum(result_emcee.params, new_wo)
    plt.plot(new_wo, spec_emcee)
    corner.corner(result_emcee.flatchain, labels=result_emcee.var_names,
                  truths=list(result_emcee.params.valuesdict().values()))
    plt.show()


if __name__ == "__main__":
    # main()
    test()
