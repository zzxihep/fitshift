#!/usr/bin/env python

import os
# import time
import glob
import math
import numpy as np
from astropy.io import fits
from astropy.constants import c
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from lmfit.printfuncs import report_fit
import corner
import rebin
import convol
import specio
import func


class Model(specio.Spectrum):
    def __init__(self, tmpname, hduid=None):
        super(Model, self).__init__(tmpname, hduid)
        self.wshift = -(self.wave[0]+self.wave[-1])/2
        self.wscale = 1.99/(self.wave[-1]-self.wave[0])

    def reset_zoom(self, wave):
        self.wshift = -(wave[0]+wave[-1])/2
        self.wscale = 1.99/(wave[-1]-wave[0])

    def trans_wave(self, wave):
        return (wave + self.wshift) * self.wscale

    def get_scale(self, wave, par):
        tmpwave = self.trans_wave(wave)
        return np.array(convol.poly(tmpwave, par))

    def get_legendre_scale(self, wave, par):
        tmpwave = self.trans_wave(wave)
        if tmpwave[0] > -1 and tmpwave[-1] < 1:
            return np.array(convol.legendre_poly(tmpwave, par))
        arg = np.where((tmpwave>-1)&(tmpwave<1))
        sewave = tmpwave[arg]
        inscale = np.array(convol.legendre_poly(sewave, par))
        outscale = np.zeros(len(tmpwave))
        outscale[arg] = inscale
        return outscale

    def convol_spectrum(self, par):
        # tmpwave = self.trans_wave(wave)
        if len(par) == 1 and par[0] == 0.0:
            return self.flux_unit
        return np.array(convol.gauss_filter(self.wave, self.flux_unit, par))

    def get_wave(self, parwave):
        return np.array(convol.map_wave(self.wave, parwave))

    def get_spectrum_pre(self, wave, arrshift, arrsigma, arrscale):
        arrsigma = np.abs(arrsigma)
        new_wave = self.get_wave(arrshift)
        new_flux = self.convol_spectrum(arrsigma)
        # scale = self.get_scale(new_wave, arrscale)
        scale = self.get_legendre_scale(wave, arrscale)
        flux_rebin = np.array(rebin.rebin(new_wave, new_flux, wave))
        flux_aftscale = flux_rebin * scale
        return flux_aftscale

    def set_lmpar_name(self, parscale, parsigma=None, parshift=None):
        self.lmscale_name = parscale
        self.lmsigma_name = parsigma
        self.lmshift_name = parshift

    def get_scale_par(self, pars):
        if self.lmscale_name is None:
            return [0.0]
        return read_lmpar(pars, self.lmscale_name)

    def get_sigma_par(self, pars):
        if self.lmsigma_name is None:
            return [0.0]
        return read_lmpar(pars, self.lmsigma_name)

    def get_shift_par(self, pars):
        if self.lmshift_name is None:
            return [0.0]
        return read_lmpar(pars, self.lmshift_name)

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
        scalevalst = [1.0, -1.0, -1.0, 0.22, -0.1, -0.13]
        scaleparname = set_pars(params, 'scale', 5, valuelst=scalevalst)
        template.set_lmpar_name(scaleparname, None, shiftparname)
    # start = time.process_time()
    # print(start)

    def residual(pars, x, data, eps):
        flux_fit1 = template.get_spectrum(pars, x)
        arrpar = read_lmpar(pars, sigmaparname)
        flux_fit2 = np.array(convol.gauss_filter(x, data, arrpar))
        return (flux_fit1 - flux_fit2)/eps

    # out = minimize(template.residual, params, args=(wave, flux, err),
    #                method='leastsq')
    # end = time.process_time()
    # print("Time used:", end-start)
    out = minimize(residual, params, args=(wave, flux, err),
                   method='leastsq')
    if isprint:
        report_fit(out)
    shift = out.params['shift1'].value * c
    shifterr = out.params['shift1'].stderr * c
    print('-'*20+' velocity '+'-'*20)
    print(shift.to('km/s'))
    print(shifterr.to('km/s'))
    if show:

        plt.figure()

        arrpar = read_lmpar(out.params, sigmaparname)
        flux_fit2 = np.array(convol.gauss_filter(wave, flux, arrpar))
        plt.plot(wave, flux_fit2)

        spec_fit = template.get_spectrum(out.params, wave)
        lower = flux_fit2 - err
        upper = flux_fit2 + err

        plt.fill_between(wave, lower, upper, alpha=0.3, color='grey')

        plt.plot(wave, spec_fit)
        plt.show()
    return out


def read_lmpar(pars, dic_parnames):
    arrpar = np.zeros(max(dic_parnames)+1, dtype=np.float64)
    for key in dic_parnames:
        parname = dic_parnames[key]
        arrpar[key] = pars[parname].value
    return arrpar


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
    temp1.reset_zoom(spec1.wave)
    temp2.reset_zoom(spec2.wave)
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
    temp1.set_lmpar_name(ascalepar, None, shiftpar)
    temp2.set_lmpar_name(bscalepar, None, shiftpar)
    wave = np.append(spec1.wave, spec2.wave)
    flux = np.append(spec1.flux_unit, spec2.flux_unit)
    err = np.append(spec1.err_unit, spec2.err_unit)
    arg_mask = func.mask(wave, mask)

    def residual(pars, x, data, eps=None):
        # print('Flag 1')
        # print(asigmapar)
        arrsigma1 = read_lmpar(pars, dic_parnames=asigmapar)
        arrsigma2 = read_lmpar(pars, dic_parnames=bsigmapar)
        flux_unit1 = np.array(convol.gauss_filter(spec1.wave, spec1.flux_unit, arrsigma1))
        flux_unit2 = np.array(convol.gauss_filter(spec2.wave, spec2.flux_unit, arrsigma2))
        # print(len(spec2.wave))
        # print(len(flux_unit1))
        # print(len(spec2.err_unit))
        residual1 = temp1.residual(pars, spec1.wave, flux_unit1, eps=spec1.err_unit)
        residual2 = temp2.residual(pars, spec2.wave, flux_unit2, eps=spec2.err_unit)
        all_res = np.append(residual1, residual2)
        all_res[arg_mask] = 1.0
        return all_res
    out = minimize(residual, pars, args=(wave, flux, err))
    # report_fit(out)
    scale1_par = temp1.get_scale_par(out.params)
    scale1 = temp1.get_scale(spec1.wave, scale1_par)
    scale2_par = temp2.get_scale_par(out.params)
    scale2 = temp2.get_scale(spec2.wave, scale2_par)
    shift_par = temp1.get_shift_par(out.params)
    temp_new_wave = temp1.get_wave(shift_par)
    plt.plot(temp_new_wave, temp1.flux)
    arrsigma1 = read_lmpar(out.params, dic_parnames=asigmapar)
    arrsigma2 = read_lmpar(out.params, dic_parnames=bsigmapar)
    flux_unit1 = convol.gauss_filter(spec1.wave, spec1.flux_unit, arrsigma1)
    flux_unit2 = convol.gauss_filter(spec2.wave, spec2.flux_unit, arrsigma2)
    plt.plot(spec1.wave, flux_unit1 / scale1)
    plt.plot(spec2.wave, flux_unit2 / scale2)
    # plt.plot(wave[arg_mask], flux[arg_mask], color='red')
    shift = out.params['shift1'].value * c
    shifterr = out.params['shift1'].stderr * c
    bluename = os.path.basename(spec1.filename)
    redname = os.path.basename(spec2.filename)
    print('-'*20+' velocity '+'-'*20)
    print(bluename + '  '+redname)
    print(shift.to('km/s'))
    print(shifterr.to('km/s'))
    plt.show()
    return out


def test():
    tempname = 'data/templates/F0_+0.5_Dwarf.fits'
    objname1 = '/home/zzx/workspace/data/xiamen/P200-Hale_spec/blue/reduce_second/specdir/fawftbblue0073.fits'
    objname2 = '/home/zzx/workspace/data/xiamen/P200-Hale_spec/red/reduced_second/specdir/fawftbred0073.fits'
    templat = Model(tempname)
    spec1 = specio.Spectrum(objname1)
    spec2 = specio.Spectrum(objname2)
    maskwindow = [
        [6270.0, 6320.0],
        [6860.0, 6970.0],
        [7150.0, 7340.0]
    ]
    fit2(templat, spec1, spec2, mask=maskwindow)


def fit_lamost():
    bluelst, redlst = [], []
    namelst = glob.glob('/home/zzx/workspace/data/stellar_X/*.fits')
    namelst = ['/home/zzx/workspace/data/stellar_X/med-58409-TD045606N223435B01_sp16-102.fits']
    for name in namelst:
        # fig1 = plt.figure()
        # fig2 = plt.figure()
        # ax1 = fig1.add_subplot(111)
        # ax2 = fig2.add_subplot(111)
        print(name)
        size = len(fits.open(name))
        for ind in range(3, size):
            spec = specio.Spectrum(name, ind)
            spec.clean_cosmic_ray()
            if 'B' in spec.header['EXTNAME']:
                bluelst.append(spec)
            else:
                redlst.append(spec)
    name = namelst[0]
    model_blue = Model(name, 3)
    model_blue.clean_cosmic_ray()
    model_red = Model(name, 11)
    model_red.clean_cosmic_ray()
    params = Parameters()
    shiftparname = set_pars(params, 'shift', [1], valuelst=[0.0])
    scalevalst = [0.99608100, -0.00931768, 0.00319284, 5.5658e-04, -4.4060e-04, 0.0]
    bscaleparname = set_pars(params, 'b_scale', 5, valuelst=scalevalst)
    rscaleparname = set_pars(params, 'r_scale', 5, valuelst=scalevalst)
    bsimgapar = set_pars(params, 'b_sigma', [1], valuelst=[0.0004])
    rsigmapar = set_pars(params, 'r_sigma', [1], valuelst=[0.0004])
    model_blue.set_lmpar_name(bscaleparname, bsimgapar, shiftparname)
    model_red.set_lmpar_name(rscaleparname, rsigmapar, shiftparname)
    shiftlst, shifterrlst = [], []
    
    def residual(pars, x1, data1, eps1, x2, data2, eps2):
        res1 = model_blue.residual(pars, x1, data1, eps1)
        res2 = model_red.residual(pars, x2, data2, eps2)
        return np.append(res2, res1)
        return res2
    
    for ind in range(len(redlst)):
        bspec = bluelst[ind]
        # bspec = redlst[ind]
        rspec = redlst[ind]
        bnw, bnf, bne = func.select(bspec.wave, bspec.flux_unit, bspec.err_unit, lw=4920, uw=5300)
        # bnw, bnf, bne = func.select(bspec.wave, bspec.flux_unit, bspec.err_unit, lw=6320, uw=6860)
        rnw, rnf, rne = func.select(rspec.wave, rspec.flux_unit, rspec.err_unit, lw=6320, uw=6860)
        bfakeerr = np.ones(len(bnw), dtype=np.float64)*0.01
        rfakeerr = np.ones(len(rnw), dtype=np.float64)*0.01
        bne = bfakeerr
        rne = rfakeerr
        # out = minimize(model_blue.residual, params, args=(nw, nf))
        # out = minimize(model_red.residual, params, args=(nw, nf))
        out = minimize(residual, params, args=(bnw, bnf, bne, rnw, rnf, rne))
        report_fit(out)
        shiftlst.append(out.params['shift1'].value*c)
        shifterrlst.append(out.params['shift1'].stderr*c)

        plt.figure()

        spec_fit_blue = model_blue.get_spectrum(out.params, model_blue.wave)
        spec_fit_red = model_red.get_spectrum(out.params, model_red.wave)

        plt.plot(bnw, bnf)
        plt.plot(model_blue.wave, spec_fit_blue)

        plt.figure()
        plt.plot(rnw, rnf)
        plt.plot(model_red.wave, spec_fit_red)
        plt.show()

    for ind, value in enumerate(shiftlst):
        # print(value.to('km/s'))
        print(value.to('km/s'), shifterrlst[ind].to('km/s'))



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
    # test()
    fit_lamost()
