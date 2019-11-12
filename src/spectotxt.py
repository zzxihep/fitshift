#!/usr/bin/env python

import os
import sys
import specio


def main():
    fname = sys.argv[1]
    dirname = os.path.dirname(fname)
    if dirname is '':
        dirname = '.'
    spec = specio.Spectrum(fname)
    outname = fname.replace('.fits', '') + '.txt'
    fulloutname = dirname + os.sep + outname
    text = 'write to file ' + fulloutname
    print(text)
    fil = open(fulloutname, 'w')
    for ind in range(len(spec.wave)):
        text = '%.4f  %.6e  %.6e' % (spec.wave[ind], spec.flux[ind], spec.err[ind])
        fil.write(text+'\n')
    fil.close()

if __name__ == "__main__":
    main()