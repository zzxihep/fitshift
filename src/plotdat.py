#!/usr/bin/env python


import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    fname = sys.argv[1]
    data = np.loadtxt(fname)
    wave = data[:, 0]
    flux = data[:, 1]
    plt.plot(wave, flux)
    plt.show()

    
if __name__ == "__main__":
    main()