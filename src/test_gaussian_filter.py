# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import convol


def main():
    x = np.arange(4000, 8000, 1.0)
    y = np.zeros(x.shape, dtype=np.float64)
    y[300] = 1.0
    y[0] = 1.0
    y[-1] = 0.5
    ny = convol.gauss_filter(x, y, [20.0])
    plt.plot(x, y)
    plt.plot(x, ny)
    plt.show()


if __name__ == '__main__':
    main()

