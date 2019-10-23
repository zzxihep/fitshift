import functools
import numpy as np

inf = float('inf')

def select(lw=-inf, uw=inf):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            if len(data) == 3:
                wave, flux, err = np.array(data)
                arg = np.where((wave > lw) & (wave < uw))
                return wave[arg], flux[arg], err[arg]
            else:
                wave, flux = data
                arg = np.where((wave > lw) & (wave < uw))
                return wave[arg], flux[arg]
        return wrapper
    return decorator


def read_data(fname):
    wave = np.arange(3000, 9000)
    return wave, wave, wave

@select(4000, 8000)
def read_data2(fname):
    wave = np.arange(3000, 9000)
    return wave, wave, wave


def main():
    wave1, flux1, err1 = read_data('abc')
    wave2, flux2, err2 = read_data2('abc')
    print(len(wave1))
    print(len(wave2))


if __name__ == '__main__':
    main()