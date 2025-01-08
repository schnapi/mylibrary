import numpy as np
from scipy.fftpack import fft, ifft
from scipy.ndimage.interpolation import shift

from typing import Optional, Tuple, Union

def get_max(data: Union[list, np.ndarray]) -> Tuple[Union[float, int], int]:
    if isinstance(data, list):
        data = np.array(data)
    maxind = np.nanargmax(data)
    return data[maxind], maxind

def crosscorr(x: np.ndarray, y: np.ndarray, max_shift: Optional[int], scale: bool = False) -> np.ndarray:
    # https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
    if max_shift is None or max_shift > max(len(x), len(y)) - 1:
        max_shift = max(len(x), len(y)) - 1
    m = max(len(x), len(y))
    max_shift1 = min(max_shift, m - 1)  # Result is clipped based on max_shift
    fft_length = find_transform_length(m)
    xfft = fft(x, fft_length)  # this will add zeros if len(y) < m2
    yfft = fft(y, fft_length)
    resifft = ifft(np.multiply(xfft, np.conj(yfft)))
    # Keep only the lags we want and move negative lags before positive lags.
    rescrosscorr = np.array((*resifft[fft_length - max_shift1 + np.arange(max_shift1)], *resifft[:max_shift1 + 1]))
    if scale:
        raise NotImplementedError()
    return rescrosscorr


def find_transform_length(m: int) -> int:
    m = 2 * m
    while True:
        r = float(m)
        for p in [2, 3, 5, 7]:
            while r > 1 and r % p == 0:
                r = r / p
        if r == 1:
            break
        m = m + 1
    return m


def finddelay(x: np.ndarray, y: np.ndarray, method="fft") -> Tuple[int, float]:
    max_shift = max(len(x), len(y)) - 1
    index_max, max_c = 0, 0.
    sumxx = sum(x**2)
    sumyy = sum(y**2)
    if sumxx == 0 or sumyy == 0:  # this is a special case if all x or y are 0
        corr_normalized = np.zeros((2 * max_shift + 1, 1))
    else:
        corr_normalized = abs(crosscorr(x, y, max_shift)) / np.sqrt(sumxx * sumyy)  # for normalization
    # find max lags on right side, negative delays
    max_right_lag, index_max_right = get_max(corr_normalized[max_shift:])
    # find max lags on left side, positive delays
    max_left_lag, index_max_left = get_max(np.flipud(corr_normalized[:max_shift]))
    if max_left_lag == 0:  # if zero
        index_max = max_shift + index_max_right + 1
    else:
        if max_right_lag > max_left_lag:  # The estimated lag is positive or zero.
            index_max = max_shift + index_max_right + 1
            max_c = max_right_lag
        elif max_right_lag < max_left_lag:  # The estimated lag is negative.
            index_max = max_shift - index_max_left
            max_c = max_left_lag
        elif max_right_lag == max_left_lag:
            if index_max_right <= index_max_left:  # The estimated lag is positive or zero.
                index_max = max_shift + index_max_right + 1
                max_c = max_right_lag
            else:  # The estimated lag is negative.
                index_max = max_shift - index_max_left
                max_c = max_left_lag
    delay: int = max_shift + 1 - index_max
    if max_c < 1e-8:
        # if max_shift != 0:
        #     raise ValueError("No significant correlation.")
        max_c = -1
        delay = 0
    return delay, max_c


def alignsignal(x: np.ndarray, y: np.ndarray):
    delay, corr = finddelay(x, y)
    # wrong implementation, you need to pad zeros or last value...
    return shift(x, delay, mode='nearest'), delay
