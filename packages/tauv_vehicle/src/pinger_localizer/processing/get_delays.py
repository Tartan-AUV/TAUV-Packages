import numpy as np
import scipy as sp
from math import ceil, log, pi, floor
import matplotlib.pyplot as plt

def get_delays_fft(times, samples, sample_frequency, max_delay, interp=1):
    n_samples = samples.shape[1]
    n_fft = 2 ** ceil(log(2 * n_samples - 1) / log(2))

    sample_period = 1 / sample_frequency

    ffts = sp.fft.rfft(samples, n=n_fft, axis=1)

    cross_spectrums = ffts[0] * np.conj(ffts[1:])
    norm_cross_spectrums = np.exp(1j * np.angle(cross_spectrums))
    cross_correlations = np.real(sp.fft.irfft(norm_cross_spectrums, n=n_fft * interp, axis=1))

    max_delay_samples = min(int(interp * max_delay / sample_period), int(interp * n_fft / 2))

    valid_cross_correlations = np.abs(np.concatenate((
        cross_correlations[:, -max_delay_samples:],
        cross_correlations[:, :max_delay_samples+1]
    ), axis=1))

    delays_samples = np.argmax(valid_cross_correlations, axis=1) - max_delay_samples
    delays = delays_samples / float(interp / sample_period)

    return delays

def get_delays_fft_angle(times, samples, sample_frequency, ping_frequency):
    interp = 4

    duration = (times[0, -1] - times[0, 0])
    cropped_duration = duration - (duration % (1 / ping_frequency))
    n_samples = floor(cropped_duration * sample_frequency)

    print(n_samples)

    n_fft = 2 ** ceil(log(2 * n_samples - 1) / log(2)) * interp

    ffts = sp.fft.rfft(samples[:n_samples], n=n_fft, axis=1)
    fft_frequencies = sp.fft.rfftfreq(n_fft, 1 / sample_frequency)

    fft_bin_i = np.searchsorted(fft_frequencies[:n_fft//2], ping_frequency)

    print(fft_frequencies[fft_bin_i])

    delays = np.array([
        np.angle(ffts[1, fft_bin_i] / ffts[0, fft_bin_i]),
        np.angle(ffts[2, fft_bin_i] / ffts[0, fft_bin_i]),
        np.angle(ffts[3, fft_bin_i] / ffts[0, fft_bin_i]),
    ])

    delays = delays / ping_frequency

    # delays = (delays + pi) % (2 * pi) - pi

    return delays


def get_delays_xcorr(times, samples, sample_frequency, max_delay):
    sample_period = 1 / sample_frequency

    cross_correlations = np.array([
        sp.signal.correlate(samples[0], samples[1]),
        sp.signal.correlate(samples[0], samples[2]),
        sp.signal.correlate(samples[0], samples[3]),
    ])

    max_delay_samples = ceil(max_delay / sample_period)

    valid_cross_correlations = cross_correlations[:, (cross_correlations.shape[1] // 2) - max_delay_samples:(cross_correlations.shape[1] // 2) + max_delay_samples]

    # plt.figure()
    # plt.plot(valid_cross_correlations[0])
    # plt.plot(valid_cross_correlations[1])
    # plt.plot(valid_cross_correlations[2])

    delays_samples = -np.argmax(valid_cross_correlations, axis=1) + max_delay_samples
    delays = delays_samples * sample_period

    return delays
