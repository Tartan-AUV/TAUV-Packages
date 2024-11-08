import numpy as np
import scipy as sp

def get_ping_frequency(times, samples, sample_frequency, min_ping_frequency, max_ping_frequency):
    sample_period = 1 / sample_frequency

    samples_fft = np.abs(sp.fft.rfft(samples[0]))
    samples_fft_freqs = sp.fft.rfftfreq(samples.shape[1], sample_period)

    ping_frequency = samples_fft_freqs[
        np.argmax(np.where((min_ping_frequency <= samples_fft_freqs) & (samples_fft_freqs <= max_ping_frequency), samples_fft, 0))
    ]

    return int(ping_frequency)
