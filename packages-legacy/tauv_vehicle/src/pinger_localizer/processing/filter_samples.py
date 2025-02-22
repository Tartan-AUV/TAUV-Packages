import numpy as np
import scipy as sp

def filter_samples(times, samples, sample_frequency, ping_frequency):
    sos = sp.signal.butter(2, np.array([max(ping_frequency - 1000, 1), min(ping_frequency + 1000, sample_frequency)]), 'bandpass', output='sos', fs=sample_frequency)
    filtered_samples = sp.signal.sosfilt(sos, samples, axis=1)

    cropped_times = times[:, :5000]
    cropped_samples = filtered_samples[:, :5000]

    normalized_samples = cropped_samples / np.max(np.abs(cropped_samples), axis=1)[..., None]

    return cropped_times, normalized_samples
