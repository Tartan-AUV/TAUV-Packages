import os
import random
from math import pi, ceil, log
import scipy as sp
import time

import numpy as np
import matplotlib.pyplot as plt
from processing.filter_samples import filter_samples
from processing.get_ping_frequency import get_ping_frequency
from processing.get_delays import get_delays_xcorr, get_delays_fft, get_delays_fft_angle
from processing.remap_channels import remap_channels
from processing.get_direction import get_direction

PLOT = True

min_ping_frequency = 10e3
max_ping_frequency = 50e3
# ping_frequency = 40e3

def process(times, samples):
    sample_period = np.average(np.diff(times[0]))
    sample_frequency = 1 / sample_period

    print(f'Sample frequency: {sample_frequency}')

    # ping_frequency = get_ping_frequency(times, samples, sample_frequency, min_ping_frequency, max_ping_frequency)
    ping_frequency = 20e3

    n_samples = samples.shape[1]

    # n_fft = 2 ** ceil(log(2 * n_samples - 1) / log(2))
    # ffts = sp.fft.rfft(samples, n=n_fft, axis=1)
    # fft_frequencies = sp.fft.rfftfreq(n_fft, d=sample_period)

    # if PLOT:
    #     _, fft_axs = plt.subplots(4)
    #     fft_axs[0].plot(fft_frequencies[:n_samples//2], (2 / n_samples) * np.abs(ffts[0, :n_samples//2]))
    #     fft_axs[1].plot(fft_frequencies[:n_samples//2], (2 / n_samples) * np.abs(ffts[1, :n_samples//2]))
    #     fft_axs[2].plot(fft_frequencies[:n_samples//2], (2 / n_samples) * np.abs(ffts[2, :n_samples//2]))
    #     fft_axs[3].plot(fft_frequencies[:n_samples//2], (2 / n_samples) * np.abs(ffts[3, :n_samples//2]))

    times_filt, samples_filt = filter_samples(times, samples, sample_frequency, ping_frequency)

    delays = get_delays_xcorr(times_filt, samples_filt, sample_frequency, 0.5 * (1 / ping_frequency))
    # delays = get_delays_fft_angle(times, samples, sample_frequency, ping_frequency)

    print('Delays:', delays)


    # if PLOT:
    #     _, filt_axs = plt.subplots(4)
    #     filt_axs[0].plot(times_filt[0], samples_filt[0])
    #     filt_axs[1].plot(times_filt[1], samples_filt[1])
    #     filt_axs[2].plot(times_filt[2], samples_filt[2])
    #     filt_axs[3].plot(times_filt[3], samples_filt[3])
    #
    #     offset = times_filt[0, 0] + 1 / ping_frequency
    #
    #     filt_axs[0].axvline(x=offset)
    #     filt_axs[1].axvline(x=offset + delays[0])
    #     filt_axs[2].axvline(x=offset + delays[1])
    #     filt_axs[3].axvline(x=offset + delays[2])
    #
    # if PLOT:
    #     plt.figure()
    #     plt.plot(times[0], samples[0])
    #     plt.plot(times[1] - delays[0], samples[1])
    #     plt.plot(times[2] - delays[1], samples[2])
    #     plt.plot(times[3] - delays[2], samples[3])

    return delays

def main():
    directory = '/Users/theo/Documents/tauv/data/pinger_localizer/'
    file_paths = [directory + file for file in os.listdir(directory) if os.path.isfile(directory + file)]
    id_length = 15
    file_ids = [os.path.split(path)[-1][:id_length] for path in file_paths if 'samples' in path]
    print(file_ids)
    # file_ids.sort(key=lambda date: date[:-6])
    file_ids = sorted(file_ids, key=lambda date: int(date[-6:]))

    n_files = 1000

    # selected_file_ids = random.choices(file_ids, k=n_files)
    selected_file_ids = file_ids[:n_files]

    accumulated_delays = np.zeros((n_files, 3))

    for i, file_id in enumerate(selected_file_ids):
        times_path = directory + file_id + '-times.npy'
        samples_path = directory + file_id + '-samples.npy'

        print(f'--- Processing {file_id} ({i + 1} of {n_files}) ---')

        times = np.load(times_path)
        samples = np.load(samples_path)

        if np.max(np.abs(samples)) > 1.0:
            continue

        delays = process(times, samples)

        accumulated_delays[i] = delays

        if PLOT:
            plt.show()

    if PLOT:
        _, delays_axs = plt.subplots(3)
        delays_axs[0].plot(accumulated_delays[:, 0])
        delays_axs[1].plot(accumulated_delays[:, 1])
        delays_axs[2].plot(accumulated_delays[:, 2])


    if PLOT:
        plt.show()


if __name__ == '__main__':
    main()
