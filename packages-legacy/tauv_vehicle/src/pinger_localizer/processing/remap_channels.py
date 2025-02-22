import numpy as np

def remap_channels(mappings, sample_times, samples):
    remapped_times = np.array([
        sample_times[mappings[0]],
        sample_times[mappings[1]],
        sample_times[mappings[2]],
        sample_times[mappings[3]],
    ])

    remapped_samples = np.array([
        samples[mappings[0]],
        samples[mappings[1]],
        samples[mappings[2]],
        samples[mappings[3]],
    ])

    return remapped_times, remapped_samples