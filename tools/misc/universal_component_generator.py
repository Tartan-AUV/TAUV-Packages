import math
import csv

VALUE_SERIES = {
    "E96": [
        1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12, 4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76
    ],
    "E48": [
        1.00, 1.05, 1.10, 1.15, 1.21, 1.27, 1.33, 1.40, 1.47, 1.54, 1.62, 1.69, 1.78, 1.87, 1.96, 2.05, 2.15, 2.26, 2.37, 2.49, 2.61, 2.74, 2.87, 3.01, 3.16, 3.32, 3.48, 3.65, 3.83, 4.02, 4.22, 4.42, 4.64, 4.87, 5.11, 5.36, 5.62, 5.90, 6.19, 6.49, 6.81, 7.15, 7.50, 7.87, 8.25, 8.66, 9.09, 9.53
    ],
    "E24": [
        1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1
    ]
}

SUFFIXES = {
    0: None, 1: 'k', 2: 'M', 3: 'G', 4: 'B', 5: 'T',
    -1: 'm', -2: 'u', -3: 'n', -4: 'p', -5: 'f',
}

def generate_parts(series, scaling, sigfigs, zero_suffix, name_fn, mfg_pn_fn=None):
    parts = []
    for value in series:
        value *= scaling
        power = int(math.floor(math.log10(value)))
        suffix = SUFFIXES[power // 3]
        if suffix is None:
            suffix = zero_suffix

        power3 = power // 3 * 3
        factor = 10 ** power3
        value = value / factor

        value_str = format_value_end_suffix(value, suffix, sigfigs)

        name = name_fn(value, suffix)

        if mfg_pn_fn is not None:
            mfg_pn = mfg_pn_fn(value, suffix)
            parts.append(
                (name, value_str, mfg_pn)
            )
        else:
            parts.append(
                (name, value_str)
            )
    return parts


def split_float(value):
    if abs(value - round(value)) >= 1e-6:
        return divmod(value, 1)
    else:
        return round(value), 0

def format_value_end_suffix(value, suffix, sigfigs):
    integer, decimal = split_float(value)
    integer = int(integer)
    decimal_l = sigfigs - len(str(integer))
    if abs(decimal - 0) > 1e-6:
        value_str = f"{value:.{decimal_l}f}{suffix}"
    else:
        value_str = f"{int(value)}{suffix}"
    return value_str


def format_value_separator_suffix(value, suffix, sigfigs):
    integer, decimal = split_float(value)
    integer = int(integer)
    decimal_l = sigfigs - len(str(integer))
    value_str = f"{value:.{decimal_l}f}".replace('.', suffix)
    if decimal_l == 0:
        value_str += suffix
    return value_str


if __name__ == "__main__":
    # generate E96 resistors with Vishay CRCW part numbers eg (4.7k: CRCW06034K70FKEA )
    name_fn = lambda v, s: f"RES-{format_value_end_suffix(v, s, 3)}-0603-1pct"
    mfg_pn_fn = lambda v, s: f"CRCW0603{format_value_separator_suffix(v, s, 3).upper()}FKEA"

    parts = []
    for i in range(0, 6):
        parts.extend(generate_parts(VALUE_SERIES['E48'], 10**i, 3, 'R', name_fn, mfg_pn_fn))


    with open('e48_resistors.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(parts)

    # generate capacitors (no supplier links)
    parts = []
    for temp_range, case, value_range, max_voltage in [('X7R', '0603', (-12, -6), 25),
                                                          ('X7R', '0805', (-6, -5), 25),
                                                          ('X7R', '1206', (-6, -4), 25),
                                                          ('C0G', '0603', (-12, -6), 16)]:
        name_fn = lambda v, s: f"CAP-{format_value_end_suffix(v, s, 2)}-{case}-{temp_range}-{max_voltage}"
        for i in range(*value_range):
            parts.extend(generate_parts(VALUE_SERIES['E24'], 10**i, 2, 'C', name_fn))

    with open('e48_caps.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(parts)
