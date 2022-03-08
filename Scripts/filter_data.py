from mne.filter import filter_data


def filt_data(raw, hp, lp):

    raw_filtered = raw.copy().filter(l_freq=hp, h_freq=lp)

    return raw_filtered
