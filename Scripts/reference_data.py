def reference_data(raw, reference):
    raw.set_eeg_reference(ref_channels=reference, projection=False, ch_type='eeg', verbose=None)
    # set_eeg_reference(ref_channels='average', projection=False, ch_type='auto', forward=None, verbose=None)

    return raw
