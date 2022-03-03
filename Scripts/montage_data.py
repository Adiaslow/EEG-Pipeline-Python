from mne.channels import compute_native_head_t, read_custom_montage, make_standard_montage


def montage_data(raw, fname_mon, ch_names):

    dig_montage = read_custom_montage(fname_mon, head_size=0.095, coord_frame="head")
    #             read_custom_montage(fname, head_size=0.095, coord_frame=None)

    raw.set_montage(dig_montage)
    #   set_montage(montage, match_case=True, match_alias=False, on_missing='raise', verbose=None)

    std_montage = make_standard_montage('standard_1005')
    #             make_standard_montage(kind, head_size='auto')

    # raw.set_montage(std_montage)

    trans = compute_native_head_t(dig_montage)

    return dig_montage, trans

def name_channels(raw, ch_names):

    mapping = {}
    for i in range(0, raw.info.get('nchan'), 1):
        mapping[str(i)] = ch_names[i]

    raw.rename_channels(mapping, allow_duplicates=False, verbose=None)

    return raw