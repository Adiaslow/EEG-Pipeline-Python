import pyxdf
import mne
from mne.io import RawArray


def import_data(fname_raw):

    streams, header = pyxdf.load_xdf(fname_raw)

    data = streams[0]["time_series"].T

    assert data.shape[0] == 19

    info = mne.create_info(19, 256, "eeg")

    raw = RawArray(data, info)

    return raw
