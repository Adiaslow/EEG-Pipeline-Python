import os.path as op

import mne
from mne.datasets import fetch_fsaverage


def forward_solution(raw):
    fs_dir = fetch_fsaverage(verbose=True)

    trans = 'fsaverage'
    src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

    """
    mne.viz.plot_alignment(
        raw.info, src=src, eeg=['original', 'projected'], trans=trans,
        show_axes=True, mri_fiducials=True, dig='fiducials')
    """

    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)

    return fwd, bem, src, trans
