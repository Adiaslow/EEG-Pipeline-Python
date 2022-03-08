from mne.minimum_norm import make_inverse_operator, apply_inverse


def inverse_solution(raw, fwd, noise_cov, inv_method):

    snr = 3.0  # use smaller SNR for raw data
    # inv_method = 'sLORETA'  # sLORETA, MNE, dSPM
    parc = 'aparc'  # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'
    loose = dict(surface=0.2, volume=1.)

    lambda2 = 1.0 / snr ** 2

    inverse_operator = make_inverse_operator(
        raw.info, fwd, noise_cov, depth=None, loose=loose, verbose=True)
    del fwd

    stc = apply_inverse(raw, inverse_operator, lambda2, inv_method,
                        pick_ori=None)

    src = inverse_operator['src']

    return stc, src
