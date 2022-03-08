from mne import compute_raw_covariance


def covariance_matrix(raw):

    cov = compute_raw_covariance(raw,scalings='auto')
    # compute_raw_covariance(raw, tmin=0, tmax=None, tstep=0.2, reject=None, flat=None, picks=None, method='empirical',
    #                        method_params=None, cv=3, scalings=None, n_jobs=1, return_estimators=False,
    #                        reject_by_annotation=True, rank=None, verbose=None)

    cov.plot(raw.info, proj=True, block=True)

    return cov
