def psd(raw):
    raw.plot_psd(fmin=0, fmax=50, estimate='power', dB=True, average=False, line_alpha=1, spatial_colors=True,
                     sphere=(0, -0.0075, 0, 0.12))  # sphere=(0, 0.015, 0, 0.085)
    #   plot_psd(fmin=0, fmax=inf, tmin=None, tmax=None, proj=False, n_fft=None, n_overlap=0, reject_by_annotation=True,
    #          picks=None, ax=None, color='black', xscale='linear', area_mode='std', area_alpha=0.33, dB=True,
    #          estimate='auto', show=True, n_jobs=1, average=False, line_alpha=None, spatial_colors=True, sphere=None,
    #          window='hamming', exclude='bads', verbose=None)


def psd_avg(raw):
    raw.plot_psd(fmin=0, fmax=50, estimate='power', dB=True, average=True)  # sphere=(0, 0.015, 0, 0.085)


def tracing(raw):
    raw.plot(duration=5, n_channels=30,scalings='auto', block=True)
    #   plot(events=None, duration=10.0, start=0.0, n_channels=20, bgcolor='w', color=None, bad_color='lightgray',
    #   event_color='cyan', scalings=None, remove_dc=True, order=None, show_options=False, title=None, show=True,
    #   block=False, highpass=None, lowpass=None, filtorder=4, clipping=1.5, show_first_samp=False, proj=True,
    #   group_by='type', butterfly=False, decim='auto', noise_cov=None, event_id=None, show_scrollbars=True,
    #   show_scalebars=True, time_format='float', precompute='auto', use_opengl=None, verbose=None)
