import import_data as imda
import montage_data as moda
import reference_data as reda
import vis_data as vida
import filter_data as fida
import forward_solution as foso
import covarriance_matrix as coma
import inverse_solution as inso


if __name__ == '__main__':

    raw = imda.import_data('../MUAD06022021EOFull.xdf')

    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4',
                'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz',
                'Pz']

    raw = moda.name_channels(raw, ch_names)

    montage = moda.montage_data(raw, '../standard_1005.elc')

    raw = reda.reference_data(raw, 'average')

    raw_filtered = fida.filt_data(raw, 0.5, 70) # FIXME

    fwd, bem, src, trans = foso.forward_solution(raw_filtered)

    labels_vol = ['Left-Amygdala',
                  'Left-Thalamus-Proper',
                  'Left-Cerebellum-Cortex',
                  'Brain-Stem',
                  'Right-Amygdala',
                  'Right-Thalamus-Proper',
                  'Right-Cerebellum-Cortex']

    cov = coma.covariance_matrix(raw)

    stc, src = inso.inverse_solution(raw, fwd, cov, 'sLORETA')

    print(raw_filtered)
    print(raw_filtered.info)

    vida.psd(raw_filtered)
    vida.psd_avg(raw_filtered)
    vida.tracing(raw_filtered)

