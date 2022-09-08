from nilearn import signal
from nilearn import input_data
from nilearn.interfaces import fmriprep

import numpy as np
import pandas as pd

def get_tr_per_run(run_bidsfile):
    metadata_per_run = run_bidsfile.get_metadata()
    tr_per_run = metadata_per_run['RepetitionTime']
    return tr_per_run


def get_keys_of_interest(run_bidsfile_per_subjects):
    image_caract = run_bidsfile_per_subjects.get_entities()
    if 'run' not in image_caract.keys():
        image_caract['run'] = 0
    keys_of_interest = [image_caract.get('task'), image_caract.get('session'), image_caract.get('run')]
    if None in keys_of_interest:
        keys_of_interest = list(filter(None, keys_of_interest))
    return tuple(keys_of_interest)


def calculate_bold_signals(nifti_path, atlas_path):
    atlas_masker = input_data.NiftiLabelsMasker(atlas_path, standardize=False)
    bold_signal = atlas_masker.fit_transform(nifti_path, confounds=None)
    return bold_signal.tolist()


def get_bold_signals_confounds(run_nifti_path, confounds_strategies):
    bold_signals_confounds_output = fmriprep.load_confounds_strategy(run_nifti_path,
                                                                     denoise_strategy=confounds_strategies)[0]
    return bold_signals_confounds_output


def clean_bold_signals(bold_signal, bold_signal_confounds, tr_per_run):
    time_series_cleaned = signal.clean(bold_signal, confounds=bold_signal_confounds, detrend=False,
                                       standardize='zscore', t_r=tr_per_run)
    return time_series_cleaned.tolist()


def store_bold_signals_in_df(bold_signals_list_per_subject, tr_runs_per_subject, runs_bidsfiles_ids_list_per_subject,
                             confounds_strategies):
    bold_signals_df_data = [bold_signals_list_per_subject, tr_runs_per_subject]
    bold_signals_df_index = [confounds_strategies] + ['tr_per_run']
    bold_signals_columns = pd.MultiIndex.from_tuples(runs_bidsfiles_ids_list_per_subject)
    bold_signals_df_output = pd.DataFrame(data=bold_signals_df_data,
                                          index=bold_signals_df_index,
                                          columns=bold_signals_columns)
    return bold_signals_df_output


def save_bold_signals_output(bold_signals_per_subject_df_output, csv_output_path):
    bold_signals_per_subject_df_output = bold_signals_per_subject_df_output.replace(r'^\s*$', np.nan,
                                                                                    regex=True)  # to delete
    if bold_signals_per_subject_df_output.isnull().any().any():
        bold_signals_per_subject_df_output = bold_signals_per_subject_df_output.dropna(axis='columns')
    bold_signals_per_subject_df_output.to_csv(csv_output_path, header=True)
    return None



