import os
import pandas as pd
from bids import BIDSLayout

import boldsignals as bold
import process_connectome as pc
import lib

import importlib

importlib.reload(lib)
importlib.reload(bold)
importlib.reload(pc)

pd.options.mode.chained_assignment = None

# %%
# Lecture du format BIDS
dataset_name = 'lightduo_sample'
derivatives_path = os.path.abspath(r'datasets_sample\lightduo-preproc-fmriprep')
layout = BIDSLayout(derivatives_path, index_metadata=True, reset_database=False, validate=False,
                    config=["bids", "derivatives"])

atlas_name = 'CAB-NP'
atlas_path = os.path.abspath(r'..\..\atlas\CAB-NP_volumetric\CAB-NP_volumetric_liberal.nii.gz')

confounds_strategies = 'compcor'
# %%

subjects_id_list = layout.get_subjects()
directory = 'timeSeries_files'
sub_dir = os.path.join(directory, dataset_name, atlas_name)
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)

bold_signals_config = pd.DataFrame(columns=subjects_id_list, index=['path', 'nlevels'])
info_csv_path = os.path.join(sub_dir, 'bold_signals_config.csv')


# %%
for subj_id in subjects_id_list:
    runs_bidsfiles_per_subjects = layout.get(subject=subj_id, suffix='bold', scope='derivatives', extension='nii.gz')
    runs_bidsfiles_ids_list_per_subject = []
    bold_signals_list_per_subject = []
    tr_runs_per_subject = []

    for run_bidsfile in runs_bidsfiles_per_subjects:
        run_bidsfile_ids = bold.get_keys_of_interest(run_bidsfile)
        tr_per_run = bold.get_tr_per_run(run_bidsfile)

        run_nifti_path = run_bidsfile.path
        bold_signals_confounds = bold.get_bold_signals_confounds(run_nifti_path, confounds_strategies)

        bold_signal = bold.calculate_bold_signals(run_nifti_path, atlas_path)
        bold_signal_cleaned = bold.clean_bold_signals(bold_signal, bold_signals_confounds, tr_per_run)

        runs_bidsfiles_ids_list_per_subject.append(run_bidsfile_ids)
        bold_signals_list_per_subject.append(str(bold_signal_cleaned))
        tr_runs_per_subject.append(tr_per_run)

    bold_signals_per_subject_df_output = bold.store_bold_signals_in_df(bold_signals_list_per_subject, tr_runs_per_subject,
                                                                  runs_bidsfiles_ids_list_per_subject,
                                                                  confounds_strategies)

    csv_output_path = os.path.join(sub_dir, 'id_boldsignals.csv'.replace('id', subj_id))
    bold.save_bold_signals_output(bold_signals_per_subject_df_output, csv_output_path)

    bold_signals_config.loc['path', subj_id] = csv_output_path
    bold_signals_config.loc['nlevels', subj_id] = bold_signals_per_subject_df_output.columns.nlevels

bold_signals_config.to_csv(info_csv_path, header=True)

