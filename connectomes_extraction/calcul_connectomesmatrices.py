# %%

import os
import pandas as pd
import connectomesmeasures as connectomes
import importlib
import argparse

importlib.reload(connectomes)

pd.options.mode.chained_assignment = None

# # Create the parser
# parser = argparse.ArgumentParser()
# # Add arguments
# parser.add_argument('--atlas', type=str, required=True)
# parser.add_argument('--dataset', type=str, required=True)
# args = parser.parse_args()

atlas_name = 'CAB-NP'
atlas_csv_path = os.path.abspath(f'../../atlas/CAB-NP_volumetric/CAB-NP_labels.csv')

times_series_dirpath = 'timeSeries_files'
# participants_info_dirpath = 'participants_info'
dataset_name = 'hcptrt'
multi_tr = False
time_chunks_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
windows_numbers = 5

# %%

atlas_config = pd.read_csv(atlas_csv_path, sep=';', header=0, names=['Index', 'Region name'], index_col=0)
nbr_region_pairwise = int(len(atlas_config) * (len(atlas_config) - 1) / 2)

bold_signals_subdirpath = os.path.join(times_series_dirpath, dataset_name, atlas_name)
bold_signals_config_csv_path = os.path.join(bold_signals_subdirpath, 'bold_signals_config.csv')
bold_signals_config = pd.read_csv(bold_signals_config_csv_path, header=0, index_col=0)
bold_signals_config = bold_signals_config.dropna(axis=1)

# %%

tasks_durations_list = []
subjects_ids_list = bold_signals_config.columns.sort_values()
for subject_id in subjects_ids_list:
    runs_info_per_subject = connectomes.get_runs_info_per_subject_df(bold_signals_config, subject_id, multi_tr)
    runs_durations_per_subject = connectomes.calculate_runs_durations(runs_info_per_subject)
    tasks_durations_per_subject = connectomes.calculate_task_durations_per_subject(runs_durations_per_subject)
    tasks_durations_list.append(tasks_durations_per_subject)

tasks_durations_df_output = pd.concat(tasks_durations_list, axis=1).T
tasks_durations_df_output.index = subjects_ids_list
tasks_durations_df_output_filtered = tasks_durations_df_output.dropna(axis=1)

# %%

outputs_directory = os.path.join('connectomes_matrix', dataset_name, atlas_name)
connectomes.create_dir(outputs_directory)

# %%
tasks_ids_list = tasks_durations_df_output_filtered.columns
tasks_weighting_coeff_df_output = connectomes.calculate_weighting_coefficients(tasks_durations_df_output_filtered)

for time_chunk in time_chunks_list:
    time_chunk_insec = time_chunk * 60
    tasks_durations_per_time_chunk = connectomes.apply_weighting_coefficients(tasks_weighting_coeff_df_output,
                                                                              time_chunk_insec)
    connectomes_matrices_number_per_time_chunk = connectomes.count_connectomes_matrices_number(
        tasks_durations_df_output_filtered, tasks_durations_per_time_chunk)

    for windows_iter in range(windows_numbers):

        connectomes_matrices_per_time_chunks = connectomes.initilize_connectomes_matrices(
            connectomes_matrices_number_per_time_chunk, nbr_region_pairwise)
        for subject_id in subjects_ids_list:
            runs_info_per_subject = connectomes.get_runs_info_per_subject_df(bold_signals_config, subject_id, multi_tr)
            tasks_durations_per_time_chunk_per_subject = tasks_durations_per_time_chunk.loc[subject_id]
            tasks_weighting_coeff_per_subject = tasks_weighting_coeff_df_output.loc[subject_id]
            runs_info_per_subject_updated = connectomes.update_runs_info_per_subject_df(runs_info_per_subject,
                                                                                        windows_iter, windows_numbers,
                                                                                        tasks_ids_list,
                                                                                        tasks_durations_per_time_chunk_per_subject)

            for connectomes_matrix_number in range(connectomes_matrices_number_per_time_chunk):
                runs_durations_per_subject = connectomes.calculate_runs_durations(runs_info_per_subject_updated)
                tasks_connectomes_per_connectomes_matrix_per_subject = []
                runs_info_per_subject_left_list = []
                for task_id in tasks_ids_list:
                    task_duration_to_extract = tasks_durations_per_time_chunk_per_subject[task_id]
                    runs_info_per_tasks = runs_info_per_subject_updated[task_id]
                    runs_durations_per_task = runs_durations_per_subject[task_id]
                    runs_to_extract_per_task = connectomes.get_runs_durations_to_extract(task_id, task_duration_to_extract,
                                                                                         runs_durations_per_task)
                    bold_signals_extracted_per_task = connectomes.extract_bold_signals(runs_to_extract_per_task,
                                                                                       runs_info_per_tasks)
                    runs_to_extract_per_task_filtered = runs_to_extract_per_task[bold_signals_extracted_per_task.index]
                    bold_signals_weighting_coefficients = connectomes.calculate_weighting_coefficients(
                        runs_to_extract_per_task_filtered)
                    connectomes_list_per_task = connectomes.calculate_connectomes(bold_signals_extracted_per_task)
                    connectome_per_task = connectomes.calculate_weighted_average_connectome(connectomes_list_per_task,
                                                                                            bold_signals_weighting_coefficients)
                    tasks_connectomes_per_connectomes_matrix_per_subject.append(connectome_per_task)
                    runs_info_per_tasks_updated = connectomes.delete_bold_signals(runs_info_per_tasks,
                                                                                  runs_to_extract_per_task)
                    runs_info_per_subject_left_list.append(runs_info_per_tasks_updated)
                runs_info_per_subject_updated = pd.concat(runs_info_per_subject_left_list, axis=1, keys=tasks_ids_list)
                subject_connectome_output = connectomes.calculate_weighted_average_connectome(
                    tasks_connectomes_per_connectomes_matrix_per_subject,
                    tasks_weighting_coeff_per_subject)
                connectomes_matrices_per_time_chunks = connectomes.fill_connectomes_matrix(subject_connectome_output,
                                                                                           connectomes_matrices_per_time_chunks,
                                                                                           connectomes_matrix_number)

        dir_csv_output_path = os.path.join(outputs_directory, str(windows_iter), str(time_chunk))
        connectomes.create_dir(dir_csv_output_path)

        for connectomes_matrix_number in range(connectomes_matrices_number_per_time_chunk):
            connectomes_matrix_to_save = connectomes_matrices_per_time_chunks[connectomes_matrix_number]
            connectomes.save_connectomes_matrices_output(connectomes_matrix_to_save, connectomes_matrix_number,
                                                         dir_csv_output_path, subjects_ids_list)

# %%
# runs_info_per_subject_list = []
# for task_id in tasks_ids_list:
#     task_duration_to_extract = tasks_durations_per_time_chunk_per_subject[task_id]
#     runs_info_per_tasks = runs_info_per_subject[task_id]
#     runs_durations_per_task = connectomes.calculate_runs_durations(runs_info_per_tasks)
#
#     onset_delay_per_task = windows_iter * (task_duration_to_extract // windows_numbers)
#     runs_to_move_slide_windows = connectomes.get_runs_durations_to_extract(task_id, onset_delay_per_task,
#                                                                runs_durations_per_task)
#     bold_signals_slide_windows = connectomes.extract_bold_signals(runs_to_move_slide_windows,
#                                                       runs_info_per_tasks)
#     if bold_signals_slide_windows is not None:
#         runs_info_per_tasks_updated = connectomes.delete_bold_signals(runs_info_per_tasks,
#                                                           runs_to_move_slide_windows)
#
#
#         bold_signals_slide_windows_df = bold_signals_slide_windows.to_frame()
#         bold_signals_slide_windows_df.index = bold_signals_slide_windows_df.index.remove_unused_levels().set_levels(
#             list(range(len(bold_signals_slide_windows))),
#             level=1)
#         bold_signals_slide_windows_df.loc[:,'tr_per_run'] = runs_info_per_tasks.loc['tr_per_run',
#                                                                               runs_to_move_slide_windows.index.tolist()].to_list()
#         bold_signals_slide_windows_df = bold_signals_slide_windows_df.T
#         runs_info_per_tasks_new = pd.concat([runs_info_per_tasks_updated, bold_signals_slide_windows_df],
#                                             axis=1)
#     else:
#         runs_info_per_tasks_new = runs_info_per_tasks
#     runs_info_per_subject_list.append(runs_info_per_tasks_new)
# runs_info_per_subject_updated = pd.concat(runs_info_per_subject_list, axis=1, keys=tasks_ids_list)
