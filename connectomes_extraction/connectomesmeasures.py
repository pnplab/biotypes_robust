import os
import ast
import math
import numpy as np
import pandas as pd
from nilearn import connectome


# %%

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return None


def get_runs_info_per_subject_df(bold_signals_config, subject_id, multi_tr):
    columns_level = int(bold_signals_config.at['nlevels', subject_id])
    bold_signals_csv_path = bold_signals_config.at['path', subject_id]
    runs_info_per_subject = pd.read_csv(bold_signals_csv_path,
                                        header=list(range(columns_level)),
                                        index_col=0)
    runs_info_per_subject = runs_info_per_subject.applymap(lambda x: np.array(ast.literal_eval(x)))
    tr_list = runs_info_per_subject.loc['tr_per_run']
    tr_number = tr_list.nunique()
    if not multi_tr and tr_number != 1:
        tr_values_counts = tr_list.value_counts()
        tr_most_common = tr_values_counts.index[0]
        runs_info_per_subject = runs_info_per_subject.loc[:,
                                runs_info_per_subject.loc['tr_per_run', :] == tr_most_common]
    return runs_info_per_subject


def calculate_runs_durations(runs_info):
    runs_durations_per_subject = runs_info.apply(
        lambda col: (len(col[0]) * col[1]),
        axis=0)
    return runs_durations_per_subject


def calculate_task_durations_per_subject(runs_durations_per_subject):
    tasks_durations_per_subject = runs_durations_per_subject.groupby(level=0).sum()
    return tasks_durations_per_subject


def calculate_weighting_coefficients(data_to_weight):
    if isinstance(data_to_weight, pd.Series):
        weighting_coefficients = data_to_weight.div(data_to_weight.sum())
    else:
        weighting_coefficients = data_to_weight.div(data_to_weight.sum(axis=1), axis=0)
    return weighting_coefficients


def apply_weighting_coefficients(weighting_coefficients, var):
    var_weighted = weighting_coefficients * var
    return var_weighted


def count_connectomes_matrices_number(tasks_durations_df_output, tasks_durations_per_time_chunk):
    count = (tasks_durations_df_output // tasks_durations_per_time_chunk).min().min()
    return int(count)


def initilize_connectomes_matrices(nbr_connectomes_matrices, nbr_region_pairwise):
    connectomes_matrices = [np.empty((0, nbr_region_pairwise)) for i in range(nbr_connectomes_matrices)]
    return connectomes_matrices


def get_runs_durations_to_extract(task_id, task_duration_to_extract, run_duration_per_task):
    #    task_duration_to_extract = tasks_durations_per_time_chunk_per_subject[task_id]
    # run_duration_per_task = runs_durations_per_subject[task_id]
    i = 0
    runs_ids_to_extract = []
    runs_durations_to_extract = []
    while task_duration_to_extract != 0:
        run_duration = run_duration_per_task.iloc[i]
        # run_id = (task_id,) + run_duration_per_task.index[i]
        run_id = run_duration_per_task.index[i]
        runs_ids_to_extract.append(run_id)
        if run_duration >= task_duration_to_extract:
            runs_durations_to_extract.append(task_duration_to_extract)
            task_duration_to_extract = 0
        else:
            runs_durations_to_extract.append(run_duration)
            task_duration_to_extract = task_duration_to_extract - run_duration
            i = i + 1
    runs_to_extract = pd.Series(data=runs_durations_to_extract, index=runs_ids_to_extract)
    return runs_to_extract


def extract_bold_signals(runs_to_extract, runs_info):
    bold_signals_extracted_list = []
    if runs_to_extract.size > 1:
        runs_ids_list = runs_to_extract.index[:-1]
        bold_signals_list = runs_info.loc['compcor', runs_ids_list]
        bold_signals_extracted_list.append(bold_signals_list)
        runs_to_extract = runs_to_extract.drop(index=runs_ids_list)
    run_duration_to_extract = runs_to_extract.iat[0]
    run_id = runs_to_extract.index
    bold_signal_tr = runs_info.loc['tr_per_run', run_id][0]
    bold_signal_nb_vol_to_extract = math.floor(run_duration_to_extract / bold_signal_tr)
    if bold_signal_nb_vol_to_extract > 1:
        bold_signal = runs_info.loc['compcor', run_id].apply(lambda x: x[:bold_signal_nb_vol_to_extract, :])
        bold_signals_extracted_list.append(bold_signal)
    if len(bold_signals_extracted_list) > 1:
        bold_signals_extracted = pd.concat(bold_signals_extracted_list)
    elif len(bold_signals_extracted_list) == 1:
        bold_signals_extracted = bold_signals_extracted_list[0]
    else:
        bold_signals_extracted = None
    return bold_signals_extracted


def delete_bold_signals(runs_info, runs_to_extract):
    if runs_to_extract.size > 1:
        runs_ids_list = runs_to_extract.index[:-1]
        runs_info = runs_info.drop(columns=runs_ids_list)
        runs_to_extract = runs_to_extract.drop(index=runs_ids_list)
    run_duration_to_extract = runs_to_extract.iat[0]
    run_id = runs_to_extract.index
    bold_signal_tr = runs_info.loc['tr_per_run', run_id][0]
    bold_signal_nb_vol_to_extract = math.floor(run_duration_to_extract / bold_signal_tr)
    bold_signal_nb_vol = runs_info.loc['compcor', run_id].apply(len)[0]
    bold_signal_nb_vol_remaining = bold_signal_nb_vol - bold_signal_nb_vol_to_extract
    if bold_signal_nb_vol_remaining > 1:
        runs_info.loc['compcor', run_id] = runs_info.loc['compcor', run_id].apply(
            lambda x: np.delete(x, slice(0, bold_signal_nb_vol_to_extract, 1), axis=0))
    else:
        runs_info = runs_info.drop(columns=run_id)
    return runs_info


def calculate_connectomes(bold_signals_list):
    correlation_measure = connectome.ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)
    connectome_list = correlation_measure.fit_transform(bold_signals_list)
    return connectome_list


def calculate_weighted_average_connectome(connectomes_list, weights_list):
    connectome_weighted_average = np.average(connectomes_list, axis=0, weights=weights_list)
    return connectome_weighted_average


def fill_connectomes_matrix(subject_connectome, connectomes_matrices, number):
    subject_connectome_transposed = np.reshape(subject_connectome, (1, -1))
    connectomes_matrices[number] = np.vstack((connectomes_matrices[number], subject_connectome_transposed))
    return connectomes_matrices


def save_connectomes_matrices_output(connectome_matrix, number, csv_output_path, subjects_ids_list):
    matrix_csv_path = os.path.join(csv_output_path, 'mat_j.csv'.replace('j', str(number)))
    connectome_matrix_df = pd.DataFrame(connectome_matrix, index=subjects_ids_list)
    connectome_matrix_df.to_csv(matrix_csv_path, header=None)
    #    np.savetxt(matrix_csv_path, connectome_matrix, delimiter=',')
    return None


def update_runs_info_per_subject_df(runs_info_per_subject, windows_iter, windows_numbers, tasks_ids_list,
                                    tasks_durations_per_time_chunk_per_subject):
    if windows_iter == 0:
        runs_info_per_subject_updated = runs_info_per_subject
    else:
        runs_info_per_subject_list = []
        if windows_iter == windows_numbers - 1:
            for task_id in tasks_ids_list:
                runs_info_per_tasks = runs_info_per_subject[task_id]
                runs_info_per_tasks_new = runs_info_per_tasks.iloc[:, ::-1]
                runs_info_per_subject_list.append(runs_info_per_tasks_new)
        else:
            for task_id in tasks_ids_list:
                task_duration_to_extract = tasks_durations_per_time_chunk_per_subject[task_id]
                runs_info_per_tasks = runs_info_per_subject[task_id]
                runs_durations_per_task = calculate_runs_durations(runs_info_per_tasks)

                onset_delay_per_task = windows_iter * (task_duration_to_extract // windows_numbers)
                runs_to_move_slide_windows = get_runs_durations_to_extract(task_id, onset_delay_per_task,
                                                                           runs_durations_per_task)
                bold_signals_slide_windows = extract_bold_signals(runs_to_move_slide_windows,
                                                                  runs_info_per_tasks)
                if bold_signals_slide_windows is not None:
                    runs_to_move_slide_windows_filtered = runs_to_move_slide_windows[bold_signals_slide_windows.index]
                    runs_info_per_tasks_updated = delete_bold_signals(runs_info_per_tasks,
                                                                      runs_to_move_slide_windows)

                    bold_signals_slide_windows_df = bold_signals_slide_windows.to_frame()
                    bold_signals_slide_windows_df.index = bold_signals_slide_windows_df.index.remove_unused_levels().set_levels(
                        list(range(len(bold_signals_slide_windows))),
                        level=1)
                    bold_signals_slide_windows_df.loc[:, 'tr_per_run'] = runs_info_per_tasks.loc['tr_per_run',
                                                                                                 runs_to_move_slide_windows_filtered.index.tolist()].to_list()
                    bold_signals_slide_windows_df = bold_signals_slide_windows_df.T
                    runs_info_per_tasks_new = pd.concat([runs_info_per_tasks_updated, bold_signals_slide_windows_df],
                                                        axis=1)
                else:
                    runs_info_per_tasks_new = runs_info_per_tasks
                runs_info_per_subject_list.append(runs_info_per_tasks_new)
        runs_info_per_subject_updated = pd.concat(runs_info_per_subject_list, axis=1, keys=tasks_ids_list)
    return runs_info_per_subject_updated
