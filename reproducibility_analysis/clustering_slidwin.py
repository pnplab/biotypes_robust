import os
import itertools
import pandas as pd
import numpy as np
import math

from scipy.stats import spearmanr, kruskal, mannwhitneyu, pearsonr
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics


# %%

#  Function to convert 1D vector to symmetric matrix

def vec_to_sym(vec):
    x = len(vec)
    n = int((math.sqrt(8 * x + 1) + 1) / 2)

    idx = np.tril_indices(n, k=-1)
    matrix = np.zeros((n, n))

    matrix[idx] = vec.values
    matrix_sym = matrix + matrix.T + np.diag([0] * n)
    return matrix_sym


# %%

def calculate_mean_connectomes_intersubjects(connectomes_matrices_joined):
    connectomes_matrices_joined_grouped = connectomes_matrices_joined.groupby(level=[0, 1])
    connectomes_mean_intrasubject = connectomes_matrices_joined_grouped.mean()
    connectomes_mean_intersubjects = connectomes_mean_intrasubject.mean(axis=0)
    return connectomes_mean_intersubjects


def calculate_stdmean_connectomes_intersubjects(connectomes_matrices_joined):
    connectomes_matrices_joined_grouped = connectomes_matrices_joined.groupby(level=[0, 1])
    connectomes_stdmean_intrasubject = connectomes_matrices_joined_grouped.std()
    connectomes_stdmean_intersubjects = connectomes_stdmean_intrasubject.mean(axis=0)
    return connectomes_stdmean_intersubjects


def calculate_pairwise_spearmanr_per_chunk(connectomes_matrices_joined):
    def calculate_pairwise_spearmanr(df):
        index_pairs = list(itertools.combinations(df.index, r=2))
        pairwise_spearmanr_coeffs = [spearmanr(df.loc[[val1]], df.loc[[val2]], axis=1)[0] for val1, val2 in index_pairs]
        return pairwise_spearmanr_coeffs

    connectomes_matrices_joined_grouped = connectomes_matrices_joined.groupby(level=[0, 1])
    pairwise_spearmanr_coeffs_per_chunk = connectomes_matrices_joined_grouped.apply(
        lambda x: calculate_pairwise_spearmanr(x))
    pairwise_spearmanr_coeffs_per_chunk_list = list(
        itertools.chain.from_iterable(pairwise_spearmanr_coeffs_per_chunk.tolist()))
    return pd.Series(pairwise_spearmanr_coeffs_per_chunk_list)


# %%

def get_dataset_participants_info_tsv(connectomes_matrices_extracted_dirpath, dataset_name):
    dataset_participants_info_path = os.path.join(connectomes_matrices_extracted_dirpath, dataset_name,
                                                  'participants.tsv')
    dataset_participants_info = pd.read_csv(dataset_participants_info_path, sep='\t', header=0, index_col=0)
    dataset_participants_info = dataset_participants_info.sort_index()
    dataset_participants_info['dataset'] = dataset_name
    return dataset_participants_info


def merge_datasets_participants_info(datasets_participants_info_list):
    participants_info_concat = pd.concat(datasets_participants_info_list, axis=0, join='inner')
    participants_info_filtered = participants_info_concat.reset_index().drop_duplicates().set_index('participant_id')
    participants_info_filtered = participants_info_filtered.dropna()
    return participants_info_filtered


def encoded_dummies_variables(participants_info, dummies_variables_names):
    participants_info_encoded = pd.get_dummies(participants_info, columns=dummies_variables_names, drop_first=True)
    return participants_info_encoded


# %%

def get_minMatrix(matrix_dir, dataset_name_list, atlas_name, slide_window, delta_t):
    min_files = float('inf')
    for dataset_name in dataset_name_list:
        directory = os.path.join(matrix_dir, dataset_name, atlas_name, slide_window, delta_t)
        dir_files = os.listdir(directory)

        dir_len = len(os.listdir(directory))
        if dir_len < min_files:
            min_files = dir_len
            files_list = dir_files
    return files_list


# %%

def create_connMatrix(matrix_dir, dataset_name_list, atlas_name, slide_window, delta_t, matrixFile_path):
    mat_list = []
    for dataset_name in dataset_name_list:
        mat_path = os.path.join(matrix_dir, dataset_name, atlas_name, slide_window, delta_t, matrixFile_path)
        mat = pd.read_csv(mat_path, header=None, index_col=0)
        mat_list.append(mat)

    connecMatrix = pd.concat(mat_list, axis=0)
    return connecMatrix


# %%

def calcul_residualMatrix(X, Y):
    linearModel = LinearRegression(normalize=True).fit(X, Y)
    prediction = linearModel.predict(X)
    error = Y - prediction
    return error


# %%

def apply_clustering_methods(data_to_cluster, nb_cluster_min, nb_cluster_max):
    clusters_labels_list = []
    iter = 10
    for nb_cluster in range(nb_cluster_min, nb_cluster_max + 1):
        for i in range(iter):
            kmeans = KMeans(n_clusters=nb_cluster, init='k-means++').fit(data_to_cluster)
            gmm = GaussianMixture(n_components=nb_cluster, covariance_type='full').fit(data_to_cluster)
            agglomerative = AgglomerativeClustering(n_clusters=nb_cluster, linkage='ward').fit(data_to_cluster)
            clusters_labels_list.append(pd.DataFrame(data=np.vstack((kmeans.labels_, gmm.predict(data_to_cluster),
                                                                     agglomerative.labels_))).T)
    clusters_labels = pd.concat(clusters_labels_list, axis=1, ignore_index=True)
    return clusters_labels


def calculate_coocurence_matrix(clusters_labels):
    n_sample = len(clusters_labels)
    coocurenceMatrix = np.zeros((n_sample, n_sample))
    for i in range(0, n_sample):
        for j in range(0, i + 1):
            clustering_i = clusters_labels.iloc[i,:]
            clustering_j = clusters_labels.iloc[j,:]
            bool_df = (clustering_i == clustering_j)
            coocurenceMatrix[i, j] = bool_df.sum().sum() + coocurenceMatrix[i, j]
            coocurenceMatrix[j, i] = coocurenceMatrix[i, j]
    return coocurenceMatrix / np.amax(coocurenceMatrix)


# %%

def match_biotypes(biotypes_input, biotypes_target):
    connectomes_mean_correlation = pd.concat([biotypes_input, biotypes_target],
                                             axis=0).T.corr(method='spearman')
    connectomes_mean_correlation_filtered = connectomes_mean_correlation.loc[list(biotypes_input.index),
                                                                             list(biotypes_target.index)]

    row_ind, col_ind = linear_sum_assignment(connectomes_mean_correlation_filtered, maximize=True)
    row_names = connectomes_mean_correlation_filtered.index[row_ind]
    col_names = connectomes_mean_correlation_filtered.columns[col_ind]
    matching_between_biotypes = dict(zip(row_names, col_names))
    return matching_between_biotypes


def apply_matching(input_to_match, mapping_dict):
    keys, choices = list(zip(*mapping_dict.items()))
    conds = np.array(keys)[:, None, None] == input_to_match
    input_matched = np.select(conds, choices)[0]
    return input_matched


def calculate_pairwise_connectomes_substracts_per_chunk(connectomes_matrices_joined):
    def calculate_pairwise_substract(df):
        index_pairs = list(itertools.combinations(df.index, r=2))
        pairwise_substract = [(abs(df.loc[val1] - df.loc[val2])).mean() for val1, val2 in index_pairs]
        return pairwise_substract

    connectomes_matrices_joined_grouped = connectomes_matrices_joined.groupby(level=[0, 1])
    pairwise_substract_per_chunk = connectomes_matrices_joined_grouped.apply(
        lambda x: calculate_pairwise_substract(x))
    pairwise_substract_per_chunk_list = list(
        itertools.chain.from_iterable(pairwise_substract_per_chunk.tolist()))
    return pd.Series(pairwise_substract_per_chunk_list)


# def calculate_pairwise_connectomes_mse_per_chunk(connectomes_matrices_joined):
#     def calculate_pairwise_mse(df):
#         index_pairs = list(itertools.combinations(df.index, r=2))
#         pairwise_mse = [(np.sum((vec_to_sym(df.loc[val1]) - vec_to_sym(df.loc[val2]))**2)/float(vec_to_sym(df.loc[val1]).shape[0] * vec_to_sym(df.loc[val1]).shape[1])) for val1, val2 in index_pairs]
#         return pairwise_mse
#
#     connectomes_matrices_joined_grouped = connectomes_matrices_joined.groupby(level=[0, 1])
#     pairwise_mse_per_chunk = connectomes_matrices_joined_grouped.apply(
#         lambda x: calculate_pairwise_mse(x))
#     pairwise_mse_per_chunk_list = list(
#         itertools.chain.from_iterable(pairwise_mse_per_chunk.tolist()))
#     return pd.Series(pairwise_mse_per_chunk_list)


# %%

def calcul_metric(clustering_results):
    metrics_results = []
    nb_clusteringResults = len(clustering_results)
    if nb_clusteringResults == 1:
        return None
    elif nb_clusteringResults == 2:
        ARI = metrics.adjusted_rand_score(clustering_results[0], clustering_results[1])
        AMI = metrics.adjusted_mutual_info_score(clustering_results[0], clustering_results[1])
        metrics_results.append([ARI, AMI])
    else:
        for i in range(nb_clusteringResults - 1):
            for j in range(i + 1, nb_clusteringResults):
                ARI = metrics.adjusted_rand_score(clustering_results[i], clustering_results[j])
                AMI = metrics.adjusted_mutual_info_score(clustering_results[i], clustering_results[j])
                metrics_results.append([ARI, AMI])

    return pd.DataFrame(metrics_results, columns=['ARI', 'AMI'])


# %%

def apply_kruskal_test(metric_per_chunks):
    metric_per_chunks_list = metric_per_chunks.transpose().values.tolist()
    metric_per_chunks_filtered = list(
        map(lambda metric_list: list(filter(lambda metric_value: str(metric_value) != 'nan', metric_list)),
            metric_per_chunks_list))
    kruskal_test = kruskal(*metric_per_chunks_filtered)
    return kruskal_test


def apply_mannwhitneyu(metric_per_chunks):
    time_chunks_filtered_copied = metric_per_chunks.columns.tolist()
    time_chunks_paired = []
    while len(time_chunks_filtered_copied) > 1:
        time_chunks_paired.append(tuple((time_chunks_filtered_copied[0], time_chunks_filtered_copied[1])))
        del time_chunks_filtered_copied[0]

    pairwise_mannwhitneyu_pvalues_list = [
        mannwhitneyu(metric_per_chunks.loc[:, val1].dropna(), metric_per_chunks.loc[:, val2].dropna())[1] for val1, val2
        in
        time_chunks_paired]
    pairwise_mannwhitneyu_pvalues = pd.DataFrame(pairwise_mannwhitneyu_pvalues_list, index=time_chunks_paired,
                                                 columns=['p_values'])
    return pairwise_mannwhitneyu_pvalues
