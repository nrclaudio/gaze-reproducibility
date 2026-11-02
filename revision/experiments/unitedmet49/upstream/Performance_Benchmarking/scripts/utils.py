import numpy as np
import pandas as pd
import torch
from statsmodels.stats.multitest import multipletests

def re_rank_2D(data):
    """
    Re-rank the data. Previously, rank 0 means the largest value, # len_samples-1 means the smallest value.
    Previously, the ranks (posterior predicted ranks) are not all integers, so I couldn't use a simple method.
    After reranking, vice vera, rank 0 means the smallest value, # len_samples-1 means the largest value.
    """
    # Sort the array and get the original indices
    sorted_indices = data.argsort(axis=0, kind='stable')
    # Create an array of zeros with the same shape as the original array
    reranked_array = np.zeros_like(data)
    # Assign ranks to the reranked array
    for i in range(data.shape[1]):
        reranked_array[sorted_indices[:, i], i] = np.arange(data.shape[0])
    # Reverse the ranks if needed (0 for the smallest, N-1 for the largest)
    reranked_array = data.shape[0] - 1 - reranked_array
    return reranked_array

def count_obs(data, n_batch, J, batch_index_vector):
    def count_obs_batch(data):
        return np.sum(~np.isnan(data), axis=0)

    n_obs = np.zeros(shape=[n_batch, J]).astype(int)
    for b in range(n_batch):
        n_obs[b, :] = count_obs_batch(data[batch_index_vector == b])
    return n_obs

# Order and Rank the normalized data (directly order, decreasing)
def order_and_rank(data, n_obs, N, J, n_batch, batch_index_vector):
    # offset = 0
    orders = np.zeros([N, J]).astype(int)
    ranks = np.zeros([N, J]).astype(int)
    for b in range(n_batch):
        batch = data[batch_index_vector == b]
        batch = np.where(np.isnan(batch), 0, batch)  # important, replace nan with 0
        indices = batch.argsort(axis=0,
                                kind='stable')  # Returns the indices that would sort an array in ascending order.
        orders[batch_index_vector == b] = indices[::-1]  # order: indices of rank 1 item, rank 2 item, etc.
        # (order of tied value is overwhelmingly behaving weird)
        ranks_temp = batch.argsort(axis=0, kind='stable')[::-1].argsort(
            axis=0, kind='stable')  # rank (largest item has rank 0, second largest has rank 1, etc.)
        # if rank > n_obs, set rank = n_obs (get censored rank)
        ranks[batch_index_vector == b] = np.where(ranks_temp > n_obs[b, :], n_obs[b, :], ranks_temp)
        # offset = offset + data[batch_index_vector == b].shape[0]
    return orders, ranks

def test_train_split_met_rna(met_data, start_row, stop_row, target, n_obs, ranks, f_test_prop):
    """
    testing and training sets are only metabolomics data.
    """
    test_sample_indices = np.arange(start_row[target-1], stop_row[target-1])
    temp = np.copy(met_data)
    # Mask all metabolite data of the target batch in the training set
    # training and testing set only have metabolomics data
    temp[test_sample_indices, :] = np.nan

    solo = np.all(np.isnan(temp), axis=0)
    missing = np.all(np.isnan(met_data[test_sample_indices, :]), axis=0)
    available_features = np.arange(met_data.shape[1])[(~missing) & (~solo)]
    len_available = len(available_features)
    test_feature_indices = np.random.choice(available_features,
                                            size=int(round(f_test_prop * len(available_features))), replace=False)
    # sort the indices in ascending order
    test_sample_indices = np.sort(test_sample_indices)
    test_feature_indices = np.sort(test_feature_indices)
    # Reset the n_obs, all metabolites in the target batch are set to 0
    n_obs[target - 1, test_feature_indices] = 0
    training = np.copy(met_data)
    testing = np.full(training.shape, np.nan)
    for sidx in test_sample_indices:
        for fidx in test_feature_indices:
            training[sidx, fidx] = np.nan
            testing[sidx, fidx] = ranks[sidx, fidx]

    return testing, training, n_obs, test_sample_indices, test_feature_indices, len_available

def split_folds_across(data, n_batch, batch_index_vector, n_folds):
    """
    The difference from the cross_validation() in the PL_single_array_cv.py code is that:
    the folds are not the same length, thus in format of list of arrays but not ndarray
    """
    folds = [[] for _ in range(n_folds)]  # _ is "throwaway" variable name in python
    for bidx in range(n_batch):
        batch = data[batch_index_vector == bidx]
        missing = np.all(np.isnan(batch), axis=0)
        solo = np.all(np.isnan(data[batch_index_vector != bidx]),
                      axis=0)  # solo are the data that don't overlap with other datasets
        available = np.arange(batch.shape[1])[(~missing) & (~solo)]  # features for cross validation in each batch
        np.random.shuffle(available)  # shuffles the features
        splits = np.array_split(available, n_folds)  # Split an array into multiple sub-arrays
        for fold_idx in range(n_folds):
            folds[fold_idx].extend([splits[fold_idx]])
            # folds[i] is a list of arrays storing all the (#column) in the #batch array in that i+1 fold across datasets
    # folds = [np.array(fold) for fold in folds]
    # Couldn't use this line because the arrays in each folds are not the same length, thus cannot be converted to ndarray
    return folds

def gumbel_sampling_3D(x):
    """
    x: logits, 3D array
    vectorized gumbel re-parameterization ~ categorical sampling without replacement
    """
    G = np.random.gumbel(0, 1, size=(x.shape[0], x.shape[1], x.shape[2]))
    Z = -(x + G)  # normal prior
    return np.argsort(Z, axis=1, kind='stable')  # return the indices of from largest Z to smallest Z

def smart_perm_2D(x, permutation):
    """
        x: 2D tensor of logits
        permutation: 2D tensor of orders
        permutae columns of x according to the permutation (adapted from the github version where they permute rows)
    """
    assert x.size() == permutation.size()
    if x.ndimension() == 2:
        d1, d2 = x.size()
        x_permuted = x[
            permutation.flatten(),
            torch.arange(d2).unsqueeze(0).repeat((1, d1)).flatten()
        ].view(d1, d2)
    else:
        ValueError("Only 2 dimension expected")
    return x_permuted

def saving_data_for_pyro_posterior(embedding_dir, N, K, J, met_names, start_row, stop_row, batch_sizes):
    with open(f'{embedding_dir}/data_for_posterior_pyro.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, N)  # save arrays in numpy binary .npy files
        np.save(f, K)
        np.save(f, J)
        np.save(f, start_row)
        np.save(f, stop_row)
        np.save(f, batch_sizes)
        f.close()
    pd.DataFrame({'met_names': met_names}).to_csv(f'{embedding_dir}/met_names_pyro.csv')

def saving_data_for_testing(embedding_dir, testing, test_sample_indices, test_feature_indices, n_obs_pre, censor_indicator):
    with open(f'{embedding_dir}/data_for_testing_pyro.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, testing)
        np.save(f, test_sample_indices)
        np.save(f, test_feature_indices)
        np.save(f, n_obs_pre)
        np.save(f, censor_indicator)
        f.close()

def correct_for_mult(data):
    data['p_adj'] = multipletests(pvals=data['pval'], method="fdr_bh", alpha=0.1)[1]
    return data
