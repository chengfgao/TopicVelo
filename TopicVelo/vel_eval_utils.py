#%%
"""
Evaluation utility functions.
This module contains util functions for computing evaluation scores.
"""

import numpy as np
from deeptime.markov.tools.analysis import stationary_distribution, mfpt

def fate_probabilities(
    adata, 
    k_transition_matrix):
    """
    Compute stationary distribution for a transition matrix
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_transition_matrix (str): 
            key to the transition matrix in adata.obsp
        
    Returns:
        stationary_distribution (np.array):
            fate probabilities  
    """
    k_st = k_transition_matrix+'_stationary_distribution'
    adata.obs[k_st] = stationary_distribution(adata.obsp[k_transition_matrix+'_T'], check_inputs=False)
    return adata.obs[k_st]

def mfpt_to_targets(
    adata,
    k_transition_matrix,
    target_cells,
    k_mfpt = None,
    rescale_and_smooth = True):
    """
    Compute mfpt to a set of targets then rescale and smooth the results
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_transition_matrix (str): 
            key to the transition matrix in adata.obsp 
        target_cells (list of int):
            indices of cells that are targets
        k_mfpt(str): 
            key to the transition matrix in adata.obsp
        rescale_and_smooth (bool):
            rescale the results by the mean in nonzero data and 
    
    Returns:
        mfpt (array of float): 
            mean-first passage time to targets
    """
    def rescale_and_smooth(adata, obs_key):
        data = adata.obs[obs_key].to_numpy()
        #separate into zeros and nonzeros
        other_indices = np.nonzero(data)
        other_data = data[other_indices]
        smoothed_data = np.zeros(adata.n_obs)
        for i in range(adata.n_obs):
            smoothed_data[i] = np.mean(adata.obs[obs_key][adata.uns['neighbors']['indices'][i]])
        other_data = smoothed_data[other_indices]
        #rescaling
        other_data = other_data/np.median(other_data)
        data[other_indices] = other_data
        adata.obs[obs_key] = data   
    if not k_mfpt:
        k_mfpt = k_transition_matrix+'_mfpt'
    adata.obs[k_mfpt] = mfpt(adata.obsp[k_transition_matrix+'_T'], target_cells)
    if rescale_and_smooth:
        rescale_and_smooth(adata, k_mfpt)    
    return adata.obs[k_mfpt]

# def mfpt_to_multiple_targets(
#     adata,
#     k_transition_matrix,
#     lists_target_cells,
#     k_mfpt = None,
#     rescale = True):
#     """
#     Compute mfpt to multiple targets then rescale the results
#     rescale everything together
    
#     Args:
#         adata (Anndata): 
#             Anndata object.
#         k_transition_matrix (str): 
#             key to the transition matrix in adata.obsp 
#         lists_target_cells (list of lists of int):
#             list of lists of indices of cells that are targets
#         k_mfpt(str): 
#             key to the transition matrix in adata.obsp
#         rescale (bool):
#             rescale the results by the mean in nonzero data and 
    
#     Returns:
#         mfpt (2d array of float): 
#             mean-first passage time to targets
#     """
#     n = len(lists_target_cells)
#     res = np.zeros((adata.n_obs, n))
#     for i in range(n):
#         res[i] = mfpt(adata.obsp[k_transition_matrix+'_T'], lists_target_cells[i])
#     res = res / (np.sum(res)/(np.count_nonzero(res)))
#     if not k_mfpt:
#         k_mfpt = k_transition_matrix+'_mfpt'
#     adata.obsm[k_mfpt] = res
#     return adata.obsm[k_mfpt]
    

def relative_flux_correctness(
    adata, 
    k_cluster, 
    k_transition_matrix, 
    cluster_transitions):
    """Relative Flux Direction Correctness Score (A->B) on the transition matrix
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame
        k_transition_matrix (str): 
            key to the transition matrix in adata.obsp
        cluster_transitions (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
        
    Returns:
        rel_flux (dict):
            relative flux from A->B
        flux (dict): 
            forward and reverse flux between A and B
    """
    flux = {}
    rel_flux = {}
    for A, B in cluster_transitions:
        A_inds = np.where(adata.obs[k_cluster] == A)[0]
        B_inds = np.where(adata.obs[k_cluster] == B)[0]
        A_to_B = 0
        for b in B_inds:
            A_to_B += np.sum(adata.obsp[k_transition_matrix][A_inds,b])  
        B_to_A = 0
        for a in A_inds:
            B_to_A += np.sum(adata.obsp[k_transition_matrix][B_inds,a])  
        #normalization
        # A_to_B = A_to_B/len(A_inds)
        # B_to_A = B_to_A/len(B_inds)
        flux[(A, B)] = A_to_B
        flux[(B, A)] = B_to_A
        rel_flux[(A,B)] = (A_to_B-B_to_A)/(A_to_B+B_to_A)
    return rel_flux, flux

def permutation_test_helper(test_dist, null_dist, n_resamples = 9999, alternative ='two-sided'):
    def run_permutation_test(pooled,test_size,null_size):
        np.random.shuffle(pooled)
        test_star = pooled[:test_size]
        null_star = pooled[-null_size:]
        return test_star.mean() - null_star.mean()
    pooled = np.hstack([test_dist,null_dist])
    delta = test_dist.mean() - null_dist.mean()
    estimates = np.array([run_permutation_test(pooled,test_dist.size,null_dist.size) for i in range(n_resamples)])
    if alternative == 'two-sided':
        p_val = (len(np.where(np.abs(estimates) >= abs(delta))[0])+1)/(n_resamples+1)
    elif alternative == 'less':
        p_val = (len(np.where(estimates <= delta)[0])+1)/(n_resamples+1)
    elif alternative == 'greater':
        p_val = (len(np.where(estimates >= delta)[0])+1)/(n_resamples+1)
    else:
        raise ValueError('Wrong alternative specification')
    return p_val


def permutation_test(
    adata, 
    k_cluster, 
    k_test,
    k_null, 
    k_compare_on,
    n_resamples = 9999,
    alternative ='two-sided'):
    """
    Compute empirical p-value on the means between a test_distribution and the null (background) distribution
    
    Args:
        adata (Anndata):
            Anndata object.
        k_cluster (str):
            key to the cluster column in adata.obs DataFrame.
        k_test (str): 
            group in adata.obs that is used as the test distribution (alternative hypothesis)
        k_null (str): 
            group in adata.obs that is used as the null distribution (null hypothesis)
        k_compare_on (str):
            key to the data in adata.obs to extract two distributions
        n_resamples (int):
            number of subsamples drawn from k_null
        alternative (str):
            "two-sided", "less", "greater"
    
    Returns:
        p_val (float):
            empirical p-value
    """
    test_dist = adata.obs[k_compare_on][np.where(adata.obs[k_cluster]==k_test)[0]]
    null_dist = adata.obs[k_compare_on][np.where(adata.obs[k_cluster]==k_null)[0]]
    return permutation_test_helper(test_dist, null_dist, n_resamples=n_resamples, alternative=alternative)