#%%
"""
Evaluation utility functions.
This module contains util functions for computing evaluation scores.
"""

import numpy as np

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
            key to the cluster column in adata.obs DataFrame.
        k_transition_matrix (str): 
            key to the transition matrix in adata.obs[.
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


def permutation_test(
    adata, 
    k_cluster, 
    k_test,
    k_null, 
    k_compare_on,
    n_resamples = 9999):
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
    
    Returns:
        p_val (float):
            empirical p-value
    """
    test_dist = adata.obs[k_compare_on][np.where(adata.obs[k_cluster]==k_test)[0]]
    null_dist = adata.obs[k_compare_on][np.where(adata.obs[k_cluster]==k_null)[0]]
    
    def run_permutation_test(pooled,test_size,null_size):
        np.random.shuffle(pooled)
        test_star = pooled[:test_size]
        null_star = pooled[-null_size:]
        return test_star.mean() - null_star.mean()
    
    pooled = np.hstack([test_dist,null_dist])
    delta = abs(test_dist.mean() - null_dist.mean())
    estimates = np.abs(np.array([run_permutation_test(pooled,test_dist.size,null_dist.size) for i in range(n_resamples)]))
    p_val = (len(np.where(estimates >= delta)[0])+1)/(n_resamples+1)
    
    return p_val