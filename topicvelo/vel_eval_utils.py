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
            key to save the mfpt in adata.obsp
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
    adata.uns[k_transition_matrix+'_flux'] = flux
    adata.uns[k_transition_matrix+'_rel_flux']=rel_flux
    return rel_flux, flux

#helper functions for the shortest_transition_paths
#helper functions for the shortest_transition_paths
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
def shortest_paths(adata, k_transition_matrix):
    cost_matrix = -np.log(adata.obsp[k_transition_matrix].A)
    cost_matrix [cost_matrix  == np.inf] = 0
    cost_matrix=csr_matrix(cost_matrix)
    return dijkstra(cost_matrix, return_predecessors=True)

def reconstruct_paths(predecessors, paths_costs, starts, ends):
    def reconstruct_path(predecessors, paths_costs, start, end):
        path = [end]
        cur = end
        cost = 0
        while cur != start:
            cur = predecessors[start, cur]
            cost+= paths_costs[start, cur]
            path.append(cur)
        return path, cost
    paths = []
    costs = []
    for s in starts:
        for e in ends:
            p, c = reconstruct_path(predecessors, paths_costs, s, e)
            paths.append(p)
            costs.append(c)
    return paths, costs


def shortest_transition_paths(adata, k_transition_matrix, starts, ends, recompute=False):
    '''
    Return the shortest paths from every point in start to every point in end
    Args:
        adata: 
        k_transition: key to the transiton matrix in adata.obsp
        starts: indices of starting cells
        end: indices of terminal cells
        recompute: to recompute the shortest paths between all cells via dijkstra
    Return
        Paths: list of list of paths
        Costs: list of list of costs of paths
    '''
    path_key = k_transition_matrix+'_shortest_paths'
    cost_key = k_transition_matrix+'_shortest_paths_cost'
    if path_key not in adata.uns:
        path_cost, shortest_path_predecessors = shortest_paths(adata, k_transition_matrix+'_T')
        adata.uns[path_key] = shortest_path_predecessors
        adata.obsp[cost_key] = path_cost
    return reconstruct_paths(adata.uns[path_key], adata.obsp[cost_key], starts, ends)

def neighborhood_compositions(adata, arr1, arr2, proportion = True):
    '''
    For each element i in arr1+arry2, find how many arr1 are in the neighborhood of i (si1)
    and find how many arr2 are in the neighborhood of j (si2)
    
    proportion: whether to normalize the composition by the cardinality of arrays (the arrays are sets) 
    
    return the composition as 
    comp1 arr1 arr2
    comp2 arr2 arr1
    '''
    def tally(arr):
        comp = np.zeros((len(arr),2))
        for ind, i in enumerate(arr):
            neigh_i = adata.uns['neighbors']['indices'][i]
            si1 = 0
            for j in arr1:
                if j in neigh_i:
                    si1+=1
            si2 = 0
            for j in arr2:
                if j in neigh_i:
                    si2+=1
            if proportion:
                comp[ind,0] = si1/len(arr1)
                comp[ind,1] = si2/len(arr2)
            else:
                comp[ind,0] = si1
                comp[ind,1] = si2
        return comp
    comp1 = tally(arr1)
    comp2 = tally(arr2)[:,[1,0]]
    return comp1, comp2

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


