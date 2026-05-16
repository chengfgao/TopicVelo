#%%
"""
Evaluation utility functions.
This module contains util functions for computing evaluation scores.
"""

import time
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix


def _report(message, verbose):
    if verbose:
        print(message)


def _matrix_density(T):
    n_rows, n_cols = T.shape
    size = n_rows * n_cols
    if size == 0:
        raise ValueError("Transition matrix must not be empty.")
    if sparse.issparse(T):
        nnz = T.nnz
    else:
        nnz = np.count_nonzero(T)
    return nnz / size, nnz


def _renormalize_sparse_rows(T):
    row_sums = np.asarray(T.sum(axis=1)).ravel()
    nonzero_rows = row_sums != 0
    inv_row_sums = np.zeros_like(row_sums, dtype=float)
    inv_row_sums[nonzero_rows] = 1.0 / row_sums[nonzero_rows]
    return sparse.diags(inv_row_sums).dot(T).tocsr()


def sparsify_transition_matrix_topn(
    T,
    max_density,
    *,
    renormalize=True,
    min_keep_per_row=1,
):
    """
    Prune a transition matrix by keeping the largest entries in each row.

    Args:
        T (array-like or scipy.sparse matrix):
            Transition matrix.
        max_density (float):
            Target matrix density after pruning.
        renormalize (bool):
            Whether to renormalize each row to sum to one after pruning.
        min_keep_per_row (int):
            Minimum number of entries to keep in each non-empty row.

    Returns:
        tuple:
            ``(T_pruned, prune_info)``, where ``T_pruned`` is CSR sparse.
    """
    if max_density <= 0 or max_density > 1:
        raise ValueError("max_density must be in the interval (0, 1].")
    if min_keep_per_row < 1:
        raise ValueError("min_keep_per_row must be at least 1.")

    T_csr = T.tocsr(copy=True) if sparse.issparse(T) else csr_matrix(T)
    T_csr.eliminate_zeros()
    n_rows, n_cols = T_csr.shape
    original_density, original_nnz = _matrix_density(T_csr)
    target_nnz = int(max_density * n_rows * n_cols)
    keep_per_row = max(min_keep_per_row, target_nnz // n_rows)
    keep_per_row = min(keep_per_row, n_cols)

    indptr = [0]
    indices = []
    data = []
    removed_mass = np.zeros(n_rows, dtype=float)

    for row in range(n_rows):
        start, end = T_csr.indptr[row], T_csr.indptr[row + 1]
        row_indices = T_csr.indices[start:end]
        row_data = T_csr.data[start:end]
        row_nnz = row_data.size

        if row_nnz == 0:
            indptr.append(len(indices))
            continue

        row_sum = row_data.sum()
        if row_nnz > keep_per_row:
            keep_positions = np.argpartition(row_data, -keep_per_row)[-keep_per_row:]
            kept_indices = row_indices[keep_positions]
            kept_data = row_data[keep_positions]
        else:
            kept_indices = row_indices
            kept_data = row_data

        kept_sum = kept_data.sum()
        if row_sum > 0:
            removed_mass[row] = max(row_sum - kept_sum, 0.0) / row_sum

        indices.extend(kept_indices)
        data.extend(kept_data)
        indptr.append(len(indices))

    T_pruned = csr_matrix(
        (
            np.asarray(data, dtype=T_csr.dtype),
            np.asarray(indices, dtype=T_csr.indices.dtype),
            np.asarray(indptr, dtype=T_csr.indptr.dtype),
        ),
        shape=T_csr.shape,
    )
    T_pruned.eliminate_zeros()
    if renormalize:
        T_pruned = _renormalize_sparse_rows(T_pruned)

    pruned_density, pruned_nnz = _matrix_density(T_pruned)
    prune_info = {
        "original_nnz": int(original_nnz),
        "original_density": float(original_density),
        "pruned_nnz": int(pruned_nnz),
        "pruned_density": float(pruned_density),
        "target_density": float(max_density),
        "target_nnz": int(target_nnz),
        "keep_per_row": int(keep_per_row),
        "mean_removed_mass": float(removed_mass.mean()),
        "std_removed_mass": float(removed_mass.std()),
        "max_removed_mass": float(removed_mass.max()),
        "rows_affected": int(np.count_nonzero(removed_mass > 0)),
    }
    return T_pruned, prune_info


def _validate_sparse_transition_matrix(T, row_sum_tol=1e-8):
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("Transition matrix must be square.")
    if not np.all(np.isfinite(T.data)):
        raise ValueError("Transition matrix contains non-finite values.")
    if np.any(T.data < 0):
        raise ValueError("Transition matrix contains negative values.")

    row_sums = np.asarray(T.sum(axis=1)).ravel()
    if np.any(row_sums <= 0):
        raise ValueError("Transition matrix contains zero-sum rows.")
    if not np.allclose(row_sums, 1.0, atol=row_sum_tol):
        min_row_sum = float(row_sums.min())
        max_row_sum = float(row_sums.max())
        raise ValueError(
            "Transition matrix rows must sum to 1. "
            f"Observed row sum range: [{min_row_sum:.6g}, {max_row_sum:.6g}]."
        )


def _stationary_distribution_power(T, tol, max_iter):
    n = T.shape[0]
    pi = np.full(n, 1.0 / n, dtype=float)
    residual = np.inf
    start_time = time.perf_counter()

    for iteration in range(1, max_iter + 1):
        pi_next = np.asarray(T.T @ pi).ravel()
        pi_next_sum = pi_next.sum()
        if not np.isfinite(pi_next_sum) or pi_next_sum <= 0:
            raise RuntimeError("Power iteration produced an invalid probability vector.")
        pi_next /= pi_next_sum
        residual = np.abs(pi_next - pi).sum()
        pi = pi_next
        if residual < tol:
            info = {
                "solver": "sparse_power_iteration",
                "iterations": int(iteration),
                "residual": float(residual),
                "tol": float(tol),
                "max_iter": int(max_iter),
                "elapsed_seconds": float(time.perf_counter() - start_time),
            }
            return pi, info

    raise RuntimeError(
        "Sparse power iteration failed to converge after "
        f"{max_iter} iterations; final residual was {residual:.6g}."
    )


def fate_probabilities_sparse(
    adata,
    k_transition_matrix,
    *,
    tol=1e-10,
    max_iter=10000,
    max_dense_density=0.15,
    approximate=False,
    verbose=True,
    return_info=False,
):
    """
    Compute fate probabilities with sparse power iteration.

    This avoids the sparse LU factorization used by deeptime's default
    stationary-distribution fallback path.

    Args:
        adata (Anndata):
            Anndata object.
        k_transition_matrix (str):
            Key to the transition matrix in ``adata.obsp``.
        tol (float):
            L1 convergence tolerance for power iteration.
        max_iter (int):
            Maximum power-iteration steps.
        max_dense_density (float):
            Maximum accepted dense-input density and pruning target.
        approximate (bool):
            Whether to prune high-density matrices to ``max_dense_density``.
        verbose (bool):
            Whether to report density and pruning statistics immediately.
        return_info (bool):
            Whether to return diagnostic information.

    Returns:
        pandas.Series or tuple:
            Stored fate probabilities, or ``(series, info)`` when
            ``return_info=True``.
    """
    T = adata.obsp[k_transition_matrix + '_T']
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("Transition matrix must be square.")
    if max_dense_density <= 0 or max_dense_density > 1:
        raise ValueError("max_dense_density must be in the interval (0, 1].")

    input_was_sparse = sparse.issparse(T)
    density, nnz = _matrix_density(T)
    _report(
        "Transition matrix density: "
        f"{density:.6f} (nnz={int(nnz)}, shape={T.shape}).",
        verbose,
    )

    info = {
        "input_was_sparse": bool(input_was_sparse),
        "converted_dense_to_sparse": False,
        "shape": tuple(int(x) for x in T.shape),
        "initial_nnz": int(nnz),
        "initial_density": float(density),
        "density": float(density),
        "max_dense_density": float(max_dense_density),
        "approximate": bool(approximate),
        "prune_info": None,
    }

    if input_was_sparse:
        T_csr = T.tocsr(copy=True)
    else:
        if density > max_dense_density and not approximate:
            raise ValueError(
                "Dense transition matrix density exceeds max_dense_density "
                f"({density:.6f} > {max_dense_density:.6f}). Provide a sparse "
                "matrix or rerun with approximate=True to prune before solving."
            )
        T_csr = csr_matrix(T)
        info["converted_dense_to_sparse"] = True

    T_csr.eliminate_zeros()
    if approximate and density > max_dense_density:
        T_csr, prune_info = sparsify_transition_matrix_topn(
            T_csr,
            max_dense_density,
            renormalize=True,
        )
        info["prune_info"] = prune_info
        _report(
            "Pruned transition matrix to density="
            f"{prune_info['pruned_density']:.4f}; removed transition mass "
            f"mean={100 * prune_info['mean_removed_mass']:.2f}%, "
            f"std={100 * prune_info['std_removed_mass']:.2f}%.",
            verbose,
        )

    final_density, final_nnz = _matrix_density(T_csr)
    info["final_nnz"] = int(final_nnz)
    info["final_density"] = float(final_density)

    _validate_sparse_transition_matrix(T_csr)
    pi, solver_info = _stationary_distribution_power(T_csr, tol, max_iter)
    info.update(solver_info)

    k_st = k_transition_matrix + '_stationary_distribution_sparse'
    adata.obs[k_st] = pi
    result = adata.obs[k_st]
    if return_info:
        return result, info
    return result


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
    from deeptime.markov.tools.analysis import stationary_distribution

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
    from deeptime.markov.tools.analysis import mfpt

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
def shortest_paths(adata, 
                   k_transition_matrix):
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
    if path_key not in adata.uns or recompute:
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
