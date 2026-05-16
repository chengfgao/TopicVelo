import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sparse

from topicvelo.vel_eval_utils import (
    fate_probabilities_sparse,
    sparsify_transition_matrix_topn,
)


class DummyAdata:
    def __init__(self, T, key="tm"):
        self.obsp = {key + "_T": T}
        self.obs = {}


def _write_trace(record):
    trace_path = os.environ.get("TOPICVELO_TRACE_JSON")
    if not trace_path:
        return

    path = Path(trace_path)
    if path.exists():
        records = json.loads(path.read_text())
    else:
        records = []
    records.append(record)
    path.write_text(json.dumps(records, indent=2, sort_keys=True))


def _random_transition_matrix(n, density, seed):
    rng = np.random.default_rng(seed)
    keep_per_row = min(n, max(3, int(round(n * density))))

    cols = rng.integers(0, n, size=(n, keep_per_row))
    rows_arange = np.arange(n)
    cols[:, 0] = rows_arange
    cols[:, 1] = (rows_arange + 1) % n
    cols[:, 2] = 0

    data = rng.random((n, keep_per_row)) + 0.1
    data[:, 0] += 0.2
    data[:, 1] += 0.2
    data /= data.sum(axis=1, keepdims=True)

    rows = np.repeat(rows_arange, keep_per_row)
    T = sparse.csr_matrix((data.ravel(), (rows, cols.ravel())), shape=(n, n))
    T.sum_duplicates()
    row_sums = np.asarray(T.sum(axis=1)).ravel()
    return sparse.diags(1.0 / row_sums).dot(T).tocsr()


def _run_fate_probabilities_with_wall_time(T, **kwargs):
    adata = DummyAdata(T)
    start_time = time.perf_counter()
    result, info = fate_probabilities_sparse(
        adata,
        "tm",
        return_info=True,
        **kwargs,
    )
    info["total_elapsed_seconds"] = float(time.perf_counter() - start_time)
    return result, info


def _structured_dense_transition_matrix(n):
    T = np.full((n, n), 1e-4, dtype=float)
    for i in range(n):
        T[i, i] = 0.4
        T[i, (i + 1) % n] = 0.3
        T[i, 0] += 0.1
    T /= T.sum(axis=1, keepdims=True)
    return T


def test_known_stationary_distribution_and_density_report(capsys):
    T = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.4, 0.2, 0.4],
            [0.0, 0.1, 0.9],
        ]
    )
    adata = DummyAdata(T)

    result, info = fate_probabilities_sparse(
        adata,
        "tm",
        max_dense_density=1.0,
        verbose=True,
        return_info=True,
    )

    captured = capsys.readouterr()
    assert "Transition matrix density:" in captured.out
    np.testing.assert_allclose(result, [4 / 9, 1 / 9, 4 / 9], atol=1e-8)
    assert "tm_stationary_distribution_sparse" in adata.obs
    assert info["converted_dense_to_sparse"] is True
    assert info["density"] == pytest.approx(7 / 9)


@pytest.mark.parametrize(
    ("n", "density", "tol"),
    [
        (50, 0.10, 1e-10),
        (500, 0.01, 1e-10),
        (5000, 0.001, 1e-8),
    ],
)
def test_sparse_synthetic_sizes_and_sparsities(n, density, tol):
    T = _random_transition_matrix(n, density, seed=n)
    result, info = _run_fate_probabilities_with_wall_time(
        T,
        tol=tol,
        max_iter=10000,
        verbose=False,
    )

    np.testing.assert_allclose(np.sum(result), 1.0, atol=1e-10)
    assert np.all(np.asarray(result) >= 0)
    assert info["input_was_sparse"] is True
    assert info["residual"] < tol
    assert info["iterations"] <= 10000

    _write_trace(
        {
            "test": "synthetic_sparse",
            "n": n,
            "requested_density": density,
            "initial_density": info["initial_density"],
            "final_density": info["final_density"],
            "nnz": info["final_nnz"],
            "iterations": info["iterations"],
            "residual": info["residual"],
            "elapsed_seconds": info["elapsed_seconds"],
            "total_elapsed_seconds": info["total_elapsed_seconds"],
            "status": "passed",
        }
    )


def test_sparse_and_dense_synthetic_head_to_head_speed():
    n = 1000
    density = 0.12
    T_sparse = _random_transition_matrix(n, density, seed=120)
    T_dense = T_sparse.toarray()

    sparse_result, sparse_info = _run_fate_probabilities_with_wall_time(
        T_sparse,
        tol=1e-9,
        max_iter=10000,
        max_dense_density=0.15,
        verbose=False,
    )
    dense_result, dense_info = _run_fate_probabilities_with_wall_time(
        T_dense,
        tol=1e-9,
        max_iter=10000,
        max_dense_density=0.15,
        verbose=False,
    )

    np.testing.assert_allclose(sparse_result, dense_result, atol=1e-10)
    assert sparse_info["input_was_sparse"] is True
    assert dense_info["input_was_sparse"] is False
    assert dense_info["converted_dense_to_sparse"] is True
    assert sparse_info["initial_density"] < 0.15
    assert dense_info["initial_density"] == pytest.approx(sparse_info["initial_density"])

    _write_trace(
        {
            "test": "synthetic_sparse_vs_dense_head_to_head",
            "n": n,
            "requested_density": density,
            "actual_density": sparse_info["initial_density"],
            "sparse": {
                "iterations": sparse_info["iterations"],
                "residual": sparse_info["residual"],
                "solver_elapsed_seconds": sparse_info["elapsed_seconds"],
                "total_elapsed_seconds": sparse_info["total_elapsed_seconds"],
            },
            "dense": {
                "iterations": dense_info["iterations"],
                "residual": dense_info["residual"],
                "solver_elapsed_seconds": dense_info["elapsed_seconds"],
                "total_elapsed_seconds": dense_info["total_elapsed_seconds"],
            },
            "status": "passed",
        }
    )


@pytest.mark.large
@pytest.mark.parametrize(
    ("n", "density"),
    [
        (2500, 0.10),
        (5000, 0.14),
    ],
)
def test_large_synthetic_sparse_and_dense_head_to_head_speed(n, density):
    if os.environ.get("TOPICVELO_RUN_LARGE_TESTS") != "1":
        pytest.skip("Set TOPICVELO_RUN_LARGE_TESTS=1 to run large synthetic tests.")

    T_sparse = _random_transition_matrix(n, density, seed=n + int(density * 1000))
    T_dense = T_sparse.toarray()

    sparse_result, sparse_info = _run_fate_probabilities_with_wall_time(
        T_sparse,
        tol=1e-8,
        max_iter=10000,
        max_dense_density=0.15,
        verbose=False,
    )
    dense_result, dense_info = _run_fate_probabilities_with_wall_time(
        T_dense,
        tol=1e-8,
        max_iter=10000,
        max_dense_density=0.15,
        verbose=False,
    )

    np.testing.assert_allclose(sparse_result, dense_result, atol=1e-8)
    assert sparse_info["initial_density"] < 0.15
    assert dense_info["converted_dense_to_sparse"] is True
    assert sparse_info["residual"] < 1e-8
    assert dense_info["residual"] < 1e-8

    _write_trace(
        {
            "test": "large_synthetic_sparse_vs_dense_head_to_head",
            "n": n,
            "requested_density": density,
            "actual_density": sparse_info["initial_density"],
            "nnz": sparse_info["final_nnz"],
            "sparse": {
                "iterations": sparse_info["iterations"],
                "residual": sparse_info["residual"],
                "solver_elapsed_seconds": sparse_info["elapsed_seconds"],
                "total_elapsed_seconds": sparse_info["total_elapsed_seconds"],
            },
            "dense": {
                "iterations": dense_info["iterations"],
                "residual": dense_info["residual"],
                "solver_elapsed_seconds": dense_info["elapsed_seconds"],
                "total_elapsed_seconds": dense_info["total_elapsed_seconds"],
            },
            "status": "passed",
        }
    )


def test_dense_sparse_like_input_is_converted():
    T = _random_transition_matrix(20, 0.1, seed=11).toarray()
    adata = DummyAdata(T)

    result, info = fate_probabilities_sparse(
        adata,
        "tm",
        max_dense_density=0.15,
        verbose=False,
        return_info=True,
    )

    np.testing.assert_allclose(np.sum(result), 1.0)
    assert info["converted_dense_to_sparse"] is True
    assert info["initial_density"] <= 0.15


def test_dense_high_density_rejected_without_approximation():
    T = np.full((8, 8), 1 / 8)
    adata = DummyAdata(T)

    with pytest.raises(ValueError, match="Dense transition matrix density exceeds"):
        fate_probabilities_sparse(
            adata,
            "tm",
            max_dense_density=0.25,
            approximate=False,
            verbose=False,
        )


def test_dense_high_density_approximation_prunes_and_reports(capsys):
    T = _structured_dense_transition_matrix(12)
    adata = DummyAdata(T)

    result, info = fate_probabilities_sparse(
        adata,
        "tm",
        max_dense_density=0.25,
        approximate=True,
        verbose=True,
        return_info=True,
    )

    captured = capsys.readouterr()
    assert "Transition matrix density:" in captured.out
    assert "Pruned transition matrix to density=" in captured.out
    np.testing.assert_allclose(np.sum(result), 1.0)
    assert info["prune_info"] is not None
    assert info["final_density"] <= 0.25
    assert info["prune_info"]["mean_removed_mass"] >= 0


def test_sparse_high_density_approximation_prunes_main_path():
    T = _random_transition_matrix(100, 0.20, seed=21)
    adata = DummyAdata(T)

    result, info = fate_probabilities_sparse(
        adata,
        "tm",
        max_dense_density=0.15,
        approximate=True,
        verbose=False,
        return_info=True,
    )

    np.testing.assert_allclose(np.sum(result), 1.0)
    assert info["input_was_sparse"] is True
    assert info["prune_info"] is not None
    assert info["final_density"] <= 0.15
    assert info["prune_info"]["rows_affected"] > 0


def test_sparsify_transition_matrix_topn_prunes_and_renormalizes():
    T = np.array(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.10, 0.50, 0.25, 0.15],
            [0.20, 0.20, 0.50, 0.10],
            [0.25, 0.25, 0.10, 0.40],
        ]
    )

    pruned, info = sparsify_transition_matrix_topn(T, 0.5)

    assert sparse.isspmatrix_csr(pruned)
    assert info["keep_per_row"] == 2
    assert info["pruned_density"] <= 0.5
    assert info["rows_affected"] == 4
    assert "mean_removed_mass" in info
    assert "std_removed_mass" in info
    np.testing.assert_allclose(np.asarray(pruned.sum(axis=1)).ravel(), 1.0)


@pytest.mark.parametrize(
    ("T", "match"),
    [
        (np.array([[0.5, -0.5], [0.2, 0.8]]), "negative"),
        (np.array([[0.5, np.nan], [0.2, 0.8]]), "non-finite"),
        (np.array([[0.0, 0.0], [0.2, 0.8]]), "zero-sum"),
        (np.array([[0.5, 0.5], [0.2, 0.2]]), "rows must sum to 1"),
        (np.ones((2, 3)) / 3, "square"),
    ],
)
def test_invalid_transition_matrices_raise(T, match):
    adata = DummyAdata(T)

    with pytest.raises(ValueError, match=match):
        fate_probabilities_sparse(
            adata,
            "tm",
            max_dense_density=1.0,
            verbose=False,
        )
