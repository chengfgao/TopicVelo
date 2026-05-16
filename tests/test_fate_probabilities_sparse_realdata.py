import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import load_npz

from topicvelo.vel_eval_utils import fate_probabilities, fate_probabilities_sparse


class DummyAdata:
    def __init__(self, T, key="topicVelo"):
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


def _run_fate_probabilities_sparse_with_wall_time(T, **kwargs):
    adata = DummyAdata(T)
    start_time = time.perf_counter()
    result, info = fate_probabilities_sparse(
        adata,
        "topicVelo",
        return_info=True,
        **kwargs,
    )
    info["total_elapsed_seconds"] = float(time.perf_counter() - start_time)
    return adata, result, info


def _run_fate_probabilities_deeptime_with_wall_time(T):
    adata = DummyAdata(T)
    start_time = time.perf_counter()
    result = fate_probabilities(adata, "topicVelo")
    info = {
        "method": "deeptime_stationary_distribution_fallback",
        "total_elapsed_seconds": float(time.perf_counter() - start_time),
    }
    return adata, result, info


@pytest.mark.large
def test_fate_probabilities_sparse_real_scntseq_combined_transition_matrix():
    if os.environ.get("TOPICVELO_RUN_LARGE_TESTS") != "1":
        pytest.skip("Set TOPICVELO_RUN_LARGE_TESTS=1 to run real-data large tests.")
    pytest.importorskip("deeptime")

    repo_root = Path(__file__).resolve().parents[1]
    matrix_path = repo_root / "results" / "scNT_HH_combined_transition_matrix.npz"
    if not matrix_path.exists():
        pytest.skip(f"Real-data transition matrix not found: {matrix_path}")

    T_sparse = load_npz(matrix_path)

    sparse_adata, sparse_result, sparse_info = _run_fate_probabilities_sparse_with_wall_time(
        T_sparse,
        tol=1e-10,
        max_iter=10000,
        verbose=True,
    )
    deeptime_adata, deeptime_result, deeptime_info = _run_fate_probabilities_deeptime_with_wall_time(T_sparse)

    np.testing.assert_allclose(np.sum(sparse_result), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.sum(deeptime_result), 1.0, atol=1e-10)
    np.testing.assert_allclose(sparse_result, deeptime_result, rtol=1e-6, atol=1e-8)
    assert np.all(np.asarray(sparse_result) >= 0)
    assert np.all(np.asarray(deeptime_result) >= 0)
    assert "topicVelo_stationary_distribution_sparse" in sparse_adata.obs
    assert "topicVelo_stationary_distribution" in deeptime_adata.obs
    assert sparse_info["input_was_sparse"] is True
    assert sparse_info["converted_dense_to_sparse"] is False
    assert sparse_info["residual"] < 1e-10

    _write_trace(
        {
            "test": "real_scntseq_sparse_power_vs_deeptime_head_to_head",
            "matrix_path": str(matrix_path),
            "shape": sparse_info["shape"],
            "initial_density": sparse_info["initial_density"],
            "final_density": sparse_info["final_density"],
            "nnz": sparse_info["final_nnz"],
            "sparse": {
                "iterations": sparse_info["iterations"],
                "residual": sparse_info["residual"],
                "solver_elapsed_seconds": sparse_info["elapsed_seconds"],
                "total_elapsed_seconds": sparse_info["total_elapsed_seconds"],
            },
            "deeptime": {
                "method": deeptime_info["method"],
                "total_elapsed_seconds": deeptime_info["total_elapsed_seconds"],
            },
            "status": "passed",
        }
    )
