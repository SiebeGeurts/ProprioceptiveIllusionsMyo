"""
One-off correctness check for the GPU spindle firing-rate pipeline
(utils/spindle_FR_gpu.py, train/raw_pcr_dataset.py) against the existing,
trusted CPU path (utils/spindle_FR_helper.py, as used offline by
extract_data/generate_train_test_data.py).

Computes firing rates for a handful of rows both ways with the same sampled
coefficients and asserts they match to float32 precision. Not wired into any
pipeline -- run manually once before trusting spindle_scaling_experiment_gpu.sh:

    python data_generation/verify_spindle_fr_gpu.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import h5py
import numpy as np
import torch

from directory_paths import SAVE_DIR
from utils.spindle_FR_gpu import build_coefficient_tensors, compute_spindle_FR_gpu
from utils.spindle_FR_helper import (
    clipped_spindle_transfer_function_coeffs,
    load_coefficients,
    normalize,
)

N_CHECK_ROWS = 8
N_AFF = 7  # deliberately not a round number
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def cpu_reference(lengths, velocities, accelerations, optimal_lengths, coefficients_i_a,
                   coefficients_ii, sampled_i_a, sampled_ii, muscles):
    """Row-by-row port of extract_data/generate_train_test_data.py's process_chunk,
    for a single already-loaded chunk (no HDF5 I/O)."""
    data = normalize(lengths, velocities, accelerations, optimal_lengths)
    n_ia = len(sampled_i_a[muscles[0]])
    n_ii = len(sampled_ii[muscles[0]])
    out = np.zeros((lengths.shape[0], n_ia + n_ii, len(muscles), lengths.shape[2]), dtype=np.float32)

    for m_pos, muscle in enumerate(muscles):
        for j, idx in enumerate(sampled_i_a[muscle]):
            coeffs = {k: coefficients_i_a[muscle][k][idx] for k in
                      ["k_l", "k_v", "e_v", "k_a", "k_c", "max_rate"]}
            out[:, j, m_pos, :] = clipped_spindle_transfer_function_coeffs(
                data["lengths"][:, muscle, :], data["velocities"][:, muscle, :],
                data["accelerations"][:, muscle, :], coeffs,
            )
        for j, idx in enumerate(sampled_ii[muscle]):
            coeffs = {k: coefficients_ii[muscle][k][idx] for k in
                      ["k_l", "k_v", "e_v", "k_a", "k_c", "max_rate"]}
            out[:, n_ia + j, m_pos, :] = clipped_spindle_transfer_function_coeffs(
                data["lengths"][:, muscle, :], data["velocities"][:, muscle, :],
                data["accelerations"][:, muscle, :], coeffs,
            )
    return out


def main():
    raw_path = os.path.join(SAVE_DIR, "flag_pcr_os_train.hdf5")
    ia_coeff_path = os.path.join(SAVE_DIR, "coefficients_i_a.csv")
    ii_coeff_path = os.path.join(SAVE_DIR, "coefficients_ii.csv")
    optimal_lengths_path = os.path.join(ROOT_DIR, "optimal_lengths.npy")

    optimal_lengths = np.load(optimal_lengths_path)
    n_muscles = len(optimal_lengths)
    muscles = list(range(n_muscles))

    coefficients_i_a = load_coefficients(ia_coeff_path)
    coefficients_ii = load_coefficients(ii_coeff_path)
    rng = np.random.RandomState(0)
    pool_size_ia = len(coefficients_i_a[0]["k_l"])
    pool_size_ii = len(coefficients_ii[0]["k_l"])
    sampled_i_a = {m: rng.choice(pool_size_ia, N_AFF, replace=False) for m in muscles}
    sampled_ii = {m: rng.choice(pool_size_ii, N_AFF, replace=False) for m in muscles}

    with h5py.File(raw_path, "r") as f:
        lengths = f["muscle_lengths"][:N_CHECK_ROWS]
        velocities = f["muscle_velocities"][:N_CHECK_ROWS]
        accelerations = f["muscle_accelerations"][:N_CHECK_ROWS]

    print(f"Computing CPU reference for {N_CHECK_ROWS} rows, n_aff={N_AFF} ...")
    expected = cpu_reference(
        lengths, velocities, accelerations, optimal_lengths,
        coefficients_i_a, coefficients_ii, sampled_i_a, sampled_ii, muscles,
    )

    print(f"Computing GPU path on {DEVICE} ...")
    device = torch.device(DEVICE)
    coeff_tensors = build_coefficient_tensors(
        coefficients_i_a, coefficients_ii, sampled_i_a, sampled_ii, muscles, device
    )
    optimal_lengths_t = torch.as_tensor(
        optimal_lengths, dtype=torch.float32, device=device
    ).view(1, -1, 1)
    got = compute_spindle_FR_gpu(
        torch.from_numpy(lengths).to(device),
        torch.from_numpy(velocities).to(device),
        torch.from_numpy(accelerations).to(device),
        optimal_lengths_t,
        coeff_tensors,
    ).cpu().numpy()

    max_abs_diff = np.max(np.abs(got - expected))
    print(f"max abs diff: {max_abs_diff:.3e}")
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    print("OK: GPU firing rates match the CPU reference path.")


if __name__ == "__main__":
    main()
