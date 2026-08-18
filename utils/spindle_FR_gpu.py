"""
GPU runtime computation of spindle firing rates -- the on-the-fly counterpart
to utils/spindle_FR_helper.py's normalize() / clipped_spindle_transfer_function_coeffs(),
used so training never has to read a precomputed per-n_aff firing-rate file
off disk (see extract_data/generate_train_test_data.py's process_chunk for
the offline equivalent this mirrors).
"""

import numpy as np
import torch


def build_coefficient_tensors(
    coefficients_i_a, coefficients_ii, sampled_i_a, sampled_ii, muscles, device
):
    """Assemble per-(afferent, muscle) coefficient tensors for the GPU transfer function.

    Mirrors the indexing in extract_data/generate_train_test_data.py's
    process_chunk: channels [0:n_ia) are Ia afferents, channels
    [n_ia:n_ia+n_ii) are II afferents, both drawn from `sampled_i_a`/
    `sampled_ii` (muscle -> list of pool indices, as produced by
    utils.spindle_FR_helper.load_sampled_coefficients).

    Parameters
    ----------
    coefficients_i_a, coefficients_ii : dict
        muscle -> {"k_l": [...], "k_v": [...], "e_v": [...], "k_a": [...],
        "k_c": [...], "max_rate": [...], "frac_zero": [...]}, as returned by
        utils.spindle_FR_helper.load_coefficients.
    sampled_i_a, sampled_ii : dict
        muscle -> list of pool indices to use, as returned by
        utils.spindle_FR_helper.load_sampled_coefficients.
    muscles : list of int
        Muscle indices, in the order they should occupy the muscle axis.
    device : torch.device or str

    Returns
    -------
    dict of "k_l", "k_v", "e_v", "k_a", "k_c", "max_rate" ->
    (n_ia + n_ii, len(muscles)) float32 tensors on `device`.
    """
    keys = ["k_l", "k_v", "e_v", "k_a", "k_c", "max_rate"]
    n_ia = len(sampled_i_a[muscles[0]])
    n_ii = len(sampled_ii[muscles[0]])
    n_muscles = len(muscles)

    arrays = {k: np.zeros((n_ia + n_ii, n_muscles), dtype=np.float32) for k in keys}
    for m_pos, muscle in enumerate(muscles):
        for j, idx in enumerate(sampled_i_a[muscle]):
            for k in keys:
                arrays[k][j, m_pos] = coefficients_i_a[muscle][k][idx]
        for j, idx in enumerate(sampled_ii[muscle]):
            for k in keys:
                arrays[k][n_ia + j, m_pos] = coefficients_ii[muscle][k][idx]

    return {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}


def compute_spindle_FR_gpu(lengths, velocities, accelerations, optimal_lengths, coeffs):
    """Vectorized, GPU-resident equivalent of spindle_FR_helper.normalize() +
    clipped_spindle_transfer_function_coeffs(), batched over afferents.

    Parameters
    ----------
    lengths, velocities, accelerations : (B, M, T) tensors, on the same
        device as `coeffs`.
    optimal_lengths : (1, M, 1) tensor, same device.
    coeffs : dict of (C, M) tensors, from build_coefficient_tensors.

    Returns
    -------
    (B, C, M, T) tensor of clipped firing rates.

    Note: there is no reduction axis here -- each output channel
    independently rescales the same (muscle, time) signal, it doesn't
    combine multiple muscles into one channel -- so this is a broadcasted
    elementwise op rather than a literal matrix multiplication. It's still a
    single fused vectorized GPU computation with no per-afferent Python
    loop, which is what actually matters for the speedup.

    Memory note: this accumulates into a single (B, C, M, T) buffer with
    in-place ops instead of writing the formula as one chained expression.
    Each `*`/`+`/`.pow()` in a chained broadcast expression materializes its
    own full (B, C, M, T) temporary (measured ~4x a single tensor's size at
    n_aff=150, batch=64 -- e.g. 8.3 GiB instead of 2.1 GiB), which is what
    was OOM-ing on a 12GB GPU at high n_aff. In-place accumulation keeps at
    most ~2 full-size tensors alive at once.
    """
    with torch.no_grad():
        norm_length = lengths / optimal_lengths - 1.0
        norm_velocity = velocities / optimal_lengths
        norm_accel = accelerations / optimal_lengths

        L = norm_length.unsqueeze(1)  # (B, 1, M, T)
        sign_v = torch.sign(norm_velocity).unsqueeze(1)
        abs_v = torch.abs(norm_velocity).unsqueeze(1)
        A = norm_accel.unsqueeze(1)

        k_l = coeffs["k_l"][None, :, :, None]  # (1, C, M, 1)
        k_v = coeffs["k_v"][None, :, :, None]
        e_v = coeffs["e_v"][None, :, :, None]
        k_a = coeffs["k_a"][None, :, :, None]
        k_c = coeffs["k_c"][None, :, :, None]
        max_rate = coeffs["max_rate"][None, :, :, None]

        B, _, M, T = L.shape
        C = k_l.shape[1]
        out_shape = (B, C, M, T)

        # The one unavoidable full-size allocation: |v|**e_v broadcast out to
        # (B, C, M, T) (expand() itself is a stride-0 view, free; .pow()
        # materializes the real buffer). Everything else below folds into it
        # in place instead of creating its own full-size temporary.
        firing_rate = abs_v.expand(out_shape).pow(e_v)
        firing_rate.mul_(sign_v).mul_(k_v)
        firing_rate.add_(k_l * L)  # one transient (B,C,M,T) temp, freed right after
        firing_rate.add_(k_a * A)  # ditto
        firing_rate.add_(k_c)
        firing_rate.clamp_(min=0.0)
        torch.minimum(firing_rate, max_rate, out=firing_rate)
    return firing_rate
