"""
Same models/checkpoints/config.yaml as train/train_model.py, but the input
data is never read as a precomputed spindle_FR HDF5 file: spindle firing
rates are computed per-batch on GPU from the raw flag_pcr_os_*.hdf5 signals
(muscle_lengths/velocities/accelerations) and the given coefficient sample,
via train/raw_pcr_dataset.py + utils/spindle_FR_gpu.py.

Reuses train.train_model.train_with_config (and therefore Trainer in
train/train_model_utils.py) completely unchanged, so model architecture,
checkpoint format, config.yaml, and the MODELS_DIR/EXPERIMENT_ID naming
convention are identical to what train_model.py produces.
"""

import argparse
import copy
import gc
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import yaml

from directory_paths import SAVE_DIR
from train.raw_pcr_dataset import RawPCRDataset
from train.train_model import NUM_MUSCLES, TIME, train_with_config
from utils.spindle_FR_gpu import build_coefficient_tensors
from utils.spindle_FR_helper import load_coefficients, load_sampled_coefficients

# raw buffer/val cache hold 3x25 channels regardless of n_aff, so one fixed
# size works for every level of a sweep -- no per-level budget math needed
DEFAULT_CHUNK_SIZE = 2048
DEFAULT_BUFFER_CHUNKS = 4
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_MAX_VAL_SAMPLES = 5000
DEFAULT_OPTIMAL_LENGTHS_PATH = os.path.join(ROOT_DIR, "optimal_lengths.npy")


def load_dataset(config, device):
    """Load the RawPCRDataset for config['RAW_DATA_PATH'].

    Loading is independent of `training_seed` (only the model init and the
    per-epoch shuffle depend on it), so callers should build this once per
    `seed` and reuse it across every training_seed for that seed, matching
    train_model.py's load_dataset.
    """
    return RawPCRDataset(
        config["RAW_DATA_PATH"],
        dataset_type="train",
        task=config["TASK"],
        start_end_idx=config["START_END_IDX"],
        chunk_size=config.get("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        buffer_chunks=config.get("BUFFER_CHUNKS", DEFAULT_BUFFER_CHUNKS),
        val_fraction=config.get("VAL_FRACTION", DEFAULT_VAL_FRACTION),
        max_val_samples=config.get("MAX_VAL_SAMPLES", DEFAULT_MAX_VAL_SAMPLES),
        seed=config.get("seed", 0),
        device=device,
        optimal_lengths=config["optimal_lengths"],
    )


def load_base_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models with GPU-computed spindle firing rates (no precomputed per-n_aff file)."
    )
    parser.add_argument(
        "--base_config", type=str, required=True, help="Path to base YAML config"
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        required=True,
        help="Path to the raw flag_pcr_os_*.hdf5 file (muscle_lengths/velocities/accelerations)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Where model checkpoints are written (mirrors train_model.py's --data_dir)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="List of seeds")
    parser.add_argument(
        "--training_seeds", type=int, nargs="+", default=[0], help="Training seeds"
    )
    parser.add_argument("--n_aff", type=int, default=5, help="Number of afferents per type")
    parser.add_argument("--i_a_coeff_path", type=str, required=True)
    parser.add_argument("--ii_coeff_path", type=str, required=True)
    parser.add_argument(
        "--i_a_sampled_coeff_path",
        type=str,
        required=True,
        help="CSV of sampled Ia coefficient pool indices for this n_aff level",
    )
    parser.add_argument(
        "--ii_sampled_coeff_path",
        type=str,
        required=True,
        help="CSV of sampled II coefficient pool indices for this n_aff level",
    )
    parser.add_argument(
        "--optimal_lengths_path", type=str, default=DEFAULT_OPTIMAL_LENGTHS_PATH
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    base_config = load_base_config(args.base_config)
    n_aff = args.n_aff
    seeds = args.seeds
    training_seeds = args.training_seeds
    device = torch.device(args.device)

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = SAVE_DIR
    if "/data" in data_dir:
        last_data_idx = data_dir.rfind("/data")
        base_dir = data_dir[:last_data_idx]
    else:
        base_dir = data_dir

    muscles = list(range(NUM_MUSCLES))
    optimal_lengths = np.load(args.optimal_lengths_path)
    assert len(optimal_lengths) == NUM_MUSCLES, (
        f"optimal_lengths at {args.optimal_lengths_path} has {len(optimal_lengths)} "
        f"entries, expected {NUM_MUSCLES}"
    )

    coefficients_i_a = load_coefficients(args.i_a_coeff_path)
    coefficients_ii = load_coefficients(args.ii_coeff_path)
    sampled_i_a = load_sampled_coefficients(args.i_a_sampled_coeff_path)
    sampled_ii = load_sampled_coefficients(args.ii_sampled_coeff_path)
    coeff_tensors = build_coefficient_tensors(
        coefficients_i_a, coefficients_ii, sampled_i_a, sampled_ii, muscles, device
    )
    assert coeff_tensors["k_l"].shape[0] == n_aff + n_aff, (
        f"sampled coefficient files gave {coeff_tensors['k_l'].shape[0]} channels, "
        f"expected {n_aff + n_aff} (n_aff={n_aff} Ia + n_aff={n_aff} II)"
    )

    print(f"Running GPU-computed-FR trainings for {len(seeds)} seeds and {n_aff} afferents.")

    for seed in seeds:
        # PATH_TO_DATA (here: RAW_DATA_PATH) and everything derived from it
        # depend only on `seed`, so build this config and load the dataset
        # once per seed and reuse it for every training_seed below -- avoids
        # re-reading the raw file from disk once per training_seed.
        seed_config = copy.deepcopy(base_config)

        seed_config["seed"] = seed
        seed_config["RAW_DATA_PATH"] = args.raw_data_path
        seed_config["optimal_lengths"] = optimal_lengths.tolist()
        data_path_prefix = seed_config.get("DATA_PATH_PREFIX", "optimized_linear_extended")
        if seed_config.get("PATH_TO_DATA") is None:
            # kept only for Trainer's summary logging / config.yaml -- RawPCRDataset
            # is loaded from RAW_DATA_PATH above, this is not read as a spindle_FR file
            seed_config["PATH_TO_DATA"] = args.raw_data_path
        seed_config["BASE_DIR"] = base_dir
        if seed_config.get("EXPERIMENT_ID") is None:
            seed_config["EXPERIMENT_ID"] = (
                f"causal_flag-pcr_{data_path_prefix}_{n_aff}_{n_aff}_"
            )
        seed_config["input_shape"] = [n_aff + n_aff, NUM_MUSCLES, TIME]
        seed_config["USE_GPU"] = device.type == "cuda"

        print(f"Loading raw dataset for seed {seed} from {seed_config['RAW_DATA_PATH']}")
        train_data = load_dataset(seed_config, device)
        train_data.set_coefficients(coeff_tensors)

        for training_seed in training_seeds:
            config = copy.deepcopy(seed_config)
            config["training_seed"] = training_seed

            train_with_config(config, train_data)

        train_data.close()
        del train_data
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
