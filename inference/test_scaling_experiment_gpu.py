"""
Evaluates every model trained by spindle_scaling_experiment_gpu.sh against the
real held-out test split (flag_pcr_os_test.hdf5) -- something the sweep itself
never does; training only ever sees a validation split carved out of the
training data (visible as validation_loss/validation_accuracy in each model's
config.yaml).

Computes the same MSE loss and L2-distance accuracy Trainer.eval() computes
during training (same normalization, using each model's own stored
train_mean/train_std/label_mean/label_std from config.yaml), so results are
directly comparable to those validation numbers, just measured on genuinely
unseen data. Spindle firing rates are computed on GPU per batch (same
utils/spindle_FR_gpu.py path train_model_gpu.py uses), using each model's own
sampled coefficients -- the test set is never precomputed to disk.

The raw test file is loaded once and reused across all 30 (n_aff, seed)
models (only ~2GB for the three raw signal arrays, comfortably reusable),
swapping in each model's coefficient tensors between evaluations.

Writes one row per (n_aff, coeff_seed, training_seed) to a single CSV that
spindle_scaling_analysis.ipynb reads.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from model.model_definitions import SpatiotemporalNetworkCausal
from train.raw_pcr_dataset import RawPCRDataset
from utils.spindle_FR_gpu import build_coefficient_tensors
from utils.spindle_FR_helper import load_coefficients, load_sampled_coefficients

NUM_MUSCLES = 25
TIME = 1152

DATA_DIR = "/media/data16/siebe/datasets"
EXPERIMENT_DIR = os.path.join(DATA_DIR, "spindle_scaling_experiment_gpu")
ROOT_MODELS_DIR = "/media/data16/siebe/trained_models_spindle_scaling"
DATA_PATH_PREFIX = "spindle_scaling_gpu"
TASK = "letter_reconstruction_joints"
N_AFF_LEVELS = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
SEED_PAIRS = [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4")]  # (coeff_seed, training_seed)
# (seeds 1/3 trained on a different machine, merged into the same ROOT_MODELS_DIR/EXPERIMENT_DIR tree)

RAW_TEST_PATH = os.path.join(DATA_DIR, "flag_pcr_os_test.hdf5")
IA_COEFF_PATH = os.path.join(DATA_DIR, "coefficients_i_a.csv")
II_COEFF_PATH = os.path.join(DATA_DIR, "coefficients_ii.csv")
OPTIMAL_LENGTHS_PATH = os.path.join(ROOT_DIR, "optimal_lengths.npy")

# same empirically-verified-safe batch_size*channels product spindle_scaling_experiment_gpu.sh
# uses for training -- evaluation has no backward pass so it's conservative here, but there's
# no need to retune it since eval is cheap either way
BATCH_CHANNELS_TARGET = 14400
BATCH_SIZE_MAX = 64
BATCH_SIZE_MIN = 8

OUTPUT_CSV = os.path.join(EXPERIMENT_DIR, "test_performance.csv")


def batch_size_for(n_aff):
    channels = 2 * n_aff
    return max(BATCH_SIZE_MIN, min(BATCH_SIZE_MAX, BATCH_CHANNELS_TARGET // channels))


def build_model(n_aff, coeff_seed, training_seed):
    experiment_id = f"causal_flag-pcr_{DATA_PATH_PREFIX}_{n_aff}_{n_aff}_{TASK}"
    return SpatiotemporalNetworkCausal(
        experiment_id=experiment_id,
        nclasses=7,
        arch_type="spatiotemporal",
        nlayers=4,
        n_skernels=[8, 8, 32, 64],
        n_tkernels=[8, 8, 32, 64],
        s_kernelsize=7,
        t_kernelsize=7,
        s_stride=1,
        t_stride=1,
        padding=3,
        input_shape=[2 * n_aff, NUM_MUSCLES, TIME],
        p_drop=0.7,
        seed=int(coeff_seed),
        # train=True (not False) even though this script only does inference: the model's
        # own __init__ appends "r" to its path name when train=False, which would resolve
        # to a different, nonexistent directory (spatiotemporal_4_0_0r) instead of the real
        # trained model's path -- matching train_model_gpu.py's train=True keeps the same
        # path convention. The checkpoint is loaded from an explicitly-built path below
        # regardless, so this only affects a stray-directory side effect, not correctness.
        train=True,
        task=TASK,
        outtime=TIME,
        # model_definitions.py resolves my_dir relative to its own parent dir (the repo
        # root), so "../trained_models_spindle_scaling" lands on the real model tree,
        # which lives next to the repo, not inside it
        my_dir="../trained_models_spindle_scaling",
        build_fc=False,
        layer_norm=False,
        training_seed=int(training_seed),
    )


def load_model_and_stats(n_aff, coeff_seed, training_seed, device):
    model_dir = os.path.join(
        ROOT_MODELS_DIR,
        f"experiment_causal_flag-pcr_{DATA_PATH_PREFIX}_{n_aff}_{n_aff}_{TASK}",
        f"spatiotemporal_4_{coeff_seed}_{training_seed}",
    )
    ckpt_path = os.path.join(model_dir, "model.ckpt")
    config_path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(ckpt_path):
        return None

    with open(config_path) as f:
        config = yaml.full_load(f)

    model = build_model(n_aff, coeff_seed, training_seed)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    train_mean = torch.tensor(config["train_mean"], dtype=torch.float32, device=device)
    train_std = torch.tensor(config["train_std"], dtype=torch.float32, device=device)
    label_mean = torch.tensor(config["label_mean"], dtype=torch.float32, device=device)
    label_std = torch.tensor(config["label_std"], dtype=torch.float32, device=device)
    return model, train_mean, train_std, label_mean, label_std


def evaluate(model, train_mean, train_std, label_mean, label_std, dataset, batch_size, device):
    epsilon = 1e-8
    n = dataset.test_size
    num_batches = int(np.ceil(n / batch_size))

    total_loss = 0.0
    total_acc = 0.0
    total_rows = 0
    with torch.no_grad():
        for step in range(num_batches):
            batch_X, batch_y = dataset.next_valbatch(batch_size, type="test", step=step)
            if batch_X.shape[0] == 0:
                continue
            batch_y = batch_y.to(device)

            batch_X = (batch_X - train_mean) / (train_std + epsilon)
            batch_y = (batch_y - label_mean) / (label_std + epsilon)

            scores, _, _ = model(batch_X)
            loss = F.mse_loss(scores, batch_y, reduction="mean")
            acc = torch.mean(
                torch.norm(
                    torch.subtract(
                        scores[:, :, :3].reshape(-1, 3), batch_y[:, :, :3].reshape(-1, 3)
                    ),
                    dim=1,
                )
            )

            rows = batch_X.shape[0]
            total_loss += loss.item() * rows
            total_acc += acc.item() * rows
            total_rows += rows

    return total_loss / total_rows, total_acc / total_rows, total_rows


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    optimal_lengths = np.load(OPTIMAL_LENGTHS_PATH)
    muscles = list(range(NUM_MUSCLES))
    coefficients_i_a = load_coefficients(IA_COEFF_PATH)
    coefficients_ii = load_coefficients(II_COEFF_PATH)

    # raw test signals loaded once, reused (with swapped coefficients) for all 30 models
    dataset = RawPCRDataset(
        RAW_TEST_PATH,
        dataset_type="test",
        task=TASK,
        device=device,
        optimal_lengths=optimal_lengths,
    )

    results = []
    for n_aff in N_AFF_LEVELS:
        batch_size = batch_size_for(n_aff)
        for coeff_seed, training_seed in SEED_PAIRS:
            loaded = load_model_and_stats(n_aff, coeff_seed, training_seed, device)
            if loaded is None:
                print(f"n_aff={n_aff} coeff_seed={coeff_seed}: no trained model yet, skipping")
                continue
            model, train_mean, train_std, label_mean, label_std = loaded

            sampled_i_a = load_sampled_coefficients(
                os.path.join(EXPERIMENT_DIR, f"sampled_coefficients_i_a_{n_aff}_{coeff_seed}.csv")
            )
            sampled_ii = load_sampled_coefficients(
                os.path.join(EXPERIMENT_DIR, f"sampled_coefficients_ii_{n_aff}_{coeff_seed}.csv")
            )
            coeff_tensors = build_coefficient_tensors(
                coefficients_i_a, coefficients_ii, sampled_i_a, sampled_ii, muscles, device
            )
            dataset.set_coefficients(coeff_tensors)

            test_loss, test_acc, n_rows = evaluate(
                model, train_mean, train_std, label_mean, label_std, dataset, batch_size, device
            )
            print(
                f"n_aff={n_aff} coeff_seed={coeff_seed} training_seed={training_seed}: "
                f"test_loss={test_loss:.6g} test_accuracy={test_acc:.6g} (n={n_rows})"
            )
            results.append(
                {
                    "n_aff": n_aff,
                    "coeff_seed": coeff_seed,
                    "training_seed": training_seed,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "n_test_samples": n_rows,
                }
            )

            del model
            torch.cuda.empty_cache()

    dataset.close()

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
