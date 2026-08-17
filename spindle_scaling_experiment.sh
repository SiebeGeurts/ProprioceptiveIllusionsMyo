#!/bin/bash
# ============================================================================
# SPINDLE-COUNT SCALING EXPERIMENT
#
# Trains 10 models -- n_aff = 10, 20, ..., 100 (i.e. 10..100 Ia AND 10..100 II
# afferents per muscle, so 20..200 input channels) -- all on ONE coefficient
# seed and ONE training seed, so the only thing that varies between the 10
# models is how many spindles they see.
#
# Results all land under one models directory: trained_models_spindle_scaling
# (via the MODELS_DIR env var directory_paths.py reads), with a distinct
# experiment subfolder per n_aff level (train_model.py's own naming, driven
# by DATA_PATH_PREFIX below).
#
# ----------------------------------------------------------------------------
# HOW THE DATA IS GENERATED ONLY ONCE
# ----------------------------------------------------------------------------
# Because n_aff=10..100 shares one coefficient seed, naively calling the
# pipeline's own sampler (extract_data.generate_train_test_data via
# utils/spindle_FR_helper.get_sampled_coefficients, which uses
# np.random.choice(..., replace=False)) separately for each n_aff level would
# NOT give nested samples -- there's no guarantee a fresh independent draw of
# size 20 starts with the same 10 elements as a fresh draw of size 10.
# So instead:
#   1. Build ONE master index list per muscle per type, length 100: a
#      permutation of the coefficient pool (IA_COEFF_PATH / II_COEFF_PATH
#      below), truncated to the first 100 entries. Prefixes of this list
#      ([:10], [:20], ...) are nested BY CONSTRUCTION and never repeat a
#      coefficient -- this assumes both pools have at least 100 entries per
#      muscle (asserted in step 1; the script aborts if not, rather than
#      silently sampling with replacement).
#   2. Generate ONE transformed dataset at n_aff=100 (200 channels) over the
#      full 30000-row training set from this master sampling. This is the
#      expensive step: ~691GB on disk, ~1-1.5h to generate.
#   3. For n_aff=10..90, build an HDF5 VIRTUAL DATASET that maps its "data"
#      to the relevant channel-prefix of the master file's "data" (Ia
#      channels [0:k], II channels [100:100+k]) and its "labels" via an
#      external link -- no bytes are copied, so these 9 files cost ~0 extra
#      disk. This was correctness- and performance-checked before writing
#      this script: a VDS read returns byte-identical data to reading the
#      same slice directly from the master file, and reads cost the same
#      per-channel time as a direct read (no VDS overhead observed).
#      n_aff=100 needs no subset file -- it just uses the master file itself.
#   4. Train each of the 10 levels via the existing, unmodified
#      train/train_model.py (so early stopping etc. behave exactly like every
#      other model in this repo), with a per-level chunk_size/max_val_samples
#      (see below) so host RAM use stays roughly constant across the sweep
#      instead of growing with the channel count.
#
# ----------------------------------------------------------------------------
# WHY chunk_size / max_val_samples ARE COMPUTED PER LEVEL, NOT FIXED
# ----------------------------------------------------------------------------
# train/chunked_dataset.py keeps a `buffer_chunks * chunk_size` row shuffle
# buffer AND a `max_val_samples`-row validation set resident in host RAM the
# whole time. Both scale linearly with channel count (2*n_aff). Using the
# pipeline's normal fixed defaults (chunk_size from the file's own on-disk
# chunking, max_val_samples=5000) at n_aff=100 would try to hold ~115GB just
# for the validation set alone (5000 rows * 200ch * 25 muscles * 1152 * 4B) --
# this machine has 78GB RAM. So both are computed here from a fixed BYTE
# budget divided by the level's channel count, keeping resident memory near
# constant (~2GB shuffle buffer, ~2GB validation set) across all 10 levels.
#
# ----------------------------------------------------------------------------
# COST AT A GLANCE (30000 samples, as configured below)
# ----------------------------------------------------------------------------
#   Master dataset (n_aff=100, 200ch):  ~691GB disk, ~1-1.5h to generate
#   9 virtual subsets (n_aff=10..90):   ~0 extra disk, seconds to create
#   Training: 10 sequential models, wall time per model grows with n_aff
#     (the 10-19 timing test at 300ch found training became I/O-bound, not
#     compute-bound -- expect the low end of the sweep to run in the same
#     ballpark as today's 5+5 baseline (~1.5-2h) and the top end (100+100) to
#     take substantially longer; there is no reliable number for this until
#     you've actually run a level or two, which is why the script is
#     resumable -- see below)
#   -> Budget on the order of DAYS wall-clock for the full sweep, and make
#      sure you have >750GB free before starting (5.3TB was free when this
#      script was written).
#
# ----------------------------------------------------------------------------
# RESUMABILITY
# ----------------------------------------------------------------------------
# Every step skips itself if its output already exists (master generation via
# extract_data.generate_train_test_data.process_data's own existing-file
# check; subset creation via an explicit file check; training via a
# `.done_naff{k}` marker touched after a level finishes). Safe to re-run this
# script after an interruption -- it picks up where it left off.
#
# This script is NOT executed automatically -- run it yourself when ready.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# $SECONDS is a bash builtin that counts up from 0 from the moment this script
# started -- used below to report both per-level and total elapsed time. Note
# this is scoped to THIS invocation: if the script is re-run after being
# interrupted (it's resumable, see the .done_naff markers below), the total
# reported here restarts from 0 and does not include time spent in earlier runs.
format_duration() {
    local total_seconds=$1
    printf "%dh%dm%ds" $((total_seconds / 3600)) $(((total_seconds % 3600) / 60)) $((total_seconds % 60))
}

DATA_DIR="/media/data16/siebe/datasets"
EXPERIMENT_DIR="${DATA_DIR}/spindle_scaling_experiment"
mkdir -p "$EXPERIMENT_DIR"

COEFF_SEED=0          # single coefficient seed shared by every n_aff level
TRAINING_SEED=0       # single training seed shared by every n_aff level
N_AFF_MAX=100
N_AFF_LEVELS=(10 20 30 40 50 60 70 80 90 100)

IA_COEFF_PATH="${DATA_DIR}/coefficients_i_a.csv"   # your new per-muscle coefficient pools (both must have
II_COEFF_PATH="${DATA_DIR}/coefficients_ii.csv"    # >= N_AFF_MAX entries per muscle -- asserted in step 1)

NUM_SAMPLES=30000     # full pipeline scale, matches train_spindles.yaml's START_END_IDX
BATCH_SIZE=64         # standard pipeline batch size; lower this if a level OOMs on GPU (watch nvidia-smi)

GEN_TARGET_CHUNK_BYTES=$((8 * 1024**3))   # ~8GiB budget for the one-time master-generation pass
TRAIN_TARGET_CHUNK_BYTES=$((1 * 1024**3)) # ~1GiB shuffle-buffer budget per level (x BUFFER_CHUNKS below)
VAL_TARGET_BYTES=$((2 * 1024**3))         # ~2GiB validation-set budget per level
BUFFER_CHUNKS=2

DATA_PATH_PREFIX="spindle_scaling"
export MODELS_DIR="trained_models_spindle_scaling"

MASTER_PATH="${EXPERIMENT_DIR}/${DATA_PATH_PREFIX}_${COEFF_SEED}_${N_AFF_MAX}_${N_AFF_MAX}_flag_pcr_train.hdf5"

echo "=== Spindle-count scaling experiment: n_aff in ${N_AFF_LEVELS[*]} ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "EXPERIMENT_DIR=$EXPERIMENT_DIR"
echo "MODELS_DIR=$MODELS_DIR"
df -h "$DATA_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv || true

# ----------------------------------------------------------------------------
echo ""
echo "=== [1/4] Building master coefficient index lists (nested, seed=${COEFF_SEED}) ==="
python3 - <<PYEOF
import csv, os, sys
sys.path.insert(0, "${REPO_ROOT}")
import numpy as np
from utils.spindle_FR_helper import load_coefficients

seed = ${COEFF_SEED}
n_max = ${N_AFF_MAX}
out_dir = "${EXPERIMENT_DIR}"

for coeff_type, coeff_path in [("i_a", "${IA_COEFF_PATH}"), ("ii", "${II_COEFF_PATH}")]:
    out_path = os.path.join(out_dir, f"master_sampled_coefficients_{coeff_type}_{n_max}_{seed}.csv")
    if os.path.exists(out_path):
        print(f"skip (exists): {out_path}")
        continue

    coeffs = load_coefficients(coeff_path)
    n_muscles = len(coeffs)
    pool_size = len(coeffs[0]["k_l"])
    assert pool_size >= n_max, (
        f"{coeff_type} pool at {coeff_path} has only {pool_size} coefficients per muscle, "
        f"but the sweep goes up to n_aff={n_max}. Lower N_AFF_LEVELS or supply a bigger pool."
    )

    rng = np.random.RandomState(seed + (0 if coeff_type == "i_a" else 1))  # distinct stream per type
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["muscle", "coefficients"])
        for muscle in range(n_muscles):
            idx = rng.permutation(pool_size)[:n_max]  # no replacement -- pool_size >= n_max is asserted above
            writer.writerow([muscle, idx])
    print(f"wrote {out_path} ({n_muscles} muscles, pool size {pool_size}, no duplicates)")
PYEOF

# ----------------------------------------------------------------------------
echo ""
echo "=== [2/4] Generating the master dataset: n_aff=${N_AFF_MAX} (${N_AFF_MAX}x2 channels), ${NUM_SAMPLES} samples ==="
if [[ -f "$MASTER_PATH" ]]; then
    echo "skip (exists): $MASTER_PATH"
else
    GEN_CHUNK_SIZE=$(python3 -c "print(max(1, ${GEN_TARGET_CHUNK_BYTES} // (2*${N_AFF_MAX}*25*1152*8)))")
    echo "GEN_CHUNK_SIZE=$GEN_CHUNK_SIZE rows/chunk (~$((2*N_AFF_MAX*25*1152*8*GEN_CHUNK_SIZE/1024/1024/1024))GiB/chunk during generation)"
    python3 - <<PYEOF
import sys, time
sys.path.insert(0, "${REPO_ROOT}")
from extract_data.generate_train_test_data import process_data, load_config

config = load_config("extract_data/configs/train_test_data_spindles_extended.yaml")
config.update({
    "num_i_a": ${N_AFF_MAX},
    "num_ii": ${N_AFF_MAX},
    "chunk_size": ${GEN_CHUNK_SIZE},
    "seed": ${COEFF_SEED},
    "i_a_coeff_path": "${IA_COEFF_PATH}",   # your new pools -- must match what step 1 sampled indices from
    "ii_coeff_path": "${II_COEFF_PATH}",
    "i_a_sampled_coeff_path": "${EXPERIMENT_DIR}/master_sampled_coefficients_i_a_${N_AFF_MAX}_${COEFF_SEED}.csv",
    "ii_sampled_coeff_path": "${EXPERIMENT_DIR}/master_sampled_coefficients_ii_${N_AFF_MAX}_${COEFF_SEED}.csv",
})

t0 = time.time()
process_data(
    "${DATA_DIR}/flag_pcr_train.hdf5",
    "${MASTER_PATH}",
    config,
    num_samples=${NUM_SAMPLES},
)
print(f"Master generation took {time.time() - t0:.1f}s")
PYEOF
fi
ls -lh "$MASTER_PATH"

# ----------------------------------------------------------------------------
echo ""
echo "=== [3/4] Building virtual-dataset subsets for n_aff=10..90 (self-checked against the master) ==="
LEVELS_CSV=$(IFS=,; echo "${N_AFF_LEVELS[*]}")
python3 - <<PYEOF
import h5py, numpy as np, os

master_path = "${MASTER_PATH}"
n_max = ${N_AFF_MAX}
levels = [k for k in [${LEVELS_CSV}] if k != n_max]

with h5py.File(master_path, "r") as f:
    n_rows, n_channels, n_muscles, n_time = f["data"].shape
    assert n_channels == 2 * n_max, f"master has {n_channels} channels, expected {2*n_max}"

for k in levels:
    out_path = f"${EXPERIMENT_DIR}/${DATA_PATH_PREFIX}_${COEFF_SEED}_{k}_{k}_flag_pcr_train.hdf5"
    if os.path.exists(out_path):
        print(f"skip (exists): {out_path}")
        continue

    layout = h5py.VirtualLayout(shape=(n_rows, 2 * k, n_muscles, n_time), dtype="float32")
    vsource = h5py.VirtualSource(master_path, "data", shape=(n_rows, n_channels, n_muscles, n_time))
    layout[:, 0:k, :, :] = vsource[:, 0:k, :, :]                    # Ia prefix
    layout[:, k:2 * k, :, :] = vsource[:, n_max:n_max + k, :, :]    # II prefix

    with h5py.File(out_path, "w", libver="latest") as f:
        f.create_virtual_dataset("data", layout, fillvalue=np.nan)
        f["labels"] = h5py.ExternalLink(master_path, "labels")

    # correctness self-check: a handful of rows must match a direct master read exactly
    with h5py.File(master_path, "r") as mf, h5py.File(out_path, "r") as vf:
        check_rows = slice(0, min(5, n_rows))
        expected = np.concatenate(
            [mf["data"][check_rows, 0:k], mf["data"][check_rows, n_max:n_max + k]], axis=1
        )
        got = vf["data"][check_rows]
        assert np.array_equal(got, expected), f"VDS self-check FAILED for n_aff={k}"
        assert np.array_equal(vf["labels"][check_rows], mf["labels"][check_rows])
    print(f"wrote + verified {out_path} ({2*k} channels)")
PYEOF

# ----------------------------------------------------------------------------
echo ""
echo "=== [4/4] Training n_aff=10..100 (seed=${COEFF_SEED}, training_seed=${TRAINING_SEED}) ==="
for K in "${N_AFF_LEVELS[@]}"; do
    DONE_MARKER="${EXPERIMENT_DIR}/.done_naff${K}"
    if [[ -f "$DONE_MARKER" ]]; then
        echo "--- n_aff=${K}: already trained, skipping (remove $DONE_MARKER to redo) ---"
        continue
    fi

    CHUNK_SIZE_K=$(python3 -c "print(max(10, ${TRAIN_TARGET_CHUNK_BYTES} // (2*${K}*25*1152*4)))")
    MAX_VAL_SAMPLES_K=$(python3 -c "print(max(50, ${VAL_TARGET_BYTES} // (2*${K}*25*1152*4)))")
    echo "--- n_aff=${K}: chunk_size=${CHUNK_SIZE_K}, max_val_samples=${MAX_VAL_SAMPLES_K} ---"

    CFG="${EXPERIMENT_DIR}/train_config_naff${K}.yaml"
    cat > "$CFG" <<EOF
PATH_TO_DATA: null
EXPERIMENT_ID: null
DATA_PATH_PREFIX: "${DATA_PATH_PREFIX}"
TASK: "letter_reconstruction_joints"
KEY: "spindle_FR"
RETRAIN: False
START_END_IDX: [0, ${NUM_SAMPLES}]
CAUSAL: True
BATCH_SIZE: ${BATCH_SIZE}
spindle_dataset: True
build_fc: False
layer_norm: False
CHUNK_SIZE: ${CHUNK_SIZE_K}
BUFFER_CHUNKS: ${BUFFER_CHUNKS}
MAX_VAL_SAMPLES: ${MAX_VAL_SAMPLES_K}
EOF

    LEVEL_START=$SECONDS
    python train/train_model.py \
        --data_dir "$EXPERIMENT_DIR" \
        --seeds "$COEFF_SEED" \
        --training_seeds "$TRAINING_SEED" \
        --n_aff "$K" \
        --base_config "$CFG"
    LEVEL_ELAPSED=$((SECONDS - LEVEL_START))

    touch "$DONE_MARKER"
    echo "--- n_aff=${K}: done. This model took $(format_duration $LEVEL_ELAPSED). Total script running time: $(format_duration $SECONDS) ---"
done

echo ""
echo "=== Spindle-count scaling experiment complete. Total running time: $(format_duration $SECONDS) ==="
echo "=== Models under: ${DATA_DIR}/${MODELS_DIR} ==="
