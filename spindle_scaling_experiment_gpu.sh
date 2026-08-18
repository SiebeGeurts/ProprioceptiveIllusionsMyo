#!/bin/bash
# ============================================================================
# SPINDLE-COUNT SCALING EXPERIMENT -- GPU on-the-fly firing rates
#
# Trains 15 models -- n_aff = 10, 20, ..., 150 (i.e. 10..150 Ia AND 10..150
# II afferents per muscle, so 20..300 input channels) -- all on ONE
# coefficient seed and ONE training seed, so the only thing that varies
# between the 15 models is how many spindles they see.
#
# Unlike spindle_scaling_experiment.sh, this script never writes a
# precomputed per-n_aff spindle_FR HDF5 file. Spindle firing rates are
# computed on GPU per training batch from the raw flag_pcr_os_train.hdf5
# signals (muscle_lengths/velocities/accelerations) via
# train/train_model_gpu.py -> train/raw_pcr_dataset.py ->
# utils/spindle_FR_gpu.py. That removes the entire "generate a master
# dataset + build virtual-dataset subsets" phase of the old script -- there
# is no multi-hundred-GB intermediate file and no disk-bound offline
# generation pass; the raw file (~34GB) is read directly by the training
# loop, chunk by chunk, the same way ChunkedSpindleDataset already streamed
# the old precomputed files.
#
# Results land under the same place as before: trained_models_spindle_scaling
# (via the MODELS_DIR env var directory_paths.py reads), one experiment
# subfolder per n_aff level, named and checkpointed exactly like every other
# model in this repo (train_model_gpu.py reuses train/train_model.py's
# train_with_config/Trainer unchanged).
#
# ----------------------------------------------------------------------------
# NESTED COEFFICIENT SAMPLING (kept, same idea as the old script)
# ----------------------------------------------------------------------------
# n_aff=10..150 shares one coefficient seed, and afferents are sampled so
# that n_aff=K's channels are always a strict prefix of n_aff=(K+10)'s -- a
# master permutation of the coefficient pool is built once (step 1) and each
# level slices its own prefix off of it (step 2, per level). This is pure
# index bookkeeping (a few KB of CSV), not data, so it costs nothing extra
# under the GPU pipeline and there's no reason to drop it.
#
# ----------------------------------------------------------------------------
# RESUMABILITY
# ----------------------------------------------------------------------------
# Training per level is skipped if its `.done_naff{k}` marker already exists
# (touched after that level finishes). Safe to re-run this script after an
# interruption -- it picks up where it left off.
#
# This script is NOT executed automatically -- run it yourself when ready.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

format_duration() {
    local total_seconds=$1
    printf "%dh%dm%ds" $((total_seconds / 3600)) $(((total_seconds % 3600) / 60)) $((total_seconds % 60))
}

DATA_DIR="/media/data16/siebe/datasets"
EXPERIMENT_DIR="${DATA_DIR}/spindle_scaling_experiment_gpu"
mkdir -p "$EXPERIMENT_DIR"

COEFF_SEED=0          # single coefficient seed shared by every n_aff level
TRAINING_SEED=0       # single training seed shared by every n_aff level
N_AFF_MAX=150
N_AFF_LEVELS=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150)

RAW_DATA_PATH="${DATA_DIR}/flag_pcr_os_train.hdf5"
IA_COEFF_PATH="${DATA_DIR}/coefficients_i_a.csv"   # per-muscle coefficient pools (both must have
II_COEFF_PATH="${DATA_DIR}/coefficients_ii.csv"    # >= N_AFF_MAX entries per muscle -- asserted in step 1)

NUM_SAMPLES=30000     # full pipeline scale, matches train_spindles.yaml's START_END_IDX
DEVICE="cuda:0"

# GPU memory (not host RAM) is the thing that scales with n_aff here: both the
# firing-rate tensor and the model's first-layer input are (batch, channels,
# muscles, time), and channels = 2*n_aff. Empirically on this 12GB card,
# n_aff=150 (300 channels) OOMs at batch_size=64 but runs cleanly (with room
# to spare) at 48 and 40 -- so batch_size is scaled down per level to keep
# batch_size*channels roughly constant instead of using one fixed value for
# the whole sweep. 14400 = 48*300, the empirically-verified-safe point.
BATCH_SIZE_MAX=64
BATCH_SIZE_MIN=8
BATCH_CHANNELS_TARGET=14400

# Fixed across every level: the shuffle buffer/val cache now hold raw
# per-muscle signals (25 channels each), not the spindle_FR channels, so
# their memory footprint no longer depends on n_aff -- no per-level
# chunk_size/max_val_samples budget math needed like the old script.
CHUNK_SIZE=2048
BUFFER_CHUNKS=4
MAX_VAL_SAMPLES=5000

DATA_PATH_PREFIX="spindle_scaling_gpu"
export MODELS_DIR="trained_models_spindle_scaling"

echo "=== Spindle-count scaling experiment (GPU firing rates): n_aff in ${N_AFF_LEVELS[*]} ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "EXPERIMENT_DIR=$EXPERIMENT_DIR"
echo "MODELS_DIR=$MODELS_DIR"
ls -lh "$RAW_DATA_PATH"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv || true

# ----------------------------------------------------------------------------
echo ""
echo "=== [1/2] Building master coefficient index lists (nested, seed=${COEFF_SEED}) ==="
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
echo "=== [2/2] Training n_aff=10..${N_AFF_MAX} (seed=${COEFF_SEED}, training_seed=${TRAINING_SEED}) ==="
for K in "${N_AFF_LEVELS[@]}"; do
    DONE_MARKER="${EXPERIMENT_DIR}/.done_naff${K}"
    if [[ -f "$DONE_MARKER" ]]; then
        echo "--- n_aff=${K}: already trained, skipping (remove $DONE_MARKER to redo) ---"
        continue
    fi

    CHANNELS_K=$((2 * K))
    BATCH_SIZE_K=$(python3 -c "print(max(${BATCH_SIZE_MIN}, min(${BATCH_SIZE_MAX}, ${BATCH_CHANNELS_TARGET} // ${CHANNELS_K})))")
    echo "--- n_aff=${K}: batch_size=${BATCH_SIZE_K} (channels=${CHANNELS_K}) ---"

    # Slice this level's prefix off the master nested permutation (no re-sampling,
    # so n_aff=K's channels stay a strict subset of n_aff=(K+10)'s, as in the old script)
    LEVEL_IA_SAMPLED="${EXPERIMENT_DIR}/sampled_coefficients_i_a_${K}_${COEFF_SEED}.csv"
    LEVEL_II_SAMPLED="${EXPERIMENT_DIR}/sampled_coefficients_ii_${K}_${COEFF_SEED}.csv"
    python3 - <<PYEOF
import sys
sys.path.insert(0, "${REPO_ROOT}")
import csv
import numpy as np
from utils.spindle_FR_helper import load_sampled_coefficients

k = ${K}
n_max = ${N_AFF_MAX}
seed = ${COEFF_SEED}
out_dir = "${EXPERIMENT_DIR}"
out_paths = {"i_a": "${LEVEL_IA_SAMPLED}", "ii": "${LEVEL_II_SAMPLED}"}

for coeff_type, out_path in out_paths.items():
    master_path = f"{out_dir}/master_sampled_coefficients_{coeff_type}_{n_max}_{seed}.csv"
    master = load_sampled_coefficients(master_path)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["muscle", "coefficients"])
        for muscle, idx in master.items():
            # write as a numpy array (not a plain list) so the on-disk format matches
            # what load_sampled_coefficients expects to parse back (space-separated,
            # no commas) -- same convention the master file itself uses
            writer.writerow([muscle, np.array(idx[:k])])
PYEOF

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
BATCH_SIZE: ${BATCH_SIZE_K}
spindle_dataset: True
build_fc: False
layer_norm: False
CHUNK_SIZE: ${CHUNK_SIZE}
BUFFER_CHUNKS: ${BUFFER_CHUNKS}
MAX_VAL_SAMPLES: ${MAX_VAL_SAMPLES}
EOF

    LEVEL_START=$SECONDS
    python train/train_model_gpu.py \
        --raw_data_path "$RAW_DATA_PATH" \
        --data_dir "$EXPERIMENT_DIR" \
        --seeds "$COEFF_SEED" \
        --training_seeds "$TRAINING_SEED" \
        --n_aff "$K" \
        --i_a_coeff_path "$IA_COEFF_PATH" \
        --ii_coeff_path "$II_COEFF_PATH" \
        --i_a_sampled_coeff_path "$LEVEL_IA_SAMPLED" \
        --ii_sampled_coeff_path "$LEVEL_II_SAMPLED" \
        --device "$DEVICE" \
        --base_config "$CFG"
    LEVEL_ELAPSED=$((SECONDS - LEVEL_START))

    touch "$DONE_MARKER"
    echo "--- n_aff=${K}: done. This model took $(format_duration $LEVEL_ELAPSED). Total script running time: $(format_duration $SECONDS) ---"
done

echo ""
echo "=== Spindle-count scaling experiment (GPU) complete. Total running time: $(format_duration $SECONDS) ==="
echo "=== Models under: ${DATA_DIR}/${MODELS_DIR} ==="
