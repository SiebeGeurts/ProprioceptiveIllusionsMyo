#!/bin/bash
# ============================================================================
# VIBRATION TEST DATA FOR THE SPINDLE-SCALING GPU SWEEP
#
# Generates vibration-illusion test data (perceived elbow-angle offset vs.
# vibration frequency) for every (n_aff, seed) model trained by
# spindle_scaling_experiment_gpu.sh -- that sweep only trains, it never runs
# vibration testing, so none of this exists yet. spindle_scaling_analysis.ipynb
# (repo root) reads the output of this script.
#
# Two-step pipeline per (n_aff, seed), mirroring test_vibrations/vibrations_Fig4.sh
# but adapted to the scaling sweep's per-n_aff nested coefficient sampling
# (Fig4.sh assumes one fixed n_aff=5):
#
#   1. Generate a small precomputed ES3D spindle_FR file, reusing the EXACT
#      nested-sampled coefficient indices (sampled_coefficients_i_a_<n_aff>_<seed>.csv
#      under EXPERIMENT_DIR) the model was actually trained on -- via the
#      existing offline CPU generator (extract_data/generate_train_test_data.py's
#      process_data). ES3D is tiny (~100 trials), so this is fast even on CPU;
#      no need for a GPU vibration pipeline. (SpindleDataset, which
#      test_model_vib_multipleFs.py reads this file with, ignores its `key`
#      argument and always reads the "data"/"labels" fields process_data
#      writes -- confirmed by reading train/new_spindle_dataset.py -- so this
#      file is compatible even though its on-disk layout was designed for
#      training data, not this "spindle_info"-keyed test path.)
#   2. Run test_vibrations/test_model_vib_multipleFs.py (unchanged) against
#      that file for each of the two muscle groups (triceps, biceps),
#      producing vib_results.h5 in the same directory layout
#      utils/analysis_plots_helper.py's load_vibration_data() already knows
#      how to read.
#
# Skips (n_aff, seed) combinations with no trained model yet, and skips
# vib_results.h5 files that already exist (test_model_vib_multipleFs.py's own
# convention, unless FORCE_RECOMPUTE=1 is exported) -- safe to re-run as more
# of the sweep's models finish training.
#
# Only run this once the scaling GPU sweep's process has exited -- both use
# the GPU, and the sweep needs all of it (see the OOM work earlier in this
# project's history).
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="/media/data16/siebe/datasets"
EXPERIMENT_DIR="${DATA_DIR}/spindle_scaling_experiment_gpu"
ROOT_DIR="/media/data16/siebe"
export MODELS_DIR="trained_models_spindle_scaling"

DATA_PATH_PREFIX="spindle_scaling_gpu"
TASK="letter_reconstruction_joints"
N_AFF_LEVELS=(15 30 45 60 75 90 105 120 135 150)
SEED_PAIRS=("0:0" "1:1" "2:2" "3:3" "4:4")   # coeff_seed:training_seed, same convention as the sweep
# (seeds 1/3 trained on a different machine, merged into the same MODEL_ROOT/EXPERIMENT_DIR tree)

IA_COEFF_PATH="${DATA_DIR}/coefficients_i_a.csv"
II_COEFF_PATH="${DATA_DIR}/coefficients_ii.csv"
ES3D_RAW="${DATA_DIR}/ES3D.hdf5"

MUSCLES_TO_VIB_LIST=(
    "TRIlat TRIlong TRImed"
    "BIClong BICshort"
)
VIB_FREQS="0 20 40 60 80 100 110 130 150 170 190"
VIB_START=200
VIB_END=900
vib_exp_id="vib_vary_multipleFs_vib_ia_only"

FORCE_FLAG=""
if [[ "${FORCE_RECOMPUTE:-}" == "1" ]]; then
    FORCE_FLAG="--force"
fi

echo "=== Vibration test data generation for the spindle-scaling GPU sweep ==="
echo "N_AFF_LEVELS=${N_AFF_LEVELS[*]}"
echo "SEED_PAIRS=${SEED_PAIRS[*]}"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv || true

for K in "${N_AFF_LEVELS[@]}"; do
    for PAIR in "${SEED_PAIRS[@]}"; do
        COEFF_SEED="${PAIR%%:*}"
        TRAINING_SEED="${PAIR##*:}"

        MODEL_PATH="${ROOT_DIR}/${MODELS_DIR}/experiment_causal_flag-pcr_${DATA_PATH_PREFIX}_${K}_${K}_${TASK}/spatiotemporal_4_${COEFF_SEED}_${TRAINING_SEED}"
        if [[ ! -f "${MODEL_PATH}/model.ckpt" ]]; then
            echo "--- n_aff=${K} coeff_seed=${COEFF_SEED} training_seed=${TRAINING_SEED}: no trained model yet, skipping ---"
            continue
        fi

        IA_SAMPLED="${EXPERIMENT_DIR}/sampled_coefficients_i_a_${K}_${COEFF_SEED}.csv"
        II_SAMPLED="${EXPERIMENT_DIR}/sampled_coefficients_ii_${K}_${COEFF_SEED}.csv"
        ES3D_PRECOMPUTED="${EXPERIMENT_DIR}/${DATA_PATH_PREFIX}_${COEFF_SEED}_${K}_${K}_ES3D.hdf5"

        if [[ ! -f "$ES3D_PRECOMPUTED" ]]; then
            echo "--- n_aff=${K} coeff_seed=${COEFF_SEED}: generating precomputed ES3D data ---"
            python3 - <<PYEOF
import sys
sys.path.insert(0, "${REPO_ROOT}")
from extract_data.generate_train_test_data import process_data, load_config

config = load_config("extract_data/configs/train_test_data_spindles_extended.yaml")
config.update({
    "num_i_a": ${K},
    "num_ii": ${K},
    "chunk_size": 100,
    "seed": ${COEFF_SEED},
    "i_a_coeff_path": "${IA_COEFF_PATH}",
    "ii_coeff_path": "${II_COEFF_PATH}",
    "i_a_sampled_coeff_path": "${IA_SAMPLED}",
    "ii_sampled_coeff_path": "${II_SAMPLED}",
})
process_data("${ES3D_RAW}", "${ES3D_PRECOMPUTED}", config, num_samples=None)
PYEOF
        fi

        for MUSCLES_TO_VIB in "${MUSCLES_TO_VIB_LIST[@]}"; do
            SAVE_PATH="${MODEL_PATH}/test/ES3D/${vib_exp_id}/$(echo ${MUSCLES_TO_VIB} | tr ' ' '_')"
            mkdir -p "$SAVE_PATH"
            channel_indices=$(seq 0 $((K - 1)))
            echo "--- n_aff=${K} coeff_seed=${COEFF_SEED}: vibrating ${MUSCLES_TO_VIB} ---"
            python test_vibrations/test_model_vib_multipleFs.py \
                --model_path "$MODEL_PATH" \
                --data_path "$ES3D_PRECOMPUTED" \
                --save_path "$SAVE_PATH" \
                --vib_freqs $VIB_FREQS \
                --vib_start $VIB_START \
                --vib_end $VIB_END \
                --muscles_to_vib $MUSCLES_TO_VIB \
                --i_a_sampled_coeff_path "$IA_SAMPLED" \
                --i_a_coeff_path "$IA_COEFF_PATH" \
                --channel_indices $channel_indices \
                $FORCE_FLAG
        done
    done
done

echo ""
echo "=== Done. Results under ${ROOT_DIR}/${MODELS_DIR}/experiment_causal_flag-pcr_${DATA_PATH_PREFIX}_<n_aff>_<n_aff>_${TASK}/spatiotemporal_4_<seed>_<seed>/test/ES3D/${vib_exp_id}/ ==="
