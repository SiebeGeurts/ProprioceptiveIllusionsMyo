import os

# path where the data is stored
PARENT_DIR = "/media/data16/siebe/datasets/"

# path where the data is stored to save the results if different
SAVE_DIR = PARENT_DIR

# override with e.g. `export MODELS_DIR=trained_models_old_coefficients` to
# point a pipeline run at a separate model tree without touching this default
MODELS_DIR = os.environ.get("MODELS_DIR", "trained_models_myo")
