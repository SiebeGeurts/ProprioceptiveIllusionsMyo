import argparse
import numpy as np
import h5py
from scipy.signal import savgol_filter


def smooth_data(input_path, output_path, batch_size=10000):
    with h5py.File(input_path, 'r') as f, h5py.File(output_path, 'w') as out:
        spindle_info = f['spindle_info']
        num_samples, num_muscles, num_timepoints, _ = spindle_info.shape
        dtype = spindle_info.dtype

        spindle_info_ds = out.create_dataset(
            'spindle_info', shape=(num_samples, num_muscles, num_timepoints, 2), dtype=dtype
        )
        muscle_lengths_ds = out.create_dataset(
            'muscle_lengths', shape=(num_samples, num_muscles, num_timepoints), dtype=dtype
        )
        muscle_velocities_ds = out.create_dataset(
            'muscle_velocities', shape=(num_samples, num_muscles, num_timepoints), dtype=dtype
        )
        muscle_accelerations_ds = out.create_dataset(
            'muscle_accelerations', shape=(num_samples, num_muscles, num_timepoints), dtype=dtype
        )

        # process in batches along the sample axis -- each sample's smoothing is
        # independent of every other sample, and loading/smoothing/stacking the
        # full (N, 25, T, 2) array at once is what was blowing up memory
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            print(f"Processing batch {start}-{end}...")
            spindle_batch = spindle_info[start:end]

            length = spindle_batch[:, :, :, 0]
            velocity = spindle_batch[:, :, :, 1]
            smoothed_velocity = savgol_filter(velocity, 31, 1, axis=2)
            smoothed_acceleration = np.gradient(smoothed_velocity, 1/240, axis=2)

            muscle_lengths_ds[start:end] = length
            muscle_velocities_ds[start:end] = smoothed_velocity
            muscle_accelerations_ds[start:end] = smoothed_acceleration
            spindle_info_ds[start:end] = np.stack([length, smoothed_velocity], axis=3)

        f.copy('endeffector_coords', out)
        f.copy('joint_coords', out)

    with h5py.File(output_path, 'r') as f:
        print(list(f.keys()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=10000, help="Number of samples to process per batch")

    params = parser.parse_args()

    smooth_data(params.input_path, params.output_path, batch_size=params.batch_size)
