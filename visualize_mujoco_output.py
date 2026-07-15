"""Visualize one extracted FLAG3D sample and compare MyoSuite output."""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_sample(file_path):
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data[key] = f[key][()]
    return data


def plot_trajectory(ax, coords, label, color):
    coords = np.asarray(coords).T
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Coordinates must be shape (T,3), got {coords.shape}")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], label=label, color=color)
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=color, marker='o', s=40)
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color=color, marker='x', s=40)


def plot_markers(data, file_path, save_path=None):
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Marker trajectories: {file_path}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    raw_shoulder = data.get('shoulder_coords')
    raw_elbow = data.get('elbow_coords')
    raw_wrist = data.get('endeffector_coords_flag3d')
    ik_elbow = data.get('elbow_coords')
    ik_wrist = data.get('endeffector_coords')
    mujoco_elbow = data.get('elbow_coords_mujoco')
    mujoco_wrist = data.get('endeffector_coords_mujoco')

    if raw_shoulder is not None:
        plot_trajectory(ax, raw_shoulder, 'FLAG3D shoulder', 'gray')
    if raw_elbow is not None:
        plot_trajectory(ax, raw_elbow, 'FLAG3D elbow', 'brown')
    if raw_wrist is not None:
        plot_trajectory(ax, raw_wrist, 'FLAG3D wrist', 'black')
    if ik_elbow is not None:
        plot_trajectory(ax, ik_elbow, 'IK elbow', 'blue')
    if ik_wrist is not None:
        plot_trajectory(ax, ik_wrist, 'IK wrist', 'cyan')
    if mujoco_elbow is not None:
        plot_trajectory(ax, mujoco_elbow, 'MyoSuite elbow', 'red')
    if mujoco_wrist is not None:
        plot_trajectory(ax, mujoco_wrist, 'MyoSuite wrist', 'magenta')

    ax.legend()
    if save_path is not None:
        fig.savefig(save_path+ "markers.png", dpi=180)
    plt.show()


def plot_muscle_lengths(data, save_path=None):
    muscle_lengths = data.get('muscle_lengths')
    if muscle_lengths is None:
        print('No muscle_lengths dataset found in file.')
        return

    muscle_lengths = np.asarray(muscle_lengths)
    muscles = [
        'CORB', 'DELT1', 'DELT2', 'DELT3', 'INFSP', 'LAT1', 'LAT2', 'LAT3',
        'PECM1', 'PECM2', 'PECM3', 'SUBSC', 'SUPSP', 'TMAJ', 'TMIN',
        'ANC', 'BIClong', 'BICshort', 'BRA', 'BRD', 'ECRL', 'PT',
        'TRIlat', 'TRIlong', 'TRImed'
    ]

    num_muscles = muscle_lengths.shape[0]
    fig, axs = plt.subplots(min(5, num_muscles), min(5, num_muscles), figsize=(16, 12))
    axs = axs.flatten()
    for i in range(min(num_muscles, len(axs))):
        axs[i].plot(muscle_lengths[i], label=muscles[i])
        axs[i].set_title(muscles[i])
        axs[i].legend()
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path+ "muscle_lengths.png", dpi=180)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize a FLAG3D HDF5 sample and compare MyoSuite output.')
    parser.add_argument('file_path', type=str, help='Path to a sample HDF5 file.')
    parser.add_argument('--save', type=str, default=None, help='Optional path to save the figure.')
    args = parser.parse_args()

    data = load_sample(args.file_path)
    plot_markers(data, args.file_path, save_path=args.save)
    plot_muscle_lengths(data, save_path=args.save)


if __name__ == '__main__':
    main()
