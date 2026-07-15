import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys
import musclemimic_models as mm
import mujoco as mj
import opensim as osim
import pickle
from pathlib import Path
import utils.spindle_FR_helper as spindler

def shape(data):
    return np.asarray(data).shape

def plot_muscle_lengths_subplots(muscle_lengths, trajectory_idx=0, save_path=None, dpi=100):
    """
    Create and save subplots of all muscle lengths over time for a single trajectory.
    
    Parameters
    ----------
    muscle_lengths : np.ndarray
        Array of shape (n_trajectories, n_muscles, n_timepoints) or (1, n_muscles, n_timepoints)
    trajectory_idx : int
        Index of trajectory to plot (default: 0)
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int
        DPI for saved figure (default: 100)
    """
    # Extract single trajectory
    if muscle_lengths.ndim == 4:
        traj_data = muscle_lengths[0, trajectory_idx, :, :]  # (n_muscles, n_timepoints)
    elif muscle_lengths.ndim == 3:
        traj_data = muscle_lengths[trajectory_idx, :, :]  # (n_muscles, n_timepoints)
    elif muscle_lengths.ndim == 2:
        traj_data = muscle_lengths  # (n_muscles, n_timepoints)
    else:
        raise ValueError(f"Expected 2D, 3D or 4D array, got shape {muscle_lengths.shape}")
    
    n_muscles = traj_data.shape[0]
    n_timepoints = traj_data.shape[1]
    timepoints = np.arange(n_timepoints)
    
    # Create subplots grid (5 rows x 5 columns for 25 muscles)
    n_cols = 5
    n_rows = int(np.ceil(n_muscles / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot each muscle
    for muscle_idx in range(n_muscles):
        ax = axes[muscle_idx]
        ax.plot(timepoints, traj_data[muscle_idx, :], 'b-', linewidth=1.5)
        ax.set_title(f'Muscle {muscle_idx}', fontsize=10)
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Length')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for muscle_idx in range(n_muscles, len(axes)):
        axes[muscle_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, format='png', bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_spindle_activity_subplots(spindle_data, trajectory_idx=0, spindle_type='ia', save_path=None, dpi=100):
    """
    Create and save subplots of spindle firing rates over time for a single trajectory.

    Parameters
    ----------
    spindle_data : np.ndarray
        Array containing spindle firing rates. Supported shapes:
        - (n_trajectories, n_muscles, n_spindle_types, n_spindles, n_timepoints)
        - (n_muscles, n_spindle_types, n_spindles, n_timepoints)
        - (n_trajectories, n_muscles, n_spindles, n_timepoints)
        - (n_muscles, n_spindles, n_timepoints)
    trajectory_idx : int
        Index of trajectory to plot (default: 0)
    spindle_type : {'ia', 'ii'}
        The spindle type being plotted. Only used for title and save naming.
    save_path : str or Path, optional
        Path to save the figure. If None, the figure is not saved.
    dpi : int
        DPI for saved figure (default: 100)
    """
    data = np.asarray(spindle_data)
    type_index = 0 if spindle_type.lower() == 'ia' else 1

    if data.ndim == 5:
        # (n_trajectories, n_muscles, n_spindle_types, n_spindles, n_timepoints)
        data = data[trajectory_idx, :, type_index, :, :]
    elif data.ndim == 4:
        # Could be either (n_muscles, n_spindle_types, n_spindles, n_timepoints)
        # or (n_trajectories, n_muscles, n_spindles, n_timepoints).
        if data.shape[1] == 2 and data.shape[2] == 5:
            data = data[:, type_index, :, :]
        else:
            data = data[trajectory_idx, :, :, :]
    elif data.ndim == 3:
        # (n_muscles, n_spindles, n_timepoints)
        data = data
    else:
        raise ValueError(f"Expected 3D, 4D, or 5D array, got shape {data.shape}")

    n_muscles, n_spindles, n_timepoints = data.shape
    if n_muscles != 25:
        raise ValueError(f"Expected 25 muscles, got {n_muscles}")
    if n_spindles != 5:
        raise ValueError(f"Expected 5 spindles per muscle, got {n_spindles}")

    timepoints = np.arange(n_timepoints)
    n_cols = 5
    n_rows = int(np.ceil(n_muscles / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.flatten()

    for muscle_idx in range(n_muscles):
        ax = axes[muscle_idx]
        for spindle_idx in range(n_spindles):
            ax.plot(timepoints, data[muscle_idx, spindle_idx, :], label=f'Sp{spindle_idx+1}', linewidth=1.2)
        ax.set_title(f'Muscle {muscle_idx}', fontsize=10)
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Firing rate')
        ax.grid(True, alpha=0.3)
        if muscle_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    for muscle_idx in range(n_muscles, len(axes)):
        axes[muscle_idx].axis('off')

    plt.suptitle(f'Spindle {spindle_type.upper()} firing rates per muscle', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, format='png', bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig, axes


def generate_spindle_firing_rates_for_trajectory(
    lengths,
    velocities,
    accelerations,
    ia_coeff_path,
    ii_coeff_path,
    trajectory_idx=0,
    optimal_lengths=None,
    coeff_indices=None,
):
    """
    Compute spindle firing rates for a single trajectory using the repository
    spindle transfer-function helper and coefficient CSV files.

    Parameters
    ----------
    lengths, velocities, accelerations : np.ndarray
        Arrays of shape (n_muscles, n_timepoints) or
        (n_trajectories, n_muscles, n_timepoints). If 3D, the selected
        trajectory is used.
    ia_coeff_path, ii_coeff_path : str or Path
        Paths to the Ia and II coefficient CSV files, such as
        newspindledata/coefficients_i_a.csv and newspindledata/coefficients_ii.csv.
    trajectory_idx : int
        Index of the trajectory to use when the input is 3D.
    optimal_lengths : array-like, optional
        Optional optimal fiber lengths used to normalize the inputs.
    coeff_indices : Sequence[int] or None
        Indices into the coefficient lists for the selected afferent channels.
        If None, the first 5 coefficients are used for each type.

    Returns
    -------
    np.ndarray
        Array of shape (n_muscles, 2, n_afferents, n_timepoints) with firing
        rates for Ia and II spindle types.
    """
    lengths = np.asarray(lengths, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    accelerations = np.asarray(accelerations, dtype=float)

    if lengths.ndim == 3:
        lengths = lengths[trajectory_idx]
        velocities = velocities[trajectory_idx]
        accelerations = accelerations[trajectory_idx]
    elif lengths.ndim != 2:
        raise ValueError(
            "Expected lengths/velocities/accelerations to be 2D or 3D; "
            f"got shape {lengths.shape}"
        )

    if optimal_lengths is not None:
        optimal_lengths = np.asarray(optimal_lengths, dtype=float)
        if optimal_lengths.shape[0] != lengths.shape[0]:
            raise ValueError(
                "optimal_lengths must have one value per muscle; "
                f"got {optimal_lengths.shape[0]} for {lengths.shape[0]} muscles"
            )
        lengths_norm = lengths / optimal_lengths[:, None] - 1
        velocities_norm = velocities / optimal_lengths[:, None]
        accelerations_norm = accelerations / optimal_lengths[:, None]
    else:
        lengths_norm = lengths
        velocities_norm = velocities
        accelerations_norm = accelerations

    coeff_paths = {"i_a": ia_coeff_path, "ii": ii_coeff_path}
    coefficients = {
        coeff_type: spindler.load_coefficients(str(path))
        for coeff_type, path in coeff_paths.items()
    }
    for coeff_type, coeffs in coefficients.items():
        print(f"{coeff_type}: {coeffs}")

    if len(coeff_indices) != 5:
        raise ValueError("coeff_indices must contain 5 indices for 5 afferents")

    n_muscles, n_timepoints = lengths_norm.shape
    firing_rates = np.zeros((n_muscles, 2, 5, n_timepoints), dtype=float)

    for muscle_idx in range(n_muscles):
        for type_idx, coeff_type in enumerate(["i_a", "ii"]):
            coeffs = coefficients[coeff_type][muscle_idx]
            for aff_idx, sample_idx in enumerate(coeff_indices):
                selected_coeffs = {
                    key: coeffs[key][sample_idx]
                    for key in ["k_l", "k_v", "e_v", "k_a", "k_c", "max_rate"]
                }
                rates = spindler.clipped_spindle_transfer_function_coeffs(
                    lengths_norm[muscle_idx],
                    velocities_norm[muscle_idx],
                    accelerations_norm[muscle_idx],
                    selected_coeffs,
                )
                firing_rates[muscle_idx, type_idx, aff_idx, :] = rates
    print(rates)
    return firing_rates


def plot_endeffector_trajectory(endeffector_coords, trajectory_idx=0, save_path=None, dpi=100):
    """
    Plot end-effector trajectory in 3D for a single trajectory.

    Parameters
    ----------
    endeffector_coords : np.ndarray
        Array of shape (n_trajectories, 3, n_timepoints) or
        (3, n_timepoints).
    trajectory_idx : int
        Trajectory index to plot when endeffector_coords is 3D.
    save_path : str or Path, optional
        Path to save the figure.
    dpi : int
        DPI for saved figure.
    """
    coords = np.asarray(endeffector_coords)
    if coords.ndim == 3:
        coords = coords[trajectory_idx]
    if coords.ndim == 2 and coords.shape[0] == 3:
        coords = coords.T
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"Expected endeffector_coords shape (3,n) or (n,3), got {coords.shape}"
        )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'b-', linewidth=2, alpha=0.8)
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color='green', s=80, marker='o', label='Start')
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color='red', s=80, marker='s', label='End')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'End-effector trajectory (trajectory {trajectory_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=30, azim=-125)

    if save_path is not None:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, format='png', bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig, ax


def plot_3d_trajectory(shoulder_coords, elbow_coords, endeffector_coords, 
                       joint_coords=None, trajectory_idx=0, save_path=None, dpi=100):
    """
    Create and save a 3D plot of shoulder, elbow, and wrist trajectory over time.
    
    Parameters
    ----------
    shoulder_coords : np.ndarray
        Shoulder coordinates of shape (n_trajectories, n_timepoints, 3) or similar
    elbow_coords : np.ndarray
        Elbow coordinates 
    endeffector_coords : np.ndarray
        End-effector (wrist) coordinates
    joint_coords : np.ndarray, optional
        Joint coordinates for visualization
    trajectory_idx : int
        Index of trajectory to plot (default: 0)
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int
        DPI for saved figure (default: 100)
    """
    # Extract trajectory data - handle different possible shapes
    def extract_traj(coords, idx):
        if coords.ndim == 3:
            return coords[idx]  # (n_timepoints, 3)
        elif coords.ndim == 4:
            return coords[0, idx]  # (n_timepoints, 3)
        else:
            # Assume already in correct shape
            return coords
    
    shoulder = extract_traj(shoulder_coords, trajectory_idx)
    elbow = extract_traj(elbow_coords, trajectory_idx)
    endeff = extract_traj(endeffector_coords, trajectory_idx)
    
    # Transpose if needed (should be n_timepoints x 3)
    if shoulder.shape[1] != 3:
        shoulder = shoulder.T
    if elbow.shape[1] != 3:
        elbow = elbow.T
    if endeff.shape[1] != 3:
        endeff = endeff.T
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the full trajectories
    ax.plot(shoulder[:, 0], shoulder[:, 1], shoulder[:, 2], 
            'r-', alpha=0.6, label='Shoulder', linewidth=2)
    ax.plot(elbow[:, 0], elbow[:, 1], elbow[:, 2], 
            'g-', alpha=0.6, label='Elbow', linewidth=2)
    ax.plot(endeff[:, 0], endeff[:, 1], endeff[:, 2], 
            'b-', alpha=0.6, label='End-effector (Wrist)', linewidth=2)
    
    # Plot start and end points
    ax.scatter(*shoulder[0], color='red', s=100, marker='o', label='Shoulder start')
    ax.scatter(*shoulder[-1], color='red', s=100, marker='s')
    ax.scatter(*elbow[0], color='green', s=100, marker='o')
    ax.scatter(*elbow[-1], color='green', s=100, marker='s')
    ax.scatter(*endeff[0], color='blue', s=100, marker='o')
    ax.scatter(*endeff[-1], color='blue', s=100, marker='s')

    # Plot starting arm segments
    ax.plot([shoulder[0, 0], elbow[0, 0]],
            [shoulder[0, 1], elbow[0, 1]],
            [shoulder[0, 2], elbow[0, 2]],
            color='gray', alpha=0.5, linewidth=2)
    ax.plot([elbow[0, 0], endeff[0, 0]],
            [elbow[0, 1], endeff[0, 1]],
            [elbow[0, 2], endeff[0, 2]],
            color='gray', alpha=0.5, linewidth=2)
    
    # Formatting
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'3D Trajectory - Trajectory {trajectory_idx}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=30, azim=-125)
    
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.zaxis.set_tick_params(labelsize=10)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, format='png', bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax



path_1 = "/media1/siebe/datasets/flag3d_keypoint.pkl"
path_2 = "/media1/siebe/datasets/PCR/pcr_dataset_train.hdf5"
endeffector_coords = []
joint_coords = []
spindle_info = []
muscle_data = []
# with h5py.File("/media1/siebe/datasets/FLAG3D/flag3d_raw_train.hdf5", 'r') as f:
#     print(list(f.keys()))
#     muscle_data.append(f['muscle_lengths'][()])
#     # endeffector_coords.append(f['endeffector_coords'][()])
#     # joint_coords.append(f['joint_coords'][()])
#     # spindle_info.append(f['spindle_info'][()])
# print(np.asarray(muscle_data).shape)


if __name__ == '__main__':
    with h5py.File("/media1/siebe/datasets/flag_pcr_train.hdf5", 'r') as f:
        trajectory_idx = 0
        muscle_lengths = f['muscle_lengths'][trajectory_idx]  # (25, 1152)
        spindle_info = f['spindle_info'][trajectory_idx]      # (25, 1152, 2)
        endeff = f['endeffector_coords'][trajectory_idx]      # (3, 1152)
        optimal_lenghts = np.load("/media1/siebe/ProprioceptiveIllusionsMyo/optimal_lengths.npy")
        velocities = spindle_info[:, :, 1]
  
        accelerations = np.gradient(velocities, 1 / 60, axis=-1)

        plot_muscle_lengths_subplots(muscle_lengths, save_path='figures/muscles.png')
        
        rates = generate_spindle_firing_rates_for_trajectory(
            muscle_lengths,
            velocities,
            accelerations,
            "/media1/siebe/datasets/coefficients_i_a.csv",
            "/media1/siebe/datasets/coefficients_ii.csv",
            optimal_lengths=optimal_lenghts,
            coeff_indices=[0, 1, 2, 3, 4],
        )

        plot_spindle_activity_subplots(rates, spindle_type='ii', save_path='figures/ii_rates.png', dpi=100)
        plot_spindle_activity_subplots(rates, spindle_type='ia', save_path='figures/ia_rates.png', dpi=100)

        plot_endeffector_trajectory(endeff, save_path='figures/trajectory.png', dpi=100)

###########################
# iteration = 0
# print(path_1)
# with h5py.File(path_2, 'r') as f:
#     print(list(f.keys()))
#     for k in list(f.keys()):
#         dset = f[k] # Replace with your dataset key
#         print("Key:",k)
#         print("Shape:", dset.shape)
#         print("Data Type:", dset.dtype)
        
#         # Read a slice of data (avoids overloading RAM)
#         # data_slice = dset[:5] 
#         # print("First 5 elements:", data_slice)
#         print("")
#     # iteration+=1
#     # if iteration==2:
#     #     sys.exit()
#########################################

def inspect_pickle(path: Path, sample_index: int = 0) -> None:
    with path.open("rb") as f:
        data = pickle.load(f)

    print(f"Loaded pickle: {path}")
    print(f"Top-level type: {type(data).__name__}")

    if isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))

        if "split" in data:
            print("split keys:", list(data["split"].keys()))
            for k, v in data["split"].items():
                print(f"  {k}: {len(v)}")

        if "annotations" in data:
            annotations = data["annotations"]
            print("annotations length:", len(annotations))
            if len(annotations) > 0:
                ann = annotations[sample_index]
                print(f"sample index: {sample_index}")
                print(f"sample type: {type(ann).__name__}")
                if isinstance(ann, dict):
                    print("sample keys:", list(ann.keys()))
                    for key in ["keypoint", "label", "keypoint_score", "total_frames", "frame_dir"]:
                        if key in ann:
                            value = ann[key]
                            print(f"  {key}: type={type(value).__name__}, shape={getattr(value, 'shape', None)}, dtype={getattr(value, 'dtype', None)}")
                    if "keypoint" in ann:
                        kp = ann["keypoint"]
                        print("  keypoint min/max:", kp.min(), kp.max())
                        print("  keypoint sample[0,0,8:11]:")
                        print(kp[0, 0, 8:11])
                else:
                    print("sample value:", repr(ann)[:500])
        elif isinstance(data, list):
            print("List length:", len(data))
            if len(data) > sample_index:
                print("sample type:", type(data[sample_index]).__name__)
    else:
        print("Loaded object does not contain a dict at top level.")


# inspect_pickle(Path(path_1))













######### MyoSuite
model,data = mm.load('bimanual')
print(model.actuator("DELT1").length0)
mj.mj_forward(model,data)
print(model.actuator("DELT1").length0)
print(data.tendon("DELT1_tendon").length)
print(data.actuator("DELT1").length)


curr_muscle =  model.actuator("DELT1")
# https://github.com/google-deepmind/mujoco/issues/216
L0 = (curr_muscle.lengthrange[1] - curr_muscle.lengthrange[0]) / (curr_muscle.gainprm[1] - curr_muscle.gainprm[0])
lt = curr_muscle.lengthrange[0] - curr_muscle.gainprm[0]*L0
print(L0, lt)

sys.exit()

print((data.actuator_length[0] - model.tendon_length0[0] + model.actuator_gainprm[0, 5]))


######### OpenSim
openmodel = osim.Model("/media1/siebe/ProprioceptiveIllusionsMyo/MOBL_ARMS_41_seb_writing_pos.osim")

state = openmodel.initSystem()
openmodel.equilibrateMuscles(state)

# Get specific muscle
muscle = openmodel.getMuscles().get("DELT1")


mtu_length = muscle.getLength(state)
fiber_length = muscle.getFiberLength(state)

# Calculate tendon length (in meters)
tendon_length = mtu_length - fiber_length
print(mtu_length,fiber_length)
print(tendon_length)


# ============================================================================
# EXAMPLE USAGE OF NEW FUNCTIONS
# ============================================================================
# Load data and create visualizations
# with h5py.File("/media1/siebe/datasets/FLAG3D/flag3d_raw_train.hdf5", 'r') as f:
#     muscle_lengths = f['muscle_lengths'][()]  # shape (1, 17156, 25, 288)
#     shoulder_coords = f['shoulder_coords'][()]
#     elbow_coords = f['elbow_coords'][()]
#     endeffector_coords = f['endeffector_coords'][()]
#
#     trajectory_idx = 0
#
#     # Plot muscle lengths
#     fig1, axes1 = plot_muscle_lengths_subplots(
#         muscle_lengths, 
#         trajectory_idx=trajectory_idx,
#         save_path='muscle_lengths_subplots.png'
#     )
#
#     # Plot 3D trajectory
#     fig2, ax2 = plot_3d_trajectory(
#         shoulder_coords,
#         elbow_coords,
#         endeffector_coords,
#         trajectory_idx=trajectory_idx,
#         save_path='trajectory_3d.png'
#     )
#
#     plt.show()