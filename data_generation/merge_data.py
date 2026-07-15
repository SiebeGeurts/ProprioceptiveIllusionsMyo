import argparse
import h5py
import numpy as np
import sys

def shape(data):
    return np.asarray(data).shape

def merge_data(path_1, path_2, num_samples):
    endeffector_coords = []
    joint_coords = []
    spindle_info = []
    with h5py.File(path_1, 'r') as f:
        endeffector_coords.append(f['endeffector_coords'][()])
        joint_coords.append(f['joint_coords'][()])
        spindle_info.append(f['spindle_info'][()])
        print(shape(f['spindle_info' ][()]))
    with h5py.File(path_2, 'r') as f:
        endeffector_coords.append(f['endeffector_coords'][:num_samples])
        joint_coords.append(f['joint_coords'][:num_samples])
        muscle_lengths = f['muscle_coords'][:num_samples] # (num_samples, 25, 1272)
        # spindle_info.append(f['spindle_info'][:num_samples])

    flag3d_len = np.asarray(endeffector_coords[0]).shape[-1]
    pcr_len = np.asarray(endeffector_coords[1]).shape[-1]

    print(shape(muscle_lengths))
    # If dimensions do not match, slice to fix
    if pcr_len != flag3d_len:
        endeffector_coords[1] = endeffector_coords[1][:,:,:flag3d_len]
        joint_coords[1] = joint_coords[1][:,:,:flag3d_len]
        muscle_lengths = muscle_lengths[:,:,:flag3d_len]

    vel_inputs = np.gradient(muscle_lengths, 1 / 60, axis=-1)
    print(f"PCR Max vel: {np.max(vel_inputs)}")
    spindle_info.append(np.stack((muscle_lengths, vel_inputs), axis=-1))

    endeffector_coords = np.concatenate(endeffector_coords, axis=0)
    joint_coords = np.concatenate(joint_coords, axis=0)
    spindle_info = np.concatenate(spindle_info, axis=0)

    return endeffector_coords, joint_coords, spindle_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--flag3d", type=str)
    parser.add_argument("--pcr", type=str)
    parser.add_argument("--num_samples", type=int, default=50_000)

    params = parser.parse_args()
    
    endeffector_coords, joint_coords, spindle_info = merge_data(params.flag3d, params.pcr, params.num_samples)

    with h5py.File(params.save_path, 'w') as f:
        f.create_dataset('endeffector_coords', data=endeffector_coords)
        f.create_dataset('joint_coords', data=joint_coords)
        f.create_dataset('spindle_info', data=spindle_info)
        
    print("done!")
      