"""
Streaming raw-PCR dataset that computes spindle firing rates on GPU per
batch, instead of reading them from a precomputed per-n_aff HDF5 file.

Structurally mirrors train/chunked_dataset.py's ChunkedSpindleDataset (same
shuffle-buffer / capped-val-cache design, same next_trainbatch/
next_valbatch/compute_train_stats interface, so Trainer in
train/train_model_utils.py needs no changes) -- the difference is *what* is
buffered and *when* the transfer function runs:

  - ChunkedSpindleDataset buffers precomputed spindle_FR channels (2*n_aff
    channels per row), computed once offline and re-read from disk for
    every epoch.
  - RawPCRDataset buffers the three raw per-muscle signals (25 channels
    each, independent of n_aff) and computes the spindle firing rates for a
    batch only when that batch is actually served, on GPU, via
    utils.spindle_FR_gpu.compute_spindle_FR_gpu. Buffer memory is therefore
    small and constant across every n_aff level of a scaling sweep.

Coefficients are swappable via set_coefficients() so a single instance (and
its shuffle buffer / val cache, both raw and therefore n_aff-independent)
could be reused across multiple n_aff levels without re-reading the raw file
from scratch -- not wired up that way in train/train_model_gpu.py (each
level runs as its own process, matching train/train_model.py's convention),
but the method is there if that changes later.
"""

import h5py
import numpy as np
import torch

from utils.spindle_FR_gpu import compute_spindle_FR_gpu


class RawPCRDataset:
    def __init__(
        self,
        path_to_data,
        dataset_type="train",
        task="letter_reconstruction_joints",
        start_end_idx=None,
        chunk_size=2048,
        buffer_chunks=4,
        val_fraction=0.1,
        max_val_samples=5000,
        seed=0,
        device="cuda:0",
        optimal_lengths=None,
        coeff_tensors=None,
        verbatim=True,
    ):
        self.path_to_data = path_to_data
        self.dataset_type = dataset_type
        self.task = task
        self.buffer_chunks = buffer_chunks
        self.chunk_size = chunk_size
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)

        if task in (
            "letter_reconstruction",
            "letter_reconstruction_joints",
            "letter_reconstruction_joints_vel",
        ):
            self.ground_truth = "endeffector_coords"
        else:
            raise ValueError(f"RawPCRDataset does not support task={task!r} yet")

        assert optimal_lengths is not None, "optimal_lengths is required"
        self.optimal_lengths = torch.as_tensor(
            optimal_lengths, dtype=torch.float32, device=self.device
        ).view(1, -1, 1)

        self.coeffs = None
        self.n_channels = None
        if coeff_tensors is not None:
            self.set_coefficients(coeff_tensors)

        self._file = None
        self._buffer_raw = None  # (lengths, velocities, accelerations)
        self._buffer_labels = None
        self._buffer_pos = 0
        self._chunk_order = None
        self._chunk_cursor = 0
        self._test_raw = self._test_labels = None

        with h5py.File(self.path_to_data, "r") as f:
            total_rows = f["muscle_lengths"].shape[0]

        start, end = start_end_idx if start_end_idx is not None else (0, total_rows)

        if dataset_type == "train":
            n = end - start
            num_train = int((1 - val_fraction) * n)
            self.train_start, self.train_end = start, start + num_train
            self.val_start, self.val_end = start + num_train, end
            self.train_size = self.train_end - self.train_start
            self.val_raw, self.val_labels = self._load_val_raw(max_val_samples)
            self.val_size = self.val_raw[0].shape[0]
        elif dataset_type == "test":
            self.test_start, self.test_end = start, end
            self.test_size = end - start

        if verbatim:
            print(
                f"RawPCRDataset -> {dataset_type} split ready over "
                f"rows [{start}:{end}] (chunk_size={self.chunk_size}, "
                f"buffer_chunks={self.buffer_chunks})"
            )

    # -- coefficients ---------------------------------------------------

    def set_coefficients(self, coeff_tensors):
        """Swap in a new (k_l, k_v, e_v, k_a, k_c, max_rate) coefficient set,
        e.g. between n_aff levels of a scaling sweep. Does not touch the
        buffered raw data or val cache -- only batches served afterwards are
        affected, and compute_train_stats() must be re-run after this since
        it depends on the active coefficients."""
        self.coeffs = {k: v.to(self.device) for k, v in coeff_tensors.items()}
        self.n_channels = self.coeffs["k_l"].shape[0]

    # -- disk access ------------------------------------------------------

    def _file_handle(self):
        if self._file is None:
            self._file = h5py.File(self.path_to_data, "r")
        return self._file

    def _read_raw_rows(self, f, lo, hi):
        lengths = torch.from_numpy(f["muscle_lengths"][lo:hi])
        velocities = torch.from_numpy(f["muscle_velocities"][lo:hi])
        accelerations = torch.from_numpy(f["muscle_accelerations"][lo:hi])

        coords = np.transpose(f["endeffector_coords"][lo:hi], (0, 2, 1))
        joints = np.transpose(f["joint_coords"][lo:hi], (0, 2, 1))
        labels = torch.from_numpy(
            np.concatenate((coords, joints), axis=2).astype(np.float32)
        )
        if self.task == "letter_reconstruction_joints_vel":
            sample_rate = 240
            velocities_labels = np.gradient(labels.numpy(), 1 / sample_rate, axis=1)
            labels = torch.cat((labels, torch.from_numpy(velocities_labels)), dim=2)

        return (lengths, velocities, accelerations), labels

    def _compute_fr(self, raw):
        lengths, velocities, accelerations = raw
        assert self.coeffs is not None, "call set_coefficients() before serving batches"
        return compute_spindle_FR_gpu(
            lengths.to(self.device),
            velocities.to(self.device),
            accelerations.to(self.device),
            self.optimal_lengths,
            self.coeffs,
        )

    def _load_val_raw(self, max_val_samples):
        f = self._file_handle()
        full_val_size = self.val_end - self.val_start
        if full_val_size <= max_val_samples:
            return self._read_raw_rows(f, self.val_start, self.val_end)

        # fixed random subsample so validation memory doesn't grow with the dataset
        idx = np.sort(
            self.rng.choice(full_val_size, size=max_val_samples, replace=False)
        )
        length_parts, vel_parts, accel_parts, label_parts = [], [], [], []
        for lo in range(0, len(idx), self.chunk_size):
            block = idx[lo : lo + self.chunk_size]
            block_lo, block_hi = int(block[0]), int(block[-1]) + 1
            (l, v, a), labels = self._read_raw_rows(
                f, self.val_start + block_lo, self.val_start + block_hi
            )
            offsets = block - block_lo
            length_parts.append(l[offsets])
            vel_parts.append(v[offsets])
            accel_parts.append(a[offsets])
            label_parts.append(labels[offsets])
        raw = (torch.cat(length_parts), torch.cat(vel_parts), torch.cat(accel_parts))
        return raw, torch.cat(label_parts)

    # -- streaming training buffer ----------------------------------------

    def _refill_buffer(self):
        """Pull the next `buffer_chunks` chunks (in shuffled order) off disk."""
        f = self._file_handle()
        if self._chunk_order is None or self._chunk_cursor >= len(self._chunk_order):
            n_chunks = int(np.ceil(self.train_size / self.chunk_size))
            # global RNG on purpose -- mirrors ChunkedSpindleDataset, see its docstring
            self._chunk_order = np.random.permutation(n_chunks)
            self._chunk_cursor = 0

        length_parts, vel_parts, accel_parts, label_parts = [], [], [], []
        for _ in range(self.buffer_chunks):
            if self._chunk_cursor >= len(self._chunk_order):
                break
            chunk_id = int(self._chunk_order[self._chunk_cursor])
            self._chunk_cursor += 1
            lo = self.train_start + chunk_id * self.chunk_size
            hi = min(lo + self.chunk_size, self.train_end)
            (l, v, a), labels = self._read_raw_rows(f, lo, hi)
            length_parts.append(l)
            vel_parts.append(v)
            accel_parts.append(a)
            label_parts.append(labels)

        lengths = torch.cat(length_parts)
        velocities = torch.cat(vel_parts)
        accelerations = torch.cat(accel_parts)
        labels = torch.cat(label_parts)
        perm = torch.randperm(lengths.shape[0])
        self._buffer_raw = (lengths[perm], velocities[perm], accelerations[perm])
        self._buffer_labels = labels[perm]
        self._buffer_pos = 0

    def next_trainbatch(self, batch_size, step=0, flag=False):
        if step == 0:
            # start of a new epoch: reshuffle chunk order and refill from scratch
            self._chunk_order = None
            self._chunk_cursor = 0
            self._refill_buffer()

        if self._buffer_pos + batch_size > self._buffer_raw[0].shape[0]:
            leftover_raw = tuple(x[self._buffer_pos :] for x in self._buffer_raw)
            leftover_labels = self._buffer_labels[self._buffer_pos :]
            self._refill_buffer()
            self._buffer_raw = tuple(
                torch.cat([lo, hi]) for lo, hi in zip(leftover_raw, self._buffer_raw)
            )
            self._buffer_labels = torch.cat([leftover_labels, self._buffer_labels])

        lo, hi = self._buffer_pos, self._buffer_pos + batch_size
        batch_raw = tuple(x[lo:hi] for x in self._buffer_raw)
        batch_labels = self._buffer_labels[lo:hi]
        self._buffer_pos = hi

        batch_data = self._compute_fr(batch_raw)
        return batch_data, batch_labels

    def next_valbatch(self, batch_size, type="val", step=0, flag=False):
        if type == "val":
            raw, labels = self.val_raw, self.val_labels
        elif type == "test":
            if self._test_raw is None:
                f = self._file_handle()
                self._test_raw, self._test_labels = self._read_raw_rows(
                    f, self.test_start, self.test_end
                )
            raw, labels = self._test_raw, self._test_labels
        lo, hi = batch_size * step, batch_size * (step + 1)
        batch_raw = tuple(x[lo:hi] for x in raw)
        batch_labels = labels[lo:hi]
        batch_data = self._compute_fr(batch_raw)
        return batch_data, batch_labels

    # -- normalization stats, streamed over the training split -------------

    def compute_train_stats(self):
        """Mean/std of the *computed firing rates* (not the raw signals),
        streamed using the currently-set coefficients -- must be called
        after set_coefficients() for the level being trained (Trainer.train()
        already does this at the right time).

        Deliberately does NOT reuse self.chunk_size for this: that knob is
        sized for the raw buffer, whose rows are a fixed 25 channels
        regardless of n_aff. A firing-rate tensor is (rows, n_channels,
        muscles, time) and n_channels scales with n_aff, so a
        self.chunk_size-sized *firing-rate* chunk can be far bigger than any
        batch ever used in training (e.g. ~70GB at n_aff=150 with
        self.chunk_size=2048) and OOM here even when training itself fits.
        Streams instead in its own row count, capped by a fixed memory
        budget so it scales down automatically as n_channels grows.
        """
        assert self.coeffs is not None, "call set_coefficients() before compute_train_stats()"
        f = self._file_handle()
        n_muscles, n_time = f["muscle_lengths"].shape[1], f["muscle_lengths"].shape[2]
        stats_target_bytes = 1 * 1024**3  # ~1GiB per firing-rate chunk, independent of self.chunk_size
        bytes_per_row = max(1, self.n_channels * n_muscles * n_time * 4)
        stats_chunk_size = max(1, min(self.chunk_size, stats_target_bytes // bytes_per_row))

        data_sum = data_sq = label_sum = label_sq = None
        data_count = label_count = 0
        for lo in range(self.train_start, self.train_end, stats_chunk_size):
            hi = min(lo + stats_chunk_size, self.train_end)
            raw, labels = self._read_raw_rows(f, lo, hi)
            d = self._compute_fr(raw).float()
            l = labels.float()
            d_sum = d.sum(dim=[0, 3], keepdim=True)
            d_sq = (d**2).sum(dim=[0, 3], keepdim=True)
            l_sum = l.sum(dim=[0, 1], keepdim=True)
            l_sq = (l**2).sum(dim=[0, 1], keepdim=True)
            if data_sum is None:
                data_sum, data_sq, label_sum, label_sq = d_sum, d_sq, l_sum, l_sq
            else:
                data_sum += d_sum
                data_sq += d_sq
                label_sum += l_sum
                label_sq += l_sq
            data_count += d.shape[0] * d.shape[3]
            label_count += l.shape[0] * l.shape[1]
            # drop references to this chunk's (rows, n_channels, muscles, time)
            # tensors before the next iteration allocates its own -- otherwise
            # the old and new chunk's tensors are briefly alive at once (the
            # loop variables aren't rebound until the next assignment
            # finishes evaluating), roughly doubling peak memory for no reason
            del d, raw, labels, l

        data_mean = data_sum / data_count
        data_var = (data_sq / data_count - data_mean**2).clamp_min(0)
        label_mean = label_sum / label_count
        label_var = (label_sq / label_count - label_mean**2).clamp_min(0)
        return data_mean, data_var.sqrt(), label_mean, label_var.sqrt()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()
