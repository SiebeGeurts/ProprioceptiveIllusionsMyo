"""
Streaming, memory-bounded replacement for `SpindleDataset` (train/new_spindle_dataset.py).

`SpindleDataset` reads the whole `data`/`labels` arrays out of the training HDF5
file in one shot (`datafile["data"][()]`), so the full dataset -- and the 90/10
train/val split of it -- has to fit in RAM for the whole training run. As the
collected dataset grows past what fits in memory, that stops being possible.

`ChunkedSpindleDataset` keeps the file on disk and only ever materializes a
bounded slice of it:
  - training data is served from a shuffle buffer of `buffer_chunks` HDF5
    chunks (~`buffer_chunks * chunk_size` rows) at a time, refilled from disk
    as batches are consumed, so resident memory stays constant no matter how
    many rows the file has.
  - the validation split is read once and kept resident (it's read on every
    `val_steps` during training for early stopping, so re-reading it from disk
    each time would be slow), but capped at `max_val_samples` rows -- a fixed
    random subsample -- so it doesn't become the new memory bottleneck as the
    dataset grows.

It exposes the same `next_trainbatch` / `next_valbatch` / `path_to_data`
interface `Trainer` (train/train_model_utils.py) already uses, so it is a
drop-in replacement wherever `SpindleDataset` was used for training.

Note on RNG use: `model_definitions.py`'s `_set_seed(training_seed)` reseeds
the *global* numpy/torch RNGs right before each training run, and the
original `next_trainbatch` shuffled with `np.random.permutation` (the global
RNG) so that different `training_seed`s reshuffle the same data differently.
Per-epoch shuffling here (`_refill_buffer`) deliberately uses that same global
RNG to preserve that behavior. The one-time validation subsample in
`_load_val_data` is different: it runs once per `seed` at construction time,
before `training_seed` is set, and must stay identical across every
`training_seed` that reuses this dataset instance -- so it uses its own
`seed`-scoped `np.random.Generator` instead.
"""

import h5py
import numpy as np
import torch


class ChunkedSpindleDataset:
    def __init__(
        self,
        path_to_data,
        dataset_type="train",
        key="spindle_firing",
        task="letter_recognition",
        new_size=None,
        start_end_idx=None,
        chunk_size=None,
        buffer_chunks=4,
        val_fraction=0.1,
        max_val_samples=5000,
        seed=0,
        verbatim=True,
    ):
        self.path_to_data = path_to_data
        self.dataset_type = dataset_type
        self.key = key
        self.task = task
        self.new_size = new_size
        self.buffer_chunks = buffer_chunks
        self.rng = np.random.default_rng(seed)

        # mirrors Dataset.__init__ (train/train_model_utils.py) -- Trainer
        # picks its loss criterion off this attribute
        if task == "letter_recognition":
            self.ground_truth = "label"
        elif task in (
            "letter_reconstruction",
            "letter_reconstruction_joints",
            "letter_reconstruction_joints_vel",
        ):
            self.ground_truth = "endeffector_coords"
        elif task in ("elbow_flex", "elbow_flex_joints"):
            self.ground_truth = "end_effector_coord"

        self._file = None
        self._buffer_data = self._buffer_labels = None
        self._buffer_pos = 0
        self._chunk_order = None
        self._chunk_cursor = 0
        self._test_data = self._test_labels = None

        with h5py.File(self.path_to_data, "r") as f:
            total_rows = f["data"].shape[0]
            native_chunks = f["data"].chunks
        # default to the file's own on-disk chunk size so reads land on chunk
        # boundaries instead of straddling them
        self.chunk_size = chunk_size or (native_chunks[0] if native_chunks else 2048)

        start, end = start_end_idx if start_end_idx is not None else (0, total_rows)

        if dataset_type == "train":
            n = end - start
            num_train = int((1 - val_fraction) * n)
            self.train_start, self.train_end = start, start + num_train
            self.val_start, self.val_end = start + num_train, end
            self.train_size = self.train_end - self.train_start
            self.val_data, self.val_labels = self._load_val_data(max_val_samples)
            self.val_size = self.val_data.shape[0]
        elif dataset_type == "test":
            self.test_start, self.test_end = start, end
            self.test_size = end - start

        if verbatim:
            print(
                f"ChunkedSpindleDataset -> {dataset_type} split ready over "
                f"rows [{start}:{end}] (chunk_size={self.chunk_size}, "
                f"buffer_chunks={self.buffer_chunks})"
            )

    # -- disk access ----------------------------------------------------

    def _file_handle(self):
        if self._file is None:
            self._file = h5py.File(self.path_to_data, "r")
        return self._file

    def _read_rows(self, f, lo, hi):
        data = torch.from_numpy(f["data"][lo:hi])
        labels = torch.from_numpy(f["labels"][lo:hi])
        if self.task == "letter_reconstruction_joints_vel":
            sample_rate = 240
            velocities = np.gradient(labels.numpy(), 1 / sample_rate, axis=1)
            labels = torch.cat((labels, torch.from_numpy(velocities)), dim=2)
        return data, labels

    def _load_val_data(self, max_val_samples):
        f = self._file_handle()
        full_val_size = self.val_end - self.val_start
        if full_val_size <= max_val_samples:
            return self._read_rows(f, self.val_start, self.val_end)

        # fixed random subsample so validation memory doesn't grow with the dataset
        idx = np.sort(
            self.rng.choice(full_val_size, size=max_val_samples, replace=False)
        )
        data_parts, label_parts = [], []
        for lo in range(0, len(idx), self.chunk_size):
            block = idx[lo : lo + self.chunk_size]
            block_lo, block_hi = int(block[0]), int(block[-1]) + 1
            d, l = self._read_rows(
                f, self.val_start + block_lo, self.val_start + block_hi
            )
            offsets = block - block_lo
            data_parts.append(d[offsets])
            label_parts.append(l[offsets])
        return torch.cat(data_parts), torch.cat(label_parts)

    # -- streaming training buffer --------------------------------------

    def _refill_buffer(self):
        """Pull the next `buffer_chunks` chunks (in shuffled order) off disk."""
        f = self._file_handle()
        if self._chunk_order is None or self._chunk_cursor >= len(self._chunk_order):
            n_chunks = int(np.ceil(self.train_size / self.chunk_size))
            # global RNG on purpose -- see module docstring
            self._chunk_order = np.random.permutation(n_chunks)
            self._chunk_cursor = 0

        data_parts, label_parts = [], []
        for _ in range(self.buffer_chunks):
            if self._chunk_cursor >= len(self._chunk_order):
                break
            chunk_id = int(self._chunk_order[self._chunk_cursor])
            self._chunk_cursor += 1
            lo = self.train_start + chunk_id * self.chunk_size
            hi = min(lo + self.chunk_size, self.train_end)
            d, l = self._read_rows(f, lo, hi)
            data_parts.append(d)
            label_parts.append(l)

        data = torch.cat(data_parts)
        labels = torch.cat(label_parts)
        perm = torch.randperm(data.shape[0])
        self._buffer_data = data[perm]
        self._buffer_labels = labels[perm]
        self._buffer_pos = 0

    def next_trainbatch(self, batch_size, step=0, flag=False):
        if step == 0:
            # start of a new epoch: reshuffle chunk order and refill from scratch
            self._chunk_order = None
            self._chunk_cursor = 0
            self._refill_buffer()

        if self._buffer_pos + batch_size > self._buffer_data.shape[0]:
            leftover_data = self._buffer_data[self._buffer_pos :]
            leftover_labels = self._buffer_labels[self._buffer_pos :]
            self._refill_buffer()
            self._buffer_data = torch.cat([leftover_data, self._buffer_data])
            self._buffer_labels = torch.cat([leftover_labels, self._buffer_labels])

        lo, hi = self._buffer_pos, self._buffer_pos + batch_size
        batch_data = self._buffer_data[lo:hi]
        batch_labels = self._buffer_labels[lo:hi]
        self._buffer_pos = hi
        return batch_data, batch_labels

    def next_valbatch(self, batch_size, type="val", step=0, flag=False):
        if type == "val":
            data, labels = self.val_data, self.val_labels
        elif type == "test":
            if self._test_data is None:
                f = self._file_handle()
                self._test_data, self._test_labels = self._read_rows(
                    f, self.test_start, self.test_end
                )
            data, labels = self._test_data, self._test_labels
        lo, hi = batch_size * step, batch_size * (step + 1)
        return data[lo:hi], labels[lo:hi]

    # -- normalization stats, streamed over the training split ----------

    def compute_train_stats(self):
        """Mean/std of train_data (dims 0,3) and train_labels (dims 0,1), computed
        chunk by chunk so the full training split never has to sit in memory."""
        f = self._file_handle()
        data_sum = data_sq = label_sum = label_sq = None
        data_count = label_count = 0
        for lo in range(self.train_start, self.train_end, self.chunk_size):
            hi = min(lo + self.chunk_size, self.train_end)
            d, l = self._read_rows(f, lo, hi)
            d, l = d.float(), l.float()
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
