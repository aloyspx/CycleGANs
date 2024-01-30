import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from helpers.transforms import get_transforms
except:
    from transforms import get_transforms


class BraTSDataset(Dataset):
    def __init__(self, dataset_h5py, A_key, B_key, transform_name, subset_keys, n_slices=64, patch_size=128):
        super().__init__()
        self.n_slices = n_slices
        self.transform = get_transforms(transform_name)
        self.dataset = h5py.File(dataset_h5py)
        self.A_key = A_key
        self.B_key = B_key

        self.A_intervals = self.create_volume_intervals([self.dataset[case][self.A_key] for case in subset_keys],
                                                        subset_keys)
        self.B_intervals = self.create_volume_intervals([self.dataset[case][self.B_key] for case in subset_keys],
                                                        subset_keys)

        self.patch_size = [patch_size, patch_size]

    def __getitem__(self, idx):

        d = self.get_slice_from_contiguous_index(idx, self.dataset, self.A_intervals, self.A_key)[:, 0, :, :]

        start_x = np.random.randint(0, d.shape[-2] - self.patch_size[0])
        start_y = np.random.randint(0, d.shape[-1] - self.patch_size[1])

        d = d[:, start_y:start_y + self.patch_size[1], start_x:start_x + self.patch_size[0]]

        A = np.expand_dims(d, axis=[0]).astype(np.float32)
        A = self.transform(data=A)

        d = self.get_slice_from_contiguous_index(idx, self.dataset, self.B_intervals, self.B_key)[:, 0, :, :]

        d = d[:, start_y:start_y + self.patch_size[1], start_x:start_x + self.patch_size[0]]

        B = np.expand_dims(d, axis=[0]).astype(np.float32)
        B = self.transform(data=B)

        return {"A": np.clip(A['data'], -1, 1), "B": np.clip(B['data'], -1, 1)}

    def create_volume_intervals(self, volumes, subset_keys):
        volume_intervals = {}
        start = 0
        for subset_key, volume in zip(subset_keys, volumes):
            num_slices = volume.shape[0]
            if num_slices >= self.n_slices:
                for i in range(num_slices - self.n_slices - 1):
                    end = start + 1
                    volume_intervals[(subset_key, i)] = (start, end)
                    start = end
        return volume_intervals

    def __len__(self):
        return sum(end - start for start, end in self.A_intervals.values())

    def get_slice_from_contiguous_index(self, contiguous_index, volumes, volume_intervals, key):
        for (case, patch_index), (start, _) in volume_intervals.items():
            if start == contiguous_index:
                base_index = patch_index
                return volumes[case][key][base_index:base_index + self.n_slices]
        raise IndexError("Contiguous index out of range")


def setup_dataloaders(dataset_h5py, A_key, B_key, batch_size, num_workers, train_transform="cyclegan"):

    tst_cases = np.load("test.npy").astype(str)
    cases = list(h5py.File(dataset_h5py).keys())

    cases = list(set(cases).difference(set(tst_cases)))

    np.random.shuffle(cases)

    val_cases = cases[0:37]
    trn_cases = cases[37:]

    trn_dataset = BraTSDataset(dataset_h5py, A_key, B_key, train_transform, trn_cases)
    trn_dataloader = DataLoader(trn_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True,
                                pin_memory=True)

    tst_dataset = BraTSDataset(dataset_h5py, A_key, B_key, 'validation', tst_cases)
    tst_dataloader = DataLoader(tst_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=True)

    val_dataset = BraTSDataset(dataset_h5py, A_key, B_key, 'validation', val_cases)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=True)

    return trn_dataloader, val_dataloader, tst_dataloader


if __name__ == "__main__":
    trn_dataloader, val_dataloader, tst_dataloader = setup_dataloaders(
        dataset_h5py="../translation_mbrats_cyclegan.h5",
        A_key='t1', B_key='t2', batch_size=1, num_workers=0)

    import time

    start = time.time()
    for idx, elem in enumerate(tqdm(tst_dataloader)):
        for i in range(elem['A'].shape[2]):
            plt.imshow(elem['A'][0][0][i], cmap="gray")
            plt.title(f"{idx}")
            plt.show()
            plt.imshow(elem['B'][0][0][i], cmap="gray")
            plt.title(f"{idx}")
            plt.show()
    print(f"{time.time() - start} seconds")
