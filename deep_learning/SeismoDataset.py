import torch
from torch.utils.data import Dataset
import h5py
import os
import skimage
import glob


class SeismoDataset(Dataset):
    def __init__(self, h5_path):
        archive = h5py.File(h5_path, 'r')
        self.data = archive['train']['X']
        self.truth = archive['train']['y']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wfs = self.data[idx]
        wfs = wfs[:, :-1]
        print(wfs.shape)
        wfs = torch.Tensor(wfs)
        #wfs = torch.unsqueeze(wfs, 1)
        picks = self.truth[idx]
        return wfs, picks
