import torch
from torch.utils.data import Dataset
import h5py
import os
import skimage
import glob
import torch.nn as nn


class SeismoDataset(Dataset):
    def __init__(self, h5_path):
        archive = h5py.File(h5_path, 'r')
        self.data = archive['train']['X']
        self.truth = archive['train']['y']
        #self.pad_data = nn.ConstantPad2d((0,7,0,0),0)
        self.pad_8 = nn.ConstantPad1d((0,7),0)

    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        wfs = self.data[idx]
        wfs = torch.Tensor(wfs)
        wfs = self.pad_8(wfs)
        picks = self.truth[idx]
        picks = torch.Tensor(picks)
        picks = self.pad_8(picks)
        return wfs, picks
