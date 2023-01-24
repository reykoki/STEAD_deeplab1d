import torch
from SeismoDataset import SeismoDataset
train_set = SeismoDataset('subset.hdf5')
print(train_set)
print(len(train_set))
