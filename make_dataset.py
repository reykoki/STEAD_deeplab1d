import h5py
import random
import pandas as pd

import numpy as np
import time


def noise_shift(wfs):
    new_cent = random.randint(1501,4499)
    start = int(new_cent-1500)
    end = int(new_cent+1501)
    wfs = wfs[:,start:end]
    return wfs

def get_new_idx(start, p_idx, s_idx):
    new_p_idx = abs(p_idx - start)
    new_s_idx = abs(s_idx - start)
    return new_p_idx, new_s_idx


def rand_shift(wfs, p_idx, s_idx):
    diff = s_idx - p_idx
    center = (s_idx - p_idx)/2 + p_idx
    if diff > 2500:
        s_idx = p_idx
    left = s_idx - 2750
    right = p_idx + 2750
    if left < 0:
        left = 250
    if right > 6000:
        right = 5750
    elif p_idx < 250:
        left = 0
        right = 3001
    c_o = left + 1500
    c_f = right - 1500
    if c_o > c_f:
        new_cent = random.randint(c_f, c_o)
    else:
        new_cent = random.randint(c_o, c_f)
    start = int(new_cent-1500)
    end = int(new_cent+1501)
    new_p_idx, new_s_idx = get_new_idx(start, p_idx, s_idx)
    wfs = wfs[:,start:end]
    if wfs.shape != (3, 3001):
        print('p_idx:', p_idx)
        print('s_idx:', s_idx)
        print('c_o:', c_o)
        print('c_f:', c_f)
        print('center:', center)
        print('new_center:', new_cent)
        x = input("STOP!!!!")
    return wfs, new_p_idx, new_s_idx


metadata = pd.read_csv("merge.csv")
metadata = metadata.sort_values('trace_start_time')
col_names = metadata.columns.values.tolist()
print(col_names)
metadata = metadata.values.tolist()
trace_name_idx = col_names.index("trace_name")
s_idx_idx = col_names.index("s_arrival_sample")
p_idx_idx = col_names.index("p_arrival_sample")

ds = h5py.File("wf_ds.hdf5", "w")

num_samples = len(metadata)
#num_samples = 100
X = np.empty((num_samples, 3, 3001))
y = np.empty((num_samples, 2, 3001))
with h5py.File("merge.hdf5") as f:
    gdata = f["data"]
    for idx, md in enumerate(metadata):

        waveforms = gdata[md[trace_name_idx]][()]
        waveforms = waveforms.T  # From WC to CW
        waveforms = waveforms[[2, 1, 0]]  # From ENZ to ZNE
        p_idx = md[p_idx_idx]
        s_idx = md[s_idx_idx]

        truth = np.zeros((2,3001))
        if p_idx == p_idx and s_idx == s_idx:
            wfs, new_s_idx, new_p_idx = rand_shift(waveforms, int(p_idx), int(s_idx))
            truth[0, new_p_idx] = 1
            truth[1, new_s_idx] = 1
        elif p_idx != p_idx and s_idx == s_idx:
            wfs, new_s_idx, new_p_idx = rand_shift(waveforms, int(s_idx), int(s_idx))
            truth[1, new_s_idx] = 1
        elif p_idx == p_idx and s_idx != s_idx:
            wfs, new_s_idx, new_p_idx = rand_shift(waveforms, int(p_idx), int(p_idx))
            truth[0, new_p_idx] = 1
        else:
            # noise random shift anywhere
            wfs = noise_shift(waveforms)

        X[idx] = wfs
        y[idx] = truth
        #if idx == num_samples-1:
        #    break
d_type = ds.create_group('train')
ds_X = d_type.create_dataset('X', shape=(num_samples,3,3001), data = X, dtype=np.float32)
ds_y = d_type.create_dataset('y', shape=(num_samples,2,3001), data = y, dtype=np.float32)


ds.close()



