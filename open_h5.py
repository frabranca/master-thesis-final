import h5py
import samna
import torch
import os
import matplotlib.pyplot as plt
import hdf5plugin

# dataset_path_new = 'dataset_sr/128_train'

dataset_path_new = 'dataset_sr/128_test'
# dataset_path_old = '../../../data/datasets/sr_dataset_train'
# dataset_path_new = '../event_based_fra/sr_dataset_undistorted_180'

files = os.listdir(dataset_path_new)

for file_name in files:
    file_path = os.path.join(dataset_path_new, file_name)
    
    with h5py.File(file_path, 'a') as file_new:
        print(file_new.keys())
