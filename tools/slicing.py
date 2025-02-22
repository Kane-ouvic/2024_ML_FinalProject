import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import xarray as xr
from PIL import Image
import imageio

def slice_xyz(input_path, output_path, direction):
    # path to your training data
    dir_path = os.path.join(os.getcwd(), input_path)
    sample_ids = []
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if os.path.isdir(path):
            sample_ids.append(name)

    sample_ids = sample_ids[:3] # Testing

    for i in range(len(sample_ids)):
        sample_id = sample_ids[i]
        sample_path = os.path.join(input_path, sample_id)
        files = os.listdir(sample_path)

        for file in files:
            if file.startswith("seismicCubes_") and file.endswith(".npy"):
                seismic = np.load(os.path.join(sample_path, file), allow_pickle=True)
            elif file.startswith("fault_") and file.endswith(".npy"):
                fault = np.load(os.path.join(sample_path, file), allow_pickle=True)
        # print(f"\n Seismic array is of the shape: {seismic.shape} and of the data type: {seismic.dtype}")
        # print(f" Faults array is of the shape: {fault.shape} and of the data type: {fault.dtype}")
        seismic_rescaled = rescale_volume(seismic, low=2, high=98)
        
        # Create saving folder
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        #slice x, y, z
        if direction == 'x':
            slices_seismic = [seismic_rescaled[i, ...] for i in range(seismic_rescaled.shape[0])]
            slices_fault = [fault[i, ...] for i in range(fault.shape[0])]
            subfolder_path = os.path.join(output_path, 'x_slices', f'{sample_id}')
        elif direction == 'y':
            slices_seismic = [seismic_rescaled[:, i, :] for i in range(seismic_rescaled.shape[1])]
            slices_fault = [fault[:, i, :] for i in range(fault.shape[1])]
            subfolder_path = os.path.join(output_path, 'y_slices', f'{sample_id}')
        elif direction == 'z':
            slices_seismic = [seismic_rescaled[..., i] for i in range(seismic_rescaled.shape[2])]
            slices_fault = [fault[...: i] for i in range(fault.shape[2])]
            subfolder_path = os.path.join(output_path, 'z_slices', f'{sample_id}')

        os.makedirs(subfolder_path, exist_ok=True)

        # print(seismic_rescaled.shape)
        # print(f"Slice with {direction} direction")
        # print(f"Number of slices: {len(slices_seismic)}")
        # print(f"slice's shape: {slices_seismic[0].shape}")
        # print('-------------------------------------')

        for i in range(len(slices_seismic)):
            slice_np = slices_seismic[i]
            slice_np = slice_np.astype(np.uint8)
            img = Image.fromarray(slice_np)
            img.save(f'{subfolder_path}/seismic_{sample_id}_{direction}_{i}.png')

        for i in range(len(slices_fault)):
            slice_np = slices_fault[i]
            slice_np = slice_np.astype(np.uint8)
            img = Image.fromarray(slice_np)
            img.save(f'{subfolder_path}/fault_{sample_id}_{direction}_{i}.png')


def restore_masks(input_path, direction):
    masks_ret = []
    dir_path = os.path.join(os.getcwd(), input_path, f'{direction}_slices')
    sample_ids = []
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if os.path.isdir(path):
            sample_ids.append(name)
    sorted(sample_ids)
    for i in range(len(sample_ids)):
        sample_id = sample_ids[i]
        sample_path = os.path.join(dir_path, sample_id)
        files = os.listdir(sample_path)
        results = []
        for file in files:
            if file.startswith('fault'):
                results.append(os.path.join(sample_path, file))

        def extract_index(filename):
            base_name = os.path.splitext(filename)[0]
            number_part = base_name.split('_')[-1]
            return int(number_part) if number_part.isdigit() else float('inf')
        
        sorted_files = sorted(results, key=extract_index)
        masks = [imageio.v2.imread(os.path.join(sample_path, file)) for file in sorted_files]
        if direction == 'x':
            axis = 0
        elif direction == 'y':
            axis = 1
        elif direction == 'z':
            axis = 2
        reconstructed = np.stack(masks, axis=axis)
        # print(sample_path)
        # ori = np.load('training_data\\2023-10-05_0283ecc5\\fault_segments_2023.76163530.npy')
        # print(reconstructed.dtype)
        # print(ori.dtype)
        # print((reconstructed == ori).all())
        masks_ret.append(reconstructed)
    return reconstructed
    

'''
Given input and output directory path and a desire slicing direction
'''
slice_xyz("training_data", 'output', 'x')

'''
Test reconstruction here
'''
restore_masks('output', 'x')
