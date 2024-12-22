import os
import numpy as np
import imageio

def restore_and_save_npy(input_dir, output_dir, direction='x'):
    """
    Restore images in subdirectories into a 3D numpy array and save as .npy file.
    
    Parameters:
    - input_dir: Path to the directory containing subdirectories of images.
    - output_dir: Path to the directory where the .npy files will be saved.
    - direction: Axis for stacking images ('x', 'y', 'z').
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine axis for stacking based on direction
    axis = {'x': 0, 'y': 1, 'z': 2}.get(direction, 0)
    
    # Traverse each subdirectory in the input directory
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue  # Skip files, only process directories
        
        # Collect image files and sort them by index in filename
        image_files = [
            f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        def extract_index(filename):
            base_name = os.path.splitext(filename)[0]
            number_part = base_name.split('_')[-1]
            return int(number_part) if number_part.isdigit() else float('inf')
        
        sorted_files = sorted(image_files, key=extract_index)
        
        # Read images and stack into a 3D array
        masks = [imageio.v2.imread(os.path.join(subdir_path, file)) for file in sorted_files]
        if not masks:
            print(f"No valid images found in {subdir_path}. Skipping...")
            continue
        
        reconstructed = np.stack(masks, axis=axis)
        reconstructed[reconstructed == 255] = 1
        
        # Save the 3D array as a .npy file
        npy_filename = os.path.join(output_dir, f"{subdir}.npy")
        np.save(npy_filename, reconstructed)
        print(f"Saved {npy_filename}")
        
        # Load and print the saved .npy file's content
        # loaded_array = np.load(npy_filename)
        # print(f"Values in {npy_filename}:")
        # print(loaded_array)

# Define input and output paths
input_directory = "/home/ouvic/ML/ML_Final/test_2d_x_result_1"  # Replace with your input directory
output_directory = "/home/ouvic/ML/ML_Final/test_2d_x_result_1_npy"  # Replace with your output directory

# Run the restoration and saving process
restore_and_save_npy(input_directory, output_directory, direction='x')
