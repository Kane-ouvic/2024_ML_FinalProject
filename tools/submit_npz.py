import os
import numpy as np

def create_combined_submission(input_dir, submission_path):
    """
    Combine all .npy files in the input directory into a single .npz file.

    Parameters:
        input_dir (str): Path to the directory containing .npy files.
        submission_path (str): Path to save the combined .npz file.

    Returns:
        None
    """
    submission = {}

    # Iterate over all files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".npy"):
            file_path = os.path.join(input_dir, file_name)
            
            # Load the .npy file
            npy_data = np.load(file_path)
            
            # Extract the sample ID from the file name (e.g., remove .npy extension)
            sample_id = os.path.splitext(file_name)[0]
            
            # Find coordinates where the value is 1
            coordinates = np.stack(np.where(npy_data == 1)).T
            coordinates = coordinates.astype(np.uint16)  # Ensure uint16 type
            
            # Add to the submission dictionary
            submission[sample_id] = coordinates

    # Save all collected data into a single .npz file
    np.savez(submission_path, **submission)
    print(f"Submission file saved at {submission_path}")

# Define input directory and submission file path
input_directory = "/home/ouvic/ML/ML_Final/test_2d_x_result_1_npy"  # Replace with the directory containing .npy files
submission_file_path = "/home/ouvic/ML/ML_Final/submission/submission_x.npz"  # Replace with your desired submission path

# Run the process
create_combined_submission(input_directory, submission_file_path)
