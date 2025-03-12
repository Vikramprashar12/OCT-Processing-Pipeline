import os
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.io import loadmat, savemat
import cupy as cp
import numpy as np

def oct_loader(input_folder, output_folder):
    """
    Load .mat files containing frequency domain OCT data,
    apply IFFT to recover spatial domain data in parallel.
    """
    os.makedirs(output_folder, exist_ok=True)

    # List all .mat files in the input folder
    mat_files = [f for f in os.listdir(input_folder) if f.endswith(".mat")]

    # Process files concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, input_folder, output_folder, file_name)
            for file_name in mat_files
        ]

        for future in futures:
            try:
                future.result()  # Ensure exceptions are raised if any
            except Exception as e:
                print(f"Error processing file: {e}")

def process_file(input_folder, output_folder, file_name):
    """
    Process a single .mat file: load, transform, and save.
    """
    file_path = os.path.join(input_folder, file_name)
    print(f"\nProcessing: {file_name}")

    # Load the .mat file
    mat_data = loadmat(file_path)

    # Get the frequency domain data
    freq_domain_data = mat_data['images']
    print(f"Frequency domain data shape: {freq_domain_data.shape}")

    # Apply IFFT to recover spatial domain in chunks
    spatial_domain = recover_spatial_domain_in_chunks(freq_domain_data)

    # Save the recovered data
    output_file_path = os.path.join(output_folder, f"raw_{file_name}")
    savemat(output_file_path, {'raw': spatial_domain})
    print(f"Saved: {output_file_path}")

def recover_spatial_domain_in_chunks(freq_domain_data):
    """
    Recover spatial domain data from frequency domain using IFFT in smaller chunks.
    """
    num_slices = freq_domain_data.shape[2]
    chunk_size = 10  # Process 10 slices at a time (adjust based on GPU memory)
    spatial_domain = np.zeros_like(freq_domain_data, dtype=np.complex64)

    for start_idx in range(0, num_slices, chunk_size):
        end_idx = min(start_idx + chunk_size, num_slices)
        print(f"Processing slices {start_idx} to {end_idx-1}...")

        # Move chunk to GPU
        freq_chunk_gpu = cp.asarray(freq_domain_data[:, :, start_idx:end_idx], dtype=cp.complex64)

        # Apply IFFT on GPU
        spatial_chunk_gpu = cp.fft.ifft2(freq_chunk_gpu, axes=(0, 1))

        # Move chunk back to CPU
        spatial_domain[:, :, start_idx:end_idx] = cp.asnumpy(spatial_chunk_gpu)

        # Free GPU memory
        cp._default_memory_pool.free_all_blocks()

    return spatial_domain

if __name__ == "__main__":
    start_time = time.time()

    input_folder = "input_mat_files"
    output_folder = "output_mat_files"

    # Run the loader
    oct_loader(input_folder, output_folder)

    print(f"Processing completed in {time.time() - start_time:.2f} seconds.")