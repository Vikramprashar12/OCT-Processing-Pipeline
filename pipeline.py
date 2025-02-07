import os
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.io import loadmat, savemat
import cupy as cp
import numpy as np

def oct_loader(input_folder, output_folder):
    """
    Load .mat files containing spatial domain OCT data,
    apply DC subtraction and FFT in parallel.
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
    Process a single .mat file: load, apply DC subtraction, FFT, and save.
    """
    file_path = os.path.join(input_folder, file_name)
    print(f"\nProcessing: {file_name}")

    # Load the .mat file
    mat_data = loadmat(file_path)

    # Get the spatial domain data
    spatial_domain_data = mat_data['raw']
    print(f"Spatial domain data shape: {spatial_domain_data.shape}")

    # Apply DC subtraction and FFT
    freq_domain = process_oct_data(spatial_domain_data)

    # Save the processed data
    output_file_path = os.path.join(output_folder, f"processed_{file_name}")
    savemat(output_file_path, {'images': freq_domain})
    print(f"Saved: {output_file_path}")

def process_oct_data(spatial_domain_data):
    """
    Perform DC subtraction and recover frequency domain using FFT.
    """
    num_slices = spatial_domain_data.shape[2]
    chunk_size = 10  # Process 10 slices at a time (adjust based on GPU memory)
    freq_domain = np.zeros_like(spatial_domain_data, dtype=np.complex64)

    for start_idx in range(0, num_slices, chunk_size):
        end_idx = min(start_idx + chunk_size, num_slices)
        print(f"Processing slices {start_idx} to {end_idx-1}...")

        # Move chunk to GPU
        spatial_chunk_gpu = cp.asarray(spatial_domain_data[:, :, start_idx:end_idx], dtype=cp.float32)

        # Subtract DC component
        spatial_chunk_gpu = subtract_dc_component(spatial_chunk_gpu)

        # Apply FFT on GPU
        freq_chunk_gpu = cp.fft.fft2(spatial_chunk_gpu, axes=(0, 1))

        # Move chunk back to CPU
        freq_domain[:, :, start_idx:end_idx] = cp.asnumpy(freq_chunk_gpu)

        # Free GPU memory
        cp._default_memory_pool.free_all_blocks()

    return freq_domain

def subtract_dc_component(spatial_chunk_gpu):
    """
    Subtract the DC component (mean intensity) from the spatial domain data.
    """
    # Compute the mean intensity across each slice (DC component)
    dc_component = cp.mean(spatial_chunk_gpu, axis=(0, 1), keepdims=True)

    # Subtract the mean from the entire image chunk
    return spatial_chunk_gpu - dc_component

if __name__ == "__main__":
    start_time = time.time()

    input_folder = "in"  # Folder containing spatial domain .mat files
    output_folder = "out"  # Folder to save frequency domain .mat files

    # Run the loader
    oct_loader(input_folder, output_folder)

    print(f"Processing completed in {time.time() - start_time:.2f} seconds.")
