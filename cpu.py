import os
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.io import loadmat, savemat
import numpy as np
from scipy.ndimage import zoom

def oct_loader(input_folder, output_folder):
    """
    Load .mat files containing spatial domain OCT data,
    apply DC subtraction and FFT in parallel on CPU.
    """
    os.makedirs(output_folder, exist_ok=True)
    mat_files = [f for f in os.listdir(input_folder) if f.endswith(".mat")]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, input_folder, output_folder, file_name) for file_name in mat_files]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")

def process_file(input_folder, output_folder, file_name):
    """
    Process a single .mat file: load, apply DC subtraction, interpolation, FFT, and save on CPU.
    """
    file_path = os.path.join(input_folder, file_name)
    print(f"\nProcessing: {file_name}")
    mat_data = loadmat(file_path)
    spatial_domain_data = mat_data['raw']
    print(f"Spatial domain data shape: {spatial_domain_data.shape}, dtype: {spatial_domain_data.dtype}")
    
    
    freq_domain = process_oct_data(spatial_domain_data)
    freq_domain_real = np.abs(freq_domain)
    print(f"Output min: {freq_domain_real.min()}, max: {freq_domain_real.max()}")
    if freq_domain_real.max() > freq_domain_real.min():
        freq_domain_real = (freq_domain_real - freq_domain_real.min()) / (freq_domain_real.max() - freq_domain_real.min()) * 255
    output_file_path = os.path.join(output_folder, f"processed_{file_name}")
    savemat(output_file_path, {'images': freq_domain_real})
    print(f"Saved: {output_file_path}")

def process_oct_data(spatial_domain_data):
    """
    Perform DC subtraction and recover frequency domain using FFT on CPU.
    """
    num_slices = spatial_domain_data.shape[2]
    chunk_size = 10  # Process 10 slices at a time (still useful for memory management on CPU)
    freq_domain = np.zeros_like(spatial_domain_data, dtype=np.complex64)

    for start_idx in range(0, num_slices, chunk_size):
        end_idx = min(start_idx + chunk_size, num_slices)
        print(f"Processing slices {start_idx} to {end_idx-1}...")
        
        # Use NumPy array directly (no GPU transfer)
        spatial_chunk = spatial_domain_data[:, :, start_idx:end_idx].astype(np.complex64)
        print(f"Initial chunk shape: {spatial_chunk.shape}, dtype: {spatial_chunk.dtype}")
        
        # Subtract DC component
        spatial_chunk = subtract_dc_component(spatial_chunk)
        print(f"After DC subtraction min: {np.min(spatial_chunk)}, max: {np.max(spatial_chunk)}")

        spatial_chunk = interpolate_bicubic(spatial_chunk, scale=1)
        # Convert to complex64 for FFT
        spatial_chunk = spatial_chunk.astype(np.complex64)
        
        # Apply FFT on CPU
        freq_chunk = np.fft.fft2(spatial_chunk, axes=(0, 1))
        
        # Store result
        freq_domain[:, :, start_idx:end_idx] = freq_chunk

    return freq_domain

def interpolate_bicubic(data, scale=1):
    """
    Apply bicubic interpolation to upscale/downscale the input 3D OCT data on CPU.
    """
    print(f"Applying bicubic interpolation with scale {scale}...")
    # Use scipy.ndimage.zoom for CPU-based interpolation
    upscaled = zoom(data, (scale, scale, 1), order=3)  # Bicubic interpolation (order=3)
    print(f"Interpolated shape: {upscaled.shape}, dtype: {upscaled.dtype}")
    return upscaled

def subtract_dc_component(spatial_chunk):
    """
    Subtract the DC component (mean intensity) per slice on CPU.
    """
    # Compute mean per slice (across height and width)
    dc_component = np.mean(spatial_chunk, axis=(0, 1), dtype=np.complex64, keepdims=True)
    return spatial_chunk - dc_component

if __name__ == "__main__":
    start_time = time.time()
    input_folder = "in"
    output_folder = "out"
    oct_loader(input_folder, output_folder)
    print(f"Processing completed in {time.time() - start_time:.2f} seconds.")