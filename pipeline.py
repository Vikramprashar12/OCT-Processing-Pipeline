import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.io import loadmat, savemat
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
from cupyx.scipy.ndimage import zoom
import numpy as np
from tqdm import tqdm
import psutil
from rich.console import Console
from rich.table import Table

console = Console()

def show_system_stats(file_name, current_slice, total_slices):
    """
    Display real-time system statistics including:
    - Current file being processed
    - Slice progress
    - CPU memory usage
    - GPU memory usage
    """
    mem = psutil.virtual_memory()
    gpu_mem = cp.get_default_memory_pool().used_bytes()

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    table.add_row("File", file_name)
    table.add_row("Slice Progress", f"{current_slice + 1}/{total_slices}")
    table.add_row("CPU Used", f"{mem.percent:.2f}%")
    table.add_row("RAM Available", f"{mem.available / (1024**3):.2f} GB")
    table.add_row("GPU Memory Used", f"{gpu_mem / (1024**3):.2f} GB")

    console.clear()
    console.print(table)

def oct_loader(input_folder, output_folder):
    """
    Load all .mat OCT files from input_folder, process them one-by-one using a thread pool
    (with a single worker due to GPU memory constraints), and save results to output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    # List all .mat files in the input folder
    mat_files = [f for f in os.listdir(input_folder) if f.endswith(".mat")]

    max_workers = 1  # Limit concurrency due to GPU memory constraints
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, input_folder, output_folder, file_name)
            for file_name in mat_files
        ]

        for future in as_completed(futures):
            try:
                future.result()  # Ensures exceptions are raised
            except Exception as e:
                print(f"Error processing file: {e}")

def process_file(input_folder, output_folder, file_name):
    """
    Process a single .mat file: load the raw spatial domain OCT data,
    perform preprocessing (bicubic interpolation, DC subtraction, dispersion compensation),
    transform to frequency domain using FFT, and save the result.
    """
    file_path = os.path.join(input_folder, file_name)
    print(f"\nProcessing: {file_name}")

    try:
        # Load the .mat file
        mat_data = loadmat(file_path)

        # Get the spatial domain data (original IFFT output from earlier stage)
        spatial_domain_data = mat_data['raw']
        print(f"Spatial domain data shape: {spatial_domain_data.shape}")

         # Step 1: Apply bicubic interpolation of A-lines (depth-wise) instead of full 2D image
        spatial_domain_data = interpolate_alines_bicubic_gpu(spatial_domain_data, scale=0.9)

        # # Uncomment below to use old bicubic interpolation (full image scaling)
        # spatial_domain_data = interpolate_bicubic(spatial_domain_data, scale=0.9)

        # Step 2: Process the interpolated data (DC subtraction, compensation, FFT)
        freq_domain = process_oct_data(spatial_domain_data, file_name)

        # Step 3: Convert complex FFT result to magnitude
        freq_domain_real = np.abs(freq_domain)

        # Step 4: Save the processed data
        output_file_path = os.path.join(output_folder, f"processed_{file_name}")
        savemat(output_file_path, {'images': freq_domain_real})
        print(f"Saved: {output_file_path}")

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")

def process_oct_data(spatial_domain_data, file_name):
    """
    Process OCT data in chunks to avoid GPU overload.
    Each chunk undergoes:
    - DC component subtraction
    - Dispersion compensation
    - 2D FFT along x, y axes
    """
    num_slices = spatial_domain_data.shape[2]
    chunk_size = 2  # Reduce chunk size to avoid GPU memory overflow
    freq_domain = np.zeros_like(spatial_domain_data, dtype=np.complex64)

    for start_idx in tqdm(range(0, num_slices, chunk_size), desc=f"Processing {file_name}", unit="chunk"):
        end_idx = min(start_idx + chunk_size, num_slices)

        # Display system stats in CLI
        show_system_stats(file_name, start_idx, num_slices)

        # Move chunk to GPU
        spatial_chunk_gpu = cp.asarray(spatial_domain_data[:, :, start_idx:end_idx], dtype=cp.complex64)

        # Step 1: Subtract DC component (mean intensity across slices)
        spatial_chunk_gpu = subtract_dc_component(spatial_chunk_gpu)

        # Step 2: Apply Dispersion Compensation (phase correction)
        spatial_chunk_gpu = apply_dispersion_compensation(spatial_chunk_gpu)

        # Step 3: Apply FFT on GPU
        freq_chunk_gpu = cp.fft.fft2(spatial_chunk_gpu, axes=(0, 1))

        # Move chunk back to CPU
        freq_domain[:, :, start_idx:end_idx] = cp.asnumpy(freq_chunk_gpu)

        # Free GPU memory
        cp.get_default_memory_pool().free_all_blocks()

    return freq_domain

# def interpolate_bicubic(data, scale=2):
#     """
#     Old method: Apply bicubic interpolation to entire 2D slices
#     This is commented out and replaced by more efficient A-line interpolation
#     """
#     print(f"Applying bicubic interpolation with scale {scale}...")
#     data_gpu = cp.asarray(data, dtype=cp.float32)
#     zoom_factors = (scale, scale, 1)
#     upscaled_gpu = zoom(data_gpu, zoom_factors, order=3)
#     return cp.asnumpy(upscaled_gpu)

def interpolate_alines_bicubic_gpu(data, scale=2):
    """
    Alternative using CuPy's zoom for bicubic interpolation along the depth axis.
    Less memory efficient than kernel-based interpolation.
    """
    data_gpu = cp.asarray(data, dtype=cp.float32)
    zoom_factors = (scale, 1, 1)  # Only upscale along height (depth axis)
    upscaled_gpu = zoom(data_gpu, zoom=zoom_factors, order=3)
    return cp.asnumpy(upscaled_gpu)

def subtract_dc_component(spatial_chunk_gpu):
    """
    Subtract DC component (mean intensity) from each pixel across all depth slices.
    This centers data around zero to prepare for spectral transformation.
    """
    dc_component = cp.mean(spatial_chunk_gpu, axis=2, dtype=cp.float32, keepdims=True)
    return spatial_chunk_gpu - dc_component

def apply_dispersion_compensation(spatial_chunk_gpu):
    """
    Apply phase correction to compensate for dispersion effects in OCT signal.
    Here we use a synthetic quadratic phase model.
    """
    x_dim, y_dim, num_slices = spatial_chunk_gpu.shape
    k = cp.linspace(-1, 1, num_slices)  # Wavenumber domain

    # Create quadratic phase correction profile
    phase_correction = cp.exp(-1j * (0.1 * k**2))

    # Apply across depth axis
    return spatial_chunk_gpu * phase_correction.reshape(1, 1, num_slices)

if __name__ == "__main__":
    start_time = time.time()

    input_folder = "in"  # Folder containing input .mat files (raw spatial domain data)
    output_folder = "out"  # Folder where processed .mat files will be saved

    # Start the GPU-accelerated OCT processing pipeline
    oct_loader(input_folder, output_folder)

    elapsed = time.time() - start_time
    print(f"\n✅ Processing completed in {elapsed:.2f} seconds.")
