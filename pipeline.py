import os
import scipy.io
import h5py
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cpx_nd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def process(file_path):

    # Try loading using scipy (for MATLAB v7 and below)
    data = scipy.io.loadmat(file_path)
    print("Loaded using scipy.io.loadmat")

    show_bscan_only(data['images'], title="Original Image")

    images_dc = dc_subtract(data['images'])
    show_bscan_only(images_dc, title="After DC Subtraction")


def show_bscan_only(images, bscan_index=50, title="B-scan"):
    """
    Interactive B-scan viewer with slider (no layers).
    """
    num_bscans = images.shape[2]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)
    img_display = ax.imshow(
        images[:, :, bscan_index], cmap='gray', aspect='auto')

    ax.set_title(f'{title} {bscan_index}')
    ax.set_xlabel('A-scan (x-axis)')
    ax.set_ylabel('Depth (z-axis)')

    # Slider setup
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'B-scan', 0, num_bscans -
                    1, valinit=bscan_index, valstep=1)

    def update(val):
        idx = int(slider.val)
        img_display.set_data(images[:, :, idx])
        ax.set_title(f'{title} {idx}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def dc_subtract(images):
    """
    Perform per-B-scan, per-A-scan DC subtraction using GPU (CuPy).
    """
    images_gpu = cp.asarray(images)
    result_gpu = cp.empty_like(images_gpu)

    for i in range(images_gpu.shape[2]):
        bscan = images_gpu[:, :, i]  # (depth, ascans)
        mean_vals = cp.mean(bscan, axis=0, keepdims=True)  # mean per A-scan
        result_gpu[:, :, i] = bscan - mean_vals

    return cp.asnumpy(result_gpu)


def dispersion_compensate(images, beta=1.0, gamma=0.0):
    """
    Apply dispersion compensation in the frequency domain using GPU.
    beta: quadratic phase term coefficient
    gamma: cubic phase term coefficient (optional)
    """
    images_gpu = cp.asarray(images)
    result_gpu = cp.empty_like(images_gpu, dtype=cp.complex64)

    N = images_gpu.shape[0]  # number of depth samples
    k = cp.fft.fftfreq(N)  # normalized frequency axis
    phase_correction = cp.exp(-1j * (beta * k**2 + gamma * k**3))

    for i in range(images_gpu.shape[2]):
        bscan = images_gpu[:, :, i]  # shape: (depth, ascans)
        bscan_fft = cp.fft.fft(bscan, axis=0)
        # apply dispersion compensation
        bscan_fft *= phase_correction[:, cp.newaxis]
        bscan_ifft = cp.fft.ifft(bscan_fft, axis=0)
        result_gpu[:, :, i] = bscan_ifft

    return cp.asnumpy(cp.abs(result_gpu))


def gaussian_lowpass(images, sigma=(3, 3, 0)):
    """
    Apply a Gaussian low-pass filter using the specified sigma values.
    """
    images_gpu = cp.asarray(images)
    lowpass = cpx_nd.gaussian_filter(images_gpu, sigma=sigma)
    return cp.asnumpy(lowpass)


def main():
    folder = 'in'
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist.")
        return

    mat_files = [f for f in os.listdir(folder) if f.endswith('.mat')]
    if not mat_files:
        print("No .mat files found in the folder.")
        return

    for file_name in mat_files:
        file_path = os.path.join(folder, file_name)
        process(file_path)


if __name__ == "__main__":
    main()

'''

Removing the functions I am not going to use

def gaussian_lowpass(images, sigma=(3, 3, 0)):
    """
    Apply a Gaussian low-pass filter using the specified sigma values.
    """
    images_gpu = cp.asarray(images)
    lowpass = cpx_nd.gaussian_filter(images_gpu, sigma=sigma)
    return cp.asnumpy(lowpass)


def auto_canny(image, sigma=0.33, blur_sigma=1.0):
    """
    Perform Canny edge detection with optional pre-blur and robust scaling.
    - sigma: threshold sensitivity
    - blur_sigma: Gaussian blur strength before edge detection
    """
    # Normalize + blur using CuPy
    image_gpu = cp.asarray(image, dtype=cp.float32)
    image_gpu -= cp.min(image_gpu)
    image_gpu /= cp.max(image_gpu)
    image_gpu *= 255.0

    blurred = cpx_nd.gaussian_filter(image_gpu, sigma=blur_sigma)

    # Compute thresholds
    v = cp.median(blurred).get()
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # Convert to 8-bit and run OpenCV Canny
    image_np = cp.asnumpy(blurred).astype(np.uint8)
    edges = cv2.Canny(image_np, lower, upper)
    return edges

'''
'''
This code was in the process function for gaussian filtering and canny edge detection

    images_lowpass_soft = gaussian_lowpass(images_dc, sigma=(3, 3, 5))
    show_bscan_only(images_lowpass_soft,
                    title=f"Gaussian Low-Pass ")

    processed_stack = []
    for i in range(images_lowpass_soft.shape[2]):
        bscan = images_lowpass_soft[:, :, i]
        edges = auto_canny(bscan)
        processed_stack.append(edges)

    processed_stack = np.stack(processed_stack, axis=-1)  # shape: (H, W, N)
    show_bscan_only(processed_stack, title="Canny Edge Stack")

    # Vary first argument (depth axis), keep second at 1
    for d in [1, 2, 3, 4, 5]:
        filtered = gaussian_lowpass(images_dc, sigma=(1, 1, 2))
        show_bscan_only(filtered, title=f"Low-Pass sigma=(1, 1, {d})")

    # Vary second argument (A-scan axis), keep first at 1
    for a in [1, 3, 5, 7, 9]:
        filtered = gaussian_lowpass(images_dc, sigma=(1, a, 5))
        show_bscan_only(filtered, title=f"Low-Pass sigma=(1,{a},5)")

'''
