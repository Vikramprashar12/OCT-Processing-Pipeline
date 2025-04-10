import os
import scipy.io
import h5py
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cpx_nd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.restoration import denoise_nl_means, estimate_sigma


def process(file_path):

    # Try loading using scipy (for MATLAB v7 and below)
    data = scipy.io.loadmat(file_path)
    print("Loaded using scipy.io.loadmat")

    # show_bscan_only(data['images'], data['layerMaps'], title="Original Image")
    images_dc = dc_subtract(data['images'])
    # show_bscan_only(images_dc, data['layerMaps'], title="After DC Subtraction")

    filtered = median_filter(images_dc, size=3)
    # show_bscan_only(filtered, layerMaps=data['layerMaps'], title="Median Filtering")

    masked = mask_with_layer_bounds(filtered, layerMaps=data['layerMaps'])
    show_bscan_only(masked, data['layerMaps'], title='Masked')


def mask_with_layer_bounds(images, layerMaps):
    """
    Zero out any pixels above the max of the first layer and below the min of the last layer.
    """
    masked = images.copy()
    for i in range(images.shape[2]):
        print(i)
        upper_bound = int(np.max(layerMaps[i, :, 50]))
        lower_bound = int(np.min(layerMaps[i, :, -1]))
        masked[:upper_bound, :, i] = 0
        masked[lower_bound+1:, :, i] = 0
    return masked


def show_bscan_only(images, layerMaps=None, bscan_index=50, title="B-scan"):
    """
    Interactive B-scan viewer with slider, optionally displaying segmentation layers.
    """
    num_bscans = images.shape[2]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)
    img_display = ax.imshow(
        images[:, :, bscan_index], cmap='gray', aspect='auto')

    layer_lines = []
    if layerMaps is not None:
        for i in range(layerMaps.shape[2]):
            line, = ax.plot(np.arange(layerMaps.shape[1]),
                            layerMaps[bscan_index, :, i], label=f'Layer {i+1}')
            layer_lines.append(line)
        ax.legend()

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
        if layerMaps is not None:
            for i in range(layerMaps.shape[2]):
                layer_lines[i].set_ydata(layerMaps[idx, :, i])
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


def median_filter(images, size=3):
    """
    Apply a median filter to each B-scan individually using CuPy.
    """
    images_gpu = cp.asarray(images)
    result_gpu = cp.empty_like(images_gpu)
    for i in range(images_gpu.shape[2]):
        result_gpu[:, :, i] = cpx_nd.median_filter(
            images_gpu[:, :, i], size=size, mode='nearest')
    return cp.asnumpy(result_gpu)


def non_local_means_denoise(images, patch_size=10, patch_distance=5, h=0.6):
    """
    Apply non-local means denoising using scikit-image for each B-scan.
    """
    result = []
    for i in range(images.shape[2]):
        img = images[:, :, i].astype(np.float32)
        sigma_est = np.mean(estimate_sigma(img, channel_axis=None))
        denoised = denoise_nl_means(
            img,
            h=h * sigma_est,
            patch_size=patch_size,
            patch_distance=patch_distance,
            channel_axis=None,
            fast_mode=True,
        )
        result.append(denoised)
    return np.stack(result, axis=-1)


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
