import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.io import loadmat

def visualize_slices(file_path):
    """
    Visualize slices from a .mat file interactively using a slider.
    """
    # Load the .mat file
    mat_data = loadmat(file_path)

    # Get the frequency domain data
    freq_domain_data = mat_data['images']  # Frequency domain data
    print(f"Frequency domain data shape: {freq_domain_data.shape}")

    # Take the magnitude of the frequency domain data (FFT output is complex)
    magnitude_data = np.abs(freq_domain_data)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)  # Adjust layout to make room for the slider

    # Display the initial slice (middle slice)
    initial_slice = magnitude_data.shape[2] // 2
    im = ax.imshow(magnitude_data[:, :, initial_slice], cmap='gray')
    ax.set_title(f"Frequency Domain\nSlice {initial_slice}")
    ax.axis('off')  # Hide axes

    # Add a slider for slice selection
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # Define slider position
    slice_slider = Slider(
        ax=ax_slider,
        label='Slice Index',
        valmin=0,
        valmax=magnitude_data.shape[2] - 1,
        valinit=initial_slice,
        valstep=1
    )

    # Update function for the slider
    def update(val):
        slice_idx = int(slice_slider.val)
        im.set_data(magnitude_data[:, :, slice_idx])
        ax.set_title(f"Frequency Domain\nSlice {slice_idx}")
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slice_slider.on_changed(update)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Specify the file name
    file_name = "processed_raw_Farsiu_Ophthalmology_2013_AMD_Subject_1001.mat"

    # Folder containing .mat files
    folder = "out"  # Folder with frequency domain data
    file_path = os.path.join(folder, file_name)

    # Visualize slices interactively
    visualize_slices(file_path)