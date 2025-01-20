from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat file
file_path = 'Farsiu_Ophthalmology_2013_AMD_Subject_1009.mat'
mat_data = loadmat(file_path)

# Inspect the keys in the loaded .mat file
print(mat_data.keys())
# Replace 'data3d' with the correct key based on your .mat file structure
volume_data = mat_data['images']  
print(volume_data.shape)  # Inspect the shape (e.g., (X, Y, Z))

slice_index = 30  # Choose a slice index
plt.imshow(volume_data[:, :, slice_index], cmap='gray')
plt.title(f"Slice {slice_index}")
plt.colorbar()
plt.show()

