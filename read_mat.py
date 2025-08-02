import h5py
import numpy as np
from PIL import Image

# Path to your .mat file
mat_file_path = 'nyu_depth_v2_labeled.mat'
output_dir_rgb = 'nyu_rgb_images'
output_dir_depth = 'nyu_depth_images'

# Create output directories if they don't exist
import os
os.makedirs(output_dir_rgb, exist_ok=True)
os.makedirs(output_dir_depth, exist_ok=True)

with h5py.File(mat_file_path, 'r') as f:
    # Access RGB images (assuming 'images' is the key)
    # The data might be transposed, so check the dimensions and transpose if needed
    images_data = f['images'][:]
    # Access depth maps (assuming 'rawDepths' is the key)
    raw_depths_data = f['rawDepths'][:]

    num_images = images_data.shape[0] # Assuming first dimension is image count

    for i in range(num_images):
        # Extract and save RGB image
        rgb_image_array = images_data[i].transpose(2, 1, 0) # Adjust transpose based on data structure
        rgb_image = Image.fromarray(rgb_image_array.astype('uint8'))
        rgb_image.save(os.path.join(output_dir_rgb, f'rgb_image_{i:04d}.jpg'))

        # Extract and save depth map (often 16-bit grayscale)
        depth_map_array = raw_depths_data[i].T # Adjust transpose based on data structure
        # Normalize depth values if necessary for visualization or specific image formats
        # For example, scale to 0-255 for 8-bit grayscale visualization
        depth_map_normalized = (depth_map_array / depth_map_array.max() * 255).astype('uint8')
        depth_image = Image.fromarray(depth_map_normalized, 'L') # 'L' for grayscale
        depth_image.save(os.path.join(output_dir_depth, f'depth_map_{i:04d}.png'))

print("Conversion complete.")