#!/usr/bin/env python3
"""
CT Dataset Grid Viewer with Interactive Cells
Use Ctrl+Enter to run each cell in VS Code Python Interactive
"""
# %% [markdown]
# # CT Dataset Visualization with Interactive Cells
# This script uses #%% to create interactive cells that run in VS Code Python Interactive window

# %% Import Libraries
import json
import numpy as np
import matplotlib.pyplot as plt
from monai import transforms
from monai.data import Dataset
import torch
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.append(os.getcwd())
from ct_prep import ExtractPixelSpacingd, DefineEmbedDim, SizeEmbed, ClinicalText, CropResized

# Set matplotlib to display inline in VS Code Interactive
plt.ion()  # Turn on interactive mode
print("Libraries imported successfully!")

# %% Define Configuration
# Configuration parameters - modify these as needed
json_file = '/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene_modified.json'
start_idx = 0
num_samples = 16  # Change this for different grid sizes
rows = 4
cols = 4
lung_side = 'right'  # 'right' or 'left' for CropResized transform

print(f"Configuration:")
print(f"- JSON file: {json_file}")
print(f"- Start index: {start_idx}")
print(f"- Samples: {num_samples} ({rows}x{cols} grid)")
print(f"- Lung side: {lung_side}")

# %% Define Transforms
# Full MONAI transform pipeline as specified
transforms_list = transforms.Compose([
    transforms.LoadImaged(keys=['image', 'mask'], allow_missing_keys=True, meta_key_postfix="meta_dict", image_only=False, reader="ITKReader"),
    ExtractPixelSpacingd(keys=['image']),  # Extract pixel spacing after loading
    transforms.EnsureChannelFirstd(keys=['image', 'mask'], allow_missing_keys=True),
    transforms.Orientationd(keys=["image", "mask"], allow_missing_keys=True, axcodes="ALS"),
    #transforms.Transposed(keys=["image", "mask"], indices=(0, 3, 1, 2), allow_missing_keys=True),
    DefineEmbedDim(keys=['image'], embed_dim=1024, allow_missing_keys=True),
    SizeEmbed(keys=['image'], allow_missing_keys=True),
    ClinicalText(keys=['image'], question='Predict the lung cancer risk over six years.', clinical_text='No patient information available.', data_name='cancer_risk', allow_missing_keys=True),
    # #transforms.Rotate90d(keys=["image", "mask"], k=1, spatial_axes=(1,2), allow_missing_keys=True),
    transforms.ScaleIntensityRanged(keys=['image'], a_min=-1300, a_max=150, b_min=0, b_max=1.0, clip=True),
    # # transforms.CropForegroundd(keys=['image', 'mask'], source_key='mask', margin=0),
    # # transforms.Resized(keys=['image', 'mask'], spatial_size=(128,320,448)),
    CropResized(keys=['image', 'mask'], crop_size=(320, 448, 128), lung=lung_side), #(width=320, height=448, depth=128)
    #transforms.Rotate90d(keys=["image"], k=2, spatial_axes=(1,2), allow_missing_keys=True),
    #transforms.Flipd(keys=['image'], spatial_axis=2, allow_missing_keys=True),
    #transforms.ResizeWithPadOrCropd(keys=['image', 'mask'], spatial_size=(448, 320, 128), mode='constant', allow_missing_keys=True),
    transforms.Transposed(keys=["image", "mask"], indices=(0, 3, 2, 1), allow_missing_keys=True),
    #SizeEmbed(keys=['image'], allow_missing_keys=True),
    transforms.ToTensord(keys=['image', 'mask', 'questions_ids', 'questions_mask', 'txt_ids', 'txt_mask', 'data_size'], allow_missing_keys=True),
])

print("Full transform pipeline defined successfully!")
print("Transforms include: Loading, Spacing, Orientation, Embedding, Clinical Text, Intensity Scaling,")
print("                   CropResize, Rotation, Flip, Transpose, ToTensor")

# %% Load Dataset
# Load JSON data and create dataset slice
print(f"Loading data from: {json_file}")
with open(json_file, 'r') as f:
    data_dict = json.load(f)

total_samples = len(data_dict)
print(f"Total samples available: {total_samples}")

# Get slice of data
end_idx = min(start_idx + num_samples, total_samples)
data_slice = data_dict[start_idx:end_idx]
actual_samples = len(data_slice)

print(f"Loading samples {start_idx} to {end_idx-1} ({actual_samples} samples)")

# Create dataset
dataset = Dataset(data=data_slice, transform=transforms_list)
print(f"Dataset created with {len(dataset)} samples")

# %% Create Visualization
# Create and display the grid visualization
print("Creating visualization...")

# Create figure
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
fig.suptitle(f'CT Dataset Grid View (Samples {start_idx}-{end_idx-1})', fontsize=14)

# Handle single subplot case
if rows == 1 and cols == 1:
    axes = [axes]
elif rows == 1 or cols == 1:
    axes = axes.flatten()
else:
    axes = axes.flatten()

print("\nProcessing samples:")
print("=" * 60)

# Process samples
for i in range(min(actual_samples, rows * cols)):
    try:
        print(f"Processing sample {i+1}/{actual_samples}...")
        
        # Load sample
        sample = dataset[i]
        
        # Print sample info for debugging
        print(f"  Sample keys: {sample.keys()}")
        
        # Get data
        image = sample['image']
        mask = sample['mask'] if 'mask' in sample else None
        
        # Print shapes and additional info
        print(f"  Image shape: {image.shape}")
        if mask is not None:
            print(f"  Mask shape: {mask.shape}")
        
        # Print additional transform results
        if 'coords' in sample:
            print(f"  Coordinates: {sample['coords']}")
        if 'pixel_size' in sample:
            print(f"  Pixel size: {sample['pixel_size']}")
        if 'clinical_txt' in sample:
            print(f"  Clinical text: {sample['clinical_txt']}")
        
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        # After full pipeline, image shape should be [C, S, H, W] due to final Transpose
        # Let's handle both cases for robustness
        if len(image.shape) == 4:
            # Check if it's [C, S, H, W] (after transpose) or [C, W, H, S] (before transpose)
            if image.shape[1] < 200:  # Assume depth dimension is smaller
                # Shape is [C, S, H, W] (transposed)
                z_middle = image.shape[1] // 2
                img_slice = image[0, z_middle, :, :]  # Shape: [H, W]
                if mask is not None:
                    mask_slice = mask[0, z_middle, :, :] if len(mask.shape) == 4 else None
                else:
                    mask_slice = None
            else:
                # Shape is [C, W, H, S] (not transposed)
                z_middle = image.shape[3] // 2
                img_slice = image[0, :, :, z_middle]  # Shape: [W, H]
                if mask is not None:
                    mask_slice = mask[0, :, :, z_middle] if len(mask.shape) == 4 else None
                else:
                    mask_slice = None
        else:
            print(f"  Unexpected image shape: {image.shape}")
            continue
            
        print(f"  Slice shape: {img_slice.shape}")
        if mask_slice is not None:
            print(f"  Mask slice shape: {mask_slice.shape}")
        
        # Display
        ax = axes[i]
        
        # Display image - handle orientation based on slice shape
        if img_slice.shape[0] > img_slice.shape[1]:
            # Width > Height, probably [W, H] format, transpose for display
            ax.imshow(img_slice.T, cmap='gray', origin='lower', aspect='equal')
            transpose_mask = True
        else:
            # Height >= Width, probably [H, W] format, display as-is
            ax.imshow(img_slice, cmap='gray', origin='lower', aspect='equal')
            transpose_mask = False
        
        # Overlay mask with 0.5 opacity
        if mask_slice is not None and np.any(mask_slice > 0):
            # Create colored overlay with same orientation as image
            if transpose_mask:
                mask_colored = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
            else:
                mask_colored = np.ma.masked_where(mask_slice == 0, mask_slice)
            ax.imshow(mask_colored, alpha=0.5, cmap='Reds', origin='lower', aspect='equal')
        
        # Get series info
        actual_idx = start_idx + i
        series = data_slice[i].get('series', 'Unknown')
        
        # Set title (truncate series for readability)
        series_short = series[:15] + "..." if len(series) > 15 else series
        ax.set_title(f'Idx: {actual_idx}\n{series_short}', fontsize=10)
        ax.axis('off')
        
        # Print for copy-paste
        print(f"Index: {actual_idx}, Series: {series}")
        
    except Exception as e:
        print(f"ERROR processing sample {start_idx + i}: {e}")
        
        # Show error in subplot
        ax = axes[i]
        ax.text(0.5, 0.5, f'Error\n{start_idx + i}', 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=10, color='red')
        ax.axis('off')

# Hide unused subplots
for i in range(actual_samples, len(axes)):
    axes[i].axis('off')

print("=" * 60)
print(f"Visualization complete! {actual_samples} samples processed")

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# %% Inspect Transform Results
# Display detailed information about the transform pipeline results
print("Transform Pipeline Results:")
print("=" * 50)

if 'dataset' in locals() and len(dataset) > 0:
    # Get first sample to inspect
    sample = dataset[0]
    
    print("\n📊 Sample Data Structure:")
    for key in sample.keys():
        if hasattr(sample[key], 'shape'):
            print(f"  {key}: {sample[key].shape} ({type(sample[key])})")
        else:
            print(f"  {key}: {type(sample[key])}")
    
    print("\n🔍 Detailed Information:")
    
    # Image info
    if 'image' in sample:
        image = sample['image']
        print(f"  Image: {image.shape}, dtype: {image.dtype}")
        if isinstance(image, torch.Tensor):
            print(f"    Min: {image.min():.3f}, Max: {image.max():.3f}, Mean: {image.mean():.3f}")
        else:
            print(f"    Min: {image.min():.3f}, Max: {image.max():.3f}, Mean: {image.mean():.3f}")
    
    # Coordinates from CropResized
    if 'coords' in sample:
        print(f"  Crop Coordinates: {sample['coords']}")
    
    # Pixel size
    if 'pixel_size' in sample:
        print(f"  Pixel Size: {sample['pixel_size']}")
    
    # Clinical data
    if 'clinical_txt' in sample:
        print(f"  Clinical Text: '{sample['clinical_txt']}'")
    
    if 'questions' in sample:
        print(f"  Question: '{sample['questions']}'")
    
    # Embedding dimensions
    if 'embed_dim' in sample:
        print(f"  Embedding Dimension: {sample['embed_dim']}")
    
    if 'size_embed' in sample:
        embed = sample['size_embed']
        if hasattr(embed, 'shape'):
            print(f"  Size Embedding Shape: {embed.shape}")
        else:
            print(f"  Size Embedding: {type(embed)}")
    
    # Data size from CropResized
    if 'data_size' in sample:
        print(f"  Target Crop Size: {sample['data_size']}")

else:
    print("No dataset loaded yet. Run the 'Load Dataset' cell first.")

print("\n" + "=" * 50)

# %% Navigation Commands
# Easy navigation - modify and run this cell to change views
print("\nNavigation options:")
print("=" * 40)
print("To change the view, modify the configuration cell above and rerun from there:")
print(f"- Next {num_samples} samples: start_idx = {start_idx + num_samples}")
print(f"- Previous {num_samples} samples: start_idx = {max(0, start_idx - num_samples)}")
print("- For 6x20 grid: rows=6, cols=20, num_samples=120")
print("- For 3x4 grid: rows=3, cols=4, num_samples=12")
print(f"- Change lung side: lung_side = 'left' or 'right'")

# Example configurations:
print("\nQuick configuration examples:")
print("# For 6x20 grid (120 samples):")
print("# start_idx = 0; num_samples = 120; rows = 6; cols = 20")
print("# For 3x4 grid (12 samples):")
print("# start_idx = 0; num_samples = 12; rows = 3; cols = 4")
print("# For left lung analysis:")
print("# lung_side = 'left'")

# %% [markdown]
# ## How to Use This Interactive Script:
# 
# 1. **Run each cell** with `Ctrl+Enter` or click the "Run Cell" button
# 2. **Modify configuration** in the "Define Configuration" cell to change:
#    - `start_idx`: Starting sample index
#    - `num_samples`: Total samples to display
#    - `rows`, `cols`: Grid dimensions
# 3. **Rerun from configuration cell** to see different samples
# 4. **Plots display inline** in VS Code Python Interactive window
# 
# ### Cell-by-cell execution:
# - Cell 1: Import libraries
# - Cell 2: Set configuration parameters
# - Cell 3: Define MONAI transforms
# - Cell 4: Load dataset from JSON
# - Cell 5: Create and display visualization
# - Cell 6: Navigation help
# %%
