#!/usr/bin/env python3
"""
CT Dataset Grid Viewer - VS Code Inline Version
"""

import matplotlib
# Set backend before importing pyplot to display inline in VS Code Python Interactive
matplotlib.use('Agg')  # For inline display in VS Code Python Interactive window

import matplotlib.pyplot as plt
import json
import numpy as np
from monai import transforms
from monai.data import Dataset
import torch
import sys
import os

# Add current directory to path for imports
sys.path.append(os.getcwd())
from ct_prep import ExtractPixelSpacingd

# For VS Code Python Interactive window
try:
    # This enables inline plotting in VS Code Python Interactive
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass

def create_grid_view_inline(json_file, start_idx=0, num_samples=12):
    """
    Create grid view that displays inline in VS Code
    """
    
    # Define transforms
    transforms_list = transforms.Compose([
        transforms.LoadImaged(keys=['image', 'mask'], allow_missing_keys=True, 
                            meta_key_postfix="meta_dict", image_only=False, reader="ITKReader"),
        ExtractPixelSpacingd(keys=['image']),
        transforms.EnsureChannelFirstd(keys=['image', 'mask'], allow_missing_keys=True),
        transforms.Orientationd(keys=["image", "mask"], allow_missing_keys=True, axcodes="RAI"),
    ])
    
    # Load JSON data
    print(f"Loading data from: {json_file}")
    with open(json_file, 'r') as f:
        data_dict = json.load(f)
    
    # Get slice
    end_idx = min(start_idx + num_samples, len(data_dict))
    data_slice = data_dict[start_idx:end_idx]
    actual_samples = len(data_slice)
    
    print(f"Total: {len(data_dict)}, Loading: {actual_samples} samples ({start_idx}-{end_idx-1})")
    
    # Create dataset
    dataset = Dataset(data=data_slice, transform=transforms_list)
    
    # Calculate grid
    cols = min(4, actual_samples)
    rows = int(np.ceil(actual_samples / cols))
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    fig.suptitle(f'CT Dataset (Samples {start_idx}-{end_idx-1})', fontsize=14)
    
    # Handle axes
    if actual_samples == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Process samples
    for i in range(actual_samples):
        try:
            sample = dataset[i]
            
            # Get data
            image = sample['image']
            mask = sample['mask'] if 'mask' in sample else None
            
            # Convert to numpy
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            if mask is not None and isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            # Get middle slice
            if len(image.shape) == 4:
                z_middle = image.shape[3] // 2
                img_slice = image[0, :, :, z_middle]
                if mask is not None:
                    mask_slice = mask[0, :, :, z_middle]
                else:
                    mask_slice = None
            
            # Display
            ax = axes[i]
            ax.imshow(img_slice.T, cmap='gray', origin='lower')
            
            # Overlay mask
            if mask_slice is not None and np.any(mask_slice > 0):
                mask_colored = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
                ax.imshow(mask_colored, alpha=0.5, cmap='Reds', origin='lower')
            
            # Info
            actual_idx = start_idx + i
            series = data_slice[i].get('series', 'Unknown')
            
            ax.set_title(f'Idx: {actual_idx}\n{series[:15]}...', fontsize=8)
            ax.axis('off')
            
            print(f"Index: {actual_idx}, Series: {series}")
            
        except Exception as e:
            print(f"Error: {e}")
            ax = axes[i]
            ax.text(0.5, 0.5, f'Error\n{start_idx + i}', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.axis('off')
    
    # Hide unused
    for i in range(actual_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()  # This will display inline in VS Code Python Interactive
    
    return fig

if __name__ == "__main__":
    json_file = '/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene_modified.json'
    
    # Test with small sample
    create_grid_view_inline(json_file, start_idx=0, num_samples=4)