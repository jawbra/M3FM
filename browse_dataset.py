#!/usr/bin/env python3
"""
Quick Dataset Browser
Loads and displays CT dataset samples in a grid format
"""

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
from ct_prep import ExtractPixelSpacingd


def quick_view(json_file, start_idx=0, num_samples=12, rows=3, cols=4):
    """
    Quick view of dataset samples
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
    print(f"Loading {num_samples} samples starting from index {start_idx}...")
    with open(json_file, 'r') as f:
        data_dict = json.load(f)
    
    print(f"Total samples available: {len(data_dict)}")
    
    # Get slice of data
    end_idx = min(start_idx + num_samples, len(data_dict))
    data_slice = data_dict[start_idx:end_idx]
    actual_samples = len(data_slice)
    
    print(f"Loading samples {start_idx} to {end_idx-1} ({actual_samples} samples)")
    
    # Create dataset
    dataset = Dataset(data=data_slice, transform=transforms_list)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    fig.suptitle(f'CT Dataset Browser (Samples {start_idx}-{end_idx-1})', fontsize=14)
    
    # Flatten axes for easier indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Process samples
    for i in range(min(actual_samples, rows * cols)):
        try:
            print(f"Processing sample {i+1}/{actual_samples}...")
            
            # Load sample
            sample = dataset[i]
            
            # Get data
            image = sample['image']
            mask = sample['mask'] if 'mask' in sample else None
            
            # Convert to numpy if tensor
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            if mask is not None and isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            # Get middle slice
            if len(image.shape) == 4:  # [C, W, H, S]
                z_middle = image.shape[3] // 2
                img_slice = image[0, :, :, z_middle]
                if mask is not None:
                    mask_slice = mask[0, :, :, z_middle]
                else:
                    mask_slice = None
            else:
                print(f"Unexpected shape: {image.shape}")
                continue
            
            # Display
            ax = axes[i]
            
            # Show image
            ax.imshow(img_slice.T, cmap='gray', origin='lower')
            
            # Overlay mask
            if mask_slice is not None:
                mask_colored = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
                ax.imshow(mask_colored, alpha=0.5, cmap='Reds', origin='lower')
            
            # Get info
            actual_idx = start_idx + i
            series = data_slice[i].get('series', 'Unknown')
            
            # Set title
            ax.set_title(f'Idx: {actual_idx}\n{series[:20]}...', fontsize=8)
            ax.axis('off')
            
            # Print for copy-paste
            print(f"Index: {actual_idx}")
            print(f"Series: {series}")
            print("---")
            
        except Exception as e:
            print(f"Error loading sample {start_idx + i}: {e}")
            ax = axes[i]
            ax.text(0.5, 0.5, f'Error\nIdx: {start_idx + i}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(actual_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Browse CT dataset')
    parser.add_argument('--json_file', type=str, 
                       default='/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene_modified.json',
                       help='Path to JSON file')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index')
    parser.add_argument('--num_samples', type=int, default=12, help='Number of samples to show')
    parser.add_argument('--rows', type=int, default=3, help='Number of rows')
    parser.add_argument('--cols', type=int, default=4, help='Number of columns')
    
    args = parser.parse_args()
    
    try:
        quick_view(args.json_file, args.start_idx, args.num_samples, args.rows, args.cols)
        
        print(f"\nTo browse more samples:")
        print(f"python browse_dataset.py --start_idx {args.start_idx + args.num_samples} --num_samples {args.num_samples}")
        print(f"\nFor 6x20 grid (120 samples):")
        print(f"python browse_dataset.py --start_idx 0 --num_samples 120 --rows 6 --cols 20")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()