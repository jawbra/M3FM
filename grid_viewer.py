#!/usr/bin/env python3
"""
CT Dataset Grid Viewer
Creates a 6x20 grid visualization of CT images with masks
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


def create_grid_view(json_file, start_idx=0):
    """
    Create 6x20 grid view of CT dataset
    """
    
    # Define transforms (exactly as specified)
    transforms_list = transforms.Compose([
        transforms.LoadImaged(keys=['image', 'mask'], allow_missing_keys=True, 
                            meta_key_postfix="meta_dict", image_only=False, reader="ITKReader"),
        ExtractPixelSpacingd(keys=['image']),  # Extract pixel spacing after loading
        transforms.EnsureChannelFirstd(keys=['image', 'mask'], allow_missing_keys=True),
        transforms.Orientationd(keys=["image", "mask"], allow_missing_keys=True, axcodes="RAI"),
    ])
    
    # Load JSON data
    print(f"Loading data from: {json_file}")
    with open(json_file, 'r') as f:
        data_dict = json.load(f)
    
    total_samples = len(data_dict)
    print(f"Total samples available: {total_samples}")
    
    # Calculate grid parameters
    rows, cols = 6, 20
    samples_to_load = rows * cols  # 120 samples
    end_idx = min(start_idx + samples_to_load, total_samples)
    
    # Get slice of data
    data_slice = data_dict[start_idx:end_idx]
    actual_samples = len(data_slice)
    
    print(f"Displaying samples {start_idx} to {end_idx-1} ({actual_samples} samples)")
    print(f"Grid: {rows}x{cols}")
    
    # Create dataset
    dataset = Dataset(data=data_slice, transform=transforms_list)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    fig.suptitle(f'CT Dataset Grid View (Samples {start_idx}-{end_idx-1})', fontsize=16)
    
    print("\n" + "="*60)
    print("PROCESSING SAMPLES - Copy indices/series from below:")
    print("="*60)
    
    # Process samples
    for i in range(min(actual_samples, rows * cols)):
        try:
            if (i + 1) % 10 == 0:
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
            
            # Get middle slice - assuming shape [C, W, H, S] after transforms
            if len(image.shape) == 4:
                z_middle = image.shape[3] // 2
                img_slice = image[0, :, :, z_middle]  # Shape: [W, H]
                if mask is not None:
                    mask_slice = mask[0, :, :, z_middle]
                else:
                    mask_slice = None
            else:
                print(f"Unexpected image shape: {image.shape}")
                continue
            
            # Calculate subplot position
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Display image (transpose for correct orientation)
            ax.imshow(img_slice.T, cmap='gray', origin='lower', aspect='equal')
            
            # Overlay mask with 0.5 opacity
            if mask_slice is not None and np.any(mask_slice > 0):
                # Create colored overlay
                mask_colored = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
                ax.imshow(mask_colored, alpha=0.5, cmap='Reds', origin='lower', aspect='equal')
            
            # Get series info
            actual_idx = start_idx + i
            series = data_slice[i].get('series', 'Unknown')
            
            # Set title (truncate series for readability)
            series_short = series[:15] + "..." if len(series) > 15 else series
            ax.set_title(f'{actual_idx}\n{series_short}', fontsize=6)
            ax.axis('off')
            
            # Print for copy-paste (every sample)
            print(f"Index: {actual_idx}, Series: {series}")
            
        except Exception as e:
            print(f"ERROR processing sample {start_idx + i}: {e}")
            
            # Show error in subplot
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            ax.text(0.5, 0.5, f'Error\n{start_idx + i}', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=8, color='red')
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(actual_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    print("="*60)
    print(f"GRID COMPLETE: {actual_samples} samples processed")
    print("="*60)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CT Dataset 6x20 Grid Viewer')
    parser.add_argument('--json_file', type=str, 
                       default='/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene_modified.json',
                       help='Path to JSON file')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index in dataset (default: 0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"Error: File not found: {args.json_file}")
        exit(1)
    
    try:
        fig = create_grid_view(args.json_file, args.start_idx)
        
        print(f"\nNavigation commands:")
        print(f"Next 120 samples: python grid_viewer.py --start_idx {args.start_idx + 120}")
        print(f"Previous 120 samples: python grid_viewer.py --start_idx {max(0, args.start_idx - 120)}")
        print(f"Jump to index: python grid_viewer.py --start_idx <index>")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()