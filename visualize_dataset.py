#!/usr/bin/env python3
"""
Dataset Visualization Script
Loads CT images and masks from JSON file and displays them in a 6x20 grid
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


def load_and_visualize_data(json_file, start_idx=0, end_idx=120):
    """
    Load data from JSON file and create visualization
    
    Args:
        json_file: Path to JSON file containing dataset
        start_idx: Starting index in dataset (default: 0)
        end_idx: Ending index in dataset (default: 120 for 6x20 grid)
    """
    
    # Define transforms
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
    
    print(f"Total samples in dataset: {len(data_dict)}")
    
    # Slice the dataset
    data_slice = data_dict[start_idx:end_idx]
    actual_end = min(end_idx, len(data_dict))
    num_samples = len(data_slice)
    
    print(f"Displaying samples {start_idx} to {actual_end-1} ({num_samples} samples)")
    
    if num_samples == 0:
        print("No samples to display!")
        return
    
    # Create dataset
    dataset = Dataset(data=data_slice, transform=transforms_list)
    
    # Calculate grid dimensions (6 rows, up to 20 columns)
    rows = 6
    cols = min(20, int(np.ceil(num_samples / rows)))
    actual_samples = min(num_samples, rows * cols)
    
    print(f"Grid: {rows}x{cols} = {actual_samples} samples")
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    fig.suptitle(f'CT Dataset Visualization (Samples {start_idx}-{actual_end-1})', fontsize=16)
    
    # Handle single row/column cases
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Process and display samples
    for i in range(actual_samples):
        try:
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
            
            # Get middle slice (assuming shape is [C, W, H, S] after transforms)
            if len(image.shape) == 4:  # [C, W, H, S]
                z_middle = image.shape[3] // 2
                img_slice = image[0, :, :, z_middle]  # First channel, middle Z slice
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
            
            # Display image
            im = ax.imshow(img_slice.T, cmap='gray', origin='lower')  # Transpose for correct orientation
            
            # Overlay mask if available
            if mask_slice is not None:
                mask_rgba = np.zeros((*mask_slice.T.shape, 4))
                mask_rgba[:, :, 0] = 1.0  # Red channel
                mask_rgba[:, :, 3] = (mask_slice.T > 0) * 0.5  # Alpha channel for opacity
                ax.imshow(mask_rgba, origin='lower')
            
            # Get series info
            actual_idx = start_idx + i
            series = data_slice[i].get('series', 'Unknown')
            
            # Set title and print info
            ax.set_title(f'Idx: {actual_idx}\nSeries: {series}', fontsize=8)
            ax.axis('off')
            
            # Print to terminal for copy-paste
            print(f"Index: {actual_idx}, Series: {series}")
            
        except Exception as e:
            print(f"Error processing sample {start_idx + i}: {e}")
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            ax.text(0.5, 0.5, f'Error\nIdx: {start_idx + i}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide empty subplots
    for i in range(actual_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize CT dataset from JSON file')
    parser.add_argument('--json_file', type=str, 
                       default='/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene_modified.json',
                       help='Path to JSON file')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index in dataset (default: 0)')
    parser.add_argument('--end_idx', type=int, default=120,
                       help='Ending index in dataset (default: 120)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.json_file):
        print(f"Error: File not found: {args.json_file}")
        return
    
    # Load and visualize
    try:
        fig = load_and_visualize_data(args.json_file, args.start_idx, args.end_idx)
        print("\nVisualization complete!")
        print(f"To view different samples, use:")
        print(f"python visualize_dataset.py --start_idx <start> --end_idx <end>")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()