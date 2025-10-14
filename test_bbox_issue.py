#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

import torch
import numpy as np
from ct_prep import CropResized

def test_compute_bbox():
    # Create a test mask similar to lung segmentation
    mask_shape = (100, 100, 50)
    mask = np.zeros(mask_shape)
    
    # Create some mock lung regions
    # Right lung (values 4-5)
    mask[20:40, 30:70, 10:40] = 4  # Right lung
    # Left lung (values 2-3)  
    mask[60:80, 30:70, 10:40] = 2  # Left lung
    # Hull (value 1)
    mask[45:55, 30:70, 10:40] = 1  # Hull to be removed
    
    print(f"Test mask shape: {mask.shape}")
    print(f"Test mask unique values: {np.unique(mask)}")
    
    # Test with numpy array directly
    crop_resized = CropResized(keys=['image', 'mask'], crop_size=(320, 448, 128), lung='right')
    
    print("\n=== Testing with numpy array ===")
    coords_numpy = crop_resized.compute_bbox(mask)
    
    print("\n=== Testing with torch tensor ===")
    mask_torch = torch.from_numpy(mask)
    coords_torch = crop_resized.compute_bbox(mask_torch)
    
    print(f"\nNumpy coords: {coords_numpy}")
    print(f"Torch coords: {coords_torch}")
    print(f"Coords equal: {coords_numpy == coords_torch}")

if __name__ == "__main__":
    test_compute_bbox()