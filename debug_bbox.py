#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

from ct_prep import get_dataloader
import argparse

class Args:
    def __init__(self):
        self.lung_side = 'right'

def main():
    args = Args()
    
    # Create a minimal input dict
    input_dict = {}
    
    try:
        # Get dataloader with debug output
        loader, _ = get_dataloader(input_dict, args)
        
        print("Starting to iterate through dataloader...")
        
        # Get just the first batch to see debug output
        batch = next(iter(loader))
        
        print(f"Batch keys: {batch.keys()}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Mask shape: {batch['mask'].shape if 'mask' in batch else 'No mask'}")
        print(f"Coords: {batch.get('coords', 'No coords')}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()