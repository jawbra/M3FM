#!/usr/bin/env python3
"""
Modify test_oct5_selene.json to update image and mask paths
"""

import json
import os

def modify_paths(input_json_file, output_json_file):
    """
    Modify image and mask paths in JSON file
    """
    # Read the original JSON file
    with open(input_json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} entries...")
    
    # Process each entry
    modified_count = 0
    for entry in data:
        if 'image' in entry:
            # Modify image path
            old_image_path = entry['image']
            
            # Replace base path
            new_image_path = old_image_path.replace(
                "/vol/cuttlefish/users/braj/nlst_npy/images",
                "/mnt/external_ssd/NLST/nlst_dicom"
            )
            
            # Remove the last element (filename) - keep only directory
            path_parts = new_image_path.split('/')
            if len(path_parts) > 0 and path_parts[-1].endswith('.npy'):
                # Remove the filename, keep the directory with trailing slash
                new_image_path = '/'.join(path_parts[:-1]) + '/'
            
            entry['image'] = new_image_path
            
        if 'mask' in entry:
            # Modify mask path
            old_mask_path = entry['mask']
            
            # Replace base path for masks
            new_mask_path = old_mask_path.replace(
                "/vol/cuttlefish/users/braj/nlst_npy/masks",
                "/mnt/external_ssd/NLST/nlst_mask"
            )
            
            # Change file extension from .npy to .nii.gz
            if new_mask_path.endswith('.npy'):
                new_mask_path = new_mask_path[:-4] + '.nii.gz'
            
            entry['mask'] = new_mask_path
            
        modified_count += 1
        
        # Show progress for every 1000 entries
        if modified_count % 1000 == 0:
            print(f"Processed {modified_count} entries...")
    
    # Write to new JSON file
    with open(output_json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nModification completed!")
    print(f"Input file: {input_json_file}")
    print(f"Output file: {output_json_file}")
    print(f"Modified {modified_count} entries")
    
    # Show example of changes
    if len(data) > 0:
        print("\nExample of modifications:")
        first_entry = data[0]
        if 'image' in first_entry:
            print(f"Image: {first_entry['image']}")
        if 'mask' in first_entry:
            print(f"Mask: {first_entry['mask']}")

def main():
    input_file = "/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene.json"
    output_file = "/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene_modified.json"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    print("=== JSON Path Modifier ===")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print()
    
    try:
        modify_paths(input_file, output_file)
        print(f"\n✅ Successfully created modified JSON file: {output_file}")
    except Exception as e:
        print(f"❌ Error processing file: {e}")

if __name__ == "__main__":
    main()