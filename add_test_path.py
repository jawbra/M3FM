#!/usr/bin/env python3
"""
Add /test/ to image and mask paths in the modified JSON file
"""

import json
import os

def add_test_path(input_json_file, output_json_file):
    """
    Add /test/ to image and mask paths in JSON file
    """
    # Read the modified JSON file
    with open(input_json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} entries...")
    
    # Process each entry
    modified_count = 0
    for entry in data:
        if 'image' in entry:
            # Add /test/ after nlst_dicom
            old_image_path = entry['image']
            new_image_path = old_image_path.replace(
                "/mnt/external_ssd/NLST/nlst_dicom/",
                "/mnt/external_ssd/NLST/nlst_dicom/test/"
            )
            entry['image'] = new_image_path
            
        if 'mask' in entry:
            # Add /test/ after nlst_mask
            old_mask_path = entry['mask']
            new_mask_path = old_mask_path.replace(
                "/mnt/external_ssd/NLST/nlst_mask/",
                "/mnt/external_ssd/NLST/nlst_mask/test/"
            )
            entry['mask'] = new_mask_path
            
        modified_count += 1
        
        # Show progress for every 1000 entries
        if modified_count % 1000 == 0:
            print(f"Processed {modified_count} entries...")
    
    # Write to output JSON file (overwriting the same file)
    with open(output_json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nModification completed!")
    print(f"Input file: {input_json_file}")
    print(f"Output file: {output_json_file}")
    print(f"Modified {modified_count} entries")
    
    # Show example of changes
    if len(data) > 0:
        print("\nExample of updated paths:")
        first_entry = data[0]
        if 'image' in first_entry:
            print(f"Image: {first_entry['image']}")
        if 'mask' in first_entry:
            print(f"Mask: {first_entry['mask']}")

def main():
    input_file = "/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene_modified.json"
    output_file = "/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene_modified.json"  # Same file
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    print("=== JSON Path Updater - Adding /test/ ===")
    print(f"File: {input_file}")
    print()
    
    try:
        add_test_path(input_file, output_file)
        print(f"\n✅ Successfully updated JSON file with /test/ paths")
    except Exception as e:
        print(f"❌ Error processing file: {e}")

if __name__ == "__main__":
    main()