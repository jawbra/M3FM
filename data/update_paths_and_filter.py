#!/usr/bin/env python3
"""
Path Updater and File Existence Checker for NLST Dataset
Updates paths from DICOM/mask format to NPY format and filters based on file existence

This script:
1. Updates image paths: /mnt/external_ssd/NLST/nlst_dicom/test/ -> /mnt/external_ssd/NLST/nlst_npy/test/images/
2. Updates mask paths: /mnt/external_ssd/NLST/nlst_mask/test/ -> /mnt/external_ssd/NLST/nlst_npy/test/masks/
3. Checks if both image and mask files exist
4. Creates filtered JSON with only entries where both files exist

Usage:
    python update_paths_and_filter.py
    python update_paths_and_filter.py --input custom_input.json --output custom_output.json
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime


def update_and_check_paths(input_file, output_file):
    """Update paths and filter entries based on file existence"""
    
    print(f"🔄 Processing dataset: {input_file}")
    print(f"📁 Output will be saved to: {output_file}")
    print()
    
    # Load the JSON data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        print(f"📊 Loaded {len(data)} entries from input file")
    except Exception as e:
        print(f"❌ Error loading input file: {str(e)}")
        return False
    
    # Path mappings
    OLD_IMAGE_BASE = "/mnt/external_ssd/NLST/nlst_dicom/test/"
    NEW_IMAGE_BASE = "/mnt/external_ssd/NLST/nlst_npy/test/images/"
    
    OLD_MASK_BASE = "/mnt/external_ssd/NLST/nlst_mask/test/"
    NEW_MASK_BASE = "/mnt/external_ssd/NLST/nlst_npy/test/masks/"
    
    print("📂 Path mappings:")
    print(f"   Image: {OLD_IMAGE_BASE} -> {NEW_IMAGE_BASE}")
    print(f"   Mask:  {OLD_MASK_BASE} -> {NEW_MASK_BASE}")
    print()
    
    # Process entries
    valid_entries = []
    invalid_entries = []
    updated_count = 0
    
    for i, entry in enumerate(data):
        if (i + 1) % 1000 == 0:
            print(f"⏳ Processing entry {i + 1}/{len(data)}...")
        
        # Create a copy of the entry
        new_entry = entry.copy()
        
        # Update image path
        if 'image' in entry and entry['image'].startswith(OLD_IMAGE_BASE):
            # Extract the relative path and update base
            relative_path = entry['image'][len(OLD_IMAGE_BASE):]
            # For images, construct full path with series ID as filename + .npy
            if relative_path.endswith('/'):
                # Extract series ID from the path (last directory component)
                path_parts = relative_path.rstrip('/').split('/')
                if len(path_parts) >= 1:
                    series_id = path_parts[-1]  # Last part is the series ID
                    # Construct full file path: base + relative_path + series_id.npy
                    new_entry['image'] = NEW_IMAGE_BASE + relative_path + series_id + '.npy'
                else:
                    new_entry['image'] = NEW_IMAGE_BASE + relative_path
            else:
                # Already has a filename, just update base and ensure .npy extension
                if not relative_path.endswith('.npy'):
                    relative_path += '.npy'
                new_entry['image'] = NEW_IMAGE_BASE + relative_path
            updated_count += 1
        
        # Update mask path
        if 'mask' in entry and entry['mask'].startswith(OLD_MASK_BASE):
            # Extract the relative path and update base
            relative_path = entry['mask'][len(OLD_MASK_BASE):]
            # For masks, change .nii.gz to .npy
            if relative_path.endswith('.nii.gz'):
                relative_path = relative_path[:-7] + '.npy'  # Remove .nii.gz, add .npy
            new_entry['mask'] = NEW_MASK_BASE + relative_path
        
        # Check if both image and mask files exist
        image_exists = False
        mask_exists = False
        
        if 'image' in new_entry:
            # For images, we now expect full absolute paths with .npy extension
            image_path = new_entry['image']
            image_exists = os.path.isfile(image_path)
        
        if 'mask' in new_entry:
            mask_path = new_entry['mask']
            mask_exists = os.path.isfile(mask_path)
        
        # Add to appropriate list
        if image_exists and mask_exists:
            valid_entries.append(new_entry)
        else:
            invalid_entries.append({
                'original_entry': entry,
                'updated_entry': new_entry,
                'image_exists': image_exists,
                'mask_exists': mask_exists
            })
    
    # Save the filtered results
    try:
        with open(output_file, 'w') as f:
            json.dump(valid_entries, f, indent=2)
        print(f"✅ Saved {len(valid_entries)} valid entries to {output_file}")
    except Exception as e:
        print(f"❌ Error saving output file: {str(e)}")
        return False
    
    # Save invalid entries for analysis
    if invalid_entries:
        invalid_file = output_file.replace('.json', '_invalid.json')
        try:
            with open(invalid_file, 'w') as f:
                json.dump(invalid_entries, f, indent=2)
            print(f"📋 Saved {len(invalid_entries)} invalid entries to {invalid_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save invalid entries: {str(e)}")
    
    # Print summary
    print()
    print("=== Summary ===")
    print(f"📊 Total entries processed: {len(data)}")
    print(f"📊 Entries with updated paths: {updated_count}")
    print(f"✅ Valid entries (both files exist): {len(valid_entries)}")
    print(f"❌ Invalid entries (missing files): {len(invalid_entries)}")
    print(f"📈 Success rate: {len(valid_entries)/len(data)*100:.1f}%")
    
    if invalid_entries:
        # Analyze missing files
        missing_images = sum(1 for e in invalid_entries if not e['image_exists'])
        missing_masks = sum(1 for e in invalid_entries if not e['mask_exists'])
        missing_both = sum(1 for e in invalid_entries if not e['image_exists'] and not e['mask_exists'])
        
        print()
        print("=== Missing Files Analysis ===")
        print(f"📂 Missing images only: {missing_images - missing_both}")
        print(f"🎭 Missing masks only: {missing_masks - missing_both}")
        print(f"❌ Missing both files: {missing_both}")
    
    return True


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Update paths and filter dataset based on file existence')
    parser.add_argument('--input', 
                       default='../test_oct5_selene_modified.json',
                       help='Input JSON file (default: ../test_oct5_selene_modified.json)')
    parser.add_argument('--output', 
                       default='test_oct5_selene_npy_filtered.json',
                       help='Output JSON file (default: test_oct5_selene_npy_filtered.json)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    print(f"🚀 Starting path update and filtering process...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run the processing
    success = update_and_check_paths(args.input, args.output)
    
    if success:
        print(f"🎉 Processing completed successfully!")
        return 0
    else:
        print(f"💥 Processing failed!")
        return 1


if __name__ == "__main__":
    exit(main())