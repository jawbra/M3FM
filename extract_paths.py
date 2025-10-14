#!/usr/bin/env python3
"""
Extract relative paths from JSON file for rsync transfer
"""

import json
import os

def extract_paths_from_json(json_file, output_txt):
    """
    Extract pid/study/series directory paths from JSON file
    """
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Base path to remove
    base_path = "/vol/cuttlefish/users/braj/nlst_npy/images/"
    
    # Extract pid/study/series directories (unique set)
    series_dirs = set()
    for entry in data:
        if 'image' in entry:
            full_path = entry['image']
            if full_path.startswith(base_path):
                relative_path = full_path[len(base_path):]
                # Split path and get pid/study/series
                path_parts = relative_path.split('/')
                if len(path_parts) >= 3:
                    pid = path_parts[0]
                    study = path_parts[1] 
                    series = path_parts[2]
                    series_path = f"{pid}/{study}/{series}"
                    series_dirs.add(series_path)
                else:
                    print(f"Warning: Path doesn't have enough levels: {relative_path}")
            else:
                print(f"Warning: Path doesn't start with expected base: {full_path}")
    
    # Convert to sorted list
    unique_series = sorted(list(series_dirs))
    
    # Write to txt file
    with open(output_txt, 'w') as f:
        for series_path in unique_series:
            f.write(series_path + '/\n')  # Add trailing slash for directories
    
    print(f"Extracted {len(unique_series)} unique series directories to {output_txt}")
    return len(unique_series)

if __name__ == "__main__":
    json_file = "/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene.json"
    output_txt = "/home/brandtj/Documents/projects/iderha/M3FM/file_paths.txt"
    
    extract_paths_from_json(json_file, output_txt)