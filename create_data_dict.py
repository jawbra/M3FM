import os
import json
from pathlib import Path
import nibabel as nib
import numpy as np

def get_dcm_folders(base_path):
    """Get all folders containing DICOM files"""
    dcm_folders = []
    for root, dirs, files in os.walk(base_path):
        if any(file.endswith('.dcm') for file in files):
            dcm_folders.append(os.path.relpath(root, base_path))
    return dcm_folders

def reorder_nifti_zooms(zooms):
    return [zooms[2], zooms[0], zooms[1]]

def create_dataset_dict(folders, base_image, base_mask):
    """Create list of dictionaries for each folder"""
    dataset = []
    for folder in folders:
        data_dict = {
            'image': str(Path(base_image) / folder),
            'mask': str(Path(base_mask) / folder / f'{folder.split("/")[-1]}.nii.gz'),
            #get pixel size from nifti header
            'pixel_size': str(reorder_nifti_zooms(nib.load(str(Path(base_mask) / folder / f'{folder.split("/")[-1]}.nii.gz')).header.get_zooms())),
            'clinical_txt': "No patient information available.",
            'question': 'Predict the lung cancer risk over six years.',
            'config_file': 'config_files/config_m3mf_cancer_risk.py'
        }
        dataset.append(data_dict)
    return dataset

def main():
    # Base path for DICOM data
    base_image = '/home/brandtj/Documents/projects/iderha/M3FM/nlst_data/dicom'
    base_mask = '/home/brandtj/Documents/projects/iderha/M3FM/nlst_data/masks'
    
    # Get all DICOM folders
    dcm_folders = get_dcm_folders(base_image)
    
    # Create dataset dictionary
    dataset = {
        'training': create_dataset_dict(dcm_folders, base_image, base_mask),
        'validation': [],
        'testing': []
    }
    
    # Save to JSON file
    save_path = '/home/brandtj/Documents/projects/iderha/M3FM/data'
    name = 'dataset.json'
    output_path = os.path.join(save_path, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Dataset created and saved to {output_path}")
    print(f"Total folders: {len(dcm_folders)}")

if __name__ == '__main__':
    main()