from monai.data import Dataset
from torch.utils.data import DataLoader
from monai import transforms
from monai.transforms import MapTransform
import torch
import numpy as np
from util import txt2embed, get_sincos_size_embed
import json
import logging
import os
from datetime import datetime


class DefineEmbedDim(MapTransform):
    def __init__(self, keys, embed_dim=1024, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.embed_dim = embed_dim
    
    def __call__(self, data):
        d = dict(data)
        # Get spacing from image metadata
        d['embed_dim'] = self.embed_dim
        return d
    
    
class ClinicalText(MapTransform):
    def __init__(self, keys, question='', clinical_text='', data_name='', allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.question = question
        self.clinical_text = clinical_text
        self.data_name = data_name
        self.data_name_list = []
    
    def __call__(self, data):
        d = dict(data)
        # Get spacing from image metadata
        question_ids, question_masks = txt2embed(self.question)
        txt_ids, txt_masks = txt2embed(self.clinical_text, max_length=160)
        self.data_name_list.append(self.data_name)

        d['questions'] = self.question,
        d['txt_ids'] = txt_ids
        d['txt_mask'] = txt_masks
        d['clinical_txt'] = self.clinical_text
        d['questions_ids'] = question_ids
        d['questions_mask'] = question_masks

        return d
    
class SizeEmbed(MapTransform):
    def __init__(self, keys, patch_size=[16, 16, 16], allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.patch_size = patch_size
    
    def __call__(self, data):
        d = dict(data)
        # Get spacing from image metadata
        d["patch_size"] = self.patch_size
        pix_size = d['pixel_size']
        embed_dim = d["embed_dim"]
        sizes = np.array([[pix_size[0] * self.patch_size[0], pix_size[1] * self.patch_size[1], pix_size[2] * self.patch_size[2]]])
        size_embed = get_sincos_size_embed(embed_dim, sizes)
        #size_embed = torch.from_numpy(size_embed).to(torch.float32)
        d['size_embed'] = size_embed
        return d

class ExtractPixelSpacingd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        # Get spacing from image metadata
        if "image_meta_dict" in d:
            spacing = d["image_meta_dict"].get("spacing", [1.4,1.4,2.5])
            # MONAI loader returns spacing as (w, h, d), we need (d, h, w)
            d["pixel_size"] = [spacing[2], spacing[1], spacing[0]] # here spacing is adjusted, since we reorient the image later in transfomrs by calling transforms.Orientationd
        return d

class CropResizedFailureException(Exception):
    """Custom exception for CropResized transform failures"""
    pass


class CropResized(MapTransform):
    """
    Custom MONAI transform for cropping and resizing 3D images with lung selection
    Robust version with error handling and failure logging
    """
    def __init__(self, keys, crop_size, lung='right', allow_missing_keys=False, 
                 failed_subjects_file='failed_inference_subjects.txt'):
        """
        Args:
            keys: Keys to be transformed
            crop_size: Target size (depth, height, width)
            lung: Which lung to select ('right' or 'left')
            allow_missing_keys: Whether to allow missing keys
            failed_subjects_file: File to log failed subjects
        """
        super().__init__(keys, allow_missing_keys)
        self.crop_size = crop_size
        self.lung = lung
        self.failed_subjects_file = failed_subjects_file

    def log_failure(self, subject_id, error_msg, error_type="CropResize"):
        """Log failed subject to file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {error_type}: {subject_id} - {error_msg}\n"
            
            with open(self.failed_subjects_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            logging.warning(f"Could not write to failed subjects file: {str(e)}")

    def compute_bbox(self, mask):
        """Compute bounding box coordinates from mask with error handling"""
        try:
            # Validate mask input
            if mask is None:
                raise ValueError("Mask is None")
            
            if mask.size == 0:
                raise ValueError("Mask is empty")
            
            # Make a copy to avoid modifying original
            mask_copy = mask.copy()
            
            # Get rid of the hull
            mask_copy[mask_copy == 1] = 0
            
            # Select appropriate lung
            if self.lung == 'right':
                mask_copy[mask_copy < 4] = 0  # select right lung only
                mask_copy[mask_copy >= 2] = 1
            else:
                mask_copy[mask_copy > 3] = 0  # select left lung only
                mask_copy[mask_copy >= 2] = 1
            
            # Find non-zero points
            points = np.array(np.where(mask_copy > 0))
            
            # Check if we found any points
            if points.size == 0 or points.shape[1] == 0:
                raise ValueError(f"No {self.lung} lung pixels found in mask")
            
            # Get min and max for each dimension
            mins = np.min(points, axis=1)
            maxs = np.max(points, axis=1)
            
            # Validate bounding box dimensions
            if len(mins) != 3 or len(maxs) != 3:
                raise ValueError(f"Invalid bounding box dimensions: mins={len(mins)}, maxs={len(maxs)}")
            
            # Check for valid coordinate ranges
            coords = [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]
            for i in range(0, len(coords), 2):
                if coords[i] >= coords[i+1]:
                    raise ValueError(f"Invalid coordinate range: {coords[i]} >= {coords[i+1]}")
            
            # Return coordinates in format [s1, s2, h1, h2, w1, w2]
            return coords
            
        except Exception as e:
            raise CropResizedFailureException(f"compute_bbox failed: {str(e)}")

    def crop_resize(self, x, coord, pix_size, crop_size):
        """Crop and resize with error handling"""
        try:
            # Validate inputs
            if x is None or x.size == 0:
                raise ValueError("Input image is None or empty")
            
            if len(coord) != 6:
                raise ValueError(f"Invalid coordinate length: {len(coord)} (expected 6)")
            
            if pix_size is None or len(pix_size) != 3:
                raise ValueError(f"Invalid pixel size: {pix_size}")
            
            s, h, w = x.shape
            ts, th, tw = crop_size

            # Validate crop coordinates are within image bounds
            if coord[0] < 0 or coord[1] > s or coord[2] < 0 or coord[3] > h or coord[4] < 0 or coord[5] > w:
                raise ValueError(f"Crop coordinates out of bounds: coords={coord}, image_shape=({s},{h},{w})")

            sl = coord[1] - coord[0]  # slice
            hl = coord[3] - coord[2]  # height  
            wl = coord[5] - coord[4]  # width

            # Check for zero or negative dimensions
            if sl <= 0 or hl <= 0 or wl <= 0:
                raise ValueError(f"Invalid crop dimensions: slice={sl}, height={hl}, width={wl}")

            # Crop the image
            x_cropped = x[coord[0]:coord[1], coord[2]:coord[3], coord[4]:coord[5]]
            
            if x_cropped.size == 0:
                raise ValueError("Cropped image is empty")

            # Calculate new pixel sizes
            ps = pix_size[0] * sl / ts
            ph = pix_size[1] * hl / th
            pw = pix_size[2] * wl / tw

            # Resize using interpolation
            x_resized = torch.nn.functional.interpolate(
                torch.from_numpy(x_cropped).unsqueeze(0).unsqueeze(0).to(torch.float32),
                size=(ts, th, tw),
                mode="trilinear",
                align_corners=False,
            ).numpy().squeeze()

            # Validate output
            if x_resized.shape != crop_size:
                raise ValueError(f"Unexpected output shape: {x_resized.shape} (expected {crop_size})")

            return x_resized, [ps, ph, pw]
            
        except Exception as e:
            raise CropResizedFailureException(f"crop_resize failed: {str(e)}")

    def __call__(self, data):
        d = dict(data)
        
        # Extract subject identifier for logging
        subject_id = "unknown"
        try:
            if 'series' in d:
                subject_id = d['series']
            elif 'image' in d and isinstance(d['image'], str):
                subject_id = os.path.basename(d['image'])
            elif 'mask' in d and isinstance(d['mask'], str):
                subject_id = os.path.basename(d['mask'])
        except:
            pass
        
        try:
            for key in self.key_iterator(d):
                if key == 'image':
                    # Get the image and mask data
                    img = d[key]
                    mask = d['mask']
                    
                    # Validate inputs exist
                    if img is None:
                        raise ValueError("Image data is None")
                    if mask is None:
                        raise ValueError("Mask data is None")
                    
                    # Convert tensors to numpy if needed
                    if isinstance(img, torch.Tensor):
                        img = img.numpy()
                    if isinstance(mask, torch.Tensor):
                        mask = mask.numpy()
                    
                    # Validate array shapes
                    if img.ndim == 0 or mask.ndim == 0:
                        raise ValueError(f"Invalid array dimensions: img={img.ndim}D, mask={mask.ndim}D")
                    
                    if img.shape[0] == 0 or mask.shape[0] == 0:
                        raise ValueError("Empty image or mask arrays")
                    
                    # Compute bounding box from mask
                    coords = self.compute_bbox(mask[0])  # Assuming first channel is the mask
                    
                    # Get pixel size from the data dictionary
                    pix_size = d.get('pixel_size', None)
                    if pix_size is None:
                        raise ValueError("Pixel size not found in data")
                    
                    # Apply crop_resize
                    img_transformed, new_pix_size = self.crop_resize(img[0], coords, pix_size, self.crop_size)
                    
                    # Update the dictionary
                    d[key] = img_transformed[None]  # Add channel dimension back
                    d['coords'] = {'cancer_risk': coords}
                    d['data_size'] = self.crop_size
            
            return d
            
        except CropResizedFailureException as e:
            # Log the failure and raise to skip this item
            self.log_failure(subject_id, str(e), "CropResized")
            logging.warning(f"CropResized failed for subject {subject_id}: {str(e)}")
            raise e
            
        except Exception as e:
            # Log unexpected errors
            error_msg = f"Unexpected error: {str(e)}"
            self.log_failure(subject_id, error_msg, "CropResized")
            logging.error(f"CropResized unexpected error for subject {subject_id}: {str(e)}")
            raise CropResizedFailureException(error_msg)



# # Define the dataset and dataloader
# data_dict =[{'image': '/home/brandtj/Documents/projects/iderha/M3FM/nlst_data/dicom/100012/1.2.840.113654.2.55.240231128564881525363489796879328810792',
#     'mask': '/home/brandtj/Documents/projects/iderha/M3FM/nlst_data/masks/100012/1.2.840.113654.2.55.240231128564881525363489796879328810792/1.2.840.113654.2.55.240231128564881525363489796879328810792.nii.gz'},]


class RobustDataset(Dataset):
    """
    Custom dataset that handles transform failures gracefully by skipping failed items
    """
    def __init__(self, data, transform, failed_subjects_file='failed_inference_subjects.txt'):
        super().__init__(data=data, transform=transform)
        self.failed_subjects_file = failed_subjects_file
        self.failed_indices = set()
        self.valid_indices = None
        
    def log_failure(self, index, subject_id, error_msg):
        """Log failed subject to file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] Dataset[{index}]: {subject_id} - {error_msg}\n"
            
            with open(self.failed_subjects_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            logging.warning(f"Could not write to failed subjects file: {str(e)}")
    
    def __getitem__(self, index):
        # Skip if we know this index failed before
        if index in self.failed_indices:
            return None
            
        try:
            return super().__getitem__(index)
        except (CropResizedFailureException, RuntimeError) as e:
            # Extract subject identifier for logging
            subject_id = "unknown"
            try:
                data_item = self.data[index]
                if 'series' in data_item:
                    subject_id = data_item['series']
                elif 'image' in data_item:
                    subject_id = os.path.basename(str(data_item['image']))
                elif 'mask' in data_item:
                    subject_id = os.path.basename(str(data_item['mask']))
            except:
                pass
            
            # Log the failure
            error_msg = str(e)
            if "CropResized" in error_msg:
                error_type = "Transform"
            else:
                error_type = "Runtime"
                
            self.log_failure(index, subject_id, f"{error_type} error: {error_msg}")
            self.failed_indices.add(index)
            
            # Return None to indicate this item should be skipped
            return None
        except Exception as e:
            # Handle other unexpected errors
            subject_id = "unknown"
            try:
                data_item = self.data[index]
                if 'series' in data_item:
                    subject_id = data_item['series']
            except:
                pass
                
            self.log_failure(index, subject_id, f"Unexpected error: {str(e)}")
            self.failed_indices.add(index)
            return None


def collate_fn_skip_none(batch):
    """Custom collate function that filters out None values (failed transforms)"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
        
    # Use the default collate for valid items
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


def get_dataloader(input_dict, args):
    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=['image', 'mask'], allow_missing_keys=True, meta_key_postfix="meta_dict", image_only=False),
        ExtractPixelSpacingd(keys=['image']),  # Extract pixel spacing after loading
        transforms.EnsureChannelFirstd(keys=['image', 'mask'], allow_missing_keys=True),
        #transforms.Orientationd(keys=["image", "mask"], allow_missing_keys=True, axcodes="RAI"),
        transforms.Transposed(keys=["image", "mask"], indices=(0, 3, 1, 2), allow_missing_keys=True),
        DefineEmbedDim(keys=['image'], embed_dim=1024, allow_missing_keys=True),
        SizeEmbed(keys=['image'], allow_missing_keys=True),
        ClinicalText(keys=['image'], question='Predict the lung cancer risk over six years.', clinical_text='No patient information available.', data_name='cancer_risk', allow_missing_keys=True),
        #transforms.Rotate90d(keys=["image", "mask"], k=1, spatial_axes=(1,2), allow_missing_keys=True),
        #transforms.ScaleIntensityRanged(keys=['image'], a_min=-1300, a_max=150, b_min=-1.0, b_max=1.0, clip=True),
        # transforms.CropForegroundd(keys=['image', 'mask'], source_key='mask', margin=0),
        # transforms.Resized(keys=['image', 'mask'], spatial_size=(128,320,448)),
        CropResized(keys=['image', 'mask'], crop_size=(128, 448, 320), lung=args.lung_side),
        #transforms.Rotate90d(keys=["image", "mask"], k=1, spatial_axes=(1,2), allow_missing_keys=True),
        transforms.ToTensord(keys=['image', 'mask', 'questions_ids', 'questions_mask', 'txt_ids', 'txt_mask', 'data_size'], allow_missing_keys=True),
    ])

    with open('/home/brandtj/Documents/projects/iderha/M3FM/data/test_oct5_selene_npy_filtered.json', 'r') as f:
        data_dict = json.load(f)
    
    # Use our robust dataset that can handle transform failures
    data = RobustDataset(data=data_dict, transform=train_transforms)
    
    # Use custom collate function that filters out None values (failed items)
    loader = DataLoader(
        data, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with error handling
        pin_memory=True, 
        collate_fn=collate_fn_skip_none
    )

    return loader, input_dict



# train_transforms = transforms.Compose([
#     transforms.LoadImaged(keys=['image', 'mask'], allow_missing_keys=True, meta_key_postfix="meta_dict", image_only=False),
#     ExtractPixelSpacingd(keys=['image']),  # Extract pixel spacing after loading
#     transforms.EnsureChannelFirstd(keys=['image', 'mask'], allow_missing_keys=True),
#     transforms.Orientationd(keys=["image", "mask"], allow_missing_keys=True, axcodes="RAI"),
#     transforms.Transposed(keys=["image", "mask"], indices=(0, 3, 1, 2), allow_missing_keys=True),
#     DefineEmbedDim(keys=['image'], embed_dim=1024, allow_missing_keys=True),
#     SizeEmbed(keys=['image'], allow_missing_keys=True),
#     ClinicalText(keys=['image'], question='Predict the lung cancer risk over six years.', clinical_text='No patient information available.', data_name=['cancer_risk'],allow_missing_keys=True),
#     transforms.Rotate90d(keys=["image", "mask"], k=1, spatial_axes=(1,2), allow_missing_keys=True),
#     transforms.ScaleIntensityRanged(keys=['image'], a_min=-1300, a_max=150, b_min=-1.0, b_max=1.0, clip=True),
#     # transforms.CropForegroundd(keys=['image', 'mask'], source_key='mask', margin=0),
#     # transforms.Resized(keys=['image', 'mask'], spatial_size=(128,320,448)),
#     CropResized(keys=['image', 'mask'], crop_size=(128, 448, 320), lung='left'),
#     #transforms.Rotate90d(keys=["image", "mask"], k=1, spatial_axes=(1,2), allow_missing_keys=True),
#     transforms.ToTensord(keys=['image', 'mask', 'questions_ids', 'question_mask', 'txt_ids', 'txt_mask'], allow_missing_keys=True),
# ])

# data = Dataset(data=data_dict, transform=train_transforms)
# loader = DataLoader(data, batch_size=1)


# batch = next(iter(loader))
# np.save('test.npy',batch['image'][0,0,:,:,:])
# np.save('test_mask.npy',batch['mask'][0,0,:,:,:])
# print('stop')


#find dicom through dataset path

#compute box for lung

#run inference with this template

# input_data = {
#     'ct_path': 'demo_data/ct_npy/cancer_risk.npy', # change this to simple numpy array
#     'pixel_size': [2.0, 0.53, 0.53],
#     'coords': {'cancer_risk': [7, 132, 112, 450, 233, 490]},
#     'clinical_txt': "No patient information available", #stays the same
#     'question': 'Predict the lung cancer risk over six years.', #stays the same
#     'config_file': 'config_files/config_m3mf_cancer_risk.py' #stays the same
# }