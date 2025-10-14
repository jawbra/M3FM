from monai.data import Dataset
from torch.utils.data import DataLoader
from monai import transforms
from monai.transforms import MapTransform
import torch
import numpy as np
from util import txt2embed, get_sincos_size_embed
import json


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
        #print(sizes)
        return d

class CreateOutcomeFields(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        # Create 'golds' and 'censors' from existing outcome data
        # 'golds' typically represents the ground truth labels (y_seq)
        # 'censors' typically represents censoring indicators (1 - y_mask, where 0=censored, 1=observed)
        
        if 'y_seq' in d and 'y_mask' in d:
            # Ensure we have proper lists, not numpy arrays or other types
            y_seq = d['y_seq'] if isinstance(d['y_seq'], list) else list(d['y_seq'])
            y_mask = d['y_mask'] if isinstance(d['y_mask'], list) else list(d['y_mask'])
            
            # Convert to numpy arrays to ensure proper format for MONAI
            d['golds'] = np.array(y_seq)
            # Censors: 1 where event is censored (y_mask=0), 0 where event is observed (y_mask=1)
            d['censors'] = np.array([1 - mask for mask in y_mask])
        else:
            # Fallback if fields don't exist
            d['golds'] = np.array([])
            d['censors'] = np.array([])
        
        return d

class DebugShape(MapTransform):
    def __init__(self, keys, name="", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.name = name
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                shape = d[key].shape if hasattr(d[key], 'shape') else 'No shape'
                print(f"DEBUG {self.name}: {key} shape = {shape}")
        return d

class ExtractPixelSpacingd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        # Get spacing from image metadata
        if "image_meta_dict" in d:
            spacing = d["image_meta_dict"].get("spacing", [1.4,1.4,2.5])
            # MONAI loader returns spacing as (w, h, d)
            # After Orientationd(RAI), data order is (W, H, S) which matches (w, h, d)
            # So we can use spacing directly: [width_spacing, height_spacing, depth_spacing]
            d["pixel_size"] = [spacing[2], spacing[0], spacing[1]]  # (W, H, S) order
        return d

class CropResized(MapTransform):
    """
    Custom MONAI transform for cropping and resizing 3D images with lung selection
    """
    def __init__(self, keys, crop_size, lung='right', allow_missing_keys=False):
        """
        Args:
            keys: Keys to be transformed
            crop_size: Target size (width, height, depth) - matching input (W, H, S) order
            lung: Which lung to select ('right' or 'left')
            allow_missing_keys: Whether to allow missing keys
        """
        super().__init__(keys, allow_missing_keys)
        self.crop_size = crop_size
        self.lung = lung

    def compute_bbox(self, mask):
        """Compute bounding box coordinates from mask with padding"""
        
        # Work with tensor directly - don't modify the original mask in place!
        if isinstance(mask, torch.Tensor):
            mask = mask.clone()
        else:
            mask = mask.copy()
            
        # Select appropriate lung based on new label scheme:
        # Left lung: labels 1, 2
        # Right lung: labels 3, 4, 5
        if self.lung == 'right':
            # Keep only right lung labels (3, 4, 5), set everything else to 0
            mask[(mask < 3) | (mask > 5)] = 0
            # Convert remaining right lung labels to binary 1
            mask[mask >= 3] = 1
        else:
            # Keep only left lung labels (1, 2), set everything else to 0  
            mask[(mask < 1) | (mask > 2)] = 0
            # Convert remaining left lung labels to binary 1
            mask[mask >= 1] = 1
        
        # Find non-zero points - handle both tensor and numpy cases
        if isinstance(mask, torch.Tensor):
            points = torch.where(mask > 0)
            if len(points[0]) == 0:
                w, h, s = mask.shape
                raise ValueError(f"No {self.lung} lung pixels found in mask after label selection. "
                               f"Mask shape: (D: {s}, H: {h}, W: {w}). "
                               f"Check if the mask contains the correct lung labels.")
            
            # Convert to numpy for min/max operations
            points_array = torch.stack(points).cpu().numpy()
            mins = np.min(points_array, axis=1)
            maxs = np.max(points_array, axis=1)
            w, h, s = mask.shape
        else:
            points = np.array(np.where(mask > 0))
            if points.size == 0:
                w, h, s = mask.shape
                raise ValueError(f"No {self.lung} lung pixels found in mask after label selection. "
                               f"Mask shape: (D: {s}, H: {h}, W: {w}). "
                               f"Check if the mask contains the correct lung labels.")
            
            mins = np.min(points, axis=1)
            maxs = np.max(points, axis=1)
            w, h, s = mask.shape
        
        # Add padding based on actual dimension order: (W, H, S)
        # 1st dim (index 0): Width - 3 voxel padding
        # 2nd dim (index 1): Height - 3 voxel padding  
        # 3rd dim (index 2): Slice/Depth - 1 voxel padding
        
        # Apply padding with boundary checks - ensure we work with scalar values
        w1 = max(0, int(mins[0]) - 3)  # width padding: -3 voxels
        w2 = min(w, int(maxs[0]) + 3)  # width padding: +3 voxels
        h1 = max(0, int(mins[1]) - 3)  # height padding: -3 voxels
        h2 = min(h, int(maxs[1]) + 3)  # height padding: +3 voxels
        s1 = max(0, int(mins[2]) - 1)  # slice padding: -1 voxel
        s2 = min(s, int(maxs[2]) + 1)  # slice padding: +1 voxel
        
        # Validate final coordinates to ensure positive dimensions
        final_wl = w2 - w1
        final_hl = h2 - h1
        final_sl = s2 - s1
        
        if final_wl <= 0 or final_hl <= 0 or final_sl <= 0:
            raise ValueError(f"Invalid bounding box dimensions after padding for {self.lung} lung: "
                           f"width={final_wl}, height={final_hl}, slices={final_sl}. "
                           f"Coords: [w:{w1}-{w2}, h:{h1}-{h2}, s:{s1}-{s2}]. "
                           f"Original mask shape: (D: {s}, H: {h}, W: {w})")
        
        # Return coordinates in format [w1, w2, h1, h2, s1, s2] for (W, H, S) order
        return [w1, w2, h1, h2, s1, s2]

    def crop_resize(self, x, coord, pix_size, crop_size):
                # Input shape is (W, H, S) - width, height, slice/depth
                w, h, s = x.shape
                tw, th, ts = crop_size  # Target size: (width, height, slices)

                wl = coord[1] - coord[0]  # width length
                hl = coord[3] - coord[2]  # height length
                sl = coord[5] - coord[4]  # slice length

                # Validate dimensions before cropping - check for zero or negative dimensions
                if wl <= 0 or hl <= 0 or sl <= 0:
                    raise ValueError(f"Invalid crop dimensions: width={wl}, height={hl}, slices={sl}. "
                                   f"Original shape: (D: {s}, H: {h}, W: {w}), "
                                   f"Crop coords: [w:{coord[0]}-{coord[1]}, h:{coord[2]}-{coord[3]}, s:{coord[4]}-{coord[5]}]")

                # Crop using coordinate order matching (W, H, S)
                x = x[coord[0]:coord[1], coord[2]:coord[3], coord[4]:coord[5]]
                
                # Validate cropped shape before interpolation
                crop_w, crop_h, crop_s = x.shape
                if crop_w <= 0 or crop_h <= 0 or crop_s <= 0:
                    raise ValueError(f"Invalid cropped image dimensions: (D: {crop_s}, H: {crop_h}, W: {crop_w}). "
                                   f"Cannot interpolate to target size (D: {ts}, H: {th}, W: {tw})")

                # Calculate new pixel sizes
                pw = pix_size[0] * wl / tw  # width pixel size
                ph = pix_size[1] * hl / th  # height pixel size
                ps = pix_size[2] * sl / ts  # slice pixel size

                # Interpolate to target size (tw, th, ts) - width, height, slices
                x = torch.nn.functional.interpolate(
                    torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(torch.float32),
                    size=(tw, th, ts),
                    mode="trilinear",
                    align_corners=False,
                ).numpy().squeeze()

                return x, [pw, ph, ps]

    def __call__(self, data):
        d = dict(data)
        
        try:
            # First, compute bounding box from mask (if available)
            coords = None
            if 'mask' in d:
                mask = d['mask']
                coords = self.compute_bbox(mask[0])  # Assuming first channel is the mask
                d['coords'] = {'cancer_risk': coords}
                d['data_size'] = self.crop_size
            
            # Get pixel size from the data dictionary
            pix_size = d.get('pixel_size', None)
            
            # Process each key in the transform
            for key in self.key_iterator(d):
                if key in ['image', 'mask'] and coords is not None:
                    # Get the data
                    data_array = d[key]
                    if isinstance(data_array, torch.Tensor):
                        data_array = data_array.numpy()
                    
                    # Apply crop_resize (same operation for both image and mask)
                    if key == 'image':
                        transformed_data, new_pix_size = self.crop_resize(data_array[0], coords, pix_size, self.crop_size)
                        #d['pixel_size'] = new_pix_size  # Only update pixel size once for image
                    else:  # key == 'mask'
                        transformed_data, _ = self.crop_resize(data_array[0], coords, pix_size, self.crop_size)
                    
                    # Update the dictionary
                    d[key] = transformed_data[None]  # Add channel dimension back
                    
        except (ValueError, RuntimeError) as e:
            # Log the error with sample information
            sample_info = ""
            if 'image' in d:
                sample_info = f" (sample: {d.get('image', 'unknown')})"
            
            # Re-raise with more context - this will be caught by MONAI's error handling
            # and properly handled by the DataLoader
            raise RuntimeError(f"CropResized transform failed for {self.lung} lung{sample_info}: {str(e)}")
                
        return d



# # Define the dataset and dataloader
# data_dict =[{'image': '/home/brandtj/Documents/projects/iderha/M3FM/nlst_data/dicom/100012/1.2.840.113654.2.55.240231128564881525363489796879328810792',
#     'mask': '/home/brandtj/Documents/projects/iderha/M3FM/nlst_data/masks/100012/1.2.840.113654.2.55.240231128564881525363489796879328810792/1.2.840.113654.2.55.240231128564881525363489796879328810792.nii.gz'},]


def get_dataloader(input_dict, args):
    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=['image', 'mask'], allow_missing_keys=True, meta_key_postfix="meta_dict", image_only=False, reader="ITKReader"),
        ExtractPixelSpacingd(keys=['image']),  # Extract pixel spacing after loading
        transforms.EnsureChannelFirstd(keys=['image', 'mask'], allow_missing_keys=True),
        transforms.Orientationd(keys=["image", "mask"], allow_missing_keys=True, axcodes="ALS"),
        #transforms.Transposed(keys=["image", "mask"], indices=(0, 3, 1, 2), allow_missing_keys=True),
        DefineEmbedDim(keys=['image'], embed_dim=1024, allow_missing_keys=True),
        SizeEmbed(keys=['image'], allow_missing_keys=True),
        ClinicalText(keys=['image'], question='Predict the lung cancer risk over six years.', clinical_text='No patient information available.', data_name='cancer_risk', allow_missing_keys=True),
        # #transforms.Rotate90d(keys=["image", "mask"], k=1, spatial_axes=(1,2), allow_missing_keys=True),
        transforms.ScaleIntensityRanged(keys=['image'], a_min=-1300, a_max=150, b_min=0, b_max=1.0, clip=True),
        # # transforms.CropForegroundd(keys=['image', 'mask'], source_key='mask', margin=0),
        # # transforms.Resized(keys=['image', 'mask'], spatial_size=(128,320,448)),
        CropResized(keys=['image', 'mask'], crop_size=(320, 448, 128), lung=args.lung_side), #(width=320, height=448, depth=128)
        #transforms.Rotate90d(keys=["image"], k=2, spatial_axes=(1,2), allow_missing_keys=True),
        #transforms.Flipd(keys=['image'], spatial_axis=2, allow_missing_keys=True),
        #transforms.ResizeWithPadOrCropd(keys=['image', 'mask'], spatial_size=(448, 320, 128), mode='constant', allow_missing_keys=True),
        transforms.Transposed(keys=["image", "mask"], indices=(0, 3, 2, 1), allow_missing_keys=True),
        #SizeEmbed(keys=['image'], allow_missing_keys=True),
        CreateOutcomeFields(keys=['y_seq', 'y_mask'], allow_missing_keys=True),  # Create golds and censors
        transforms.ToTensord(keys=['image', 'mask', 'questions_ids', 'questions_mask', 'txt_ids', 'txt_mask', 'data_size'], allow_missing_keys=True),
    ])


    with open('/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene_modified.json', 'r') as f:
        data_dict = json.load(f)
    # data_dict = data_dict['testing']  # Use the training set
    #data_dict = data_dict[4500:4550]
    data = Dataset(data=data_dict, transform=train_transforms)
    # Use single process to enable proper error handling and debugging
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8)

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