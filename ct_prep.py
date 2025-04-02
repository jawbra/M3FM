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
        return d

class ExtractPixelSpacingd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        # Get spacing from image metadata
        if "image_meta_dict" in d:
            spacing = d["image_meta_dict"].get("spacing", [1.0, 1.0, 1.0])
            # MONAI loader returns spacing as (w, h, d), we need (d, h, w)
            d["pixel_size"] = [spacing[2], spacing[1], spacing[0]] # here spacing is adjusted, since we reorient the image later in transfomrs by calling transforms.Orientationd
        return d

class CropResized(MapTransform):
    """
    Custom MONAI transform for cropping and resizing 3D images with lung selection
    """
    def __init__(self, keys, crop_size, lung='right', allow_missing_keys=False):
        """
        Args:
            keys: Keys to be transformed
            crop_size: Target size (depth, height, width)
            lung: Which lung to select ('right' or 'left')
            allow_missing_keys: Whether to allow missing keys
        """
        super().__init__(keys, allow_missing_keys)
        self.crop_size = crop_size
        self.lung = lung

    def compute_bbox(self, mask):
        """Compute bounding box coordinates from mask"""
        # Select appropriate lung
        mask = mask.copy()
        if self.lung == 'right':
            mask[mask < 3] = 0  # select right lung only
            mask[mask >= 1] = 1
        else:
            mask[mask > 2] = 0  # select left lung only
            mask[mask >= 1] = 1
        
        # Find non-zero points
        points = np.array(np.where(mask > 0))
        
        # Get min and max for each dimension
        mins = np.min(points, axis=1)
        maxs = np.max(points, axis=1)
        
        # Return coordinates in format [s1, s2, h1, h2, w1, w2]
        return [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]

    def crop_resize(self, x, coord, pix_size, crop_size):
                s, h, w = x.shape
                ts, th, tw = crop_size

                sl = coord[1] - coord[0]  # slice
                hl = coord[3] - coord[2]  # height
                wl = coord[5] - coord[4]  # width

                cs = ts
                ch = th
                cw = tw

                # Changed order of slicing to match coordinate order (s, h, w)
                x = x[coord[0]:coord[1], coord[2]:coord[3], coord[4]:coord[5]]

                ps = pix_size[0] * sl / ts
                ph = pix_size[1] * hl / th
                pw = pix_size[2] * wl / tw

                x = torch.nn.functional.interpolate(
                    torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(torch.float32),
                    size=(ts, th, tw),
                    mode="trilinear",
                    align_corners=False,
                ).numpy().squeeze()

                return x, [ps, ph, pw]

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if key == 'image':
                # Get the image and mask data
                img = d[key]
                mask = d['mask']
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()
                
                # Compute bounding box from mask
                coords = self.compute_bbox(mask[0])  # Assuming first channel is the mask
                
                # Get pixel size from the data dictionary
                pix_size = d.get('pixel_size', None)
                
                # Apply crop_resize
                img_transformed, new_pix_size = self.crop_resize(img[0], coords, pix_size, self.crop_size)
                
                # Update the dictionary
                d[key] = img_transformed[None]  # Add channel dimension back
                #d['pixel_size'] = new_pix_size
                d['coords'] = {'cancer_risk': coords}
                d['data_size'] = self.crop_size
                
        return d



# # Define the dataset and dataloader
# data_dict =[{'image': '/home/brandtj/Documents/projects/iderha/M3FM/nlst_data/dicom/100012/1.2.840.113654.2.55.240231128564881525363489796879328810792',
#     'mask': '/home/brandtj/Documents/projects/iderha/M3FM/nlst_data/masks/100012/1.2.840.113654.2.55.240231128564881525363489796879328810792/1.2.840.113654.2.55.240231128564881525363489796879328810792.nii.gz'},]


def get_dataloader(input_dict, args):
    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=['image', 'mask'], allow_missing_keys=True, meta_key_postfix="meta_dict", image_only=False),
        ExtractPixelSpacingd(keys=['image']),  # Extract pixel spacing after loading
        transforms.EnsureChannelFirstd(keys=['image', 'mask'], allow_missing_keys=True),
        transforms.Orientationd(keys=["image", "mask"], allow_missing_keys=True, axcodes="RAI"),
        transforms.Transposed(keys=["image", "mask"], indices=(0, 3, 1, 2), allow_missing_keys=True),
        DefineEmbedDim(keys=['image'], embed_dim=1024, allow_missing_keys=True),
        SizeEmbed(keys=['image'], allow_missing_keys=True),
        ClinicalText(keys=['image'], question='Predict the lung cancer risk over six years.', clinical_text='No patient information available.', data_name='cancer_risk', allow_missing_keys=True),
        transforms.Rotate90d(keys=["image", "mask"], k=1, spatial_axes=(1,2), allow_missing_keys=True),
        transforms.ScaleIntensityRanged(keys=['image'], a_min=-1300, a_max=150, b_min=-1.0, b_max=1.0, clip=True),
        # transforms.CropForegroundd(keys=['image', 'mask'], source_key='mask', margin=0),
        # transforms.Resized(keys=['image', 'mask'], spatial_size=(128,320,448)),
        CropResized(keys=['image', 'mask'], crop_size=(128, 448, 320), lung=args.lung_side),
        #transforms.Rotate90d(keys=["image", "mask"], k=1, spatial_axes=(1,2), allow_missing_keys=True),
        transforms.ToTensord(keys=['image', 'mask', 'questions_ids', 'questions_mask', 'txt_ids', 'txt_mask', 'data_size'], allow_missing_keys=True),
    ])


    with open('/home/brandtj/Documents/projects/iderha/M3FM/data/dataset.json', 'r') as f:
        data_dict = json.load(f)
    data_dict = data_dict['testing']  # Use the training set
    data = Dataset(data=data_dict, transform=train_transforms)
    loader = DataLoader(data, batch_size=1)

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