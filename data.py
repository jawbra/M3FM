import numpy as np
import torch
from util import txt2embed, get_sincos_size_embed
from monai.data import Dataset
from torch.utils.data import DataLoader
from monai import transforms
from monai.transforms import MapTransform
import torch
import numpy as np
from monai.transforms import MapTransform
import json

class AddPixelSpacingd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        # Get spacing from image metadata
        # if "image_meta_dict" in d:
        #     spacing = d["image_meta_dict"].get("spacing", [1.0, 1.0, 1.0]) d[1.4 , 1.4 , 2.5]
        #     # MONAI loader returns spacing as (w, h, d), we need (d, h, w)
        d["pixel_size"] = [1.4,1.4,2.5] # here spacing is adjusted, since we reorient the image later in transfomrs by calling transforms.Orientationd
        return d
    
# class Add(MapTransform):
#     def __init__(self, keys, allow_missing_keys=False):
#         super().__init__(keys, allow_missing_keys)
    
#     def __call__(self, data):
#         d = dict(data)
#         # Get spacing from image metadata
#         if "image_meta_dict" in d:
#             spacing = d["image_meta_dict"].get("spacing", [1.0, 1.0, 1.0])
#             # MONAI loader returns spacing as (w, h, d), we need (d, h, w)
#             d["pixel_size"] = [spacing[2], spacing[1], spacing[0]] # here spacing is adjusted, since we reorient the image later in transfomrs by calling transforms.Orientationd
#         return d

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
                
        return d



def normalize(x, data_min, data_max):
    x = [np.clip(xi, data_min, data_max) for xi in x]
    x = [(xi - data_min) / (data_max - data_min) for xi in x]
    x = [xi * 2 - 1 for xi in x]
    return x


def to_tensor(x):
    assert isinstance(x, list)
    return [torch.from_numpy(xi.copy()).unsqueeze(0) for xi in x]


def crop_resize(x, coord, pix_size, crop_size):
    s, h, w = x.shape
    ts, th, tw = crop_size

    sl = coord[1] - coord[0]
    hl = coord[3] - coord[2]
    wl = coord[5] - coord[4]

    cs = ts
    ch = th
    cw = tw

    if sl < cs:
        ms = (cs - sl) // 2
        ss = max(0, coord[0] - ms)
        se = min(s, coord[1] + ms)
    else:
        ss = coord[0]
        se = coord[1]

    if hl < ch:
        mh = (ch - hl) // 2
        hs = max(0, coord[2] - mh)
        he = min(h, coord[3] + mh)
    else:
        hs = coord[2]
        he = coord[3]

    if wl < cw:
        mw = (cw - wl) // 2
        ws = max(0, coord[4] - mw)
        we = min(w, coord[5] + mw)
    else:
        ws = coord[4]
        we = coord[5]

    x = x[ss:se, hs:he, ws:we]

    ps = pix_size[0] * cs / ts
    ph = pix_size[1] * ch / th
    pw = pix_size[2] * cw / tw

    x = torch.nn.functional.interpolate(
        torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=(ts, th, tw),
        mode="trilinear",
        align_corners=False,
    ).numpy().squeeze()

    return x, [ps, ph, pw]




def get_data(input_dict, args):

    pix_size = input_dict['pixel_size']
    ct_path = input_dict['ct_path']
    data_name = args.data_name
    coords = input_dict['coords'][data_name[0]]
    crop_size = args.crop_size
    patch_size = args.cube_size
    hu_range = args.hu_range
    embed_dim = args.embed_dim
    question = input_dict['question']
    clinical_txt = input_dict['clinical_txt']

    data_ori = np.load(ct_path)
    data, pix_size = crop_resize(data_ori, coords, pix_size, crop_size)

    data = normalize([data], hu_range[0], hu_range[1])[0]
    data = to_tensor([data])[0]

    sizes = np.array([[pix_size[0] * patch_size[0], pix_size[1] * patch_size[1], pix_size[2] * patch_size[2]]])
    size_embed = get_sincos_size_embed(embed_dim, sizes)
    size_embed = torch.from_numpy(size_embed).to(torch.float32)

    question_ids, question_masks = txt2embed(question)
    question_ids = torch.LongTensor(question_ids).unsqueeze(0)
    question_masks = torch.LongTensor(question_masks).unsqueeze(0)

    txt_ids, txt_masks = txt2embed(clinical_txt, max_length=160)
    txt_ids = torch.LongTensor(txt_ids).unsqueeze(0)
    txt_masks = torch.LongTensor(txt_masks).unsqueeze(0)
    data_dict = {'data': data.unsqueeze(0), 'questions': question,
                 'questions_ids': question_ids.unsqueeze(0), 'questions_mask': question_masks.unsqueeze(0),
                 'data_size': torch.LongTensor(crop_size).unsqueeze(0),
                 'txt_ids': txt_ids.unsqueeze(0), 'txt_mask': txt_masks.unsqueeze(0),
                 'size_embed': size_embed.unsqueeze(0), "clinical_txt": clinical_txt,
                 'patch_size': patch_size,
                 'data_name': [data_name]}

    return data_dict


def get_dataloader(input_dict, args):
    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=['image', 'mask'], allow_missing_keys=True, image_only=False),
        AddPixelSpacingd(keys=['image']),  # Extract pixel spacing after loading
        transforms.EnsureChannelFirstd(keys=['image', 'mask'], allow_missing_keys=True),
        transforms.Orientationd(keys=["image", "mask"], allow_missing_keys=True, axcodes="RAI"),
        transforms.Transposed(keys=["image", "mask"], indices=(0, 3, 1, 2), allow_missing_keys=True),
        transforms.Rotate90d(keys=["image", "mask"], k=1, spatial_axes=(1,2), allow_missing_keys=True),
        transforms.ScaleIntensityRanged(keys=['image'], a_min=-1300, a_max=150, b_min=-1.0, b_max=1.0, clip=True),
        # transforms.CropForegroundd(keys=['image', 'mask'], source_key='mask', margin=0),
        # transforms.Resized(keys=['image', 'mask'], spatial_size=(128,320,448)),
        CropResized(keys=['image', 'mask'], crop_size=(128, 448, 320), lung='left'),
        #transforms.Rotate90d(keys=["image", "mask"], k=1, spatial_axes=(1,2), allow_missing_keys=True),
        transforms.ToTensord(keys=['image', 'mask'], allow_missing_keys=True),
    ])


    with open('/home/brandtj/Documents/projects/iderha/M3FM/data/dataset.json', 'r') as f:
        data_dict = json.load(f)
    data_dict = data_dict['testing']  # Use the training set
    data = Dataset(data=data_dict, transform=train_transforms)
    loader = DataLoader(data, batch_size=1)

    return loader, input_dict


    # data_dict = {'data': data.unsqueeze(0), 'questions': question,
    #              'questions_ids': question_ids.unsqueeze(0), 'questions_mask': question_masks.unsqueeze(0),
    #              'data_size': torch.LongTensor(crop_size).unsqueeze(0),
    #              'txt_ids': txt_ids.unsqueeze(0), 'txt_mask': txt_masks.unsqueeze(0),
    #              'size_embed': size_embed.unsqueeze(0), "clinical_txt": clinical_txt,
    #              'patch_size': patch_size,
    #              'data_name': [data_name]}